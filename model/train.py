import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import os
import logging
import datetime
from tqdm import tqdm
from model.architecture import TinyUNetMultiStem

# Asegurar que la carpeta 'log' exista
os.makedirs('log', exist_ok=True)

# Generar el nombre del archivo basado en el timestamp actual
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f"training_{timestamp}.log")

# Configuración del logger para escribir en el archivo con timestamp y mostrar en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class MultiResolutionSTFTLoss(nn.Module):
    """
    Pérdida multi-resolución STFT: evalúa la calidad de separación
    a distintas escalas temporales/frecuenciales.
    Combina Spectral Convergence + Log Magnitude Loss.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], 
                 hop_sizes=[128, 256, 512], 
                 win_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, x, y, fft_size, hop_size, win_size):
        window = torch.hann_window(win_size, device=x.device)
        x_stft = torch.stft(x, fft_size, hop_size, win_size, window, return_complex=True)
        y_stft = torch.stft(y, fft_size, hop_size, win_size, window, return_complex=True)
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # Spectral Convergence: Frobenius norm de la diferencia / norm del target
        sc_loss = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-7)
        # Log Magnitude Loss
        mag_loss = torch.mean(torch.abs(torch.log(x_mag + 1e-7) - torch.log(y_mag + 1e-7)))
        
        return sc_loss + mag_loss

    def forward(self, x, y):
        """x, y: (batch, stems, samples) en dominio del tiempo"""
        B, S, T_len = x.shape
        x_flat = x.reshape(B * S, T_len)
        y_flat = y.reshape(B * S, T_len)
        loss = 0.0
        for fs, hs, ws in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self._stft_loss(x_flat, y_flat, fs, hs, ws)
        return loss / len(self.fft_sizes)


def orthogonality_loss(pred_stems):
    """
    Penaliza el solapamiento de energía entre stems predichos.
    Mejora SIR al forzar que cada stem ocupe bins tiempo-frecuencia distintos.
    """
    B, S, F, T = pred_stems.shape
    loss = 0.0
    count = 0
    for i in range(S):
        for j in range(i + 1, S):
            loss += torch.mean(torch.abs(pred_stems[:, i] * pred_stems[:, j]))
            count += 1
    return loss / count


def train_model(train_dataloader, val_dataloader, device, epochs=50, patience=5):
    logging.info(f"Entrenando en: {device}")
    logging.info(f"Archivo de log creado en: {log_filename}")

    # Hiperparámetros
    learning_rate = 0.001

    # Inicialización del modelo y optimizador
    model = TinyUNetMultiStem().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Pérdidas: L1 (espectrograma, en GPU) + Multi-Resolution STFT (en CPU, evita NVRTC)
    criterion_l1 = nn.L1Loss()
    criterion_mrstft = MultiResolutionSTFTLoss()          # CPU: evita libnvrtc-builtins error
    istft_cpu = T.InverseSpectrogram(n_fft=1024, hop_length=256)  # CPU

    # Variables para el control de Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model_checkpoint.pt"
    torch.save(model.state_dict(), best_model_path)

    # Inicialización de Mixed Precision (AMP) y Scheduler — menos agresivo que antes
    scaler = torch.amp.GradScaler(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4)

    # Bucle de Epochs
    for epoch in range(epochs):
        
        # ==========================================
        # FASE 1: ENTRENAMIENTO
        # ==========================================
        model.train()
        train_running_loss = 0.0
        
        # 1. Envolvemos el dataloader con tqdm
        loop_train = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}] Entreno", leave=False)
        
        for batch_idx, (mix, true_stems, mix_phase, true_audio) in enumerate(loop_train):
            mix = mix.to(device, non_blocking=True)
            true_stems = true_stems.to(device, non_blocking=True)
            mix_phase = mix_phase.to(device, non_blocking=True)
            true_audio = true_audio.to(device, non_blocking=True)

            # Forward Pass con Mixed Precision (L1 Loss)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                l1_loss = criterion_l1(pred_stems, true_stems)
                ortho = orthogonality_loss(pred_stems)
            
            # Multi-Resolution STFT Loss en CPU (evita NVRTC / libnvrtc-builtins error)
            pred_stems_cpu = pred_stems.detach().float().cpu()
            mix_phase_cpu  = mix_phase.float().cpu()
            true_audio_cpu = true_audio.float().cpu()
            
            pred_mag_cpu = torch.expm1(pred_stems_cpu * 7.0)
            phase_exp_cpu = mix_phase_cpu.expand_as(pred_mag_cpu)
            # torch.complex en vez de exp(1j*...) para evitar compilación CUDA JIT
            pred_complex_cpu = torch.complex(
                pred_mag_cpu * torch.cos(phase_exp_cpu),
                pred_mag_cpu * torch.sin(phase_exp_cpu)
            )
            pred_padded_cpu = F.pad(pred_complex_cpu, (0, 0, 0, 1))
            Bc, Sc, Fbc, Tfc = pred_padded_cpu.shape
            pred_audio_cpu = istft_cpu(pred_padded_cpu.reshape(Bc*Sc, Fbc, Tfc))
            pred_audio_cpu = pred_audio_cpu.reshape(Bc, Sc, -1)
            
            min_len = min(pred_audio_cpu.shape[-1], true_audio_cpu.shape[-1])
            mrstft_loss = criterion_mrstft(
                pred_audio_cpu[..., :min_len], true_audio_cpu[..., :min_len]
            ).to(device)  # Devolver escalar a GPU para backprop
            
            loss = l1_loss + 0.5 * mrstft_loss + 0.1 * ortho
            
            # Backpropagation con AMP
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            
            # 2. Actualizamos la barra de progreso con el loss actual
            loop_train.set_postfix(loss=loss.item(), l1=l1_loss.item(), stft=mrstft_loss.item(), ortho=ortho.item())

        avg_train_loss = train_running_loss / len(train_dataloader)
        
        # Liberar memoria GPU antes de validación
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ==========================================
        # FASE 2: VALIDACIÓN
        # ==========================================
        model.eval()
        val_running_loss = 0.0
        
        # 1. Envolvemos el dataloader de validación
        loop_val = tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{epochs}] Valid", leave=False)
        
        # Desactivamos gradientes para la validación
        with torch.no_grad():
            for batch_idx, (mix, true_stems, mix_phase, true_audio) in enumerate(loop_val):
                mix = mix.to(device, non_blocking=True)
                true_stems = true_stems.to(device, non_blocking=True)
                mix_phase = mix_phase.to(device, non_blocking=True)
                true_audio = true_audio.to(device, non_blocking=True)
                
                # Inferir máscaras y predecir con Mixed Precision
                with torch.amp.autocast(device_type=device.type):
                    masks = model(mix)
                    mix_expanded = mix.expand_as(masks)
                    pred_stems = masks * mix_expanded
                    l1_loss = criterion_l1(pred_stems, true_stems)
                    ortho = orthogonality_loss(pred_stems)
                
                # Multi-Resolution STFT Loss en CPU
                pred_stems_cpu = pred_stems.float().cpu()
                mix_phase_cpu  = mix_phase.float().cpu()
                true_audio_cpu = true_audio.float().cpu()
                
                pred_mag_cpu = torch.expm1(pred_stems_cpu * 7.0)
                phase_exp_cpu = mix_phase_cpu.expand_as(pred_mag_cpu)
                pred_complex_cpu = torch.complex(
                    pred_mag_cpu * torch.cos(phase_exp_cpu),
                    pred_mag_cpu * torch.sin(phase_exp_cpu)
                )
                pred_padded_cpu = F.pad(pred_complex_cpu, (0, 0, 0, 1))
                Bc, Sc, Fbc, Tfc = pred_padded_cpu.shape
                pred_audio_cpu = istft_cpu(pred_padded_cpu.reshape(Bc*Sc, Fbc, Tfc))
                pred_audio_cpu = pred_audio_cpu.reshape(Bc, Sc, -1)
                
                min_len = min(pred_audio_cpu.shape[-1], true_audio_cpu.shape[-1])
                mrstft_loss = criterion_mrstft(
                    pred_audio_cpu[..., :min_len], true_audio_cpu[..., :min_len]
                ).to(device)
                
                loss = l1_loss + 0.5 * mrstft_loss + 0.1 * ortho
                val_running_loss += loss.item()
                
                # 2. Actualizamos la barra de progreso
                loop_val.set_postfix(loss=loss.item())
                
        avg_val_loss = val_running_loss / len(val_dataloader)

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss L1 Entreno: {avg_train_loss:.4f} | Loss L1 Validación: {avg_val_loss:.4f}")
        
        # Actualizar el Scheduler basado en el loss de validación
        scheduler.step(avg_val_loss)

        # ==========================================
        # FASE 3: EARLY STOPPING
        # ==========================================
        if avg_val_loss < best_val_loss:
            # Si hay mejora, actualizamos el mejor récord y guardamos los pesos
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info("Mejora detectada. Guardando estado del modelo...")
        else:
            # Si no hay mejora, aumentamos el contador
            epochs_no_improve += 1
            logging.info(f"Sin mejora ({epochs_no_improve}/{patience}).")
            
            # Si agotamos la paciencia, detenemos el bucle
            if epochs_no_improve >= patience:
                logging.warning(f"EARLY STOPPING ACTIVADO en la epoch {epoch+1}.")
                break

    logging.info("Entrenamiento finalizado.")
    
    # Restauramos siempre los mejores pesos antes de devolver el modelo, 
    # ya sea por Early Stopping o por haber completado todas las epochs.
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # Limpiar checkpoint temporal
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    return model


def finetune_model(model, train_dataloader, val_dataloader, device, epochs=10, patience=3):
    """
    Fine-tuning post-poda: reajusta los pesos del modelo podado con un learning rate
    reducido para recuperar la calidad perdida durante la poda estructural.
    A diferencia de train_model, NO crea un modelo nuevo, sino que recibe uno existente.
    """
    logging.info(f"Iniciando Fine-Tuning post-poda en: {device}")
    logging.info(f"Epochs: {epochs} | Patience: {patience}")

    # Learning rate reducido (10x menor que el entrenamiento original)
    # para ajustes finos sin destruir lo aprendido
    learning_rate = 0.0001

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_finetuned_checkpoint.pt"
    torch.save(model.state_dict(), best_model_path)

    scaler = torch.amp.GradScaler(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)

    for epoch in range(epochs):
        
        # ==========================================
        # FASE 1: FINE-TUNING
        # ==========================================
        model.train()
        train_running_loss = 0.0
        
        loop_train = tqdm(train_dataloader, desc=f"FT Epoch [{epoch+1}/{epochs}] Entreno", leave=False)
        
        for batch_idx, (mix, true_stems, _, _) in enumerate(loop_train):
            mix, true_stems = mix.to(device, non_blocking=True), true_stems.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                loss = criterion(pred_stems, true_stems)
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            loop_train.set_postfix(loss=loss.item())

        avg_train_loss = train_running_loss / len(train_dataloader)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ==========================================
        # FASE 2: VALIDACIÓN
        # ==========================================
        model.eval()
        val_running_loss = 0.0
        
        loop_val = tqdm(val_dataloader, desc=f"FT Epoch [{epoch+1}/{epochs}] Valid", leave=False)
        
        with torch.no_grad():
            for batch_idx, (mix, true_stems, _, _) in enumerate(loop_val):
                mix, true_stems = mix.to(device, non_blocking=True), true_stems.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type):
                    masks = model(mix)
                    mix_expanded = mix.expand_as(masks)
                    pred_stems = masks * mix_expanded
                    loss = criterion(pred_stems, true_stems)
                val_running_loss += loss.item()
                loop_val.set_postfix(loss=loss.item())
                
        avg_val_loss = val_running_loss / len(val_dataloader)

        logging.info(f"FT Epoch [{epoch+1}/{epochs}] - Loss L1 Entreno: {avg_train_loss:.4f} | Loss L1 Validación: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)

        # ==========================================
        # FASE 3: EARLY STOPPING
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info("FT: Mejora detectada. Guardando estado del modelo...")
        else:
            epochs_no_improve += 1
            logging.info(f"FT: Sin mejora ({epochs_no_improve}/{patience}).")
            
            if epochs_no_improve >= patience:
                logging.warning(f"FT: EARLY STOPPING ACTIVADO en la epoch {epoch+1}.")
                break

    logging.info("Fine-Tuning post-poda finalizado.")
    
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    return model