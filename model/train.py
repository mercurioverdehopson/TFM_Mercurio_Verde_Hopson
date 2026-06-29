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


def si_sdr_loss(pred, target):
    """
    Scale-Invariant Signal-to-Distortion Ratio Loss.
    Mide directamente la distorsión en dominio temporal (audio).
    Es la métrica que más correlaciona con SDR de BSS Eval.
    
    Args:
        pred: (B, S, T) audio predicho
        target: (B, S, T) audio real
    Returns:
        Negativo del SI-SDR medio (para minimizar)
    """
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    
    # Zero-mean (Scale-Invariant)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Proyección: s_target = <pred, target> * target / ||target||^2
    dot = torch.sum(pred * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    s_target = dot * target / s_target_energy
    
    # Ruido residual
    e_noise = pred - s_target
    
    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8) + 1e-8
    )
    
    
    return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.windows = {}

    def forward(self, pred, target):
        B, S, T = pred.shape
        pred = pred.reshape(B * S, T)
        target = target.reshape(B * S, T)
        
        loss = 0.0
        for f, h, w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            if f not in self.windows or self.windows[f].device != pred.device:
                self.windows[f] = torch.hann_window(w, device=pred.device)
            
            p_stft = torch.stft(pred, n_fft=f, hop_length=h, win_length=w, window=self.windows[f], return_complex=True)
            t_stft = torch.stft(target, n_fft=f, hop_length=h, win_length=w, window=self.windows[f], return_complex=True)
            
            p_mag = torch.abs(p_stft) + 1e-7
            t_mag = torch.abs(t_stft) + 1e-7
            
            # Spectral Convergence
            sc_loss = torch.norm(t_mag - p_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-7)
            # Log Magnitude
            log_loss = F.l1_loss(torch.log(p_mag), torch.log(t_mag))
            
            loss += (sc_loss + log_loss)
            
        return loss / len(self.fft_sizes)

def reconstruct_audio(pred_log_mag, mix_phase, n_fft=1024, hop_length=256):
    """
    Reconstruye audio temporal para evaluar SI-SDR sin gradientes.
    """
    B, S, F, T = pred_log_mag.shape
    
    mag = torch.expm1(pred_log_mag * 7.0)
    
    pad = torch.zeros(B, S, 1, T, device=mag.device, dtype=mag.dtype)
    mag = torch.cat([mag, pad], dim=2)
    
    phase = mix_phase.expand(-1, S, -1, -1)
    pad_phase = torch.zeros(B, S, 1, T, device=phase.device, dtype=phase.dtype)
    phase = torch.cat([phase, pad_phase], dim=2)
    
    complex_spec = mag * torch.exp(1j * phase)
    complex_spec = complex_spec.reshape(B * S, F + 1, T)
    
    window = torch.hann_window(n_fft, device=complex_spec.device)
    audio = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                        window=window, length=T * hop_length)
    
    return audio.reshape(B, S, -1)


def prepare_batch(complex_stems, true_audio, device, randomize_stems=False):
    """
    Realiza In-Batch Stem Randomization y prepara los tensores para la GPU.
    """
    complex_stems = complex_stems.to(device, non_blocking=True)
    true_audio = true_audio.to(device, non_blocking=True)
    
    B, S, F, T = complex_stems.shape
    
    if randomize_stems and B > 1:
        # In-Batch Stem Randomization (Frankenstein mix)
        # S = 4: ['vocals', 'drums', 'bass', 'other']
        # Dejamos vocals fijos y mezclamos aleatoriamente los demás
        for i in range(1, S):
            idx = torch.randperm(B, device=device)
            complex_stems[:, i] = complex_stems[idx, i]
            true_audio[:, i] = true_audio[idx, i]
            
    # Crear mezcla sumando los stems
    complex_mix = torch.sum(complex_stems, dim=1, keepdim=True)
    
    # Separar Magnitud y Fase
    mix_mag = torch.abs(complex_mix)
    mix_phase = torch.angle(complex_mix)
    y_true_mag = torch.abs(complex_stems)
    
    # Compresión Logarítmica
    mix = torch.log1p(mix_mag) / 7.0
    true_stems = torch.log1p(y_true_mag) / 7.0
    
    return mix, true_stems, mix_phase, true_audio


def train_model(train_dataloader, val_dataloader, device, epochs=50, patience=5):
    logging.info(f"Entrenando en: {device}")
    logging.info(f"Archivo de log creado en: {log_filename}")

    # Hiperparámetros
    learning_rate = 0.001

    # Inicialización del modelo y optimizador
    model = TinyUNetMultiStem().to(device)
    
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logging.info("Modelo optimizado con torch.compile().")
        except Exception as e:
            logging.warning(f"No se pudo usar torch.compile: {e}")
            
    if device.type == 'cuda':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, fused=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Pérdida: L1, MR-STFT
    criterion_l1 = nn.L1Loss()
    criterion_mrstft = MultiResolutionSTFTLoss().to(device)

    # Variables para el control de Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model_checkpoint.pt"
    torch.save(model.state_dict(), best_model_path)

    # Inicialización de Mixed Precision (AMP) y Scheduler (Cosine Annealing Warm Restarts)
    scaler = torch.amp.GradScaler(device=device)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    # Bucle de Epochs
    for epoch in range(epochs):
        
        # ==========================================
        # FASE 1: ENTRENAMIENTO
        # ==========================================
        model.train()
        train_running_loss = 0.0
        
        # 1. Envolvemos el dataloader con tqdm
        loop_train = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}] Entreno", leave=False)
        
        for batch_idx, (complex_stems, true_audio) in enumerate(loop_train):
            mix, true_stems, mix_phase, true_audio = prepare_batch(complex_stems, true_audio, device, randomize_stems=True)

            # Forward Pass con Mixed Precision (L1 + Ortho + SI-SDR Waveform)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                l1_loss = criterion_l1(pred_stems, true_stems)
                ortho = orthogonality_loss(pred_stems)
                
                # SI-SDR Real en dominio temporal (entrenamiento)
                pred_audio = reconstruct_audio(pred_stems.float(), mix_phase.float())
                min_len = min(pred_audio.shape[-1], true_audio.shape[-1])
                si_sdr = si_sdr_loss(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
                mr_stft = criterion_mrstft(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
            
            loss = l1_loss + 0.5 * si_sdr + 0.1 * ortho + 0.5 * mr_stft
            
            # Backpropagation con AMP
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            
            # 2. Actualizamos la barra de progreso con el loss actual
            loop_train.set_postfix(loss=loss.item(), l1=l1_loss.item(), sdr=si_sdr.item(), ortho=ortho.item())

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
            for batch_idx, (complex_stems, true_audio) in enumerate(loop_val):
                mix, true_stems, mix_phase, true_audio = prepare_batch(complex_stems, true_audio, device, randomize_stems=False)
                
                # Inferir máscaras y predecir con Mixed Precision
                with torch.amp.autocast(device_type=device.type):
                    masks = model(mix)
                    mix_expanded = mix.expand_as(masks)
                    pred_stems = masks * mix_expanded
                    l1_loss = criterion_l1(pred_stems, true_stems)
                    ortho = orthogonality_loss(pred_stems)
                
                # SI-SDR Real en dominio temporal (rápido por estar en torch.no_grad)
                pred_audio = reconstruct_audio(pred_stems.float(), mix_phase.float())
                min_len = min(pred_audio.shape[-1], true_audio.shape[-1])
                si_sdr = si_sdr_loss(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
                mr_stft = criterion_mrstft(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
                
                loss = l1_loss + 0.5 * si_sdr + 0.1 * ortho + 0.5 * mr_stft
                val_running_loss += loss.item()
                
                # 2. Actualizamos la barra de progreso
                loop_val.set_postfix(loss=loss.item(), l1=l1_loss.item(), sdr=si_sdr.item())
                
        avg_val_loss = val_running_loss / len(val_dataloader)

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss Total Entreno: {avg_train_loss:.4f} | Loss Total Validación: {avg_val_loss:.4f}")
        
        # Actualizar el Scheduler (Cosine Annealing)
        scheduler.step()

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
    
    # Restauramos siempre los mejores pesos antes de devolver el modelo
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
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
        
        for batch_idx, (complex_stems, true_audio) in enumerate(loop_train):
            mix, true_stems, mix_phase, true_audio = prepare_batch(complex_stems, true_audio, device, randomize_stems=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                l1_loss = criterion(pred_stems, true_stems)
                
                # SI-SDR Real en dominio temporal (entrenamiento de Fine-Tuning)
                pred_audio = reconstruct_audio(pred_stems.float(), mix_phase.float())
                min_len = min(pred_audio.shape[-1], true_audio.shape[-1])
                si_sdr = si_sdr_loss(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
            
            # Entrenamiento (L1 + SI-SDR)
            loss = l1_loss + 0.2 * si_sdr
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            loop_train.set_postfix(loss=loss.item(), l1=l1_loss.item(), sdr=si_sdr.item())

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
            for batch_idx, (complex_stems, true_audio) in enumerate(loop_val):
                mix, true_stems, mix_phase, true_audio = prepare_batch(complex_stems, true_audio, device, randomize_stems=False)
                
                with torch.amp.autocast(device_type=device.type):
                    masks = model(mix)
                    mix_expanded = mix.expand_as(masks)
                    pred_stems = masks * mix_expanded
                    l1_loss = criterion(pred_stems, true_stems)
                
                # SI-SDR Real temporal solo en validación
                pred_audio = reconstruct_audio(pred_stems.float(), mix_phase.float())
                min_len = min(pred_audio.shape[-1], true_audio.shape[-1])
                si_sdr = si_sdr_loss(pred_audio[:, :, :min_len], true_audio[:, :, :min_len])
                
                loss = l1_loss + 0.2 * si_sdr
                val_running_loss += loss.item()
                loop_val.set_postfix(loss=loss.item(), l1=l1_loss.item(), sdr=si_sdr.item())
                
        avg_val_loss = val_running_loss / len(val_dataloader)

        logging.info(f"FT Epoch [{epoch+1}/{epochs}] - Loss Total Entreno: {avg_train_loss:.4f} | Loss Total Validación: {avg_val_loss:.4f}")
        
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
    
    return model