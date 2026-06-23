import torch
import torch.nn as nn
import torch.optim as optim
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

def train_model(train_dataloader, val_dataloader, device, epochs=50, patience=5):
    logging.info(f"Entrenando en: {device}")
    logging.info(f"Archivo de log creado en: {log_filename}")

    # Hiperparámetros
    learning_rate = 0.001

    # Inicialización del modelo y optimizador
    model = TinyUNetMultiStem().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Error Absoluto Medio (MAE) tal y como dicta la metodología
    criterion = nn.L1Loss() 

    # Variables para el control de Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model_checkpoint.pt"
    torch.save(model.state_dict(), best_model_path)

    # Inicialización de Mixed Precision (AMP) y Scheduler — API moderna device-aware
    scaler = torch.amp.GradScaler(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Bucle de Epochs
    for epoch in range(epochs):
        
        # ==========================================
        # FASE 1: ENTRENAMIENTO
        # ==========================================
        model.train()
        train_running_loss = 0.0
        
        # 1. Envolvemos el dataloader con tqdm
        loop_train = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}] Entreno", leave=False)
        
        for batch_idx, (mix, true_stems, _, _) in enumerate(loop_train):
            mix, true_stems = mix.to(device, non_blocking=True), true_stems.to(device, non_blocking=True)

            # Forward Pass con Mixed Precision
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                loss = criterion(pred_stems, true_stems)
            
            # Backpropagation con AMP
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            
            # 2. Actualizamos la barra de progreso con el loss actual
            loop_train.set_postfix(loss=loss.item())

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
            for batch_idx, (mix, true_stems, _, _) in enumerate(loop_val):
                mix, true_stems = mix.to(device, non_blocking=True), true_stems.to(device, non_blocking=True)
                
                # Inferir máscaras y predecir con Mixed Precision
                with torch.amp.autocast(device_type=device.type):
                    masks = model(mix)
                    mix_expanded = mix.expand_as(masks)
                    pred_stems = masks * mix_expanded
                    loss = criterion(pred_stems, true_stems)
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_finetuned_checkpoint.pt"
    torch.save(model.state_dict(), best_model_path)

    scaler = torch.amp.GradScaler(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

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