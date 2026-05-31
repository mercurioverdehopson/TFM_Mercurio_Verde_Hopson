import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import os
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
    best_model_wts = copy.deepcopy(model.state_dict())

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
            mix, true_stems = mix.to(device), true_stems.to(device)

            # Forward Pass
            optimizer.zero_grad()
            masks = model(mix)
            
            # Multiplicar las máscaras por la mezcla para estimar los stems
            mix_expanded = mix.expand_as(masks)
            pred_stems = masks * mix_expanded
            
            # Calcular la pérdida (L1) y Backpropagation
            loss = criterion(pred_stems, true_stems)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
            # 2. Actualizamos la barra de progreso con el loss actual
            loop_train.set_postfix(loss=loss.item())

        avg_train_loss = train_running_loss / len(train_dataloader)

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
                mix, true_stems = mix.to(device), true_stems.to(device)
                
                # Inferir máscaras y predecir
                masks = model(mix)
                mix_expanded = mix.expand_as(masks)
                pred_stems = masks * mix_expanded
                
                # Calcular pérdida L1 en validación
                loss = criterion(pred_stems, true_stems)
                val_running_loss += loss.item()
                
                # 2. Actualizamos la barra de progreso
                loop_val.set_postfix(loss=loss.item())
                
        avg_val_loss = val_running_loss / len(val_dataloader)

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss L1 Entreno: {avg_train_loss:.4f} | Loss L1 Validación: {avg_val_loss:.4f}")

        # ==========================================
        # FASE 3: EARLY STOPPING
        # ==========================================
        if avg_val_loss < best_val_loss:
            # Si hay mejora, actualizamos el mejor récord y guardamos los pesos
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
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
    model.load_state_dict(best_model_wts)
    
    return model