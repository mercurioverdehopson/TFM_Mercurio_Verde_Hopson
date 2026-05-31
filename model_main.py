import torch
from torch.utils.data import DataLoader
import logging
import os
import datetime

# Importaciones adaptadas a las nuevas funciones y arquitecturas
from model.train import train_model
from model.export import export_and_quantize
from model.test import test_model
from model.pruning import apply_structural_pruning
from model.data import MUSDB18RandomMixDataset

# Configuración del logger principal
os.makedirs('log', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('log', f"main_pipeline_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Iniciando pipeline principal en dispositivo: {device}")
    
    ruta_dataset = r"C:\Users\mercu\Desktop\tfm\TFM_Mercurio_Verde_Hopson-app\dataset"
    
    # ==========================================
    # 1. PREPARAR DATASETS
    # ==========================================
    logging.info("Preparando Datasets...")
    
    # Train: Utiliza el split 'train' para activar el Data Augmentation (Pitch y Gain)
    dataset_train = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='train', samples_per_epoch=1000)
    
    # Validación: Utiliza el split 'test' (sin augmentations) para evaluar el overfitting rápido
    dataset_val = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='test', samples_per_epoch=200)
    
    # Test: Evaluación final estricta
    dataset_test = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='test', samples_per_epoch=100)
    
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=0)

    # ==========================================
    # 2. ENTRENAMIENTO (Con Early Stopping)
    # ==========================================
    logging.info("--- Fase 1: Entrenamiento ---")
    trained_model = train_model(
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        device=device, 
        epochs=1,       # Ajustado al estándar, el Early Stopping lo detendrá si es necesario
        patience=5
    )
    
    # ==========================================
    # 3. COMPRESIÓN: PODA ESTRUCTURAL
    # ==========================================
    logging.info("--- Fase 2: Poda Estructural ---")
    pruned_model = apply_structural_pruning(trained_model, pruning_amount=0.2)
    
    # ==========================================
    # 4. EVALUACIÓN (Métricas BSS EVAL)
    # ==========================================
    # IMPORTANTE: Evaluamos el modelo DESPUÉS de la poda para obtener 
    # las métricas reales que tendrá el modelo ligero en producción.
    logging.info("--- Fase 3: Evaluacion BSS EVAL ---")
    test_model(pruned_model, test_loader, device=device)
    
    # ==========================================
    # 5. EXPORTACIÓN Y CUANTIZACIÓN INT8
    # ==========================================
    logging.info("--- Fase 4: Exportacion y Cuantizacion INT8 ---")
    export_and_quantize(pruned_model, device=device)
    
    logging.info("Pipeline principal ejecutado con exito.")