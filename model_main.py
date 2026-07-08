import torch
from torch.utils.data import DataLoader
import logging
import os
import datetime

# Importaciones adaptadas a las nuevas funciones y arquitecturas
from model.train import train_model, finetune_model
from model.export import export_and_quantize
from model.test import test_model
from model.pruning import apply_structural_pruning
from model.data import MUSDB18RandomMixDataset, create_leave_p_out_splits

# Configuración centralizada del logging (root logger)
# Todos los módulos usan logging.getLogger(__name__) y heredan esta configuración.
os.makedirs('log', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('log', f"training_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Iniciando pipeline principal en dispositivo: {device}")
    
    ruta_dataset = r"/workspace/dataset"
    
    # ==========================================
    # 1. PREPARAR DATASETS (Leave-15-Out)
    # ==========================================
    logging.info("Preparando Datasets...")
    
    # Leave-15-Out: De las 100 pistas del split 'train' de MUSDB18,
    # 85 se usan para entrenar y 15 se reservan para validación.
    # Las 50 pistas del split 'test' quedan intactas para evaluación final.
    train_indices, val_indices = create_leave_p_out_splits(ruta_dataset, p=15, seed=42)
    
    # Train: Pistas de entrenamiento con augmentation activada
    dataset_train = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='train', is_training=True,
        samples_per_epoch=3000, track_indices=train_indices
    )
    
    # Validación: Pistas reservadas del train, sin augmentation, determinista
    dataset_val = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='train', is_training=False,
        samples_per_epoch=200, track_indices=val_indices
    )
    
    # Test: Split 'test' de MUSDB18 (50 pistas independientes, sin data leakage)
    dataset_test = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='test', is_training=False,
        samples_per_epoch=100
    )
    
    # Configuración agresiva para aprovechar los 57GB de RAM y la CPU AMD EPYC
    loader_kwargs = {
        'num_workers': 12,          # 12 procesos paralelos para descodificar MP4 a toda velocidad
        'pin_memory': True,         # Memoria fija para transferencia rápida CPU→GPU
        'persistent_workers': True, # Reusar procesos entre epochs (evita fork overhead)
        'prefetch_factor': 4,       # Pre-cargar 4 batches por worker (mucha RAM disponible)
    }
    
    # Batch size 16: más steps de gradiente por epoch (187 vs 16 anterior)
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(dataset_val, batch_size=16, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(dataset_test, batch_size=16, shuffle=False, **loader_kwargs)

    # ==========================================
    # 2. ENTRENAMIENTO (Con Early Stopping)
    # ==========================================
    logging.info("--- Fase 1: Entrenamiento ---")
    trained_model = train_model(
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        device=device, 
        epochs=150,     # Más epochs con scheduler más suave para convergencia real
        patience=15     # Más paciencia acorde al scheduler menos agresivo
    )
    
    # Descompilar modelo si fue optimizado con torch.compile()
    # Necesario para que la poda, test y export operen sobre el modelo original
    if hasattr(trained_model, '_orig_mod'):
        trained_model = trained_model._orig_mod
        logging.info("Modelo descompilado (torch.compile) para fases posteriores.")
    
    # Guardar checkpoint del modelo pre-poda
    torch.save(trained_model.state_dict(), "modelo_pre_poda.pt")
    logging.info("Checkpoint pre-poda guardado: modelo_pre_poda.pt")
    
    # Evaluación BSS EVAL del modelo ANTES de la poda (para comparativa en la memoria)
    logging.info("--- Evaluación Pre-Poda (BSS EVAL) ---")
    test_model(trained_model, test_loader, device=device)
    
    # ==========================================
    # 3. COMPRESIÓN: PODA ESTRUCTURAL (10%)
    # ==========================================
    logging.info("--- Fase 2: Poda Estructural (10%) ---")
    pruned_model = apply_structural_pruning(trained_model, pruning_amount=0.1)
    
    # ==========================================
    # 4. FINE-TUNING POST-PODA
    # ==========================================
    # Reajustar los pesos tras la poda con un LR reducido (0.0001)
    # para que el modelo recupere la calidad perdida al eliminar canales.
    logging.info("--- Fase 3: Fine-Tuning post-poda ---")
    finetuned_model = finetune_model(
        model=pruned_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        epochs=10,
        patience=3
    )
    
    # Guardar checkpoint del modelo post-poda
    torch.save(finetuned_model.state_dict(), "modelo_post_poda.pt")
    logging.info("Checkpoint post-poda guardado: modelo_post_poda.pt")
    
    # ==========================================
    # 5. EVALUACIÓN (Métricas BSS EVAL)
    # ==========================================
    # IMPORTANTE: Evaluamos el modelo DESPUÉS del fine-tuning para obtener 
    # las métricas reales que tendrá el modelo ligero en producción.
    logging.info("--- Fase 4: Evaluacion BSS EVAL ---")
    test_model(finetuned_model, test_loader, device=device)
    
    # ==========================================
    # 6. EXPORTACIÓN Y CUANTIZACIÓN INT8
    # ==========================================
    logging.info("--- Fase 5: Exportacion y Cuantizacion INT8 ---")
    export_and_quantize(finetuned_model, device=device)
    
    logging.info("Pipeline principal ejecutado con exito.")