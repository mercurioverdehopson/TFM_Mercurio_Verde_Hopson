import torch
from torch.utils.data import DataLoader
import logging
import os
import datetime

# Importaciones
from model.train import finetune_model
from model.export import export_and_quantize
from model.test import test_model
from model.pruning import apply_structural_pruning
from model.data import MUSDB18RandomMixDataset, create_leave_p_out_splits

# Configuración centralizada del logging (root logger)
os.makedirs('log', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('log', f"resume_pipeline_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Iniciando reanudación del pipeline en dispositivo: {device}")
    
    ruta_dataset = r"/workspace/dataset"
    
    # ==========================================
    # 1. PREPARAR DATASETS (Leave-15-Out)
    # ==========================================
    logging.info("Preparando Datasets...")
    
    # Mismos splits que model_main.py (misma seed → mismos índices)
    train_indices, val_indices = create_leave_p_out_splits(ruta_dataset, p=15, seed=42)
    
    dataset_train = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='train', is_training=True,
        samples_per_epoch=3000, track_indices=train_indices
    )
    dataset_val = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='train', is_training=False,
        samples_per_epoch=200, track_indices=val_indices
    )
    dataset_test = MUSDB18RandomMixDataset(
        root_dir=ruta_dataset, subset='test', is_training=False,
        samples_per_epoch=100
    )
    
    loader_kwargs = {
        'num_workers': 12,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
    }
    
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(dataset_val, batch_size=16, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(dataset_test, batch_size=16, shuffle=False, **loader_kwargs)

    # ==========================================
    # 2. CARGAR EL MODELO PRE-PODA (CHECKPOINT)
    # ==========================================
    logging.info("Cargando el checkpoint pre-poda guardado...")
    
    from model.architecture import TinyUNetMultiStem
    trained_model = TinyUNetMultiStem().to(device)
    
    # Cargar pesos (Eliminamos '_orig_mod.' por si fue guardado con torch.compile)
    state_dict = torch.load("modelo_pre_poda.pt", map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    trained_model.load_state_dict(state_dict)
    
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
    logging.info("--- Fase 3: Fine-Tuning post-poda ---")
    finetuned_model = finetune_model(
        model=pruned_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        epochs=10,
        patience=3
    )
    
    torch.save(finetuned_model.state_dict(), "modelo_post_poda.pt")
    logging.info("Checkpoint post-poda guardado: modelo_post_poda.pt")
    
    # ==========================================
    # 5. EVALUACIÓN (Métricas BSS EVAL)
    # ==========================================
    logging.info("--- Fase 4: Evaluacion BSS EVAL ---")
    test_model(finetuned_model, test_loader, device=device)
    
    # ==========================================
    # 6. EXPORTACIÓN Y CUANTIZACIÓN INT8
    # ==========================================
    logging.info("--- Fase 5: Exportacion y Cuantizacion INT8 ---")
    export_and_quantize(finetuned_model, device=device)
    
    logging.info("Pipeline de reanudacion ejecutado con exito.")
