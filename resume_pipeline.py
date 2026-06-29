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
from model.data import MUSDB18RandomMixDataset

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
    # 1. PREPARAR DATASETS
    # ==========================================
    logging.info("Preparando Datasets...")
    dataset_train = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='train', samples_per_epoch=3000)
    dataset_val = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='test', samples_per_epoch=200)
    dataset_test = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='test', samples_per_epoch=100)
    
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
    
    # Instanciar el modelo vacío (Asegúrate de que esta sea la arquitectura que usaste)
    # Ejemplo: Si usas UNetD3(stems=4), debes importarlo de model.model
    # model_pre = UNetD3(stems=4).to(device) 
    
    # COMO NO TENGO CERTEZA DE CÓMO SE INICIALIZA, SE PUEDE USAR LA MISMA LÓGICA DE train_model
    # PERO PARA SIMPLIFICAR, DEBES REEMPLAZAR ESTO CON TU CLASE:
    from model.architecture import TinyUNetMultiStem
    trained_model = TinyUNetMultiStem().to(device)
    
    # Cargar pesos
    trained_model.load_state_dict(torch.load("modelo_pre_poda.pt", map_location=device))
    
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
