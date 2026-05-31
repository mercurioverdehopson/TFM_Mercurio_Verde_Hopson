import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import logging
import datetime
from model.architecture import TinyUNetMultiStem

# Asegurar que la carpeta 'log' exista
os.makedirs('log', exist_ok=True)

# Generar el nombre del archivo basado en el timestamp actual
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f"pruning_{timestamp}.log")

# Configuración del logger para escribir en el archivo con timestamp y mostrar en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def apply_structural_pruning(model, pruning_amount=0.2):
    """
    Aplica poda estructural orientada a canales (L1-norm) en las capas convolucionales
    del codificador y decodificador según lo especificado en la metodología del TFM.
    """
    logging.info(f"Iniciando el pipeline de poda estructural (Cantidad: {pruning_amount * 100}%)")
    logging.info(f"Archivo de log creado en: {log_filename}")
    
    pruned_layers_count = 0
    
    # Recorrer todos los módulos para buscar capas convolucionales de los bloques DoubleConv
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # No podamos la capa de salida final para no alterar los 4 canales de stems musicales
            if name == "out_conv":
                logging.info(f"Saltando capa de salida: {name}")
                continue
                
            logging.info(f"Aplicando poda estructural L1 en los canales de la capa: {name}")
            
            # Poda estructural orientada a canales basada en la norma L1 (dim=0 es el canal de salida)
            prune.ln_structured(
                module, 
                name="weight", 
                amount=pruning_amount, 
                n=1, 
                dim=0
            )
            
            # Hacer la poda permanente eliminando las estructuras de máscara de PyTorch
            prune.remove(module, "weight")
            pruned_layers_count += 1

    logging.info(f"Poda estructural completada con éxito. Se han podado {pruned_layers_count} capas convolucionales.")
    return model

if __name__ == "__main__":
    # Ejemplo de inicialización y prueba del script de poda
    model = TinyUNetMultiStem()
    logging.info("Modelo base cargado para el test de poda.")
    pruned_model = apply_structural_pruning(model, pruning_amount=0.2)
