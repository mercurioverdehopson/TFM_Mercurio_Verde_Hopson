import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import logging
from model.architecture import TinyUNetMultiStem

logger = logging.getLogger(__name__)

def apply_structural_pruning(model, pruning_amount=0.2):
    """
    Aplica poda estructural orientada a canales (L1-norm) en las capas convolucionales
    del codificador y decodificador según lo especificado en la metodología del TFM.
    
    Nota: Se saltan las convoluciones depthwise (groups > 1) porque podar sus canales
    sin ajustar coherentemente las pointwise asociadas crearía inconsistencias dimensionales.
    También se salta la capa de salida para no alterar los 4 canales de stems musicales,
    y capas con pocos canales (< 8) para no eliminar demasiada capacidad.
    """
    logger.info(f"Iniciando el pipeline de poda estructural (Cantidad: {pruning_amount * 100}%)")
    
    pruned_layers_count = 0
    skipped_layers = []
    
    # Recorrer todos los módulos para buscar capas convolucionales de los bloques DoubleConv
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # No podamos la capa de salida final para no alterar los 4 canales de stems musicales
            if 'out_conv' in name:
                skipped_layers.append((name, "capa de salida"))
                continue
            
            # No podamos convoluciones depthwise (groups > 1)
            # En Depthwise Separable Convolutions, cada canal de salida del depthwise
            # depende de un solo canal de entrada. Podarlos sin ajustar las pointwise
            # asociadas crearía inconsistencias dimensionales silenciosas.
            if module.groups > 1:
                skipped_layers.append((name, f"depthwise (groups={module.groups})"))
                continue
            
            # No podamos capas con muy pocos canales para preservar capacidad mínima
            if module.out_channels < 8:
                skipped_layers.append((name, f"pocos canales ({module.out_channels})"))
                continue
                
            logger.info(f"Aplicando poda estructural L1 en los canales de la capa: {name}")
            
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

    # Registrar capas saltadas para trazabilidad
    for name, reason in skipped_layers:
        logger.info(f"Saltando capa: {name} ({reason})")
    
    logger.info(f"Poda estructural completada con éxito. Se han podado {pruned_layers_count} capas convolucionales.")
    return model

if __name__ == "__main__":
    # Ejemplo de inicialización y prueba del script de poda
    model = TinyUNetMultiStem()
    logger.info("Modelo base cargado para el test de poda.")
    pruned_model = apply_structural_pruning(model, pruning_amount=0.2)
