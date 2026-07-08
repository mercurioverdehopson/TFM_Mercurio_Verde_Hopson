import torch
import os
import logging
from onnxruntime.quantization import quantize_dynamic, QuantType

logger = logging.getLogger(__name__)

def export_and_quantize(model, fp32_filepath="tiny_unet_musdb.onnx", int8_filepath="tiny_unet_musdb_quantized.onnx", device="cpu"):
    logger.info("Iniciando el pipeline de exportacion y optimizacion del modelo")
    
    model.eval()
    
    # Descompilar modelo si fue optimizado con torch.compile()
    # torch.compile envuelve el modelo en un OptimizedModule incompatible con ONNX export
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
        logger.info("Modelo descompilado (torch.compile) para exportación ONNX.")
    
    # 1. Exportación a ONNX estándar (FP32)
    logger.info("Fase 1: Exportando modelo base a formato ONNX (FP32)...")
    # Mover a CPU para asegurar un export a ONNX y cuantización estables
    model = model.cpu()
    dummy_input = torch.randn(1, 1, 512, 528, device="cpu") # Ajustado a los 528 frames (~6 seg)
    
    try:
        torch.onnx.export(
            model, dummy_input, fp32_filepath,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['mix_spectrogram'], output_names=['stem_masks'],
            dynamic_axes={
                'mix_spectrogram': {0: 'batch_size', 3: 'time_frames'},
                'stem_masks': {0: 'batch_size', 3: 'time_frames'}
            }
        )
        logger.info(f"Modelo base FP32 exportado correctamente en: {fp32_filepath}")
    except Exception as e:
        logger.error(f"Error durante la exportacion a ONNX: {str(e)}")
        return

    # 2. Cuantización Post-Entrenamiento a INT8 mediante ONNX Runtime
    logger.info("Fase 2: Iniciando cuantizacion dinamica post-entrenamiento a INT8...")
    
    if not os.path.exists(fp32_filepath):
        logger.error("No se encontro el archivo ONNX base para cuantizar.")
        return

    try:
        # Se cuantizan los pesos de las capas lineales y convolucionales a enteros de 8 bits
        quantize_dynamic(
            model_input=fp32_filepath,
            model_output=int8_filepath,
            weight_type=QuantType.QInt8
        )
        
        # Validación de reducción de espacio en disco
        size_fp32 = os.path.getsize(fp32_filepath) / (1024 * 1024)
        size_int8 = os.path.getsize(int8_filepath) / (1024 * 1024)
        compression_ratio = (1 - (size_int8 / size_fp32)) * 100
        
        logger.info(f"Modelo cuantizado INT8 guardado correctamente en: {int8_filepath}")
        logger.info(f"Tamano original FP32: {size_fp32:.2f} MB")
        logger.info(f"Tamano comprimido INT8: {size_int8:.2f} MB")
        logger.info(f"Reduccion de la huella de memoria: {compression_ratio:.1f}%")
        
    except Exception as e:
        logger.error(f"Error durante el proceso de cuantizacion: {str(e)}")