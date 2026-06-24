import torch
import os
import logging
import datetime
from onnxruntime.quantization import quantize_dynamic, QuantType

# Asegurar que la carpeta 'log' exista
os.makedirs('log', exist_ok=True)

# Generar el nombre del archivo basado en el timestamp actual
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f"export_{timestamp}.log")

# Configuración del logger para escribir en el archivo con timestamp y mostrar en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def export_and_quantize(model, fp32_filepath="tiny_unet_musdb.onnx", int8_filepath="tiny_unet_musdb_quantized.onnx", device="cpu"):
    logging.info("Iniciando el pipeline de exportacion y optimizacion del modelo")
    logging.info(f"Archivo de log creado en: {log_filename}")
    
    model.eval()
    
    # 1. Exportación a ONNX estándar (FP32)
    logging.info("Fase 1: Exportando modelo base a formato ONNX (FP32)...")
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
        logging.info(f"Modelo base FP32 exportado correctamente en: {fp32_filepath}")
    except Exception as e:
        logging.error(f"Error durante la exportacion a ONNX: {str(e)}")
        return

    # 2. Cuantización Post-Entrenamiento a INT8 mediante ONNX Runtime
    logging.info("Fase 2: Iniciando cuantizacion dinamica post-entrenamiento a INT8...")
    
    if not os.path.exists(fp32_filepath):
        logging.error("No se encontro el archivo ONNX base para cuantizar.")
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
        
        logging.info(f"Modelo cuantizado INT8 guardado correctamente en: {int8_filepath}")
        logging.info(f"Tamano original FP32: {size_fp32:.2f} MB")
        logging.info(f"Tamano comprimido INT8: {size_int8:.2f} MB")
        logging.info(f"Reduccion de la huella de memoria: {compression_ratio:.1f}%")
        
    except Exception as e:
        logging.error(f"Error durante el proceso de cuantizacion: {str(e)}")