import torch
import torchaudio.transforms as T
import numpy as np
import mir_eval
import os
import logging

from model.train import prepare_batch

logger = logging.getLogger(__name__)

target_sr = 22050
n_fft = 1024
hop_length = 256
istft = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

def spectrogram_to_audio(log_mag, phase):
    """Deshace la normalización, el logaritmo y aplica la iSTFT"""
    # 1. Deshacer escala (x10) y logaritmo (expm1 es la inversa de log1p)
    mag = torch.expm1(log_mag * 7.0)
    
    # 2. Re-añadir la frecuencia de Nyquist (fila de ceros)
    pad_mag = torch.zeros((mag.shape[0], 1, mag.shape[2]), dtype=mag.dtype, device=mag.device)
    mag = torch.cat([mag, pad_mag], dim=1)
    
    # phase tiene forma (1, 512, 256). Hacemos lo mismo.
    pad_phase = torch.zeros((phase.shape[0], 1, phase.shape[2]), dtype=phase.dtype, device=phase.device)
    phase = torch.cat([phase, pad_phase], dim=1)
    
    # 3. Reconstruir el espectrograma complejo (Magnitud * e^(i * Fase))
    complex_spec = mag * torch.exp(1j * phase)
    
    # 4. Inversa de Fourier para volver a onda de tiempo
    audio_wave = istft(complex_spec)
    return audio_wave

def wiener_post_process(pred_stems_log, complex_mix):
    """
    Post-procesamiento Wiener: refina la separación usando filtros 
    basados en densidad espectral de potencia (PSD).
    Reduce artefactos (mejora SAR) al suavizar las máscaras y preservar
    las relaciones de fase originales de la mezcla.
    
    Usa la mezcla compleja original directamente, evitando la pérdida de
    precisión del round-trip log1p/expm1.
    
    Args:
        pred_stems_log: (4, F, T) magnitudes predichas (dominio log-comprimido)
        complex_mix: (1, F, T) espectrograma complejo original de la mezcla
    
    Returns:
        refined_audio: (4, time_samples) audio refinado
    """
    # 1. Convertir predicciones a magnitud lineal
    pred_linear = torch.expm1(pred_stems_log * 7.0)  # (4, F, T)
    
    # 2. Filtro Wiener: W_i = |S_i|^2 / (Σ_j |S_j|^2 + ε)
    power = pred_linear ** 2                           # (4, F, T)
    power_sum = torch.sum(power, dim=0, keepdim=True) + 1e-10  # (1, F, T)
    wiener_masks = power / power_sum                   # (4, F, T) suman ~1.0
    
    # 3. Aplicar máscaras directamente al espectrograma complejo original
    # Esto preserva la fase real de la mezcla sin round-trip de compresión
    refined_complex = wiener_masks * complex_mix       # (4, F, T) broadcast
    
    # 4. Re-añadir Nyquist bin y aplicar iSTFT
    pad = torch.zeros((refined_complex.shape[0], 1, refined_complex.shape[2]),
                       dtype=refined_complex.dtype, device=refined_complex.device)
    refined_complex = torch.cat([refined_complex, pad], dim=1)
    
    refined_audio = istft(refined_complex)
    return refined_audio

def test_model(model, test_loader, device):
    logger.info("Iniciando Fase de Evaluación (Test)")
    model.eval()
    
    metrics = {'SDR': [], 'SIR': [], 'SAR': []}
    
    # Desactivamos el cálculo de gradientes para ahorrar memoria y CPU
    with torch.no_grad():
        for batch_idx, (complex_stems, true_audio) in enumerate(test_loader):
            # Mover stems complejos a device para computar la mezcla compleja original
            complex_stems_device = complex_stems.to(device, non_blocking=True)
            
            # Computar la mezcla compleja original (antes de compresión logarítmica)
            # para usarla en el filtro Wiener con máxima fidelidad
            complex_mix_original = torch.sum(complex_stems_device, dim=1, keepdim=True)
            
            # Preparar batch para el modelo (compresión logarítmica, etc.)
            mix, true_stems, mix_phase, true_audio_device = prepare_batch(
                complex_stems, true_audio, device, randomize_stems=False
            )
            
            # 1. Predicción
            masks = model(mix)
            mix_expanded = mix.expand_as(masks)
            pred_stems_mag = masks * mix_expanded
            
            # 2. Reconstrucción de Audio
            for i in range(pred_stems_mag.shape[0]): 
                est_audio = wiener_post_process(
                    pred_stems_mag[i].cpu(),
                    complex_mix_original[i].cpu()  # Mezcla compleja original (sin round-trip)
                )
                ref_audio = true_audio[i].numpy()
                est_audio = est_audio.numpy()
                
                # Igualar tamaños
                min_len = min(est_audio.shape[-1], ref_audio.shape[-1])
                est_audio = est_audio[:, :min_len]
                ref_audio = ref_audio[:, :min_len]
                
                # --- FILTRO DE SILENCIO ---
                is_silent = False
                for canal in range(ref_audio.shape[0]):
                    if np.max(np.abs(ref_audio[canal])) < 1e-5:
                        is_silent = True
                        break
                
                if is_silent:
                    # Si hay silencio, saltamos este fragmento y no lo contamos para la media
                    continue
                # ---------------------------------
                
                # Evaluamos los 4 stems juntos (solo llegará aquí si hay sonido en los 4)
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                    ref_audio, 
                    est_audio, 
                    compute_permutation=False
                )
                
                metrics['SDR'].append(np.mean(sdr))
                metrics['SIR'].append(np.mean(sir))
                metrics['SAR'].append(np.mean(sar))

            # Imprimir progreso del test (BSS eval es lento) utilizando logging
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"Test Batch [{batch_idx+1}/{len(test_loader)}] procesado...")

    # Promedios finales
    final_sdr = np.mean(metrics['SDR'])
    final_sir = np.mean(metrics['SIR'])
    final_sar = np.mean(metrics['SAR'])
    
    logger.info("==================================")
    logger.info("      RESULTADOS FINALES TEST      ")
    logger.info("==================================")
    logger.info(f"SDR (Calidad Global)       : {final_sdr:.2f} dB")
    logger.info(f"SIR (Aislamiento/Sangrado) : {final_sir:.2f} dB")
    logger.info(f"SAR (Artefactos robóticos) : {final_sar:.2f} dB")
    logger.info("==================================")