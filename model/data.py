import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import musdb
import random
import logging

logger = logging.getLogger(__name__)

def create_leave_p_out_splits(root_dir, p=15, seed=42):
    """
    Crea los índices para Leave-P-Out a partir del split 'train' de MUSDB18.
    
    MUSDB18 tiene 100 pistas en 'train' y 50 en 'test'.
    Se reservan P pistas del train como validación, dejando el resto para entrenamiento.
    Los 50 tracks de 'test' quedan intactos como evaluación final independiente.
    
    Args:
        root_dir: Ruta al dataset MUSDB18
        p: Número de pistas a reservar para validación (default: 15 → 85 train / 15 val)
        seed: Semilla para reproducibilidad
    
    Returns:
        train_indices: Lista de índices para entrenamiento
        val_indices: Lista de índices para validación
    """
    mus = musdb.DB(root=root_dir, subsets='train', is_wav=True)
    n_tracks = len(mus.tracks)
    
    rng = random.Random(seed)
    all_indices = list(range(n_tracks))
    rng.shuffle(all_indices)
    
    val_indices = sorted(all_indices[:p])
    train_indices = sorted(all_indices[p:])
    
    logger.info(f"Leave-{p}-Out: {len(train_indices)} pistas train, {len(val_indices)} pistas val (de {n_tracks} totales)")
    logger.info(f"Índices de validación: {val_indices}")
    
    return train_indices, val_indices


class MUSDB18RandomMixDataset(Dataset):
    def __init__(self, root_dir, subset='train', is_training=True, 
                 samples_per_epoch=2000, track_indices=None):
        """
        Dataset para separación de fuentes musicales con MUSDB18.
        
        Args:
            root_dir: Ruta al dataset MUSDB18
            subset: 'train' o 'test' — qué subset de MUSDB18 cargar
            is_training: True → augmentation + chunks aleatorios; False → determinista sin augmentation
            samples_per_epoch: Número de muestras por epoch
            track_indices: Lista de índices de pistas a usar (para Leave-P-Out).
                          Si es None, usa todas las pistas del subset.
        """
        self.mus = musdb.DB(root=root_dir, subsets=subset, is_wav=True)
        self.is_training = is_training
        self.samples_per_epoch = samples_per_epoch
        self.instruments = ['vocals', 'drums', 'bass', 'other']
        
        # Seleccionar pistas específicas si se proporcionan índices (Leave-P-Out)
        if track_indices is not None:
            self.tracks = [self.mus.tracks[i] for i in track_indices]
        else:
            self.tracks = self.mus.tracks
        
        self.orig_sr = 44100
        self.target_sr = 22050
        self.n_fft = 1024
        self.hop_length = 256
        
        # Segmentación adaptada a ~6 segundos (528 frames).
        # 528 es múltiplo de 16, necesario para que los 4 MaxPool2d del Tiny U-Net funcionen.
        # (528 * 256) / 22050 = 6.13 segundos. Más contexto temporal mejora la separación.
        self.time_frames = 528 
        
        self.chunk_duration = (self.time_frames * self.hop_length) / self.target_sr 
        
        self.resample = T.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        self.stft = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)

    def __len__(self):
        return self.samples_per_epoch

    def _process_audio(self, audio_array):
        # 1. Preparar audio crudo
        audio_tensor = torch.from_numpy(audio_array).float().t()
        mono_audio = torch.mean(audio_tensor, dim=0, keepdim=True)
        resampled_audio = self.resample(mono_audio)
        
        # --- DATA AUGMENTATION (solo en entrenamiento) ---
        if self.is_training:
            # Variación de ganancia estocástica
            gain = random.uniform(0.3, 2.0)
            resampled_audio = resampled_audio * gain
            
            # Pitch Shifting (Comentado porque satura la memoria RAM con 12 workers)
            # n_steps = random.randint(-2, 2)
            # if n_steps != 0:
            #     pitch_shift = T.PitchShift(self.target_sr, n_steps=n_steps)
            #     resampled_audio = pitch_shift(resampled_audio)
            
        # 2. STFT (Tensor complejo)
        complex_spec = self.stft(resampled_audio) 
        complex_spec = complex_spec[:, :-1, :] # Descartar Nyquist (512 bandas)
        
        # 3. Recortar/Rellenar espectrograma y audio para que coincidan exactamente
        if complex_spec.shape[-1] > self.time_frames:
            complex_spec = complex_spec[:, :, :self.time_frames]
            resampled_audio = resampled_audio[:, :self.time_frames * self.hop_length]
        elif complex_spec.shape[-1] < self.time_frames:
            pad_amount = self.time_frames - complex_spec.shape[-1]
            complex_spec = torch.nn.functional.pad(complex_spec, (0, pad_amount))
            
            audio_pad = (self.time_frames * self.hop_length) - resampled_audio.shape[-1]
            resampled_audio = torch.nn.functional.pad(resampled_audio, (0, audio_pad))
            
        return complex_spec, resampled_audio

    def __getitem__(self, idx):
        # Pre-alocar tensores para evitar fragmentación de memoria
        stems_complex = torch.empty(4, 512, self.time_frames, dtype=torch.cfloat)
        stems_audio = torch.empty(4, self.time_frames * self.hop_length)
        
        # 1. Seleccionar UNA canción y UN punto temporal para todos los stems (mezcla coherente)
        if not self.is_training:
            # Generador determinista para validación/test. Garantiza que el batch 'idx'
            # SIEMPRE devuelve el mismo fragmento de audio en todos los epochs.
            rng = random.Random(idx + 42)
            track = rng.choice(self.tracks)
            max_start = max(0, track.duration - self.chunk_duration)
            start_time = rng.uniform(0, max_start)
        else:
            # Aleatoriedad total para el entrenamiento
            track = random.choice(self.tracks)
            max_start = max(0, track.duration - self.chunk_duration)
            start_time = random.uniform(0, max_start)
        
        track.chunk_start = start_time
        track.chunk_duration = self.chunk_duration
        
        # Optimización masiva de I/O: track.stems lee todos los instrumentos a la vez
        # mediante una sola llamada a ffmpeg (devuelve array de forma 5, samples, channels).
        # Índices MUSDB18: 0=mezcla, 1=drums, 2=bass, 3=other, 4=vocals
        stems_array = track.stems
        target_indices = [4, 1, 2, 3] # Orden de self.instruments: ['vocals', 'drums', 'bass', 'other']
        
        for i, stem_idx in enumerate(target_indices):
            complex_spec, audio_wave = self._process_audio(stems_array[stem_idx])
            stems_complex[i] = complex_spec[0]
            stems_audio[i] = audio_wave[0]
            
        # 2. Agrupar stems
        # Shape resultante: (4, 512, 528)
        complex_stems_tensor = stems_complex
        true_audio = stems_audio
        
        # Devolvemos los tensores complejos y el audio para ensamblar la mezcla
        # en la GPU (train.py), lo que nos permite hacer In-Batch Stem Randomization.
        return complex_stems_tensor, true_audio