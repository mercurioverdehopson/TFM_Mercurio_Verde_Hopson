import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import musdb
import random

class MUSDB18RandomMixDataset(Dataset):
    def __init__(self, root_dir, split='train', samples_per_epoch=2000):
        self.mus = musdb.DB(root=root_dir, subsets=split, is_wav=True)
        self.split = split  # NUEVO: Guardamos la fase para saber cuándo aplicar augmentations
        self.samples_per_epoch = samples_per_epoch
        self.instruments = ['vocals', 'drums', 'bass', 'other']
        
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
        
        # --- NUEVO: DATA AUGMENTATION (Ganancia y Pitch) ---
        # Solo aplicamos estas transformaciones estocásticas si estamos en la fase de 'train'
        if self.split == 'train':
            # a) Variación de Ganancia: rango amplio para mayor robustez
            gain = random.uniform(0.3, 2.0)
            resampled_audio = resampled_audio * gain
            
            # Time shift eliminado: con mezclas coherentes (misma canción),
            # desplazar un stem rompería la sincronización entre instrumentos.
        # ---------------------------------------------------
        
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
        track = random.choice(self.mus.tracks)
        max_start = max(0, track.duration - self.chunk_duration)
        start_time = random.uniform(0, max_start)
        
        track.chunk_start = start_time
        track.chunk_duration = self.chunk_duration
        
        for i, inst in enumerate(self.instruments):
            complex_spec, audio_wave = self._process_audio(track.targets[inst].audio)
            stems_complex[i] = complex_spec[0]
            stems_audio[i] = audio_wave[0]
            
        # 2. Agrupar stems
        # Shape resultante: (4, 512, 352)
        complex_stems_tensor = stems_complex
        true_audio = stems_audio
        
        # 3. Crear la mezcla sumando los stems coherentes
        complex_mix = torch.sum(complex_stems_tensor, dim=0, keepdim=True) 
        
        # 4. Separar Magnitud y Fase
        mix_mag = torch.abs(complex_mix)
        mix_phase = torch.angle(complex_mix)
        y_true_mag = torch.abs(complex_stems_tensor)
        
        # 5. Compresión Logarítmica (sin clamp para preservar rango dinámico)
        x_mix = torch.log1p(mix_mag) / 7.0
        
        y_true = torch.log1p(y_true_mag) / 7.0
        
        return x_mix, y_true, mix_phase, true_audio