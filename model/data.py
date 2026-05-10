import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import musdb
import random

class MUSDB18RandomMixDataset(Dataset):
    def __init__(self, root_dir, split='train', samples_per_epoch=2000):
        self.mus = musdb.DB(root=root_dir, subsets=split, is_wav=False)
        self.samples_per_epoch = samples_per_epoch
        self.instruments = ['vocals', 'drums', 'bass', 'other']
        
        self.orig_sr = 44100
        self.target_sr = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.time_frames = 256
        
        self.chunk_duration = (self.time_frames * self.hop_length) / self.target_sr 
        
        self.resample = T.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        
        # CAMBIO CLAVE: power=None devuelve un tensor de números complejos (Magnitud + Fase)
        self.stft = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)

    def __len__(self):
        return self.samples_per_epoch

    def _process_audio(self, audio_array):
        # 1. Preparar audio crudo
        audio_tensor = torch.from_numpy(audio_array).float().t()
        mono_audio = torch.mean(audio_tensor, dim=0, keepdim=True)
        resampled_audio = self.resample(mono_audio)
        
        # 2. STFT (Ahora es un tensor complejo)
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
            
        # Devolvemos el espectrograma complejo y la onda de audio real
        return complex_spec, resampled_audio

    def __getitem__(self, idx):
        stems_complex = []
        stems_audio = []
        
        # 1. Extraer fragmentos aleatorios
        for inst in self.instruments:
            track = random.choice(self.mus.tracks)
            max_start = max(0, track.duration - self.chunk_duration)
            start_time = random.uniform(0, max_start)
            
            track.chunk_start = start_time
            track.chunk_duration = self.chunk_duration
            
            complex_spec, audio_wave = self._process_audio(track.targets[inst].audio)
            stems_complex.append(complex_spec)
            stems_audio.append(audio_wave)
            
        # 2. Agrupar stems
        complex_stems_tensor = torch.cat(stems_complex, dim=0) # (4, 512, 256)
        true_audio = torch.cat(stems_audio, dim=0)             # (4, Muestras_de_audio)
        
        # 3. Crear la mezcla "Frankenstein" sumando los espectrogramas complejos
        complex_mix = torch.sum(complex_stems_tensor, dim=0, keepdim=True) # (1, 512, 256)
        
        # 4. Separar Magnitud y Fase (El paso fundamental para el Test)
        mix_mag = torch.abs(complex_mix)
        mix_phase = torch.angle(complex_mix)
        
        y_true_mag = torch.abs(complex_stems_tensor)
        
        # 5. Compresión Logarítmica (Solo a la magnitud, la fase se queda en radianes)
        x_mix = torch.log1p(mix_mag) / 10.0
        x_mix = torch.clamp(x_mix, 0.0, 1.0)
        
        y_true = torch.log1p(y_true_mag) / 10.0
        y_true = torch.clamp(y_true, 0.0, 1.0)
        
        # Retorna los 4 elementos exactos que necesita el bucle de Test
        return x_mix, y_true, mix_phase, true_audio