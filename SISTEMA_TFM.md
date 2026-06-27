# Arquitectura y Funcionamiento del Sistema (TFM)

Este documento centraliza la arquitectura, el flujo de datos y el funcionamiento matemático del sistema de separación de fuentes musicales (Music Source Separation) desarrollado para el Trabajo Fin de Máster.

> **Importante:** Este archivo actúa como fuente de la verdad del proyecto. Cualquier modificación futura en la arquitectura, hiperparámetros o pérdida debe reflejarse aquí.

---

## 1. Propósito del Sistema
El objetivo del sistema es tomar una canción mezclada (Mix) y separar sus **4 instrumentos principales (Stems)**:
1. Vocals (Voz)
2. Drums (Batería)
3. Bass (Bajo)
4. Other (Resto de instrumentos)

Para lograrlo con bajos recursos computacionales, se utiliza una red neuronal **Tiny U-Net**, diseñada para inferencia rápida y bajo consumo de memoria (parámetros reducidos).

---

## 2. Arquitectura del Modelo (`model/architecture.py`)

El núcleo del sistema es la clase `TinyUNetMultiStem`. Trabaja en el **dominio de la frecuencia** procesando Espectrogramas de Magnitud.

### 2.1. Estructura U-Net
* **Encoder (Contracción):** Reduce la resolución espacial temporal/frecuencial del espectrograma a la vez que extrae características profundas. Utiliza *Depthwise Separable Convolutions* para minimizar drásticamente el número de parámetros frente a las convoluciones estándar.
* **Bottleneck:** El punto de mayor compresión de características.
* **Decoder (Expansión):** Reconstruye la resolución original combinando la información profunda con los detalles espaciales extraídos mediante *Skip Connections* desde el Encoder.

### 2.2. Estimación de Máscaras
En lugar de predecir el audio directamente, el modelo predice **4 máscaras de filtrado** (una por instrumento) con valores entre 0 y 1 gracias a una función de activación final **Sigmoide**.
* `Stem_Estimado = Mix_Input * Máscara_Sigmoide`

---

## 3. Procesamiento de Datos (`model/data.py`)

El encargado de alimentar a la red neuronal es el `MUSDB18RandomMixDataset`.

### 3.1. Extracción y Segmentación
* Utiliza el dataset oficial **MUSDB18**.
* Extrae mezclas **coherentes** usando `track.stems` para descargar los 4 instrumentos y la mezcla del mismo instante de tiempo.
* Trabaja con segmentos de audio de **~6.13 segundos** (`time_frames = 528`), lo que permite al modelo ver patrones temporales largos (vibratos, ritmos) para diferenciar mejor los instrumentos.

### 3.2. Transformada de Fourier (STFT)
El audio crudo se convierte en Espectrogramas usando la Transformada de Fourier a Corto Plazo (STFT):
* **n_fft = 1024**, **hop_length = 512**.
* El modelo solo ingiere la **Magnitud** (en escala logarítmica) del espectrograma (`1 canal x 512 bins x 528 frames`).
* La Fase (`mix_phase`) se descarta durante el entrenamiento por ser ruidosa e inútil para la U-Net.

---

## 4. Entrenamiento y Pérdidas (`model/train.py` y `model_main.py`)

El bucle de entrenamiento está optimizado para ejecutarse velozmente en GPU.

### 4.1. Funciones de Pérdida
El error del modelo se calcula combinando dos métricas en fase `train`:
1. **L1 Loss (Error Absoluto Medio):** Compara la magnitud predicha con la magnitud real del instrumento. Es robusta frente a ruido atípico.
2. **Orthogonality Loss (Peso = 0.1):** Penaliza fuertemente el "sangrado" (interferencia) entre instrumentos. Obliga a que si un instrumento domina una frecuencia, los demás tengan valores cercanos a cero en esa misma frecuencia.

### 4.2. Validación (SI-SDR y Early Stopping)
Al terminar cada epoch de `train`, el modelo entra en fase `val` (`torch.no_grad()`):
* Se reconstruye el audio temporal usando `istft` y la fase original de la mezcla.
* Se calcula el **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)** directo sobre el audio.
* El sistema guarda el modelo (`best_model_checkpoint.pt`) **solo si el SI-SDR mejora**, garantizando la máxima pureza acústica.

### 4.3. Optimizador y Scheduler
* **Optimizador:** `AdamW`, con una tasa de aprendizaje de `1e-3` (y weight decay para evitar sobreajuste).
* **Scheduler:** `CosineAnnealingLR`. Va reduciendo suavemente la tasa de aprendizaje en forma de curva coseno para que los últimos epochs hagan un ajuste fino microscópico de los pesos.

---

## 5. Inferencia y Post-Procesamiento (`model/test.py`)

Una vez entrenado, el modelo predice las máscaras base, pero se le aplica un filtro matemático clásico para rozar la perfección.

### Filtro Wiener Suavizado
Se aplica de forma iterativa **después** de que el modelo haya escupido sus máscaras:
* Utiliza la densidad espectral de potencia para refinar las predicciones.
* **Beneficios:** Suaviza los bordes ásperos de las máscaras (reduciendo los "artefactos robóticos" y mejorando la métrica SAR) y garantiza que la suma de energía de los stems sea exactamente igual a la energía de la mezcla original.

---

## 6. Estructura de Ficheros

* `model_main.py`: Punto de entrada. Orquesta el entrenamiento base y el *Fine-tuning*.
* `model/architecture.py`: Definición de la red `TinyUNetMultiStem`.
* `model/data.py`: Clase del Dataset y lógicas de STFT e I/O de disco con `stempeg`.
* `model/train.py`: Funciones de bucle de entrenamiento, validación y cálculos de pérdida.
* `model/test.py`: Lógica de separación de pistas finales, Filtro Wiener y métricas de evaluación BSS.
* `log/`: Carpeta autogenerada con el registro de métricas y los pesos del modelo `.pt`.
