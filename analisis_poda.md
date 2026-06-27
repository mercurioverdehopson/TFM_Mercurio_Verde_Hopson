# Análisis del Impacto de la Poda Estructural

Debido a la naturaleza del pipeline de entrenamiento diseñado para la optimización directa hacia entornos de producción, la validación del proceso de compresión del modelo se evaluó directamente sobre la función de coste (Loss L1) en el conjunto de validación, aislando así el efecto puro de la reducción paramétrica.

## Resultados de Validación (Loss L1)

Se comparó el rendimiento del modelo en su punto óptimo de convergencia original frente a su estado óptimo tras el proceso de poda estructural (10% de reducción de canales) y reajuste (*fine-tuning*).

| Fase del Pipeline | Época de parada | Loss L1 (Entrenamiento) | Loss L1 (Validación) |
| :--- | :---: | :---: | :---: |
| **Modelo Base Original** | 75 / 150 | 0.0074 | **0.0067** |
| **Modelo Podado + Fine-Tuning** | 2 / 10 | 0.0073 | **0.0071** |

### Conclusiones de la comparativa

1. **Retención de capacidad predictiva:** Tras someter al modelo a una poda estructural agresiva guiada por la norma L1 (eliminando el 10% de los mapas de características), el modelo experimentó un incremento marginal en la pérdida de validación de apenas **0.0004**. Este diferencial mínimo indica que la arquitectura poseía redundancia suficiente y que los canales podados no eran críticos para la separación espacial de los instrumentos.
2. **Eficiencia del Fine-Tuning:** La rápida convergencia del modelo podado (alcanzando su óptimo en la época 2) demuestra que el paso de *fine-tuning* con una tasa de aprendizaje reducida ($1 \times 10^{-4}$) fue altamente efectivo para redistribuir la representación del espacio latente sin destruir los pesos preentrenados, consolidando un modelo significativamente más ligero casi sin penalización en la reconstrucción del espectrograma.

Por consiguiente, se justifica plenamente la aplicación del paso de compresión: los beneficios en velocidad de inferencia y reducción de huella de memoria superan holgadamente la degradación ínfima observada en la pérdida de validación.
