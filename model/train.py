import torch
import torch.nn as nn
import torch.optim as optim
from model.architecture import TinyUNetMultiStem

# Ahora la función recibe el dataloader, el device y las epochs desde el main
def train_model(dataloader, device, epochs=50):
    print(f"Entrenando en: {device}")

    # Hiperparámetros
    learning_rate = 0.001

    # Inicialización del modelo y optimizador
    model = TinyUNetMultiStem().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss() # Error Absoluto Medio (MAE)

    # Bucle de Epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # IMPORTANTE: Ahora el dataloader devuelve 4 cosas. 
        # Ponemos '_' en la fase y el audio real porque no los usamos para entrenar, solo en el test.
        for batch_idx, (mix, true_stems, _, _) in enumerate(dataloader):
            mix, true_stems = mix.to(device), true_stems.to(device)

            # 1. Forward Pass (Obtener las 4 máscaras)
            optimizer.zero_grad()
            masks = model(mix)
            
            # 2. Multiplicar las máscaras por la mezcla para estimar los stems
            mix_expanded = mix.expand_as(masks)
            pred_stems = masks * mix_expanded
            
            # 3. Calcular la pérdida (L1)
            loss = criterion(pred_stems, true_stems)
            
            # 4. Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss L1: {running_loss/len(dataloader):.4f}")

    print("Entrenamiento completado.")
    return model