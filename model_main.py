from model.train import train_model
from model.export import export_to_onnx
from model.test import test_model
from model.data import MUSDB18RandomMixDataset
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preparar Datasets
    # Nota: Asegúrate de que MUSDB18RandomMixDataset devuelva también la 'fase' 
    # de la mezcla y el 'audio real' para que el test funcione.
    ruta_dataset = r"C:\Users\mercu\Desktop\tfm\TFM_Mercurio_Verde_Hopson-app\dataset"
    
    # 1. Preparar Datasets
    dataset_train = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='train', samples_per_epoch=1000)
    dataset_test  = MUSDB18RandomMixDataset(root_dir=ruta_dataset, split='test', samples_per_epoch=100)
    
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=0)
    test_loader  = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=0) # Batch más pequeño para test (Consume RAM)

    # 2. Entrenar (Usando la función que te di en mensajes anteriores)
    print("--- Iniciando Entrenamiento ---")
    trained_model = train_model(train_loader, device=device, epochs=1) # Reducido a 10 para probar
    
    # 3. Testear con las métricas
    test_model(trained_model, test_loader, device=device)
    
    # 4. Exportar
    export_to_onnx(trained_model, device=device)