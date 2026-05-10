import torch

def export_to_onnx(model, filepath="tiny_unet_musdb.onnx", device="cpu"):
    print("--- Exportando modelo a ONNX ---")
    model.eval()
    dummy_input = torch.randn(1, 1, 512, 256, device=device)
    
    torch.onnx.export(
        model, dummy_input, filepath,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['mix_spectrogram'], output_names=['stem_masks'],
        dynamic_axes={
            'mix_spectrogram': {0: 'batch_size', 3: 'time_frames'},
            'stem_masks': {0: 'batch_size', 3: 'time_frames'}
        }
    )
    print(f"✅ ÉXITO: Modelo exportado a {filepath}")