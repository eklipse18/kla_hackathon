from .model import RealESRGAN
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_real_esrgan(model_path: str = "../models/RealESRGAN_x2plus.pth", scale: int = 2, device: str = DEVICE) -> RealESRGAN:
    """
    Build the Real-ESRGAN model.

    Args:
        model_path (str): Path to the pre-trained model weights.
        scale (int): The upscaling factor.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        RealESRGAN: The Real-ESRGAN model.
    """
    model = RealESRGAN(device, scale)
    model.load_weights(model_path)
    print("Real-ESRGAN model loaded successfully.")
    return model