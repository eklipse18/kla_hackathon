from typing import Annotated

import typer
import torch
from pathlib import Path
from inference import build_espcn, build_restormer
import matplotlib.pyplot as plt
import numpy as np

def process_path(image_path: str) -> torch.Tensor:
    """
    Process the input image and convert it to a tensor.

    Args:
        image_path (str): Path to the input image.
    Returns:
        torch.Tensor: Processed image tensor.
    """
    path = Path(image_path)
    if path.absolute().is_file():
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Implement the image processing logic here
            # For example, you can use PIL or OpenCV to read the image, resize it, normalize it, and convert it to a tensor.
            # This is a placeholder implementation; replace it with your actual processing code.
            from PIL import Image
            import torchvision.transforms as transforms

            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return transform(image).unsqueeze(0)  # Add batch dimension
        elif path.suffix.lower() == ".npy":
            image = np.load(image_path)
            if image.shape != (128, 128):
                raise ValueError(f"Invalid image shape: {image.shape}. Expected shape is (128, 128).")
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            return image_tensor
    raise ValueError(f"Invalid image path: {image_path}. Please provide a valid image file. (Image or numpy dump)")

def run_inference(image_path: Annotated[str, typer.Argument()], save_path: Annotated[str, typer.Option("--save")] = "") -> None:
    """
    Run inference on a single image using the trained model.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Optional path to save the output image. If not provided, the output will not be saved.
    """
    input_tensor = process_path(image_path)  # Implement this function to process the image and convert it to a tensor

    ESPCN_model = build_espcn()
    Restormer_model = build_restormer()
    DEVICE = next(ESPCN_model.parameters()).device  # Assuming both models are on the same device
    input_tensor = input_tensor.to(DEVICE)
    denoised_image = Restormer_model(input_tensor)
    upscaled_image = ESPCN_model(denoised_image)
    # print("Input image shape:", input_tensor.shape)
    # print("Denoised image shape:", denoised_image.shape)
    # print("Upscaled image shape:", upscaled_image.shape)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Inference Result", fontsize=16)
    axes[0].imshow(upscaled_image.squeeze().detach().cpu().numpy(), cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Upscaled Image", fontsize=14)

    axes[1].imshow(input_tensor.squeeze().detach().cpu().numpy(), cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Input Image", fontsize=14)

    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)

if __name__ == "__main__":
    typer.run(run_inference)