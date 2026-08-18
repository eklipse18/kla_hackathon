from typing import Annotated

import typer
import torch
from pathlib import Path
from inference import build_espcn, build_restormer
import matplotlib.pyplot as plt
import numpy as np

def process_single_path(image_path: str) -> torch.Tensor:
    """
    Process the input image and convert it to a tensor.

    Args:
        image_path (str): Path to the input image.
    Returns:
        torch.Tensor: Processed image tensors.
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
    if path.absolute().isdir():
        raise ValueError("Trying to run on a directory. Use python standalone.py batch to run batch processing")
    raise ValueError(f"Invalid image path: {image_path}. Please provide a valid image file. (Image or numpy dump)")

def iter_image_paths(image_path: str):
    """Yield individual image file paths (files or directory contents)."""
    path = Path(image_path)
    if path.is_file():
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".npy"]:
            yield path
        else:
            raise ValueError(f"Invalid image path: {image_path}")
    elif path.is_dir():
        for file in sorted(path.iterdir()):
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".npy"]:
                yield file
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def load_single_tensor(file_path: Path) -> torch.Tensor:
    """Load one image file into a single (1, C, H, W) tensor."""
    if file_path.suffix.lower() == ".npy":
        image = np.load(file_path)
        if image.shape != (128, 128):
            raise ValueError(f"Invalid image shape: {image.shape}. Expected (128, 128).")
        return torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        from PIL import Image
        import torchvision.transforms as transforms

        image = Image.open(file_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(image).unsqueeze(0)

def run_batch(image_path: Annotated[str, typer.Argument()], save_path: Annotated[str, typer.Argument()] = "", batch_size: Annotated[int, typer.Option("--batch-size")] = 2) -> None:
    """
    Run inference on an image directory using the trained model.

    Args:
        image_path (str): Path to the input directory.
        save_path (str): Optional path to save the output directory, by default same directory + /output
        batch_size (int): Size of the batch for inference.
    """
    files = list(iter_image_paths(image_path))
    ESPCN_model = build_espcn()
    Restormer_model = build_restormer()
    DEVICE = next(ESPCN_model.parameters()).device  # Assuming both models are on the same device
    _save_path = Path(save_path) if save_path else Path(image_path).parent / 'output'
    _save_path.mkdir(exist_ok=True, parents=True)

    for i in range(0, len(files), batch_size):
        batch_files = files[i: i+batch_size]

        _input_tensor = []
        for file in batch_files:
            _input_tensor.append(load_single_tensor(file).to(DEVICE))
        input_tensor = torch.cat(_input_tensor, dim=0)
        denoised_images = Restormer_model(input_tensor)
        upscaled_images = ESPCN_model(denoised_images).detach().cpu().numpy()
        for j in range(batch_size):
            np.save(_save_path/f'{batch_files[j].stem}.npy', upscaled_images[j])

if __name__ == "__main__":
    typer.run(run_batch)