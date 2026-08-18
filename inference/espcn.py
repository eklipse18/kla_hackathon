import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AntiAliasBlock(nn.Module): 
    def __init__(self, channels: int, kernel_size: int = 5, identity_weight: float = 0.9):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.channels = channels
 
        self.conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False
        )
 
        # Near-identity kernel: mostly a delta at the center, blended
        # with a small amount of uniform averaging.
        with torch.no_grad():
            k = kernel_size
            center = k // 2
            identity_kernel = torch.zeros(k, k)
            identity_kernel[center, center] = 1.0
 
            avg_kernel = torch.ones(k, k) / (k * k)
 
            blended = identity_weight * identity_kernel + (1 - identity_weight) * avg_kernel
            blended = blended / blended.sum()  # preserve DC gain = 1.0
 
            weight = blended.view(1, 1, k, k).repeat(channels, 1, 1, 1)
            self.conv.weight.copy_(weight)
 
    def forward(self, x):
        return self.conv(x)
 
 
class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1,
                 aa_kernel_size: int = 5, aa_identity_weight: float = 0.9):
        super(ESPCN, self).__init__()
 
        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
 
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()
 
        # Sub-pixel convolution layer
        self.conv3 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
 
        # Anti-aliasing block, applied after the sub-pixel shuffle
        self.anti_alias = AntiAliasBlock(
            channels=num_channels,
            kernel_size=aa_kernel_size,
            identity_weight=aa_identity_weight,
        )
 
    def forward(self, x):
        x = self.tanh1(self.conv1(x))
        x = self.tanh2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        x = self.anti_alias(x)
        return x

def build_espcn(scale_factor: int = 2, num_channels: int = 1):
    """
    Build the ESPCN model.

    Args:
        scale_factor (int): The upscaling factor.
        num_channels (int): Number of input channels.

    Returns:
        ESPCN: The ESPCN model.
    """
    model = ESPCN(scale_factor=scale_factor, num_channels=num_channels)
    model.load_state_dict(torch.load("models/espcn_model_restormer.pth", map_location=DEVICE))
    print("ESPCN model loaded successfully.")
    return model.to(DEVICE)