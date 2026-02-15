import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)
    
def calc_ssim_shuffled_packed(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 4,
    stride: int = 4,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute SSIM on pixels selected by mask, after random shuffling and reshaping
    into a square image.

    If mask is None, treat it as full image (all pixels selected).

    Args:
        img1, img2:  (C, H, W) input images
        mask:        (H, W) boolean or 0/1 mask, or None
        window_size: SSIM Gaussian window size
        size_average: whether to return mean SSIM

    Returns:
        SSIM over shuffled and packed region
    """
    assert img1.shape == img2.shape and img1.dim() == 3
    C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype

    # Step 1: handle None as full mask
    if mask is None:
        mask = torch.ones((H, W), dtype=torch.bool, device=device)

    mask = mask.to(dtype=torch.bool, device=device)
    coords = mask.nonzero(as_tuple=False)  # (N, 2)

    N = coords.shape[0]
    if N < 4:
        raise ValueError("Too few selected pixels for SSIM computation.")

    # Step 2: shuffle the pixel indices
    perm = torch.randperm(N, device=device)
    coords = torch.cat([coords, coords[perm]], dim=0)

    # Step 3: reshape to square
    # patch_height = int(torch.floor(torch.sqrt(torch.tensor(N*2, dtype=torch.float32, device=device))))
    patch_height = 64
    patch_width = N * 2 // patch_height
    usable = patch_height * patch_width
    coords = coords[:usable]

    # Step 4: flatten pixel indices
    flat_idx = coords[:, 0] * W + coords[:, 1]  # (usable,)

    # Step 5: gather pixels and reshape
    img1_flat = img1.view(C, -1)[:, flat_idx]  # (C, usable)
    img2_flat = img2.view(C, -1)[:, flat_idx]

    img1_patch = img1_flat.view(C, patch_height, patch_width).unsqueeze(0)
    img2_patch = img2_flat.view(C, patch_height, patch_width).unsqueeze(0)

    # Step 6: compute SSIM
    loss_fn = SSIM(window_size=window_size, size_average=size_average, stride=stride)
    return loss_fn(img1_patch, img2_patch)