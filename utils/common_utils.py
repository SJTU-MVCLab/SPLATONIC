import os

import numpy as np
import random
import torch
import torch.nn as nn

def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)

class ModuleTimer:
    def __init__(self, rasterizer: nn.Module, name: str = None):
        self.module = rasterizer
        self.name = name if name else rasterizer.__class__.__name__
        self.fwd_start = torch.cuda.Event(enable_timing=True)
        self.fwd_end = torch.cuda.Event(enable_timing=True)
        self.bwd_start = torch.cuda.Event(enable_timing=True)
        self.bwd_end = torch.cuda.Event(enable_timing=True)
        rasterizer.debug_event = self.bwd_end

        def hook_wrapper(event, htype):
            if htype == "module_pre":
                def hook(module, input):
                    event.record()
                return hook
            elif htype == "module_post":                
                def hook(module, input, output):
                    event.record()
                return hook
        
        rasterizer.register_forward_pre_hook(hook_wrapper(self.fwd_start, htype="module_pre"))
        rasterizer.register_forward_hook(hook_wrapper(self.fwd_end, htype="module_post"))
        rasterizer.register_full_backward_pre_hook(hook_wrapper(self.bwd_start, htype="module_pre"))

    def get_fwd_time(self):
        torch.cuda.synchronize()
        return self.fwd_start.elapsed_time(self.fwd_end)

    def get_bwd_time(self):
        torch.cuda.synchronize()
        return self.bwd_start.elapsed_time(self.bwd_end)
    
    def get_full_time(self):
        self.bwd_end.synchronize()
        fwd = self.fwd_start.elapsed_time(self.fwd_end)
        bwd = self.bwd_start.elapsed_time(self.bwd_end)
        # print(f"{self.name} - Forward Time: {fwd} ms, Backward Time: {bwd} ms")
        return fwd + bwd