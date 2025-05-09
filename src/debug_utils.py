"""
Debug utilities for PyTorch models, specifically designed to help identify and resolve 
in-place operation issues that cause gradient computation errors.
"""

import torch
import contextlib
import functools
import warnings
from typing import Union, Any, Tuple, Dict, List, Optional, Callable

@contextlib.contextmanager
def autograd_anomaly_detection(enabled=True):
    """
    Context manager that enables PyTorch autograd anomaly detection.
    
    This helps identify the specific operation that's causing gradient computation issues.
    
    Args:
        enabled (bool): Whether to enable anomaly detection.
        
    Example:
        ```python
        with autograd_anomaly_detection():
            loss = model(input)
            loss.backward()  # Will show detailed error if there's an issue
        ```
    """
    prev_state = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(enabled)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(prev_state)

def check_tensor_integrity(tensor, name="tensor"):
    """
    Check if a tensor contains NaN or Inf values.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): Name to use in warning messages.
        
    Returns:
        bool: True if the tensor is valid, False otherwise.
    """
    if not isinstance(tensor, torch.Tensor):
        warnings.warn(f"{name} is not a tensor, it's a {type(tensor)}")
        return False
        
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        warnings.warn(f"{name} contains NaN values!")
    if has_inf:
        warnings.warn(f"{name} contains Inf values!")
        
    return not (has_nan or has_inf)

def safe_clone_for_backward(tensor):
    """
    Safely clone a tensor to ensure it can be used in backward pass.
    
    Args:
        tensor (torch.Tensor): The tensor to clone.
        
    Returns:
        torch.Tensor: Cloned tensor.
    """
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if not tensor.requires_grad:
        return tensor.detach()
    return tensor.clone()

def safe_min(tensor1, tensor2):
    """
    A safe replacement for torch.min() that avoids in-place operations.
    
    Args:
        tensor1 (torch.Tensor): First tensor
        tensor2 (torch.Tensor or float): Second tensor or scalar
        
    Returns:
        torch.Tensor: Element-wise minimum of tensor1 and tensor2
    """
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1, device=tensor2.device)
    elif not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2, device=tensor1.device)
    
    # Create a new tensor rather than modifying in-place
    return torch.min(tensor1, tensor2)

def track_tensor_versions(func):
    """
    Function decorator that tracks tensor versions during operations,
    helping to identify in-place operations.
    
    Args:
        func (callable): The function to wrap.
    
    Returns:
        callable: The wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tensor_versions = {}
        
        # Track tensor versions before function call
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                tensor_versions[f"arg_{i}"] = arg._version
        
        for key, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                tensor_versions[key] = arg._version
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Check if any tensor versions changed
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                if tensor_versions.get(f"arg_{i}") != arg._version:
                    warnings.warn(f"Tensor (arg {i}) was modified in-place during {func.__name__}")
        
        for key, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                if tensor_versions.get(key) != arg._version:
                    warnings.warn(f"Tensor ({key}) was modified in-place during {func.__name__}")
        
        return result
    return wrapper

def find_in_place_operations(model, input_tensor, grad_enabled=True):
    """
    Execute a model forward pass while tracking tensor operations to find in-place modifications.
    
    Args:
        model (torch.nn.Module): The model to analyze
        input_tensor (torch.Tensor): The input tensor
        grad_enabled (bool): Whether to enable gradients during analysis
    
    Returns:
        List[str]: Names of layers/operations that may contain in-place operations
    """
    suspicious_layers = []
    hooks = []
    
    def hook_fn(name):
        def fn(module, input_tensors, output_tensors):
            if isinstance(output_tensors, torch.Tensor):
                orig_version = output_tensors._version
                # Wait a tiny bit to see if the tensor gets modified in-place
                torch._C._sleep_for(0.0001)
                if output_tensors._version != orig_version:
                    suspicious_layers.append(name)
            return None
        return fn
    
    # Register forward hooks on all modules
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run the forward pass
    with torch.set_grad_enabled(grad_enabled):
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return suspicious_layers

def replace_inplace_ops(model):
    """
    Replace common in-place operations in a model with their out-of-place equivalents.
    
    Args:
        model (torch.nn.Module): The model to modify
    
    Returns:
        torch.nn.Module: The modified model
    """
    # ReLU with inplace=True is a common culprit
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            print(f"Converting inplace ReLU to out-of-place in {name}")
            module.inplace = False
        
        # Check for other modules with inplace parameter
        if hasattr(module, 'inplace') and module.inplace:
            print(f"Converting inplace operation to out-of-place in {name} ({type(module).__name__})")
            module.inplace = False
    
    return model

def clone_tensor_for_inplace(tensor):
    """
    Clone a tensor before performing operations that might modify it in-place.
    
    Args:
        tensor (torch.Tensor): The tensor that needs to be cloned
        
    Returns:
        torch.Tensor: A cloned tensor
    """
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
        return tensor.clone()
    return tensor

def add_clone_hooks(model):
    """
    Add hooks to automatically clone outputs of model modules.
    This helps prevent in-place operations from breaking the computation graph.
    
    Args:
        model (torch.nn.Module): The model to modify
    
    Returns:
        List[torch.utils.hooks.RemovableHandle]: List of hook handles
    """
    hooks = []
    
    def clone_output_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.requires_grad:
            return output.clone()
        elif isinstance(output, tuple):
            return tuple(o.clone() if isinstance(o, torch.Tensor) and o.requires_grad else o for o in output)
        elif isinstance(output, list):
            return [o.clone() if isinstance(o, torch.Tensor) and o.requires_grad else o for o in output]
        return output
        
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Sequential):  # Skip container modules
            hooks.append(module.register_forward_hook(clone_output_hook))
    
    return hooks