import cv2
import torch
import numpy as np


def check_gpu_availability():
    """
    Check and report GPU availability for PyTorch and OpenCV

    Returns:
        tuple: (torch_gpu_available, cv_gpu_available)
    """
    # Check PyTorch GPU
    torch_gpu_available = torch.cuda.is_available()
    if torch_gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        print(f"PyTorch GPU available: {gpu_name} (Count: {gpu_count}, Memory: {gpu_mem:.2f}GB)")
    else:
        print("PyTorch GPU not available, using CPU")

    # Check OpenCV GPU (CUDA)
    cv_gpu_available = False
    try:
        cv_gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cv_gpu_count > 0:
            cv_gpu_available = True
            print(f"OpenCV CUDA enabled devices: {cv_gpu_count}")
        else:
            print("OpenCV CUDA not available")
    except:
        print("OpenCV CUDA support not compiled")

    return torch_gpu_available, cv_gpu_available


def gpu_adaptive_threshold(img, max_value, adaptive_method, threshold_type, block_size, c, cv_gpu_available=False):
    """
    GPU-accelerated adaptive threshold if available

    Args:
        img: Input grayscale image
        max_value: Maximum value for threshold
        adaptive_method: Adaptive method (e.g., cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        threshold_type: Threshold type (e.g., cv2.THRESH_BINARY_INV)
        block_size: Block size for adaptive threshold
        c: Constant subtracted from mean
        cv_gpu_available: Whether OpenCV GPU is available

    Returns:
        Result image
    """
    if cv_gpu_available:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            # Process on GPU
            gpu_result = cv2.cuda.adaptiveThreshold(
                gpu_img, max_value, adaptive_method, threshold_type, block_size, c)

            # Download result
            result = gpu_result.download()
            return result
        except Exception as e:
            print(f"GPU threshold error: {e}, falling back to CPU")

    # CPU fallback
    return cv2.adaptiveThreshold(img, max_value, adaptive_method, threshold_type, block_size, c)


def gpu_resize(img, size, cv_gpu_available=False):
    """
    GPU-accelerated resize if available

    Args:
        img: Input image
        size: Target size (width, height)
        cv_gpu_available: Whether OpenCV GPU is available

    Returns:
        Resized image
    """
    if cv_gpu_available:
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_resized = cv2.cuda.resize(gpu_img, size)
            return gpu_resized.download()
        except Exception as e:
            print(f"GPU resize error: {e}, falling back to CPU")

    return cv2.resize(img, size)


def diagnose_gpu():
    """Run comprehensive GPU diagnostics and return results as a string"""
    results = []

    try:
        # Check PyTorch GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda

            results.append(f"PyTorch CUDA available: Yes")
            results.append(f"CUDA Version: {cuda_version}")
            results.append(f"GPU Device: {gpu_name}")
            results.append(f"GPU Count: {gpu_count}")
            results.append(f"Current GPU Device: {torch.cuda.current_device()}")

            # Test GPU memory
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                results.append(f"GPU Memory: Currently Allocated: {allocated:.2f}MB")
                results.append(f"GPU Memory: Max Allocated: {max_allocated:.2f}MB")
                results.append(f"GPU Memory: Total: {total:.2f}GB")

                # Test simple GPU operation
                try:
                    test_tensor = torch.tensor([1., 2., 3.], device='cuda')
                    results.append(f"GPU Test: Created test tensor on GPU: {test_tensor.device}")
                except Exception as e:
                    results.append(f"GPU Test Failed: {str(e)}")

            except Exception as e:
                results.append(f"GPU Memory Check Failed: {str(e)}")

        else:
            results.append("PyTorch CUDA not available")
            if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                results.append(f"PyTorch was built with CUDA: {torch.version.cuda}")

            # Check if CUDA is installed but not being found
            import subprocess
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
                results.append("NVIDIA GPU detected by system but not by PyTorch!")
                results.append("This indicates a PyTorch/CUDA version mismatch")
            except:
                results.append("NVIDIA driver tools (nvidia-smi) not found")

    except Exception as e:
        results.append(f"GPU Diagnostics failed: {str(e)}")

    return "\n".join(results)