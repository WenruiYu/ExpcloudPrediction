import torch
import time
import psutil
import GPUtil

def get_gpu_memory():
    """Get GPU memory usage"""
    gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
    return gpu.memoryUsed, gpu.memoryTotal

def stress_test(duration_minutes=5):
    """
    Run GPU stress test for specified duration
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Please check your GPU setup.")
        return

    print(f"Starting GPU stress test for {duration_minutes} minutes...")
    print(f"Using device: {torch.cuda.get_device_name()}")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # Initial tensor size
    size = 1000
    max_size = 10000
    
    try:
        while time.time() < end_time:
            # Create random tensors
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            # Perform computations
            for _ in range(50):
                c = torch.matmul(a, b)
                c = torch.nn.functional.relu(c)
                del c
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get current GPU stats
            mem_used, mem_total = get_gpu_memory()
            
            print(f"\rMatrix size: {size}x{size} | GPU Memory: {mem_used:.1f}MB/{mem_total:.1f}MB "
                  f"({(mem_used/mem_total)*100:.1f}%) | "
                  f"Time remaining: {int(end_time - time.time())}s", end="")
            
            # Increase tensor size, but cap at max_size
            size = min(size + 500, max_size)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        torch.cuda.empty_cache()
        print("\nStress test completed")

if __name__ == "__main__":
    # Run stress test for 5 minutes
    stress_test(duration_minutes=5)
