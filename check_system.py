"""
check_system.py -- Check Your System for Fine-Tuning Readiness
===============================================================
Run this FIRST to see what hardware you have and whether PyTorch
can use your GPU. This helps you pick the right model size and
training settings in config.yaml.

Usage:
  python check_system.py
"""

import platform
import sys
import os


def main():
    print("=" * 60)
    print("  SYSTEM CHECK — Fine-Tuning Readiness")
    print("=" * 60)

    # ---- Python & OS Info ----
    print(f"\n  Python:    {sys.version.split()[0]}")
    print(f"  Platform:  {platform.platform()}")
    print(f"  Machine:   {platform.machine()}")
    print(f"  Processor: {platform.processor() or 'N/A'}")

    # ---- RAM ----
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"\n  RAM Total:     {ram.total / 1e9:.1f} GB")
        print(f"  RAM Available: {ram.available / 1e9:.1f} GB")
        print(f"  RAM Used:      {ram.percent}%")
    except ImportError:
        print("\n  RAM: (install psutil for RAM info: pip install psutil)")

    # ---- PyTorch ----
    print("\n" + "-" * 60)
    print("  PyTorch & GPU Status")
    print("-" * 60)

    has_mps = False  # default if torch isn't installed
    try:
        import torch
        print(f"\n  PyTorch version: {torch.__version__}")

        # CUDA (NVIDIA GPU)
        print(f"\n  CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:    {torch.version.cuda}")
            print(f"  GPU count:       {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            print(f"\n  --> You have an NVIDIA GPU! Use LoRA for efficient training.")
            print(f"  --> Recommended: batch_size 4, fp16 training enabled.")
        else:
            print("  (No NVIDIA GPU detected)")

        # MPS (Apple Silicon)
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"\n  MPS available:   {has_mps}")
        if has_mps:
            print("  --> Apple Silicon GPU detected!")
            print("  --> MPS can accelerate training but has some limitations.")
            print("  --> float32 is more stable than float16 on MPS.")

        # Best device
        if torch.cuda.is_available():
            device = "cuda"
        elif has_mps:
            device = "mps"
        else:
            device = "cpu"
        print(f"\n  Best device:     {device}")

        # Quick test
        print(f"\n  Running quick tensor test on '{device}'...")
        t = torch.randn(100, 100, device=device)
        result = torch.mm(t, t)
        print(f"  Tensor test:     PASSED (matrix multiply on {device})")

    except ImportError:
        print("\n  PyTorch not installed! Run: pip install -r requirements.txt")

    # ---- Transformers ----
    print("\n" + "-" * 60)
    print("  Key Libraries")
    print("-" * 60)

    libs = {
        "transformers": "transformers",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "gradio": "gradio",
        "bitsandbytes": "bitsandbytes",
    }

    for display_name, import_name in libs.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {display_name:20s} {version}")
        except ImportError:
            print(f"  {display_name:20s} NOT INSTALLED")

    # ---- Recommendations ----
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            if gpu_mem >= 15:
                print("  Your GPU can handle most models up to 3B parameters.")
                print("  Recommended models: TinyLlama-1.1B, Qwen3-0.6B, SmolLM2-1.7B")
            elif gpu_mem >= 8:
                print("  Your GPU can handle small models with LoRA.")
                print("  Recommended: TinyLlama-1.1B, Qwen3-0.6B")
            else:
                print("  Limited GPU memory. Use the smallest models.")
                print("  Recommended: Qwen3-0.6B with LoRA, batch_size: 1")
        elif has_mps:
            print("  Apple Silicon: Training works but is slower than NVIDIA GPUs.")
            print("  Recommended: Qwen3-0.6B or TinyLlama-1.1B with LoRA")
            print("  Tip: Use batch_size: 2 and epochs: 1 for faster runs")
            print("  Tip: Google Colab (free) gives you a T4 GPU — much faster!")
        else:
            print("  CPU-only: Training will be slow but works for small models.")
            print("  Recommended: Qwen3-0.6B with LoRA, batch_size: 1, epochs: 1")
            print("  Tip: Use Google Colab for free GPU access!")
    except Exception:
        print("  Install PyTorch first, then re-run this script.")

    print("=" * 60)


if __name__ == "__main__":
    main()
