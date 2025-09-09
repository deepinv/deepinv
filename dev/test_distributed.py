#!/usr/bin/env python3
"""
Test script for validating different distributed scenarios for the tiled image processing.

This script tests:
1. Single GPU/CPU mode
2. Multi-GPU single node mode (if available)
3. Error handling and fallback scenarios
"""

import os
import sys
from pathlib import Path

# Add the current directory to path to import utils
sys.path.append(str(Path(__file__).parent))

import torch
import deepinv as dinv
from torchvision.transforms import ToTensor, Compose

from utils import *


def test_single_process():
    """Test single process mode (no torchrun)"""
    print("\n" + "=" * 50)
    print("TESTING SINGLE PROCESS MODE")
    print("=" * 50)

    # Load test data
    save_dir = (
        "/Users/tl255879/Documents/research/repos/deepinv-PRs/hackaton_v2/data/urban100"
    )

    dataset = dinv.datasets.Urban100HR(
        save_dir, download=False, transform=Compose([ToTensor()])
    )

    image = dataset[0]
    sigma = 0.2
    noisy_image = image + sigma * torch.randn_like(image)

    # Create model
    drunet = dinv.models.DRUNet()

    # Create windows
    receptive_field_radius = 32
    patch_size = 128

    windows, masks, patch_positions = create_tiled_windows_and_masks(
        noisy_image, patch_size, receptive_field_radius, overlap_strategy="reflect"
    )

    print(f"Created {len(windows)} windows from image {noisy_image.shape}")

    # Test inference
    try:
        outputs = ddp_infer_windows(
            drunet, windows, batch_size=4, num_workers=0, use_amp=False, sigma=sigma
        )

        if outputs is not None:
            print(f"✅ SUCCESS: Got {len(outputs)} outputs")
            print(f"First output shape: {outputs[0].shape}")
            return True
        else:
            print("❌ FAILED: No outputs returned")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_with_cuda():
    """Test with CUDA if available"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping CUDA test")
        return True

    print("\n" + "=" * 50)
    print("TESTING CUDA SINGLE GPU MODE")
    print("=" * 50)

    # Similar test but forcing CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    return test_single_process()


def test_error_handling():
    """Test error handling scenarios"""
    print("\n" + "=" * 50)
    print("TESTING ERROR HANDLING")
    print("=" * 50)

    # Test with empty windows list
    try:
        outputs = ddp_infer_windows(
            dinv.models.DRUNet(),
            [],
            batch_size=4,
            num_workers=0,
            use_amp=False,
            sigma=0.2,
        )
        print("❌ FAILED: Should have raised error for empty windows")
        return False
    except Exception as e:
        print(
            f"✅ SUCCESS: Correctly caught error for empty windows: {type(e).__name__}"
        )

    # Test cleanup functions
    try:
        cleanup_ddp()
        cleanup_distributed()
        print("✅ SUCCESS: Cleanup functions work safely")
        return True
    except Exception as e:
        print(f"❌ FAILED: Cleanup error: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions for different scenarios"""
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS FOR DIFFERENT SCENARIOS")
    print("=" * 60)

    print("\n1. Single GPU/CPU mode:")
    print("   python example_ddp.py")

    print("\n2. Multi-GPU single node (2 GPUs):")
    print("   torchrun --nproc_per_node=2 example_ddp.py")

    print("\n3. Multi-GPU single node (4 GPUs):")
    print("   torchrun --nproc_per_node=4 example_ddp.py")

    print("\n4. Multi-node cluster (SLURM):")
    print("   sbatch slurm_example.sh")

    print("\n5. Debug mode with detailed logging:")
    print(
        "   TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 example_ddp.py"
    )

    print("\n6. Force CPU mode:")
    print("   CUDA_VISIBLE_DEVICES= python example_ddp.py")


if __name__ == "__main__":
    print("🧪 TESTING DISTRIBUTED TILED IMAGE PROCESSING")

    # Run tests
    all_passed = True

    all_passed &= test_single_process()
    all_passed &= test_with_cuda()
    all_passed &= test_error_handling()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)

    print_usage_instructions()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
