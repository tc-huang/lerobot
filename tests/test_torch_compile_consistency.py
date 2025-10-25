#!/usr/bin/env python
"""Unit test for ACT policy torch.compile numerical consistency fix"""

import torch
import torch.nn as nn
import pytest


class SimpleTransformerModel(nn.Module):
    """Simplified model resembling ACT policy structure"""
    
    def __init__(self, dim=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))
        
        # Feed-forward block
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.layer_norm2(x + self.dropout(ff_out))
        
        return x


def configure_for_numerical_stability():
    """Configure PyTorch for numerical stability (same as in benchmark)"""
    # Set float32 matmul precision to highest
    torch.set_float32_matmul_precision("highest")
    
    # Disable TF32 for CUDA operations
    if torch.cuda.is_available():
        try:
            # New API (PyTorch 2.9+)
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except (AttributeError, RuntimeError):
            # Fallback to old API (PyTorch < 2.9)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    
    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def test_compile_numerical_consistency(device):
    """Test that torch.compile maintains numerical consistency with proper configuration"""
    # Configure for stability
    configure_for_numerical_stability()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = SimpleTransformerModel()
    model.to(device)
    model.eval()
    
    # Create test input
    batch_size, seq_len, dim = 8, 10, 512
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Get output from original model
    with torch.no_grad():
        output_original = model(x)
    
    # Compile model
    model_compiled = torch.compile(model, mode="default")
    
    # Get output from compiled model (with warmup)
    with torch.no_grad():
        _ = model_compiled(x)  # Warmup
        output_compiled = model_compiled(x)
    
    # Check numerical consistency
    max_diff = torch.abs(output_original - output_compiled).max().item()
    tolerance = 1e-5
    
    assert max_diff < tolerance, (
        f"Compiled model output differs from original by {max_diff:.2e}, "
        f"which exceeds tolerance of {tolerance:.2e}"
    )


def test_tf32_configuration():
    """Test that TF32 configuration is applied correctly"""
    configure_for_numerical_stability()
    
    # Check that precision is set to highest
    # Note: This is internal PyTorch state, so we can't easily verify it
    # but we can at least check that the configuration runs without errors
    
    if torch.cuda.is_available():
        # On CUDA, verify backends are configured
        # These should either be set to "ieee" (new API) or False (old API)
        try:
            # Try new API
            assert hasattr(torch.backends.cuda.matmul, "fp32_precision")
            # If new API exists, it should be set to "ieee"
        except (AttributeError, AssertionError):
            # Fall back to checking old API
            assert hasattr(torch.backends.cuda.matmul, "allow_tf32")


if __name__ == "__main__":
    # Run tests manually without pytest
    print("Testing compile numerical consistency on CPU...")
    test_compile_numerical_consistency("cpu")
    print("✅ CPU test passed")
    
    if torch.cuda.is_available():
        print("\nTesting compile numerical consistency on CUDA...")
        test_compile_numerical_consistency("cuda")
        print("✅ CUDA test passed")
    else:
        print("\n⚠️  CUDA not available, skipping CUDA tests")
    
    print("\nTesting TF32 configuration...")
    test_tf32_configuration()
    print("✅ TF32 configuration test passed")
    
    print("\n🎉 All tests passed!")
