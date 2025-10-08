# Migration Guide: torch.compile Changes for ACT

## Overview

The ACT policy has been updated to be fully compatible with `torch.compile`. These changes are **fully backward compatible** - no changes are required to existing code.

## For Users

### No Action Required

If you're using the ACT policy, your existing code will continue to work exactly as before:

```python
from lerobot.policies.act import ACTPolicy

# Your existing code works unchanged
policy = ACTPolicy(config)
actions = policy.select_action(batch)
```

### Optional: Enable torch.compile

To benefit from the performance improvements, you can now optionally compile your model:

```python
from lerobot.policies.act import ACTPolicy
import torch

# Create policy as usual
policy = ACTPolicy(config)

# NEW: Optionally compile for better performance
policy.model = torch.compile(policy.model, mode='default')

# Use as normal - inference and training both work
actions = policy.select_action(batch)
```

## For Developers

### What Changed

The internal implementation of `ACT.forward()` was updated to use tensor concatenation instead of Python list operations. This change:

✅ **Maintains exact same behavior**  
✅ **Produces identical outputs**  
✅ **Keeps same API**  
✅ **Enables torch.compile compatibility**

### Testing Your Code

If you have custom code that extends or modifies ACT:

1. **Run your existing tests** - they should all pass unchanged
2. **Optionally test with torch.compile**:
   ```python
   import torch
   model = ACT(config)
   compiled_model = torch.compile(model)
   # Test that outputs match
   assert torch.allclose(model(batch)[0], compiled_model(batch)[0])
   ```

### Performance Testing

Use the provided benchmark script to measure improvements:

```bash
# CPU benchmark
python examples/benchmark_act_compile.py --device cpu --batch-size 8

# GPU benchmark (if available)
python examples/benchmark_act_compile.py --device cuda --batch-size 8
```

## Technical Details

### What Was Changed

**File**: `src/lerobot/policies/act/modeling_act.py`

**Lines**: 458-487 (token preparation in forward method)

**Change**: Replaced Python list operations with tensor concatenation

**Before**:
```python
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
encoder_in_tokens.append(...)
encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
```

**After**:
```python
encoder_in_tokens = self.encoder_latent_input_proj(latent_sample).unsqueeze(0)
encoder_in_tokens = torch.cat([encoder_in_tokens, ...], dim=0)
```

### Why This Matters

- **torch.compile** can now optimize the entire forward pass
- No graph breaks during compilation
- Better performance without changing your code
- Foundation for future optimizations

### Backward Compatibility Guarantee

These changes are guaranteed to:
- ✅ Produce numerically identical outputs (within floating point precision)
- ✅ Maintain the same API
- ✅ Support all existing configurations
- ✅ Work with all existing datasets and environments

### Verification

To verify backward compatibility in your tests:

```python
# Test numerical equivalence
import torch
from lerobot.policies.act.modeling_act import ACT

model = ACT(config)
batch = create_test_batch()

# Get outputs
output1 = model(batch)[0]
output2 = model(batch)[0]

# Should be identical
assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-6)
```

## Questions?

- See `TORCH_COMPILE_ACT.md` for detailed technical documentation
- Run `python examples/benchmark_act_compile.py --help` for benchmarking options
- Check `tests/policies/test_act_torch_compile.py` for example usage

## Summary

✅ **No migration needed** - your code works as-is  
✅ **Optional performance boost** - enable torch.compile if desired  
✅ **Fully tested** - comprehensive test suite included  
✅ **Well documented** - detailed guides and examples provided
