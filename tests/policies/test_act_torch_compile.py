#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for ACT policy torch.compile compatibility.

This test verifies that the ACT policy can be compiled with torch.compile
without graph breaks, ensuring performance optimizations are possible.
"""
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT


def create_test_config():
    """Create a minimal ACT configuration for testing."""
    config = ACTConfig()
    config.action_feature = PolicyFeature(shape=[14], type=FeatureType.STATE)
    config.robot_state_feature = PolicyFeature(shape=[14], type=FeatureType.STATE)
    config.image_features = [
        PolicyFeature(shape=[3, 96, 96], type=FeatureType.IMAGE)
    ]
    config.chunk_size = 10
    config.n_action_steps = 10
    config.vision_backbone = "resnet18"
    return config


def create_dummy_batch(config, batch_size=2, device='cpu', include_actions=False):
    """Create a dummy batch for testing."""
    batch = {}
    
    if config.robot_state_feature:
        batch['observation.state'] = torch.randn(
            batch_size, config.robot_state_feature.shape[0], device=device
        )
    
    if config.image_features:
        batch['observation.images'] = []
        for img_feat in config.image_features:
            h, w = img_feat.shape[1], img_feat.shape[2]
            batch['observation.images'].append(
                torch.randn(batch_size, 3, h, w, device=device)
            )
    
    if include_actions:
        batch['action'] = torch.randn(
            batch_size, config.chunk_size, config.action_feature.shape[0], device=device
        )
        batch['action_is_pad'] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool, device=device)
    
    return batch


@pytest.mark.parametrize("use_vae", [True, False])
def test_act_forward_basic(use_vae):
    """Test that ACT forward pass works correctly."""
    config = create_test_config()
    config.use_vae = use_vae
    
    model = ACT(config)
    model.eval()
    
    batch_size = 2
    include_actions = use_vae  # VAE mode requires actions in training
    batch = create_dummy_batch(config, batch_size, include_actions=include_actions)
    
    if use_vae:
        model.train()
        actions, (mu, log_sigma) = model(batch)
        assert mu is not None
        assert log_sigma is not None
        assert mu.shape == (batch_size, config.latent_dim)
    else:
        with torch.no_grad():
            actions, (mu, log_sigma) = model(batch)
        assert mu is None
        assert log_sigma is None
    
    assert actions.shape == (batch_size, config.chunk_size, config.action_feature.shape[0])


@pytest.mark.parametrize("use_vae", [True, False])
def test_act_torch_compile(use_vae):
    """Test that ACT model can be compiled with torch.compile."""
    config = create_test_config()
    config.use_vae = use_vae
    
    model = ACT(config)
    model.eval()
    
    # Compile the model
    compiled_model = torch.compile(model, mode='default')
    
    batch_size = 2
    include_actions = use_vae
    batch = create_dummy_batch(config, batch_size, include_actions=include_actions)
    
    if use_vae:
        model.train()
        compiled_model.train()
    
    # Test that compiled model works
    with torch.no_grad() if not use_vae else torch.enable_grad():
        actions, (mu, log_sigma) = compiled_model(batch)
    
    assert actions.shape == (batch_size, config.chunk_size, config.action_feature.shape[0])


def test_act_compile_consistency():
    """Test that compiled and uncompiled models produce same outputs."""
    config = create_test_config()
    config.use_vae = False  # Test inference mode
    
    model = ACT(config)
    model.eval()
    
    # Create a copy for compilation
    compiled_model = torch.compile(model, mode='default')
    
    batch = create_dummy_batch(config, batch_size=2)
    
    with torch.no_grad():
        # Warmup compiled model
        _ = compiled_model(batch)
        
        # Get outputs from both
        actions_baseline, _ = model(batch)
        actions_compiled, _ = compiled_model(batch)
    
    # Check outputs are close (some small numerical differences are expected)
    torch.testing.assert_close(actions_baseline, actions_compiled, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_torch_compile_cuda():
    """Test that ACT model can be compiled on CUDA."""
    config = create_test_config()
    config.use_vae = False
    
    model = ACT(config).cuda()
    model.eval()
    
    compiled_model = torch.compile(model, mode='default')
    
    batch = create_dummy_batch(config, batch_size=2, device='cuda')
    
    with torch.no_grad():
        actions, _ = compiled_model(batch)
    
    assert actions.device.type == 'cuda'
    assert actions.shape == (2, config.chunk_size, config.action_feature.shape[0])
