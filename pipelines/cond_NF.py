import torch
import torch.nn as nn
from nflows import distributions, flows, transforms, distributions

# Custom transform that clamps the scale parameters
class ClampedMaskedAffineAutoregressiveTransform(transforms.MaskedAffineAutoregressiveTransform):
    def __init__(self, *args, scale_clamp=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_clamp = scale_clamp

    def _autoregressive_function(self, inputs, context):
        # Compute the raw parameters (concatenated scale and shift)
        params = super()._autoregressive_function(inputs, context)
        # Split into scale and shift components
        scale, shift = torch.chunk(params, 2, dim=-1)
        # Clamp the scale values to a reasonable range
        scale = torch.clamp(scale, min=-self.scale_clamp, max=self.scale_clamp)
        # Concatenate back and return
        return torch.cat([scale, shift], dim=-1)

class ConditionalMAF(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim=64, num_layers=5, dropout_prob=0.1, scale_clamp=2.0):
        super().__init__()
        # Regularization: apply dropout to the context to encourage the use of latent noise.
        self.context_dropout = nn.Dropout(p=dropout_prob)
        
        # Build a list of flow transformations, stacking multiple layers.
        transforms_list = []
        for _ in range(num_layers):
            maf = ClampedMaskedAffineAutoregressiveTransform(
                features=input_dim,
                hidden_features=hidden_dim,
                context_features=context_dim,
                num_blocks=2,
                scale_clamp=scale_clamp  # Clamping parameter
            )
            perm = transforms.RandomPermutation(features=input_dim)
            transforms_list.extend([maf, perm])
        
        self.flow_transform = transforms.CompositeTransform(transforms_list)
        self.base_distribution = distributions.StandardNormal([input_dim])
        self.flow = flows.Flow(transform=self.flow_transform, distribution=self.base_distribution)

    def forward(self, x, context):
        # Apply dropout to the context during training
        context = self.context_dropout(context)
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples, context):
        # Sampling without dropout (set the module to eval mode during inference)
        return self.flow.sample(num_samples, context=context)
    
    def regularization_loss(self, l2_coef=1e-4):
        # Compute L2 regularization on all parameters
        l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
        return l2_coef * l2_loss