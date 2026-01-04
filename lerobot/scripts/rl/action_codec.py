
import torch

from lerobot.common.policies.sac.configuration_sac import SACConfig


def encode_to_policy_action(action: torch.Tensor, cfg: SACConfig) -> torch.Tensor:
    """
    Encode from raw action space to policy action space.
    """
    action_encoded = action.abs() ** (1.0 / cfg.action_gamma) * torch.sign(action) / cfg.action_scaling
    # 0.5+\frac{1}{k}\ln\left(\frac{x}{1-x}\right)
    # action_scaled = action / cfg.action_scaling
    # action_scaled = torch.sign(action_scaled) * torch.clamp(action_scaled.abs(), 1e-6, 1.0 - 1e-6)
    # action_encoded = (0.5 + (1.0 / cfg.action_sigmoid_scaling) * torch.log((action_scaled.abs() / (1.0 - action_scaled.abs())).clamp(1e-6))) * torch.sign(action)
    # action_encoded = torch.sign(action_encoded) * torch.clamp(action_encoded.abs(), 0.0, 1.0)

    if cfg.enable_action_scaling:
        # compute norm for action[:, :3], norm shape is (batch_size, 1)
        norm = action[:, :3].norm(p=2, dim=1, keepdim=True)
        action_encoded[:, :3] = action_encoded[:, :3] / norm.clamp(1e-6)
        # concat the norm back to action_encoded
        action_encoded = torch.cat(
            [action_encoded, (norm * 2 - 1) / cfg.action_scaling], dim=1
        )

    return action_encoded

def decode_from_policy_action(action: torch.Tensor, cfg: SACConfig) -> torch.Tensor:
    """
    Decode from policy action space to raw action space.
    """
    unsqueeze_action = False
    if action.dim() == 1:
        action = action.unsqueeze(0)
        unsqueeze_action = True

    action_decoded = action
    if cfg.enable_action_scaling:
        action_decoded = action_decoded[:, :-1]
        norm = action[:, :3].norm(p=2, dim=1, keepdim=True).clamp(1e-6)
        norm_desired = (action[:, -1] + 1) / 2.0  # [-1, 1] mapping to [0, 1]
        norm_desired = norm_desired.clamp(0.0, 1.0)
        action_decoded[:, :3] = action_decoded[:, :3] / norm * norm_desired

    action_decoded = action_decoded.abs() ** cfg.action_gamma * torch.sign(action_decoded) * cfg.action_scaling
    # \frac{1}{1+\exp\left(-k\left(x-0.5\right)\right)}
    # action_decoded = 1.0 / (1.0 + torch.exp(-cfg.action_sigmoid_scaling * (action_decoded.abs() - 0.5))) * torch.sign(action_decoded) * cfg.action_scaling

    if unsqueeze_action:
        action_decoded = action_decoded.squeeze(0)
    
    return action_decoded