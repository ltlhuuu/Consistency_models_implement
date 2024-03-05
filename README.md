# Consistency_models_implement

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![arXiv](https://img.shields.io/badge/arXiv-2310.14189-<COLOR>.svg)](https://arxiv.org/abs/2310.14189) [![GitHub Repo stars](https://img.shields.io/github/stars/ltlhuuu/Consistency_models_implement?style=social) ](https://github.com/ltlhuuu/Consistency_models_implement)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is the code base for implementing consistency models. It can be used to train all kinds of consistency models.

## Different design choices

There are some different design choices between the original consistency model and the improved CT in the paper ["Improved techniques for training consistency models"](https://arxiv.org/abs/2310.14189).
![image](https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/38dbee6e-f4f4-420a-94a5-df32a2b4b501)

## Sampling
Starting from an initial random noise $x_{t_{max}}$ $\sim \mathcal N(0,t^2_{max}I)$ , the consistency model can be used to sample a point in a single step: $x_{t_{min}}$ $= f_\theta$ $(x_{t_{max}},t_{max})$. For iterative refinement, the following algorithm can be used:
```python
def sample(state):
    ts = list(reversed(self.t_seq))
    action_shape = list(state.shape)
    action_shape[-1] = self.action_dim
    action = torch.randn(action_shape).to(device=state.device) * self.max_T
    if self.action_norm:
        action = self.max_action * torch.tanh(action)

    action = self.predict_consistency(state, action, ts[0])

    for t in ts[1:]:
        z = torch.randn_like(action)
        action = action + z * math.sqrt(t**2 - self.eps**2)
        if self.action_norm:
            action = self.max_action * torch.tanh(action)
        action = self.predict_consistency(state, action, t)

    action.clamp_(-self.max_action, self.max_action)
    return action
```
