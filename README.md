# Consistency_models_implement

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![arXiv](https://img.shields.io/badge/arXiv-2310.14189-<COLOR>.svg)](https://arxiv.org/abs/2310.14189) [![GitHub Repo stars](https://img.shields.io/github/stars/ltlhuuu/Consistency_models_implement?style=social) ](https://github.com/ltlhuuu/Consistency_models_implement)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is the code base for implementing consistency models. It can be used to train all kinds of consistency models.

## Different design choices

There are some different design choices between the original consistency model and the improved CT in the paper ["Improved techniques for training consistency models"](https://arxiv.org/abs/2310.14189).
![image](https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/38dbee6e-f4f4-420a-94a5-df32a2b4b501)

## Sampling
Starting from an initial random noise $x_{t_{max}}$ $\sim \mathcal N(0,t^2_{max}I)$, the consistency model can be used to sample a point in a single-step: $x_{t_{min}}$ $= f_\theta$ $(x_{t_{max}},t_{max})$. Importantly, one can also evaluate the consistency model multiple times by alternating denoising and noise injection steps for improved sample quality. Summarized in Algorithm 1, this multistep sampling procedure provides the flexibility to trade compute for sample quality.
For iterative refinement, the following algorithm can be used:
<img src="https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/3b0f4d0d-f042-4abe-a3ba-d5eaa4ba795b" width="500">



```python

def sample(self, state):
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

def predict_consistency(self, state, action, t) -> torch.Tensor:
    if isinstance(t, float):
        t = (
            torch.Tensor([t] * action.shape[0], dtype=torch.float32).to(action.device).unsqueeze(1)
        )
    action_ori = action
    action = self.model(action, t, state)

    sigma_data = torch.Tensor(0.5)
    t_ = t - self.eps
    c_skip_t = sigma_data.pow(2) / (t_.pow(2) + sigma_data.pow(2))
    c_out_t = sigma_data * t_ / (sigma_data.pow(2) + t.pow(2)).pow(0.5)

    output = c_skip_t * action_ori + c_out_t * action
    if self.action_norm:
        output = self.max_action * torch.tanh(output)
    return output
```
