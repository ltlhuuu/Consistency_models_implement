# Consistency_models_implement

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![arXiv](https://img.shields.io/badge/arXiv-2310.14189-<COLOR>.svg)](https://arxiv.org/abs/2310.14189) [![GitHub Repo stars](https://img.shields.io/github/stars/ltlhuuu/Consistency_models_implement?style=social) ](https://github.com/ltlhuuu/Consistency_models_implement)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is the code base for implementing consistency models. It can be used to train all kinds of consistency models.

## Different design choices

There are some different design choices between the original consistency model and the improved CT in the paper ["Improved techniques for training consistency models"](https://arxiv.org/abs/2310.14189).
![image](https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/38dbee6e-f4f4-420a-94a5-df32a2b4b501)

## Train the consistency model
Before we use the consistency model, we should train the consistency model. Specifically, given a data point $x$, we can generate a pair of adjacement data points $(\hat{x}^\phi_{t_{n}},$x_{t_{n+1}})$ on the PF ODE trajectory efficiently by sampling $x$ from the dataset, followed by sampling $x_{t_{n+1}$ from the transition density of the SDE $\mathcal N(x, t^2_{n+1}I)$, and then computing $\hat{x}^\phi_{t_{n}}$ using one discretization step of the numerical ODE solver according to: 
<center>
  <img src="https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/fc7d1102-679b-4446-a8c3-590517039c95" width="500">
</center>
Afterwards, we train the consistency model by minimizing its output differences on the pair $(\hat{x}^\phi_{t_{n}},$x_{t_{n+1}})$. This motivates our following consistency distillation loss for training consistency models. The consistency distillation loss is defined as:

<center>
  <img src="https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/1265fcac-fb4d-46db-ad64-ee999bb72467" width="500">
</center>
```python
def loss(self, state, action, z, t1, t2, ema_model=None, weights=torch.tensor(1.0)):
    x2 = action + z * t2
    if self.action_norm:
        x2 = self.max_action * torch.tanh(x2)
    x2 = self.predict_consistency(state, x2, t2)

    with torch.no_grad():
        x1 = action + z * t1
        if self.action_norm:
            x1 = self.max_action * torch.tanh(x1)
        if ema_model is None:
            x1 = self.predict_consistency(state, x1, t1)
        else:
            x1 = ema_model(state, x1, t1)
    loss = self.loss_fn(x2, x1, weights, take_mean=False)

    return loss
```

## Sample from the consistency model
Starting from an initial random noise $x_{t_{max}}$ $\sim \mathcal N(0,t^2_{max}I)$, the consistency model can be used to sample a point in a single-step: $x_{t_{min}}$ $= f_\theta$ $(x_{t_{max}},t_{max})$. Importantly, one can also evaluate the consistency model multiple times by alternating denoising and noise injection steps for improved sample quality. Summarized in Algorithm 1, this multistep sampling procedure provides the flexibility to trade compute for sample quality.
For iterative refinement, the following algorithm can be used:

<center>
  <img src="https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/3b0f4d0d-f042-4abe-a3ba-d5eaa4ba795b" width="500">
</center>




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
