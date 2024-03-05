# Consistency_models_implement

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![arXiv](https://img.shields.io/badge/arXiv-2310.14189-<COLOR>.svg)](https://arxiv.org/abs/2310.14189) [![GitHub Repo stars](https://img.shields.io/github/stars/ltlhuuu/Consistency_models_implement?style=social) ](https://github.com/ltlhuuu/Consistency_models_implement)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is the code base for implementing consistency models. It can be used to train all kinds of consistency models.

## Different design choices

There are some different design choices between the original consistency model and the improved CT in the paper ["Improved techniques for training consistency models"](https://arxiv.org/abs/2310.14189).
![image](https://github.com/ltlhuuu/Consistency_models_implement/assets/70466570/38dbee6e-f4f4-420a-94a5-df32a2b4b501)

## Sampling
starting from an initial random noise $\hat{x}_{t_{max}}\sim \mathcal N(0,t^2_{max}I)$ , the consistency model can be used to sample a point in a single step: $\hat{x}_{t_{min}} = f_\theta(x_{t_{max}},t_{max})$ . For iterative refinement, the following algorithm can be used:
```python
def sample():
```
