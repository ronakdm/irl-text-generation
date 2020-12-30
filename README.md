# irl-text-generation

**Joint work with [Cailin Winston](https://github.com/cailinw) and [Peter Michael](https://github.com/ptrmcl).** 
This repo contains code to run inverse reinforcement learning (IRL) applied to the problem of text generation. The full write-up is in `irl_text_generation.pdf`. This project was the response to an open-ended final project in the Generative Models course at the University of Washington. The project idea is based on the paper 'Toward Diverse Text Generation with Inverse Reinforcement Learning' https://arxiv.org/abs/1804.11258 IJCAI2018 with some additional modifications.

- [Overview](#overview)
- [System Requirements](#system-requirements)

# Overview
To reproduce the results, you run the `imagecoco/train.ipynb` notebook within a GPU-accelerated Google Colab enviornment, filling in the `TODO`s that specify save folders.

# System Requirements
## Hardware requirements
The code requires a CUDA-enabled GPU.

### Python Dependencies
The code mainly depends on the Python machine learning stack. Requirements can be found in `environment.yml`, and can be installed via `conda`.

```bash
conda create -n rl-project python=3.6
conda env update -n rl-project -f environment.yml
conda activate rl-project
```