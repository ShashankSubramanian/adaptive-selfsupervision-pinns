## adaptive-selfsupervision-pinns
This repository contains PyTorch code to run adaptive resampling strategies for physics informed neural networks.

## training
The config options for any PDE system can be set in config/pinns.yaml
Runs can be logged to weights and biases if the wandb option is set to true
example training launch script:
```
python train.py --config poisson
```
See run.sh and --help for other run options


