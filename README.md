## Adaptive self-supervision for PINNs
This repository contains PyTorch code to run adaptive resampling strategies for the self-supervision loss term in physics informed neural networks.

## Run scripts
The config options for any PDE system can be set in config/pinns.yaml. See the default config for all possible options; specific configs for the
poisson and (steady state) advection-diffusion configs are set.

We use weights and biases to log all error metrics and visualizations (set the wandb option to true in the config)

example training launch script:
```
python train.py --config poisson-asym-sched --run_num 0
```

Inference is performed at the end of training. See --help for other run options.


