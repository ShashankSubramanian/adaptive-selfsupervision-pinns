## Adaptive self-supervision for PINNs
This repository contains PyTorch code to run adaptive resampling strategies for the self-supervision loss term in physics informed neural networks.

## Run scripts
The config options for any PDE system can be set in config/pinns.yaml. See the default config for all possible options; specific configs for the
poisson and (steady state) advection-diffusion configs are set (see below).

We use weights and biases (wandb) to log all error metrics and visualizations (set the wandb option to true in the config).

example training launch script:
```
python train.py --config poisson-tc2
```

The four test-case configs are included in the config/pinns.yaml file. The config names are poisson-tc1, poisson-tc2, poisadv-tc3, poisadv-tc4. Change
the config to the required name to run any specific test-case.

All parameters are highlighted in the default config. Important parameters include:
```
* system                PDE system (poisson, poisadv (advection-diffusion)) 
* nx                    number of x points
* ny                    number of y points
* force_std             std of gaussian source
* force_mean            mean of gaussian source
* lr                    learning rate
* reg                   PDE regularization parameter 
* n_col                 number of collocation points
* n_val                 number of validation points
* use_grad_resample     boolean to use loss gradients for resampling
* resample              boolean to switch on adaptive sampling (R or G)
* sample_freq           frequency of sampling
* resample_schedule_T   period of cosine-annealing schedule
* res_momentum          momentum for proxy function
* diff_tensor_type      anisotropy or isotropy in diffusion ('aniso' or 'iso')
* optimizer             optimizer for the NN (lbfgs or adam)
* kill_opt              boolean (false will enable resampling for optimizer stalls)
* save_checkpoint       boolean to save the best model weights
* test_every_epoch      runs an inference after every epoch (only for logging purposes)
* max_epochs            maximum number of epochs to train
* seed                  seed for the random number generator
```

Other parameters are defaulted. See the default config for the full list.
Inference is performed at the end of training. See --help for other run options.
To run hyperparameter sweeps, we use wandb's grid sweep configuration. See sweeps/ for an example hyperparameter sweep config.


