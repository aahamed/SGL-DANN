# Small Group Learning for Domain Adaptation

* This project applies the small group learning framework to the problem of domain adaptation. In this  
implementation each learner is an adversarial network.

SGL paper: https://arxiv.org/abs/2012.12502
DANN paper: https://arxiv.org/abs/1409.7495

## Usage

* Configuration for the experiment are passed as arguments to the train_search_sgl.py driver file.
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

### Search on MNIST ( src domain ) to MNIST-M ( target domain )
* Perform small group learning with DANN model.  
* Note, only first order search is supported.
```
python train_search_da.py --epochs=5 --pretrain_steps=2  --batch_size=32 --src_set=mnist --tgt_set=mnistm --save=my_exp
```
* --save specifies the save name for this experiment

## Files

### New Files
- train_search_sgl.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset.py: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files
- SGL.py: Wrapper class that wraps multiple learners into a small group. It applies any action on the SGL wrapper class to all the learners in the group.
- plot.py: Utility functions for plotting.

### Files from DANN
- functions.py: Contains the implementation of the gradient reversal layer

### Files from SGL
- model_search_coop.py: Contains the differentiable model architectures. We integrate the DANN model by using a differentiable feature architecture  
based on DARTS but keep the lable classifier and domain classifier as fixed feedforward network similar to the baseline DANN implementation.
- architect_coop.py: Contains the code for architecture search. We make some minor updates here to take into account the different loss function  
used by DANN
- utils.py: Contains utility functions. Mostly unchanged.
- operations.py: Contains candidate operations for architecture search. Unchanged.
- genotypes.py: Contains genotype for the searched architecture. Unchanged.

