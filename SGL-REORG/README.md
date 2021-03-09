# Small Group Learning - Reorganized code

* This is the small group learning code reorganized to make it easier to use and extend.

## Usage

* Configuration for the experiment are passed as arguments to the train_search_sgl.py driver file.
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- train_search_sgl.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
