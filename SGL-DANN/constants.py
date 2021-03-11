import torch

ROOT_STATS_DIR = './experiment_data'
NUM_CLASSES = 10
NUM_LEARNERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Put your other constants here.
