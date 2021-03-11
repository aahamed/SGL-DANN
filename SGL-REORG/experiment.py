import matplotlib.pyplot as plt
import numpy as np
import torch
import constants
import utils
import logging
import sys
import torch.nn.functional as F

from datetime import datetime
from constants import ROOT_STATS_DIR, \
        NUM_LEARNERS, DEVICE
from file_utils import *
from model_factory import get_model, \
        get_optimizers, get_schedulers
from dataset import get_dataloaders
from SGL import SGL, SGLStats
from architect_coop import ArchitectSGL

def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]

class Experiment(object):
    def __init__(self, args):
        self.args = args
        self.name = args.save
        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        # Load Datasets
        self.train_queue, self.ul_queue, self.val_queue = \
                get_dataloaders( args )

        # Setup Experiment
        self.epochs = args.epochs
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.best_model = None

        # Init Model
        self.models, self.models_pretrain = get_model(args)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizers, self.optimizers_pretrain = \
                get_optimizers( args, self.models, self.models_pretrain )
        self.schedulers, self.schedulers_pretrain = \
                get_schedulers( args, self.optimizers, self.optimizers_pretrain )

        # learner group for weights V_k
        self.sgl_pretrain = SGL( self.models_pretrain,
                self.optimizers_pretrain, self.schedulers_pretrain,
                self.criterion, self.experiment_dir, self.args )
        # learner group for weights W_k
        self.sgl = SGL( self.models, self.optimizers,
                self.schedulers, self.criterion, 
                self.experiment_dir, self.args )
        # architect uses weights W_k
        self.architect = ArchitectSGL( self.sgl, args )

        # stats
        self.train_stats = SGLStats( self.sgl, 'train',
                self.experiment_dir )
        self.val_stats = SGLStats( self.sgl, 'validation',
                self.experiment_dir )

        self.init_model()

        # Load Experiment Data if available
        self.load_experiment()

    def setup_logger( self ):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.experiment_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.log( f'Exp Name: {self.name}\n\n' )

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.experiment_dir):
            self.train_stats.read_stats_from_dir()
            self.val_stats.read_stats_from_dir()
            load_path = self.experiment_dir
            load_name = 'latest_model_pretrain'
            self.sgl_pretrain.load_models( load_name )
            load_name = 'latest_model'
            self.sgl.load_models( load_name )
            self.current_epoch = self.train_stats.current_epoch()
        else:
            os.makedirs(self.experiment_dir)
        self.setup_logger()

    def init_model(self):
        self.criterion.to( DEVICE )
        self.sgl_pretrain.to( DEVICE )
        self.sgl.to( DEVICE )

    # main loop
    def run(self):
        start_epoch = self.current_epoch
        self.sgl.log_genotype()
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            self.log( 'Starting Epoch: {epoch}' )
            start_time = datetime.now()
            self.current_epoch = epoch
            self.train( epoch )
            self.train_stats.log_last_stats()
            self.sgl_pretrain.schedulers_step()
            if epoch >= self.args.pretrain_steps:
                self.sgl.log_genotype()
                self.sgl.schedulers_step()
                self.val()
                self.val_stats.log_last_stats()
            self.record_stats()
            self.log_epoch_stats(start_time)
            self.save_model()

    # Perform one training iteration on the whole dataset and return loss value
    def train( self, epoch ):
        self.sgl.reset_stats()
        self.sgl_pretrain.reset_stats()
        self.sgl.train()
        self.sgl_pretrain.train()
        lbda = self.args.weight_lambda
        N = len( self.train_queue ) // 1
        train_queue_iter = iter( self.train_queue )
        ul_queue_iter = iter( self.ul_queue )
        val_queue_iter = iter( self.val_queue )
        for step in range( N ):
            # zero out gradients
            self.sgl_pretrain.zero_grad()
            self.sgl.zero_grad() 
            # get data
            train_images, train_labels = next( train_queue_iter )
            ul_images, _ = next( ul_queue_iter )
            val_images, val_labels = next( val_queue_iter )
            batch_size = len( train_images )
            train_images, train_labels, ul_images, val_images, val_labels = \
                    train_images.to( DEVICE ), train_labels.to( DEVICE ), \
                    ul_images.to( DEVICE ), val_images.to( DEVICE ), \
                    val_labels.to( DEVICE )
            
            # STAGE 1: Update weights V_k of each learner
            # using the training data
            outputs = self.sgl_pretrain( train_images )
            losses = self.sgl_pretrain.loss( outputs, train_labels )
            total_loss = sum( losses )
            total_loss.backward()
            # import pdb; pdb.set_trace()
            # clip gradients and optimize
            self.sgl_pretrain.optimize()
            # update stats
            self.sgl_pretrain.accuracy( outputs, train_labels )
            if step % self.args.report_freq == 0 and \
                    epoch < self.args.pretrain_steps:
                self.sgl_pretrain.log_stats( prefix='pretrain', step=step )

            # skip STAGE2 and STAGE3 if we are in pretraining
            if epoch < self.args.pretrain_steps:
                continue

            # STAGE 2: Update weights W_k of each learner
            # using the pseudolabeled data from every other
            # learner and the training data.

            # get psuedolabeled data
            raw_labels = self.sgl_pretrain( ul_images )
            pseudo_labels = []
            with torch.no_grad():
                pseudo_labels = [ F.softmax( l, dim=1 ) for l in raw_labels ]
            # get predictions using weights W_k
            outputs = self.sgl( ul_images )
            # get psuedolabeled loss
            pseudo_losses = []
            for i in range( NUM_LEARNERS ):
                p_loss = 0
                for j in range( NUM_LEARNERS ):
                    if i == j:
                        # skip self generated pseudolabels
                        continue
                    p_loss += softXEnt( outputs[i], pseudo_labels[j] )
                pseudo_losses.append( p_loss )
            # get loss for training data
            outputs = self.sgl( train_images )
            losses = self.sgl.loss( outputs, train_labels )
            # optimize
            total_loss = sum( losses ) + lbda * sum( pseudo_losses )
            total_loss.backward()
            self.sgl.optimize()
            # update stats
            self.sgl.accuracy( outputs, train_labels )
            if step % self.args.report_freq == 0:
                self.sgl.log_stats( prefix='train', step=step )

            # STAGE 3: Architecture Search
            self.architect.step( val_images, val_labels )
        
        self.train_stats.update_stats()


    # Perform one Pass on the validation set and return loss value.
    def val(self):
        self.sgl_pretrain.eval()
        self.sgl.eval()
        val_loss = 0
        N = len( self.val_queue ) // 1
        val_queue_iter = iter( self.val_queue )
        with torch.no_grad():
            for step in range( N ):
                val_images, val_labels = next( val_queue_iter )
                val_images, val_labels = val_images.to( DEVICE ), \
                        val_labels.to( DEVICE )
                outputs = self.sgl( val_images )
                losses = self.sgl.loss( outputs, val_labels )
                self.sgl.accuracy( outputs, val_labels )
                # log stats
                if step % self.args.report_freq == 0:
                    self.sgl.log_stats( prefix='validation', step=step )
        
        self.val_stats.update_stats()


    def test(self):
        self.sgl.eval()
        self.sgl_pretrain.eval()
        test_loss = 0
        return

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.test_loader):
                raise NotImplementedError()
        
        result_str = "TODO"
        self.log(result_str)

        return test_loss

    def save_model(self):
        save_name = 'latest_model_pretrain'
        self.sgl_pretrain.save_models( save_name )
        save_name = 'latest_model'
        self.sgl.save_models( save_name )

    def record_stats( self ):
        self.train_stats.write_stats_to_dir()
        self.val_stats.write_stats_to_dir()

    def log( self, log_str ):
        logging.info( log_str )

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        summary_str = "Epoch: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1,
                str(time_elapsed),
                str(time_to_completion))
        self.log( summary_str )

    def plot_stats(self):
        e = len(self.training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.training_losses, label="Training Loss")
        plt.plot(x_axis, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.name + " Stats Plot")
        plt.savefig(os.path.join(self.experiment_dir, "stat_plot.png"))
        plt.show()
