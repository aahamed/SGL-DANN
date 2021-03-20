import matplotlib.pyplot as plt
import numpy as np
import torch
import constants
import utils
import logging
import sys
import torch.nn.functional as F

from datetime import datetime
from constants import *
from file_utils import *
from model_factory import get_model, \
        get_optimizers, get_schedulers
from dataset import get_train_dataloaders, \
        get_test_dataloaders
from SGL import SGL, SGLStats
from architect_coop import ArchitectDA

def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]

class Experiment(object):
    def __init__(self, args):
        # import pdb; pdb.set_trace()
        self.args = args
        self.name = args.save
        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        # get train dataloader for src domain
        self.src_train_queue, self.src_ul_queue, self.src_val_queue = \
                get_train_dataloaders( args.src_set, args )
        # get train dataloader for tgt domain
        self.tgt_train_queue, self.tgt_ul_queue, self.tgt_val_queue = \
                get_train_dataloaders( args.tgt_set, args )

        # get test dataloader for src domain
        self.src_test_queue = get_test_dataloaders( args.src_set, args )
        # get test dataloader for tgt domain
        self.tgt_test_queue = get_test_dataloaders( args.tgt_set, args )


        # Setup Experiment
        self.epochs = args.epochs
        self.current_epoch = 0
        self.best_model = None
       
        # use same criterion for label and domain loss
        self.criterion = torch.nn.NLLLoss()

        # Init Model
        self.models, self.models_pretrain = get_model(args,
                self.criterion, self.criterion )

        self.optimizers, self.optimizers_pretrain = \
                get_optimizers( args, self.models, self.models_pretrain )
        self.schedulers, self.schedulers_pretrain = \
                get_schedulers( args, self.optimizers, self.optimizers_pretrain )

        # learner group for weights V_k
        self.sgl_pretrain = SGL( self.models_pretrain,
                self.optimizers_pretrain, self.schedulers_pretrain,
                self.criterion, self.criterion, self.experiment_dir,
                self.args )
        # learner group for weights W_k
        self.sgl = SGL( self.models, self.optimizers,
                self.schedulers, self.criterion, 
                self.criterion, self.experiment_dir, self.args )
        # architect uses weights W_k
        self.architect = ArchitectDA( self.sgl, args )

        # stats
        self.pretrain_stats = SGLStats( self.sgl_pretrain,
                'pretrain', self.experiment_dir )
        self.train_stats = SGLStats( self.sgl, 'train',
                self.experiment_dir )
        self.val_stats = SGLStats( self.sgl, 'validation',
                self.experiment_dir )
        self.test_stats = SGLStats( self.sgl, 'test',
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
            self.pretrain_stats.read_stats_from_dir()
            self.train_stats.read_stats_from_dir()
            self.val_stats.read_stats_from_dir()
            load_path = self.experiment_dir
            load_name = 'latest_model_pretrain'
            self.sgl_pretrain.load_models( load_name )
            load_name = 'latest_model'
            self.sgl.load_models( load_name )
            self.current_epoch = self.pretrain_stats.current_epoch()
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
            self.log( f'Starting Epoch: {epoch+1}' )
            self.sgl.log_stats_header()
            start_time = datetime.now()
            self.current_epoch = epoch
            self.train( epoch )
            # self.sgl_pretrain.schedulers_step()
            if epoch < self.args.pretrain_steps:
                self.pretrain_stats.log_last_stats()
            else:
                self.train_stats.log_last_stats()
                self.sgl.log_genotype()
                # self.sgl.schedulers_step()
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
        N = min( len( self.src_train_queue ),
                len( self.tgt_train_queue ) ) // 1
        src_train_queue_iter = iter( self.src_train_queue )
        src_ul_queue_iter = iter( self.src_ul_queue )
        src_val_queue_iter = iter( self.src_val_queue )
        tgt_train_queue_iter = iter( self.tgt_train_queue )
        tgt_ul_queue_iter = iter( self.tgt_ul_queue )
        tgt_val_queue_iter = iter( self.tgt_val_queue )
        for step in range( N ):
            # zero out gradients
            self.sgl_pretrain.zero_grad()
            self.sgl.zero_grad() 
            # compute alpha used in gradient reversal layer
            p = float(step + epoch * N) / self.args.epochs / N
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # get training src domain data
            src_train_images, src_train_labels = next( src_train_queue_iter )
            src_train_images, src_train_labels = src_train_images.to( DEVICE ), \
                    src_train_labels.to( DEVICE )
            src_batch_size = len( src_train_labels )
            # get training tgt domain data
            tgt_train_images, tgt_train_labels = next( tgt_train_queue_iter )
            tgt_train_images, tgt_train_labels = tgt_train_images.to( DEVICE ), \
                tgt_train_labels.to( DEVICE )
            tgt_batch_size = len( tgt_train_labels )
            
            # STAGE 1: Update weights V_k of each learner
            # using the training data

            # feed src data
            # import pdb; pdb.set_trace()
            src_domain = torch.zeros( src_batch_size ).long().to( DEVICE )
            src_labels_out, src_domain_out = \
                    self.sgl_pretrain( src_train_images, alpha )
            src_label_losses = self.sgl_pretrain.label_loss( 
                    src_labels_out, src_train_labels )
            src_domain_losses = self.sgl_pretrain.domain_loss(
                    src_domain_out, src_domain )
            # feed tgt data
            tgt_domain = torch.ones( tgt_batch_size ).long().to( DEVICE )
            tgt_labels_out, tgt_domain_out = self.sgl_pretrain( tgt_train_images, alpha )
            tgt_domain_losses = self.sgl_pretrain.domain_loss(
                    tgt_domain_out, tgt_domain )
            # optimize
            total_loss = sum( src_label_losses ) + sum( src_domain_losses ) + \
                    sum( tgt_domain_losses )
            total_loss.backward()
            # clip gradients and optimize
            self.sgl_pretrain.optimize()
            # update stats
            self.sgl_pretrain.accuracy( src_labels_out, src_train_labels )
            self.sgl_pretrain.tgt_accuracy( tgt_labels_out, tgt_train_labels )
            if step % self.args.report_freq == 0 and \
                    epoch < self.args.pretrain_steps:
                self.sgl_pretrain.log_stats( prefix='pretrain', step=step )

            # skip STAGE2 and STAGE3 if we are in pretraining
            if epoch < self.args.pretrain_steps:
                continue
            
            # get unlabeled src domain data
            src_ul_images, _ = next( src_ul_queue_iter )
            src_ul_images = src_ul_images.to( DEVICE )
            # get unlabeled tgt domain data
            tgt_ul_images, _ = next( tgt_ul_queue_iter )
            tgt_ul_images = tgt_ul_images.to( DEVICE )

            # STAGE 2: Update weights W_k of each learner
            # using the pseudolabeled data from every other
            # learner and the training data. We only do
            # pseudolabeling for the class labels not the
            # domain labels.

            #import pdb; pdb.set_trace()
            # get psuedo class labels using weights V_k
            src_raw_labels, _ = self.sgl_pretrain( src_ul_images, alpha )
            src_pseudo_labels = []
            with torch.no_grad():
                src_pseudo_labels = [ F.softmax( l, dim=1 ) for l in src_raw_labels ]
            # get predictions using weights W_k
            src_labels_out, src_domain_out = self.sgl( src_ul_images, alpha )
            _, tgt_domain_out = self.sgl( tgt_ul_images, alpha )
            # get psuedolabeled loss
            pseudo_losses = []
            for i in range( NUM_LEARNERS ):
                p_loss = 0
                for j in range( NUM_LEARNERS ):
                    if i == j:
                        # skip self generated pseudolabels
                        continue
                    p_loss += softXEnt( src_labels_out[i], src_pseudo_labels[j] )
                pseudo_losses.append( p_loss )
            pseudo_loss = sum( pseudo_losses )
            # get domain loss for unlabeled data
            ul_src_domain_loss = self.sgl.domain_loss( src_domain_out, src_domain )
            ul_tgt_domain_loss = self.sgl.domain_loss( tgt_domain_out, tgt_domain )
            ul_domain_loss = sum( ul_src_domain_loss ) + sum( ul_tgt_domain_loss )
            # get loss for training data
            src_labels_out, src_domain_out = self.sgl( src_train_images, alpha )
            tgt_labels_out, tgt_domain_out = self.sgl( tgt_train_images, alpha )
            src_label_loss = self.sgl.label_loss( src_labels_out, src_train_labels )
            src_domain_loss = self.sgl.domain_loss( src_domain_out, src_domain )
            tgt_domain_loss = self.sgl.domain_loss( tgt_domain_out, tgt_domain )
            train_loss = sum( src_label_loss ) + sum( src_domain_loss ) + \
                    sum( tgt_domain_loss )
            # optimize
            total_loss = train_loss + lbda * ( pseudo_loss + ul_domain_loss )
            total_loss.backward()
            self.sgl.optimize()
            # update stats
            self.sgl.accuracy( src_labels_out, src_train_labels )
            self.sgl.tgt_accuracy( tgt_labels_out, tgt_train_labels )
            if step % self.args.report_freq == 0:
                # import pdb; pdb.set_trace()
                self.sgl.log_stats( prefix='train', step=step )

            # STAGE 3: Architecture Search
            
            # get validation src domain data
            src_val_images, src_val_labels = next( src_val_queue_iter )
            src_val_images, src_val_labels = src_val_images.to( DEVICE ), \
                    src_val_labels.to( DEVICE )
            # get validation tgt domain data
            tgt_val_images, _ = next( tgt_val_queue_iter )
            tgt_val_images = tgt_val_images.to( DEVICE )
            if not IS_ARCH_FIXED():
                # update architecture
                self.architect.step( src_val_images, src_val_labels,
                    tgt_val_images, alpha )
        
        self.train_stats.update_stats()
        self.pretrain_stats.update_stats()

    # Perform one Pass on the validation set and return loss value.
    def val(self):
        # import pdb; pdb.set_trace()
        self.sgl.reset_stats()
        self.sgl_pretrain.reset_stats()
        self.sgl_pretrain.eval()
        self.sgl.eval()
        val_loss = 0
        N = min( len( self.src_val_queue ),
                len( self.tgt_val_queue ) ) // 1
        src_val_queue_iter = iter( self.src_val_queue )
        tgt_val_queue_iter = iter( self.tgt_val_queue )
        alpha = 1
        with torch.no_grad():
            for step in range( N ):
                # get src validation data
                src_val_images, src_val_labels = next( src_val_queue_iter )
                src_val_images, src_val_labels = src_val_images.to( DEVICE ), \
                        src_val_labels.to( DEVICE )
                src_batch_size = len( src_val_images )
                # get tgt validation data
                tgt_val_images, tgt_val_labels = next( tgt_val_queue_iter )
                tgt_val_images, tgt_val_labels = tgt_val_images.to( DEVICE ), \
                        tgt_val_labels.to( DEVICE )
                tgt_batch_size = len( tgt_val_images )
                # feed validation data
                src_domain = torch.zeros( src_batch_size ).long().to( DEVICE )
                src_labels_out, src_domain_out = self.sgl( src_val_images, alpha )
                tgt_domain = torch.zeros( tgt_batch_size ).long().to( DEVICE )
                tgt_labels_out, tgt_domain_out = self.sgl( tgt_val_images, alpha )
                # get loss
                self.sgl.label_loss( src_labels_out, src_val_labels )
                self.sgl.domain_loss( src_domain_out, src_domain )
                self.sgl.domain_loss( tgt_domain_out, tgt_domain )
                self.sgl.accuracy( src_labels_out, src_val_labels )
                self.sgl.tgt_accuracy( tgt_labels_out, tgt_val_labels )
                # log stats
                if step % self.args.report_freq == 0:
                    self.sgl.log_stats( prefix='validation', step=step )
        
        self.val_stats.update_stats()


    def test(self):
        # import pdb; pdb.set_trace()
        self.sgl.reset_stats()
        self.sgl_pretrain.reset_stats()
        self.sgl_pretrain.eval()
        self.sgl.eval()
        val_loss = 0
        N = min( len( self.src_test_queue ),
                len( self.tgt_test_queue ) ) // 1
        src_test_queue_iter = iter( self.src_test_queue )
        tgt_test_queue_iter = iter( self.tgt_test_queue )
        alpha = 1
        with torch.no_grad():
            for step in range( N ):
                # get src test data
                src_test_images, src_test_labels = next( src_test_queue_iter )
                src_test_images, src_test_labels = src_test_images.to( DEVICE ), \
                        src_test_labels.to( DEVICE )
                src_batch_size = len( src_test_images )
                # get tgt test data
                tgt_test_images, tgt_test_labels = next( tgt_test_queue_iter )
                tgt_test_images, tgt_test_labels = tgt_test_images.to( DEVICE ), \
                        tgt_test_labels.to( DEVICE )
                tgt_batch_size = len( tgt_test_images )
                # feed test data
                src_domain = torch.zeros( src_batch_size ).long().to( DEVICE )
                src_labels_out, src_domain_out = self.sgl( src_test_images, alpha )
                tgt_domain = torch.zeros( tgt_batch_size ).long().to( DEVICE )
                tgt_labels_out, tgt_domain_out = self.sgl( tgt_test_images, alpha )
                # get loss
                self.sgl.label_loss( src_labels_out, src_test_labels )
                self.sgl.domain_loss( src_domain_out, src_domain )
                self.sgl.domain_loss( tgt_domain_out, tgt_domain )
                self.sgl.accuracy( src_labels_out, src_test_labels )
                self.sgl.tgt_accuracy( tgt_labels_out, tgt_test_labels )
                # log stats
                if step % self.args.report_freq == 0:
                    self.sgl.log_stats( prefix='test', step=step )
        
        self.test_stats.update_stats()
        self.test_stats.log_last_stats()

    def save_model(self):
        save_name = 'latest_model_pretrain'
        self.sgl_pretrain.save_models( save_name )
        save_name = 'latest_model'
        self.sgl.save_models( save_name )

    def record_stats( self ):
        self.pretrain_stats.write_stats_to_dir()
        self.train_stats.write_stats_to_dir()
        self.val_stats.write_stats_to_dir()
        self.plot_stats()

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

    def plot_stats( self ):
        self.pretrain_stats.plot_stats()
        self.train_stats.plot_stats()

