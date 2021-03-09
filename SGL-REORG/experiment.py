import matplotlib.pyplot as plt
import numpy as np
import torch
import constants
import utils

from datetime import datetime
from constants import ROOT_STATS_DIR, \
        NUM_LEARNERS, DEVICE
from file_utils import *
from model_factory import get_model, \
        get_optimizers, get_schedulers
from dataset import get_dataloaders
from SGL import SGL


class Experiment(object):
    def __init__(self, args):
        #config_data = read_file_in_dir('./', name + '.json')
        #if config_data is None:
        #    raise Exception("Configuration file doesn't exist: ", name)
        import pdb; pdb.set_trace()
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

        self.sgl_pretrain = SGL( self.models_pretrain,
                self.optimizers_pretrain, self.criterion, self.args )

        self.init_model()

        # Load Experiment Data if available
        # self.load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.experiment_dir):
            self.training_losses = read_file_in_dir(self.experiment_dir, 'training_losses.txt')
            self.val_losses = read_file_in_dir(self.experiment_dir, 'val_losses.txt')
            self.current_epoch = len(self.training_losses)

            state_dict = torch.load(os.path.join(self.experiment_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.experiment_dir)

    def init_model(self):
        self.criterion.to( DEVICE )
        for i in range( NUM_LEARNERS ):
            self.models[ i ].to( DEVICE )
            self.models_pretrain[ i ].to( DEVICE )

    def run(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.current_epoch = epoch
            train_loss = self.train( epoch )
            val_loss = self.val()
            self.record_stats(train_loss, val_loss)
            self.log_epoch_stats(start_time)
            self.save_model()

    # Perform one training iteration on the whole dataset and return loss value
    def train2( self, epoch ):
        ### TODO set model.train
        import pdb; pdb.set_trace()
        training_loss = 0
        N = len( self.train_queue )
        train_queue_iter = iter( self.train_queue )
        ul_queue_iter = iter( self.ul_queue )
        val_queue_iter = iter( self.val_queue )
        loss_meter = []
        top1_meter = []
        top5_meter = []
        for i in range( NUM_LEARNERS ):
            loss_meter.append( utils.AverageMeter() )
            top1_meter.append( utils.AverageMeter() )
            top5_meter.append( utils.AverageMeter() )

        for step in range( N ):
            # zero gradients
            for j in range( NUM_LEARNERS ):
                self.optimizers_pretrain[ j ].zero_grad()
                self.optimizers[ j ].zero_grad()
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
            outputs = []
            losses = []
            for j in range( NUM_LEARNERS ):
                output = self.models_pretrain[j]( train_images )
                loss = self.criterion( output, train_labels )
                outputs.append( output )
                losses.append( loss )
            total_loss = sum( losses )
            total_loss.backward()
            # clip gradients and optimize
            for j in range( NUM_LEARNERS ):
                nn.utils.clip_grad_norm_( 
                        self.models_pretrain[j].parameters(),
                        args.grad_clip )
                self.optimizers_pretrain[j].step()
            # update stats
            for j in range( NUM_LEARNERS ):
                acc1, acc5 = utils.accuracy( outputs[j], 
                        labels, topk=(1,5) )
                top1_meter[j].update( acc1.item(), batch_size )
                top5_meter[j].update( acc5.item(), batch_size )
                loss_meter[j].update( losses[j].item(), batch_size )
                print( f'pretrain model {j} {step:03d} {loss_meter[j]:.3e} ' + 
                        f'{top1_meter[j]:.3f} {top5_meter[j]:.3f}' )

        return training_loss

    def train( self, epoch ):
        import pdb; pdb.set_trace()
        training_loss = 0
        N = len( self.train_queue )
        train_queue_iter = iter( self.train_queue )
        ul_queue_iter = iter( self.ul_queue )
        val_queue_iter = iter( self.val_queue )
        for step in range( N ):
            # zero out gradients
            self.sgl_pretrain.zero_grad()
            
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
            import pdb; pdb.set_trace()
            # clip gradients and optimize
            self.sgl_pretrain.optimize()
            # update stats
            self.sgl_pretrain.accuracy( outputs, train_labels )
            # log stats
            self.sgl_pretrain.log_stats( prefix='pretrain', step=step )


    # Perform one Pass on the validation set and return loss value.
    def val(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.val_loader):
                raise NotImplementedError()

        return val_loss

    def test(self):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.test_loader):
                raise NotImplementedError()
        
        result_str = "TODO"
        self.log(result_str)

        return test_loss

    def save_model(self):
        root_model_path = os.path.join(self.experiment_dir, 'latest_model.pt')
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def record_stats(self, train_loss, val_loss):
        self.training_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.experiment_dir, 'training_losses.txt', self.training_losses)
        write_to_file_in_dir(self.experiment_dir, 'val_losses.txt', self.val_losses)

    def log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.experiment_dir, file_name, log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        train_loss = self.training_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1,
                train_loss, val_loss, str(time_elapsed),
                str(time_to_completion))
        self.log(summary_str, 'epoch.log')

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
