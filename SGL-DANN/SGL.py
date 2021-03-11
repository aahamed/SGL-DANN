import torch
import utils
import logging
import os
from torch import nn
from file_utils import *


class SGL( object ):
    '''
    Class representing a small group of learners.
    It contains utility functions to apply actions
    to the small group.
    '''

    def __init__( self, models, optimizers,
            schedulers, label_criterion, domain_criterion,
            exp_dir, args ):
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.label_criterion = label_criterion
        self.domain_criterion = domain_criterion
        self.exp_dir = exp_dir
        self.args = args
        self.N = len( models )
        self.outputs = []
        self.label_out = []
        self.domain_out = []
        #self.losses = []
        self.label_losses = []
        self.domain_losses = []
        #self.loss_meter = []
        self.label_loss_meter = []
        self.domain_loss_meter = []
        self.top1_meter = []
        self.top5_meter = []
        for i in range( self.N ):
            self.label_loss_meter.append( utils.AverageMeter() )
            self.domain_loss_meter.append( utils.AverageMeter() )
            self.top1_meter.append( utils.AverageMeter() )
            self.top5_meter.append( utils.AverageMeter() )

    def to( self, device ):
        for i in range( self.N ):
            self.models[ i ] = self.models[ i ].to( device )

    def train( self ):
        for i in range( self.N ):
            self.models[ i ].train()

    def eval( self ):
        for i in range( self.N ):
            self.models[ i ].eval()

    def __call__( self, x, alpha ):
        return self.forward( x, alpha )

    def __len__( self ):
        return len( self.models )

    def zero_grad( self ):
        for i in range( self.N ):
            self.optimizers[ i ].zero_grad()

    def forward( self, x, alpha ):
        self.label_out.clear()
        self.domain_out.clear()
        for i in range( self.N ):
            label_out, domain_out = self.models[ i ]( x, alpha )
            self.label_out.append( label_out )
            self.domain_out.append( domain_out )
        return self.label_out, self.domain_out

    def label_loss( self, label_out, labels ):
        self.label_losses.clear()
        batch_size = len( labels )
        for i in range( self.N ):
            loss = self.label_criterion( label_out[i], labels )
            self.label_losses.append( loss )
            self.label_loss_meter[i].update( loss.item(), batch_size )
        return self.label_losses

    def domain_loss( self, domain_out, labels ):
        self.domain_losses.clear()
        batch_size = len( labels )
        for i in range( self.N ):
            loss = self.domain_criterion( domain_out[i], labels )
            self.domain_losses.append( loss )
            self.domain_loss_meter[i].update( loss.item(), batch_size )
        return self.domain_losses

    #def loss( self, outputs, labels ):
    #    self.losses.clear()
    #    batch_size = len( labels )
    #    for i in range( self.N ):
    #        loss = self.criterion( outputs[ i ], labels )
    #        self.losses.append( loss )
    #        self.loss_meter[ i ].update( loss.item(), batch_size )
    #    return self.losses

    def optimize( self ):
        for i in range( self.N ):
            nn.utils.clip_grad_norm_( 
                    self.models[i].parameters(),
                    self.args.grad_clip )
            self.optimizers[i].step()
    
    def accuracy( self, outputs, labels ):
        batch_size = len( labels )
        for i in range( self.N ):
            acc1, acc5 = utils.accuracy( outputs[i],
                    labels, topk=(1,5) )
            self.top1_meter[i].update( acc1.item(), batch_size )
            self.top5_meter[i].update( acc5.item(), batch_size )

    def log_stats( self, prefix, step ):
        for i in range( self.N ):
            label_loss = self.label_loss_meter[i].avg
            domain_loss = self.domain_loss_meter[i].avg
            top1 = self.top1_meter[i].avg
            top5 = self.top5_meter[i].avg
            stats_str = f'{prefix} model {i} {step:03d} {label_loss:.3e} ' + \
                    f'{domain_loss:.3e} {top1:.3f} {top5:.3f}'
            logging.info( stats_str )

    def reset_stats( self ):
        for i in range( self.N ):
            self.label_loss_meter[i].reset()
            self.domain_loss_meter[i].reset()
            self.top1_meter[i].reset()
            self.top5_meter[i].reset()

    def log_genotype( self ):
        for i in range( self.N ):
            logging.info( f'genotype {i} = {self.models[i].genotype()}' )

    def schedulers_step( self ):
        for i in range( self.N ):
            self.schedulers[i].step()

    def save_models( self, save_name ):
        for i in range( self.N ):
            save_path = os.path.join( self.exp_dir, f'{save_name}_{i}.pt' )
            model_dict = self.models[i].state_dict()
            opt_dict = self.optimizers[i].state_dict()
            state_dict = {'model': model_dict, 'optimizer': opt_dict}
            torch.save(state_dict, save_path)

    def load_models( self, load_name ):
        for i in range( self.N ):
            load_path = os.path.join( self.exp_dir, f'{load_name}_{i}.pt' )
            state_dict = torch.load( load_path )
            self.models[i].load_state_dict( state_dict['model'] )
            self.optimizers[i].load_state_dict( state_dict['optimizer'] )

class SGLStats( object ):
    '''
    Class to keep track of loss and accuracy stats
    associated with a small group of learners
    '''

    def __init__( self, sgl, stat_type, exp_dir ):
        self.sgl = sgl
        self.N = sgl.N
        self.stat_type = stat_type
        self.exp_dir = exp_dir
        self.label_losses = [ [] for _ in range( self.N ) ]    
        self.domain_losses = [ [] for _ in range( self.N ) ]    
        self.top1 = [ [] for _ in range( self.N ) ]
        self.top5 = [ [] for _ in range( self.N ) ]

    def update_losses( self ):
        for i in range( self.N ):
            self.label_losses[i].append( self.sgl.label_loss_meter[i].avg )
            self.domain_losses[i].append( self.sgl.domain_loss_meter[i].avg )

    def update_accuracy( self ):
        for i in range( self.N ):
            self.top1[i].append( self.sgl.top1_meter[i].avg )
            self.top5[i].append( self.sgl.top5_meter[i].avg )

    def update_stats( self ):
        self.update_losses()
        self.update_accuracy()

    def log_last_stats( self ):
        prefix = self.stat_type
        for i in range( self.N ):
            label_loss = self.label_losses[i][-1]
            domain_loss = self.domain_losses[i][-1]
            top1 = self.top1[i][-1]
            top5 = self.top5[i][-1]
            stat_str = f'{prefix} model {i} avg: {label_loss:.3e} {domain_loss:.3e} ' +\
                    f'{top1:.3f} {top5:.3f}'
            logging.info( stat_str )

    def write_stats_to_dir( self ):
        prefix = self.stat_type
        for i in range( self.N ):
            fname = f'{prefix}_label_losses_{i}.txt'
            write_to_file_in_dir( self.exp_dir, fname, self.label_losses[i] )
            fname = f'{prefix}_domain_losses_{i}.txt'
            write_to_file_in_dir( self.exp_dir, fname, self.domain_losses[i] )
    
    def read_stats_from_dir( self ):
        prefix = self.stat_type
        for i in range( self.N ):
            fname = f'{prefix}_label_losses_{i}.txt'
            self.label_losses[i] = read_file_in_dir( self.exp_dir, fname )
            fname = f'{prefix}_domain_losses_{i}.txt'
            self.domain_losses[i] = read_file_in_dir( self.exp_dir, fname )

    def current_epoch( self ):
        return len( self.label_losses[0] )

