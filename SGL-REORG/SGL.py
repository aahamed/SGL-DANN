import torch
import utils
from torch import nn

class SGL( object ):
    '''
    Class representing a small group of learners.
    It contains utility functions to apply actions
    to the small group.
    '''

    def __init__( self, models, optimizers,
            criterion, args ):
        self.models = models
        self.optimizers = optimizers
        self.criterion = criterion
        self.args = args
        self.N = len( models )
        self.outputs = []
        self.losses = []
        self.loss_meter = []
        self.top1_meter = []
        self.top5_meter = []
        for i in range( self.N ):
            self.loss_meter.append( utils.AverageMeter() )
            self.top1_meter.append( utils.AverageMeter() )
            self.top5_meter.append( utils.AverageMeter() )

    def __call__( self, x ):
        return self.forward( x )

    def zero_grad( self ):
        for i in range( self.N ):
            self.optimizers[ i ].zero_grad()

    def forward( self, x ):
        self.outputs.clear()
        for i in range( self.N ):
            output = self.models[ i ]( x )
            self.outputs.append( output )
        return self.outputs

    def loss( self, outputs, labels ):
        self.losses.clear()
        batch_size = len( labels )
        for i in range( self.N ):
            loss = self.criterion( outputs[ i ], labels )
            self.losses.append( loss )
            self.loss_meter[ i ].update( loss.item(), batch_size )
        return self.losses

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
            self.top5_meter[i].udpate( acc5.item(), batch_size )
        # top1 = [ m.avg for m in top1_meter ]
        # top5 = [ m.avg for m in top5_meter ]
        # return top1, top5

    def log_stats( self, prefix, step ):
        for i in range( self.N ):
            loss = loss_meter[i].avg
            top1 = top1_meter[i].avg
            top5 = top5_meter[i].avg
            print( f'{prefix} model {i} {step:03d} {loss:.3e} ' +
                    f'{top1:.3f} {top5:.3f}' )


