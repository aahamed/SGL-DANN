import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_da import OPS
import numpy as np
from torch.autograd import Variable
from genotypes_da import NORMAL_PRIMITIVES, \
        REDUCE_PRIMITIVES, Genotype
from functions import ReverseLayerF
from constants import DEVICE


class MixedOpNormal(nn.Module):

    def __init__(self, C_prev, C):
        super(MixedOpNormal, self).__init__()
        self._ops = nn.ModuleList()
        self.C = C
        self.C_prev = C_prev
        for primitive in NORMAL_PRIMITIVES:
            op = OPS[primitive](C_prev, C)
            self._ops.append(op)

    def forward(self, x, weights):
        # import pdb; pdb.set_trace()
        y = []
        for w, op, primitive in zip( weights, self._ops, NORMAL_PRIMITIVES ):
            if ( self.C != self.C_prev ) and \
                    ( primitive in [ 'none', 'skip_connect' ] ):
                # skip none, skip_connect for ops
                # that change the number of channels
                continue
            y.append( w * op( x ) )
        return sum( y )
        # return sum(w * op(x) for w, op in zip(weights, self._ops))

class NormalCell( nn.Module ):
    '''
    A NormalCell is a DAG consisting of num_nodes.
    It only has one input, which is the output of the previous
    cell. Each edge is of type MixedOpNormal.
    '''

    def __init__( self, num_nodes, C_prev, C ):
        super( NormalCell, self ).__init__()
        # number of channels
        self.C = C
        # number of nodes in DARTS cell
        assert num_nodes == 1 and 'Only 1 node supported currently'
        self.num_nodes = num_nodes
        self._ops = nn.ModuleList()
        # setup the ops for first node
        # to update channel dimension
        op = MixedOpNormal( C_prev, C )
        self._ops.append( op )
        # setup ops for rest of nodes which
        # maintain channel dimension
        for i in range( 1, num_nodes ):
            for j in range( i+1 ):
                op = MixedOp( C, C )
                self._ops.append( op )

    def forward( self, x, weights ):
        x = self._ops[0]( x, weights[0] )
        return x

class MixedOpReduce( nn.Module ):
    '''
    Mixing Operation for REDUCE_PRIMITIVES
    '''

    def __init__(self):
        super(MixedOpReduce, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in REDUCE_PRIMITIVES:
            op = OPS[primitive](0, 0)
            self._ops.append(op)

    def forward(self, x, weights):
        y = []
        for w, op in zip( weights, self._ops ):
            y.append( w * op( x ) )
        return sum( y )
        # return sum(w * op(x) for w, op in zip(weights, self._ops))

class ReduceCell( nn.Module ):
    '''
    A ReduceCell consists of a single node with one incoming
    edge of type MixedOpReduce.
    '''

    def __init__( self ):
        super( ReduceCell, self ).__init__()
        op = MixedOpReduce()
        self._op = op

    def forward( self, x, weights ):
        y = self._op( x, weights[0] )
        return y

class Cell( nn.Module ):
    '''
    Cell consists of a ReduceCell stacked on top of 
    a NormalCell with dropout in between and ReLU
    non linearity
    '''

    def __init__( self, num_normal_nodes, C_prev, C, cell_id ):
        super( Cell, self ).__init__()
        self.num_normal_nodes = num_normal_nodes
        self.cell_id = cell_id
        self.dropout_prob = 0.01 if cell_id == 0 else 0.5
        self.normal_cell = NormalCell( num_normal_nodes, C_prev, C )
        self.reduce_cell = ReduceCell()
        self.dropout = nn.Dropout2d( p=self.dropout_prob )
        self.relu = nn.ReLU(True)

    def forward( self, x, weights_normal, weights_reduce ):
        # import pdb; pdb.set_trace()
        x = self.normal_cell( x, weights_normal )
        x = self.dropout( x )
        x = self.reduce_cell( x, weights_reduce )
        x = self.relu( x )
        return x

class NetworkFE( nn.Module ):
    '''
    Differentiable network for feature extractor.
    It is a stack of Cells.
    '''

    def __init__( self, C, layers, num_normal_nodes,
            feature_dim=4 ):
        super( NetworkFE, self ).__init__()
        # import pdb; pdb.set_trace()
        self._C = C
        self._layers = layers
        self._feature_dim = feature_dim
        self.alphas_normal = None
        self.alphas_reduce = None
        # self._criterion = criterion
        # number of nodes in a normal cell
        self.num_normal_nodes = num_normal_nodes
        self.cells = nn.ModuleList()
        # setup first cell
        C_prev = 3
        cell = Cell( num_normal_nodes, C_prev, C, 0 )
        self.cells.append( cell )
        C_prev, C = C, 2*C
        # setup rest of cells
        for i in range( 1, layers ):
            cell = Cell( num_normal_nodes, C_prev, C, i )
            self.cells.append( cell )
            C_prev, C = C, 2*C
        self.out_ch = C_prev
        self.global_pooling = nn.AdaptiveAvgPool2d( feature_dim )
        self.out_vol = self.out_ch * feature_dim * feature_dim
        self._initialize_alphas()
    
    def num_edges( self, num_nodes ):
        num_edges = 0
        for i in range( num_nodes ):
            for _ in range( i+1 ):
                num_edges += 1
        return num_edges

    def genotype( self ):
        '''
        Create genotype for network
        '''
        # import pdb; pdb.set_trace()
        normal_op_idx, edge = torch.argmax( self.alphas_normal[ 0 ] ), 0
        reduce_op_idx, edge = torch.argmax( self.alphas_reduce[ 0 ] ), 0
        normal_op = NORMAL_PRIMITIVES[ normal_op_idx.item() ]
        reduce_op = REDUCE_PRIMITIVES[ reduce_op_idx.item() ]
        normal_gene=[ ( normal_op, edge ) ]
        reduce_gene=[ (reduce_op, edge ) ]
        genotype = Genotype( normal=normal_gene,
                reduce=reduce_gene )
        return genotype

    def forward( self, x ):
        # import pdb; pdb.set_trace()
        # N is batch_size, H is height of image, W is width of image
        N, _, H, W = x.shape
        # expand input to 3 channels ( this is for mnist images )
        x = x.expand(N, 3, H, W)
        # DARTS forward logic
        weights_normal = F.softmax( self.alphas_normal, dim=1 )
        weights_reduce = F.softmax( self.alphas_reduce, dim=1 )
        for i, cell in enumerate( self.cells ):
            x = cell( x, weights_normal, weights_reduce )
        x = self.global_pooling( x )
        return x
        
    def arch_parameters( self ):
        return self._arch_parameters
  
    def _initialize_alphas(self):
        num_edges = self.num_edges( self.num_normal_nodes )
        num_normal_ops = len( NORMAL_PRIMITIVES )
        num_reduce_ops = len( REDUCE_PRIMITIVES )
        self.alphas_normal = Variable(
                1e-3*torch.randn(num_edges, num_normal_ops).to( DEVICE ), 
                requires_grad=True)
        self.alphas_reduce = Variable(
                1e-3*torch.randn(1, num_reduce_ops).to( DEVICE ),
                requires_grad=True )
        #self.alphas_normal = Variable( 
        #        torch.tensor( [[0.1,0.1,0.4,0.4]] ).to( DEVICE ),
        #        requires_grad=True )
        #self.alphas_reduce = Variable( 
        #        torch.tensor( [[0.5, 0.5]] ).to( DEVICE ),
        #        requires_grad=True )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        return self._arch_parameters

    def finalize_model( self ):
        normal_shape = self.alphas_normal.shape
        reduce_shape = self.alphas_reduce.shape
        weights_normal = torch.zeros( normal_shape,
                requires_grad=False ).to( DEVICE )
        weights_reduce = torch.zeros( reduce_shape,
                requires_grad=False ).to( DEVICE )
        for i in range( len( self.alphas_normal ) ):
            best_op = torch.argmax( self.alphas_normal[i] )
            weights_normal[ i, best_op ] = 1
            best_op = torch.argmax( self.alphas_reduce[i] )
            weights_reduce[ i, best_op ] = 1
        self.alphas_normal = weights_normal
        self.alphas_reduce = weights_reduce

class DANN( nn.Module ):

    def __init__( self, C, num_classes, layers, label_criterion=None,
            domain_criterion=None ):
        super( DANN, self ).__init__()
        self.label_criterion = label_criterion
        self.domain_criterion = domain_criterion
        self.num_classes = num_classes
        self.num_normal_nodes = 1
        # feature extractor
        self.fe = NetworkFE( C, layers, self.num_normal_nodes ) 
        out_vol = self.fe.out_vol
        
        # label classifier
        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(out_vol, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, num_classes))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(out_vol, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
  
    def genotype(self):
        return self.fe.genotype()

    def arch_parameters(self):
        return self.fe._arch_parameters

    def forward( self, x, alpha ):
        # import pdb; pdb.set_trace()
        batch_size = len( x )
        fe_out = self.fe( x )
        fe_out = fe_out.view( batch_size, -1 )
        reverse_fe = ReverseLayerF.apply( fe_out, alpha )
        label_out = self.label_classifier( fe_out )
        domain_out = self.domain_classifier( reverse_fe )

        return label_out, domain_out
    
    def _loss(self, src_images, src_labels, tgt_images, alpha ):
        src_batch_size = len( src_images )
        tgt_batch_size = len( tgt_images )
        src_domain = torch.zeros( src_batch_size ).long().to( DEVICE )
        tgt_domain = torch.ones( tgt_batch_size ).long().to( DEVICE )
        src_labels_out, src_domain_out = self( src_images, alpha )
        _, tgt_domain_out = self( tgt_images, alpha )
        src_label_loss = self.label_criterion( src_labels_out, src_labels )
        src_domain_loss = self.domain_criterion( src_domain_out, src_domain )
        tgt_domain_loss = self.domain_criterion( tgt_domain_out, tgt_domain )
        domain_loss = src_domain_loss + tgt_domain_loss 
        return src_label_loss + tgt_domain_loss
    
    def new(self):
        assert False and 'new not supported'


def test_network_fe():
    C = 48
    layers = 2
    num_normal_nodes = 1
    net = NetworkFE( C, layers, num_normal_nodes )
    net.eval()
    print( 'genotype:', net.genotype() )

    # random data
    x = torch.randn( 1, 3, 28, 28 )
    y = net( x )
    f = net._feature_dim
    assert y.shape == ( 1, C*2, f, f )

def test_dann():
    C, layers, num_classes = 48, 2, 10
    net = DANN( C, num_classes, layers )
    net.eval()
    
    # random data
    x = torch.randn( 1, 3, 28, 28 )
    y = net( x, alpha=0.1 )

def main():
    test_network_fe()
    test_dann()

if __name__ == '__main__':
    main()
