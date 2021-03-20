import torch
import torch.nn as nn

OPS = {
    'none' : lambda C_prev, C: Zero( stride=1 ),
    'avg_pool_2x2' : lambda C_prev, C: nn.AvgPool2d( 2 ),
    'max_pool_2x2' : lambda C_prev, C: nn.MaxPool2d( 2 ),
    'skip_connect' : lambda C_prev, C: Identity(),
    'conv_3x3' : lambda C_prev, C: Conv( C_prev, C, 3, 1, 1 ),
    'conv_5x5' : lambda C_prev, C: Conv( C_prev, C, 5, 1, 2 ),
}

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class Conv( nn.Module ):

    def __init__( self, C_in, C_out, kernel_size, stride, padding, affine=True ):
        super( Conv, self ).__init__()
        self.op = nn.Sequential(
            nn.Conv2d( C_in, C_out, kernel_size=kernel_size,
                stride=stride, padding=padding ),
            nn.BatchNorm2d( C_out, affine=affine ),
            # TODO: maybe add relu
        )

    def forward( self, x ):
        return self.op( x )
