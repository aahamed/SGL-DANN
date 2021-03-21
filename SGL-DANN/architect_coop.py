import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from SGL import SGL

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, model1, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.model1 = model1
    self.args = args
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.optimizer1 = torch.optim.Adam(self.model1.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  # def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
  #   self.optimizer.zero_grad()
  #   if unrolled:
  #       self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
  #   else:
  #       self._backward_step(input_valid, target_valid)
  #   self.optimizer.step()

  # def _backward_step(self, input_valid, target_valid):
  #   loss = self.model._loss(input_valid, target_valid)
  #   loss.backward()
  def step(self,
           input_train,
           target_train,
           input_external,
           target_external,
           input_valid,
           target_valid,
           eta,
           eta1,
           network_optimizer,
           network_optimizer1,
           unrolled):
    self.optimizer.zero_grad()
    self.optimizer1.zero_grad()
    if unrolled:
      self._backward_step_unrolled(
          input_train, target_train,
          input_external, target_external,
          input_valid, target_valid, eta,
          eta1,
          network_optimizer, network_optimizer1)
    else:
      self._backward_step(
          input_valid,
          target_valid)
    nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.args.grad_clip)
    nn.utils.clip_grad_norm_(self.model1.arch_parameters(), self.args.grad_clip)
    self.optimizer.step()
    self.optimizer1.step()

  def _backward_step(self,
                     input_valid,
                     target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss1 = self.model1._loss(input_valid, target_valid)
    loss = loss + loss1
    loss.backward()
    # loss1.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


class ArchitectSGL( Architect ):
    
    def __init__( self, sgl, args ):
        assert isinstance( sgl, SGL )
        # only first order approx. currently supported
        assert args.unrolled == False and 'unrolled not supported'
        # only support 2 learners currently
        assert len( sgl ) == 2
        model0 = sgl.models[0]
        model1 = sgl.models[1]
        self.sgl = sgl
        super( ArchitectSGL, self ).__init__( model0, model1, args )
  
    def step( self, val_images, val_labels ):
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        self._backward_step( val_images, val_labels )
        nn.utils.clip_grad_norm_(self.model.arch_parameters(),
                self.args.grad_clip)
        nn.utils.clip_grad_norm_(self.model1.arch_parameters(), 
                self.args.grad_clip)
        self.optimizer.step()
        self.optimizer1.step()

class ArchitectDA( ArchitectSGL ):

    def __init__( self, sgl, args ):
        super( ArchitectDA, self ).__init__( sgl, args )

    def step( self, val_src_imgs, val_src_labels, val_tgt_imgs,
            alpha ):
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        self._backward_step( val_src_imgs, val_src_labels, val_tgt_imgs, alpha )
        nn.utils.clip_grad_norm_(self.model.arch_parameters(),
                self.args.grad_clip)
        nn.utils.clip_grad_norm_(self.model1.arch_parameters(), 
                self.args.grad_clip)
        self.optimizer.step()
        self.optimizer1.step()
  
    def _backward_step( self, src_imgs, src_labels, tgt_imgs, alpha ):
        loss0 = self.model._loss( src_imgs, src_labels, tgt_imgs, alpha )
        loss1 = self.model1._loss( src_imgs, src_labels, tgt_imgs, alpha )
        loss = loss0 + loss1
        loss.backward()
