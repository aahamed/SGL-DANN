import torch
from constants import NUM_CLASSES, \
        NUM_LEARNERS, DEVICE
from torch.autograd import Variable
from torch import nn
from model_search_coop import Network
from genotypes import PRIMITIVES

def initialize_alphas(steps=4):
    k = sum(1 for i in range(steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    alphas_normal = Variable(
        1e-3 * torch.randn(k, num_ops).to( DEVICE ), requires_grad=True)
    alphas_reduce = Variable(
        1e-3 * torch.randn(k, num_ops).to( DEVICE ), requires_grad=True)
    betas_normal = Variable(1e-3 * torch.randn(k).to( DEVICE ), requires_grad=True)
    betas_reduce = Variable(1e-3 * torch.randn(k).to( DEVICE ), requires_grad=True)
    _arch_parameters = [
        alphas_normal,
        alphas_reduce,
        betas_normal,
        betas_reduce,
    ]
    return _arch_parameters, alphas_normal, alphas_reduce, \
        betas_normal, betas_reduce

def update_model_arch_attrs( 
        model, arch, alphas_reduce,
        alphas_normal, betas_reduce,
        betas_normal ):
    model._arch_parameters = arch
    model.alphas_reduce = alphas_reduce
    model.alphas_normal = alphas_normal
    model.betas_reduce = betas_reduce
    model.betas_normal = betas_normal
    return model

# Build and return the model here based on the configuration.
def get_model(args):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to( DEVICE )
    models = []
    models_pretrain = []
    for i in range( NUM_LEARNERS ):
        models.append( Network(args.init_channels,
            NUM_CLASSES, args.layers, criterion) )
        models_pretrain.append( Network(args.init_channels, 
            NUM_CLASSES, args.layers, criterion) )
        arch_attrs = initialize_alphas()
        models[i] = update_model_arch_attrs( models[i], *arch_attrs )
        models_pretrain[i] = update_model_arch_attrs( models_pretrain[i],
                *arch_attrs )
    return models, models_pretrain

def get_optimizers( args, models, models_pretrain ):
    optimizers = []
    optimizers_pretrain = []
    for i in range( len( models ) ):
        optimizer = torch.optim.SGD(
            models[i].parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizers.append( optimizer )
        optimizer_pretrain = torch.optim.SGD(
            models_pretrain[i].parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizers_pretrain.append( optimizer_pretrain )
    return optimizers, optimizers_pretrain

def get_schedulers( args, optimizers, optimizers_pretrain ):
    schedulers = []
    schedulers_pretrain = []
    for i in range( NUM_LEARNERS ):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[i], float(args.epochs), eta_min=args.learning_rate_min)
        schedulers.append( scheduler )
        scheduler_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers_pretrain[i], float(args.epochs + args.pretrain_steps), 
            eta_min=args.learning_rate_min)
        schedulers_pretrain.append( scheduler_pretrain )
    return schedulers, schedulers_pretrain
