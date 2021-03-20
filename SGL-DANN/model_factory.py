import torch
from constants import *
from torch.autograd import Variable
from torch import nn
from model_search_coop import DANN, DANNFixed
from model_search_da import DANN as DANNSimple
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
    model.fe._arch_parameters = arch
    model.fe.alphas_reduce = alphas_reduce
    model.fe.alphas_normal = alphas_normal
    model.fe.betas_reduce = betas_reduce
    model.fe.betas_normal = betas_normal
    return model

# Build and return the model here based on the configuration.
def get_model( args, label_criterion, domain_criterion ):
    if IS_ARCH_FIXED():
        return get_model_fixed()
    elif ARCH_TYPE == 'darts-simple':
        return get_model_darts_simple( args, label_criterion, domain_criterion )
    else:
        return get_model_darts( args, label_criterion, domain_criterion )

# Build model with fixed feature extractor
def get_model_fixed():
    models = []
    models_pretrain = []
    for i in range( NUM_LEARNERS ):
        models.append( DANNFixed() )
        models_pretrain.append( DANNFixed() )
    return models, models_pretrain


# Build model with darts feature extractor
def get_model_darts(args, label_criterion, domain_criterion):
    #criterion = nn.CrossEntropyLoss()
    #criterion = criterion.to( DEVICE )
    models = []
    models_pretrain = []
    for i in range( NUM_LEARNERS ):
        models.append( DANN(args.init_channels,
            NUM_CLASSES, args.layers, label_criterion, domain_criterion ) )
        models_pretrain.append( DANN(args.init_channels, 
            NUM_CLASSES, args.layers, label_criterion, domain_criterion ) )
        arch_attrs = initialize_alphas()
        models[i] = update_model_arch_attrs( models[i], *arch_attrs )
        models_pretrain[i] = update_model_arch_attrs( models_pretrain[i],
                *arch_attrs )
    return models, models_pretrain

# Build model with simplified darts feature extractor
def get_model_darts_simple( args, label_criterion, domain_criterion ):
    models = []
    models_pretrain = []
    for i in range( NUM_LEARNERS ):
        models.append( DANNSimple(args.init_channels,
            NUM_CLASSES, args.layers, label_criterion, domain_criterion ) )
        models_pretrain.append( DANNSimple(args.init_channels, 
            NUM_CLASSES, args.layers, label_criterion, domain_criterion ) )
    return models, models_pretrain

def get_optimizers( args, models, models_pretrain ):
    if OPTIM == 'Adam':
        return get_optimizers_adam( args, models, models_pretrain )
    else:
        return get_optimizers_sgd( args, models, models_pretrain )

def get_optimizers_sgd( args, models, models_pretrain ):
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

def get_optimizers_adam( args, models, models_pretrain ):
    optimizers = []
    optimizers_pretrain = []
    lr = 1e-3
    for i in range( len( models ) ):
        optimizer = torch.optim.Adam(
            models[i].parameters(),
            lr,
            weight_decay=args.weight_decay)
        optimizers.append( optimizer )
        optimizer_pretrain = torch.optim.Adam(
            models_pretrain[i].parameters(),
            lr,
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
