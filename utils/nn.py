import math
import torch
import os
from .logger import get_timestamp

def adjust_learning_rate(optimizer, epoch, lr, cos, epochs, schedule):
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#             adjust_learning_rate(optimizer_D, epoch, lr['D'], lr_cos, epochs, schedule)

# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if args.cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def save_model(epoch, model, optimizer, args):
    if (epoch > 0 and epoch % args.ckp_interval == 0) or epoch + 1 == args.epochs:
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args': args}

        prefix = ''
        if args.arch == 'dcgan_G':
            prefix = 'G_'
        if args.arch == 'dcgan_D':
            prefix = 'D_'

        # print(prefix)

        torch.save(state, os.path.join(args.save_path, 'ckp_{}{}.pth.tar'.format(prefix,epoch)))
    pass

def get_netname(args):
    net_name = get_timestamp()
    net_name += f'_{args.task}'

    if args.task in ['mri']:
        net_name += f'{args.mri_acceleration}x_tag{args.mri_tag}'
    if args.task in ['inpainting']:
        net_name += f'{args.mask_rate}'
    if args.task in ['ct']:
        net_name += f'{args.ct_views}'


    net_name += f'_{args.mode}'

    # ei
    if args.__contains__('transform'):
        if args.transform:
            net_name +=f'_{args.transform}'
    if args.__contains__('group_size'):
        if args.group_size:
            net_name +=f'_G{args.group_size}'
    if args.__contains__('n_trans'):
        if args.n_trans:
            net_name +=f'_t{args.n_trans}'

    # mi
    if args.__contains__('n_operators'):
        if args.n_operators:
            net_name +=f'_G{args.n_operators}'
    if args.__contains__('n_sub_operators'):
        if args.n_trans:
            net_name +=f'g{args.n_sub_operators}'

    # ei & mi
    if args.__contains__('alpha'):
        if args.alpha:
            net_name += f'_alpha{args.alpha}'

    # arch

    net_name += f'_{args.arch}'
    if args.__contains__('compact'):
        if args.compact:
            net_name += f'{args.compact}'
    if args.__contains__('residual'):
        if args.residual:
            net_name += '_res'
    if args.__contains__('cat'):
        if args.cat:
            net_name += '_cat'

    # unrolling, equilibrium
    if args.__contains__('unrolling') and args.__contains__('time_step') and args.__contains__('step_size')and args.__contains__('_tied'):
        if args.unrolling:
            net_name +=f'_{args.unrolling}_{args.time_step}'
        if args.tied:
            net_name += f'_tied'

    if args.__contains__('equilibrium') and args.__contains__('step_size'):
        if args.equilibrium:
            net_name +=f'_dem_{args.equilibrium}_tied'

    return net_name
