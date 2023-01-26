import math
import torch
import os

def adjust_learning_rate(optimizer, epoch, lr, cos, epochs, schedule):
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(epoch, model, optimizer, ckp_interval, epochs, loss, save_path):
    if (epoch > 0 and epoch % ckp_interval == 0) or epoch + 1 == epochs:
        os.makedirs(save_path, exist_ok=True)

        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'loss':loss,
                 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
    pass
