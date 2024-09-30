import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision
import deepinv as dinv
from deepinv.utils import plot #, rescale_img
from argparse import ArgumentParser
import wandb
import json
from pathlib import Path

from utils.distributed_setup import setup_distributed
from utils.dataloaders import get_drunet_dataset
from utils.utils import get_model, rescale_img, get_wandb_setup

torch.backends.cudnn.benchmark = True

with open('config/config.json') as json_file:
    config = json.load(json_file)

TRAIN_DATASET_PATH = config['TRAIN_DATASET_PATH']
VAL_DATASET_PATH = config['VAL_DATASET_PATH']
WANDB_LOGS_PATH = config['WANDB_LOGS_PATH']
PRETRAINED_PATH = config['PRETRAINED_PATH']
OUT_DIR = Path(".")
CKPT_DIR = OUT_DIR / "ckpts"  # path to store the checkpoints
WANDB_PROJ_NAME = 'shared_repo'  # Name of the wandb project


class MyTrainer(dinv.training.Trainer):
    def __init__(self, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)

    def to_image(self, x):
        r"""
        Convert the tensor to an image. Necessary for complex images (2 channels)

        :param torch.Tensor x: input tensor
        :return: image
        """
        if x.shape[1] == 2:
            out = torch.moveaxis(x, 1, -1).contiguous()
            out = torch.view_as_complex(out).abs().unsqueeze(1)
        else:
            out = x
        return out

    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            if len(y.shape) == len(x.shape) and y.shape != x.shape:
                y_reshaped = torch.nn.functional.interpolate(y, size=x.shape[2])
                if hasattr(physics_cur, "A_adjoint"):
                    imgs = [y_reshaped, physics_cur.A_adjoint(y), x_net, x]
                    caption = (
                        "From top to bottom: input, backprojection, output, target"
                    )
                    titles = ["Input", "Backprojection", "Output", "Target"]
                else:
                    imgs = [y_reshaped, x_net, x]
                    titles = ["Input", "Output", "Target"]
                    caption = "From top to bottom: input, output, target"
            else:
                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        back = physics_cur.module.A_adjoint(y)
                    else:
                        back = physics_cur.A_adjoint(y)
                    imgs = [back, x_net, x]
                    titles = ["Backprojection", "Output", "Target"]
                    caption = "From top to bottom: backprojection, output, target"
                elif y.shape == x.shape:
                    imgs = [y, x_net, x]
                    titles = ["Measurement", "Output", "Target"]
                    caption = "From top to bottom: measurement, output, target"
                else:
                    imgs = [x_net, x]
                    caption = "From top to bottom: output, target"
                    titles = ["Output", "Target"]

            # Concatenate the images along the batch dimension
            for i in range(len(imgs)):
                imgs[i] = self.to_image(imgs[i])

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return imgs, titles, grid_image, caption

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot the images.

        It plots the images at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"
        if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
            imgs, titles, grid_image, caption = self.prepare_images(
                physics, x, y, x_net
            )

            # normalize the grid image
            # grid_image = rescale_img(grid_image, rescale_mode="min_max")

            # if MRI in class name, rescale = min-max
            if "MRI" in str(physics):
                rescale_mode = "min_max"
            else:
                rescale_mode = "clip"
            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode=rescale_mode,
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch)




def load_data(train_patch_size, train_batch_size, num_workers, device='cpu', train=True):
    """
    Load the training and validation datasets and create the corresponding dataloaders.

    :param torchvision.transforms train_transform: torchvision transform to be applied to the training data
    :param torchvision.transform val_transform: torchvision transform to be applied to the validation data
    :param int train_batch_size: training batch size
    :param int num_workers: number of workers
    :return: training and validation dataloaders
    """
    pin_memory = True if torch.cuda.is_available() else False

    denoising_dataset, noise_generator, physics_noise = get_drunet_dataset(train_patch_size, device=device, train=train,
                                                                           train_pth=TRAIN_DATASET_PATH,
                                                                           val_pth=VAL_DATASET_PATH,
                                                                           sigma_min=0.2,
                                                                           sigma_max=0.2)

    shuffle = True if train else False
    if dist.is_initialized():
        # batch_size= train_batch_size * dist.get_world_size()
        train_batch_size = train_batch_size // dist.get_world_size()
        sampler = DistributedSampler(denoising_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                     shuffle=shuffle)
        denoising_dataloader = DataLoader(denoising_dataset, batch_size=train_batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)

    else:
        batch_size = train_batch_size
        denoising_dataloader = DataLoader(
            denoising_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
            drop_last=True
        )


    dataloaders = [denoising_dataloader]
    physics = [physics_noise]
    generators = [noise_generator]

    return dataloaders, physics, generators


def train_denoiser(grayscale=False,
                   gpu_num=1,
                   model_name='drunet',
                   ckpt_resume=None,
                   wandb_resume_id=None,
                   seed=0,
                   wandb_vis=True,
                   epochs=None,
                   train_batch_size=None,
                   train_patch_size=None,
                   lr=1e-4):

    device, global_rank = setup_distributed(seed)

    operation = model_name + "_denoising_"

    num_workers = 8 if torch.cuda.is_available() else 0

    if train_patch_size is None:
        train_patch_size = 128

    if train_batch_size is None:
        base_bs = 32 if 'unext' in model_name else 64
        train_batch_size = base_bs*gpu_num  # TODO: check redundancy with multigpu distributed

    train_dataloader, physics, physics_generator = load_data(train_patch_size, train_batch_size, num_workers,
                                                             device=device,
                                                             train=True)
    val_dataloader, _, _ = load_data(train_patch_size, train_batch_size, num_workers, device=device, train=False)

    model = get_model()

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # For DRUNet, the authors suggest using 800k training steps
    # This corresponds to 1501 epochs with batch size 16 and 6004 epochs with batch size 64  (800k steps)
    if epochs is None:
        epochs = 1000 * train_batch_size

    # setup wandb

    # choose training losses
    losses = dinv.loss.SupLoss(metric=dinv.metric.l1())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=int(9 * epochs / 10))

    wandb_setup = get_wandb_setup(WANDB_LOGS_PATH, WANDB_PROJ_NAME, mode='offline', wandb_resume_id=wandb_resume_id)

    print('Start training on ', device)
    if global_rank == 0:
        show_progress_bar = True
        verbose = True
        plot_images = True
    else:
        show_progress_bar = False
        verbose = False
        plot_images = False

    print('The model has ', sum(p.numel() for p in model.parameters() if p.requires_grad), 'parameters')

    trainer = MyTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        epochs=epochs,
        scheduler=scheduler,
        losses=losses,
        physics=physics,
        physics_generator=physics_generator,
        optimizer=optimizer,
        device=device,
        save_path=str(CKPT_DIR / operation),
        verbose=verbose,
        wandb_vis=wandb_vis,
        wandb_setup=wandb_setup,
        plot_images=plot_images,
        eval_interval=20,
        ckp_interval=20,
        online_measurements=True,
        check_grad=True,
        ckpt_pretrained=ckpt_resume,
        freq_plot=1,
        show_progress_bar=show_progress_bar
    )

    trainer.train()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--grayscale', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--ckpt_resume', type=str, default='')
    parser.add_argument('--model_name', type=str, default='drunet')
    parser.add_argument('--wandb_resume_id', type=str, default='')
    parser.add_argument('--lr_scheduler', type=str, default='multistep')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=0)
    parser.add_argument('--train_patch_size', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)

    args = parser.parse_args()

    grayscale = False if args.grayscale == 0 else True
    ckpt_resume = None if args.ckpt_resume == '' else args.ckpt_resume
    wanddb_resume_id = None if args.wandb_resume_id == '' else args.wandb_resume_id
    lr_scheduler = None if args.lr_scheduler == '' else args.lr_scheduler
    epochs = None if args.epochs == 0 else args.epochs
    train_batch_size = None if args.train_batch_size == 0 else args.train_batch_size
    train_patch_size = None if args.train_patch_size == 0 else args.train_patch_size

    train_denoiser(model_name=args.model_name, epochs=epochs, train_batch_size=train_batch_size,
                   train_patch_size=train_patch_size, seed=args.seed, gpu_num=args.gpu_num,lr=args.lr)
