import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision
import deepinv as dinv
from deepinv.utils import plot  # , rescale_img
from argparse import ArgumentParser
import wandb
import json
from pathlib import Path
from models.unrolled_dual_MD import get_unrolled_architecture, MirrorLoss
from utils.distributed_setup import setup_distributed
from utils.dataloaders import get_drunet_dataset
from utils.utils import rescale_img, get_wandb_setup
from deepinv.physics.generator import SigmaGenerator, MotionBlurGenerator

torch.backends.cudnn.benchmark = True

with open("config/config.json") as json_file:
    config = json.load(json_file)

TRAIN_DATASET_PATH = config["TRAIN_DATASET_PATH"]
VAL_DATASET_PATH = None
WANDB_LOGS_PATH = config["WANDB_LOGS_PATH"]
PRETRAINED_PATH = config["PRETRAINED_PATH"]
OUT_DIR = Path(".")
CKPT_DIR = OUT_DIR / "ckpts"  # path to store the checkpoints
WANDB_PROJ_NAME = "learned_MD_denoising"  # Name of the wandb project


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


def load_denoising_data(
    patch_size,
    train_batch_size,
    val_batch_size,
    num_workers,
    device="cpu",
    split_val=0.9,
    distribute=False,
    max_num_images=1e6,
    noise_model = 'Gaussian',
    noise_level_min=0.,
    noise_level_max=0.2,
    gaussian_blur = False,
    psf_size = 31,
    motion_blur = False,

):
    """
    Load the training and validation datasets and create the corresponding dataloaders.

    :param torchvision.transforms train_transform: torchvision transform to be applied to the training data
    :param torchvision.transform val_transform: torchvision transform to be applied to the validation data
    :param int train_batch_size: training batch size
    :param int num_workers: number of workers
    :return: training and validation dataloaders
    """
    pin_memory = True if torch.cuda.is_available() else False
    dataset = get_drunet_dataset(
        patch_size,
        device=device,
        pth=TRAIN_DATASET_PATH,
        max_num_images=max_num_images,
    )

    # Calculate lengths for training and datasets (80% and 20%)
    train_size = int(split_val * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    if distribute and dist.is_initialized():
        # batch_size= train_batch_size * dist.get_world_size()
        train_batch_size = train_batch_size // dist.get_world_size()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            dataset=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            dataset=num_workers,
            pin_memory=pin_memory,
            sampler=val_sampler,
        )

    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            train_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
        )

    if noise_model == 'Gaussian':
        noise = dinv.physics.GaussianNoise()
    elif noise_model == 'Poisson':
        noise = dinv.physics.PoissonNoise(clip_positive = True, normalize=False)

    physics_noise = dinv.physics.DecomposablePhysics(
        device=device, noise_model=noise
    )
    noise_generator = SigmaGenerator(sigma_min=noise_level_min, sigma_max=noise_level_max, device=device)

    if gaussian_blur :
        gaussian_blur_generator = GaussianBlurGenerator(psf_size=(psf_size, psf_size), num_channels=1, device=device) 
        + SigmaGenerator(sigma_min=noise_level_min, sigma_max=noise_level_max, device=device)
    if motion_blur :
        motion_generator = MotionBlurGenerator((psf_size, psf_size), l=0.6, sigma=1, device=device)
        + SigmaGenerator(sigma_min=noise_level_min, sigma_max=noise_level_max, device=device)

    val_dataloader = [val_dataloader]
    train_dataloaders = [train_dataloader]
    physics = [physics_noise]
    generators = [noise_generator]

    return train_dataloaders, val_dataloader, physics, generators


def train_model(
    n_layers=10,
    grayscale=False,
    gpu_num=1,
    model_name="dual_DDMD",
    prior_name="wavelet",
    denoiser_name="DRUNET",
    stepsize_init=1.0,
    lamb_init=1.0,
    ckpt_resume=None,
    wandb_resume_id=None,
    seed=0,
    wandb_vis=True,
    epochs=100,
    train_batch_size=16,
    val_batch_size=16,
    patch_size=64,
    lr=1e-4,
    distribute=False,
    num_workers=8 if torch.cuda.is_available() else 0,
    max_num_images=1e6,
    use_mirror_loss=False,
    data_fidelity="L2",
    noise_model="Gaussian",
    noise_level_min=0.,
    noise_level_max=0.2,
    strong_convexity_backward=0.5,
    strong_convexity_forward=0.1,
    strong_convexity_potential='L2'
):

    if distribute:
        device, global_rank = setup_distributed(seed)
    else:
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    operation = "denoising_MD"

    train_batch_size = train_batch_size * gpu_num
    train_dataloader, val_dataloader, physics, physics_generator = load_denoising_data(
        patch_size,
        train_batch_size,
        val_batch_size,
        num_workers,
        device=device,
        distribute=distribute,
        max_num_images=max_num_images,
        noise_model=noise_model,
        noise_level_min=noise_level_min,
        noise_level_max=noise_level_max 
    )

    if not "dual" in model_name:
        use_mirror_loss = True
    else:
        use_dual_iterations = True

    model = get_unrolled_architecture(
        max_iter=n_layers,
        data_fidelity=data_fidelity,
        prior_name=prior_name,
        denoiser_name=denoiser_name,
        stepsize_init=stepsize_init,
        lamb_init=lamb_init,
        device=device,
        use_mirror_loss=use_mirror_loss,
        use_dual_iterations=use_dual_iterations,
        strong_convexity_backward = strong_convexity_backward,
        strong_convexity_forward = strong_convexity_forward,
        strong_convexity_potential = strong_convexity_potential
    )

    losses = [dinv.loss.SupLoss()]
    if use_mirror_loss:
        losses.append(MirrorLoss())

    if dist.is_initialized() and distribute:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=0.1, step_size=int(9 * epochs / 10)
    )

    wandb_setup = get_wandb_setup(
        WANDB_LOGS_PATH, WANDB_PROJ_NAME, mode="online", wandb_resume_id=wandb_resume_id
    )

    if distribute:
        print("Start training on ", device)
        if global_rank == 0:
            show_progress_bar = True
            verbose = True
            plot_images = True
        else:
            show_progress_bar = False
            verbose = False
            plot_images = False
    else:
        show_progress_bar = True
        verbose = True
        plot_images = True

    print(
        "The model has ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters",
    )

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
        eval_interval=1,
        ckp_interval=20,
        online_measurements=True,
        check_grad=True,
        ckpt_pretrained=ckpt_resume,
        freq_plot=1,
        show_progress_bar=show_progress_bar,
        display_losses_eval=True,
    )

    trainer.test(val_dataloader)

    trainer.train()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--grayscale", type=int, default=0)
    parser.add_argument("--data_fidelity", type=str, default="L2")
    parser.add_argument("--noise_model", type=str, default="Gaussian")
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--ckpt_resume", type=str, default="")
    parser.add_argument("--model_name", type=str, default="dual_DDMD")
    parser.add_argument("--use_mirror_loss", type=int, default=0)
    parser.add_argument("--denoiser_name", type=str, default="DRUNET")
    parser.add_argument("--prior_name", type=str, default="wavelet")
    parser.add_argument("--wandb_resume_id", type=str, default="")
    parser.add_argument("--lr_scheduler", type=str, default="multistep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stepsize_init", type=float, default=0.01)
    parser.add_argument("--lamb_init", type=float, default=1.0)
    parser.add_argument("--distribute", type=int, default=0)
    parser.add_argument("--max_num_images", type=int, default=1e6)
    parser.add_argument("--noise_level_min", type=float, default=0.)
    parser.add_argument("--noise_level_max", type=float, default=0.2)
    parser.add_argument("--strong_convexity_backward", type=float, default=1.)
    parser.add_argument("--strong_convexity_forward", type=float, default=1.)
    parser.add_argument("--strong_convexity_potential", type=str, default='L2')

    args = parser.parse_args()

    grayscale = False if args.grayscale == 0 else True
    distribute = False if args.distribute == 0 else True
    use_mirror_loss = False if args.use_mirror_loss == 0 else True
    ckpt_resume = None if args.ckpt_resume == "" else args.ckpt_resume
    wanddb_resume_id = None if args.wandb_resume_id == "" else args.wandb_resume_id
    lr_scheduler = None if args.lr_scheduler == "" else args.lr_scheduler
    epochs = None if args.epochs == 0 else args.epochs

    train_model(
        data_fidelity=args.data_fidelity,
        noise_model=args.noise_model,
        model_name=args.model_name,
        prior_name=args.prior_name,
        denoiser_name=args.denoiser_name,
        stepsize_init=args.stepsize_init,
        lamb_init=args.lamb_init,
        distribute=distribute,
        epochs=epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        patch_size=args.patch_size,
        seed=args.seed,
        gpu_num=args.gpu_num,
        lr=args.lr,
        max_num_images=args.max_num_images,
        use_mirror_loss=use_mirror_loss,
        noise_level_min=args.noise_level_min,
        noise_level_max=args.noise_level_max,
        strong_convexity_backward=args.strong_convexity_backward,
        strong_convexity_forward=args.strong_convexity_forward,
        strong_convexity_potential=args.strong_convexity_potential
    )