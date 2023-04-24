import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def torch2cpu(img):
    return (
        img[0, :, :, :]
        .clamp(min=0.0, max=1.0)
        .detach()
        .permute(1, 2, 0)
        .squeeze()
        .cpu()
        .numpy()
    )


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def numpy2uint(img):
    img = img.clip(0, 1)
    return np.uint8((img * 255.0).round())


def im_save(save_img_path, img):
    img = numpy2uint(img)
    plt.imsave(save_img_path, img)


def plot_batch(imgs_in, titles=None, save_dir=None, tight=True, max_imgs=8):
    if save_dir:
        if not os.path.exists(save_dir.split("/")[0]):
            print("Creating ", save_dir.split("/")[0], " folder...")
            os.makedirs(save_dir.split("/")[0])

    imgs = []
    for im in imgs_in:
        col_imgs = []
        for i in range(min(im.shape[0], max_imgs)):
            if im.shape[1] == 2:  # for complex images
                pimg = im[i, :, :, :].pow(2).sum(dim=0).sqrt().unsqueeze(0)
            else:
                pimg = im[i, :, :, :]

            col_imgs.append(
                pimg.clamp(min=0.0, max=1.0)
                .detach()
                .permute(1, 2, 0)
                .squeeze()
                .cpu()
                .numpy()
            )

        imgs.append(col_imgs)

    plt.figure(figsize=(len(imgs), len(imgs[0]) * 1.3))

    for i, row_imgs in enumerate(imgs):
        for r, img in enumerate(row_imgs):
            plt.subplot(len(imgs[0]), len(imgs), r * len(imgs) + i + 1)

            plt.imshow(img, cmap="gray")
            if titles and r == 0:
                plt.title(titles[i], size=8)
            plt.axis("off")

    if tight:
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

    if save_dir:
        plt.savefig(save_dir, dpi=1200)

    plt.show()


def plot_debug(
    imgs, shape=None, titles=None, row_order=False, save_dir=None, tight=True, show=True
):
    if save_dir:
        if not os.path.exists(save_dir.split("/")[0]):
            print("Creating ", save_dir.split("/")[0], " folder...")
            os.makedirs(save_dir.split("/")[0])

    if torch.is_tensor(imgs[0]):
        imgs = [torch2cpu(im) for im in imgs]

    if not shape:
        shape = (1, len(imgs))

    plt.figure(figsize=(shape[1], 1.2 * shape[0]))

    for i, img in enumerate(imgs):
        if row_order:
            r = i % shape[0]
            c = int((i - r) / shape[0])
            idx = r * shape[1] + c
        else:
            r = int(i / shape[1])
            idx = i

        plt.subplot(shape[0], shape[1], idx + 1)

        plt.imshow(img, cmap="gray")
        if titles and r == 0:
            plt.title(titles[i], size=8)
        plt.axis("off")

    if tight:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if save_dir:
        plt.savefig(save_dir, dpi=1200)
    if show:
        plt.show()
