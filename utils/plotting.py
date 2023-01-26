import matplotlib.pyplot as plt
import torch


def torch2cpu(img):
    return img[0, :, :, :].clamp(min=0., max=1.).detach().permute(1, 2, 0).cpu().numpy()


def plot_debug(imgs, shape=None, titles=None, row_order=False, save_dir=None, tight=True):

    if torch.is_tensor(imgs[0]):
        imgs = [torch2cpu(i) for i in imgs]

    if not shape:
        shape = (len(imgs), 1)

    plt.figure(figsize=(shape[1], shape[0]))

    for i, img in enumerate(imgs):
        if row_order:
            r = i % shape[0]
            c = int((i - r)/shape[0])
            idx = r * shape[1] + c
        else:
            c = int(i/shape[1])
            idx = i

        plt.subplot(shape[0], shape[1], idx + 1)

        plt.imshow(img, cmap='gray')
        if titles and c == 0:
            plt.title(titles[i], size=8)
        plt.axis('off')

    if tight:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if save_dir:
        plt.savefig(save_dir)

    plt.show()
