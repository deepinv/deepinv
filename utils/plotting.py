import matplotlib.pyplot as plt

def plot_debug(img,titles=None):
    imgs = []
    for input in img:
        imgs.append(input[0,:,:,:].clamp(min=0.,max=1.).detach().permute(1, 2, 0).cpu().numpy())

    plt.figure(figsize=(len(imgs),1))

    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap='gray')
        if titles:
            plt.title(titles[i], size=8)
        plt.axis('off')

    plt.show()
