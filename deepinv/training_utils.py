from deepinv.utils import save_model, AverageMeter, ProgressMeter, get_timestamp, cal_psnr
from deepinv.utils.plotting import plot_debug, torch2cpu
import numpy as np
from tqdm import tqdm
import torch


def train(model,
          train_dataloader,
          epochs,
          loss_closure,
          physics=None,
          scheduler=None,
          optimizer=None,
          device=torch.device(f"cuda:0"),
          ckp_interval=100,
          save_path='.',
          verbose=False,
          unsupervised=False,
          plot=False,):
    """
    Trains a reconstruction model with the train dataloader.
    ----------
    train_dataloader
        A string indicating the name of the person.
    learning_rate
        learning rate of the optimizer
    """

    losses = AverageMeter('loss', ':.2e')
    meters = [losses]
    losses_verbose = []
    psnr_net = []
    psnr_linear = []

    if verbose:
        losses_verbose = [AverageMeter('loss_' + l.name, ':.2e') for l in loss_closure]
        psnr_net = AverageMeter('psnr_net', ':.2f')
        psnr_linear = AverageMeter('psnr_linear', ':.2f')

        for loss in losses_verbose:
            meters.append(loss)
        meters.append(psnr_linear)
        meters.append(psnr_net)

    progress = ProgressMeter(epochs, meters)

    save_path = save_path + f'/{get_timestamp()}'

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params} trainable parameters')

    if type(physics) is not list:
        physics = [physics]

    if type(loss_closure) is not list:
        loss_closure = [loss_closure]

    if type(train_dataloader) is not list:
        train_dataloader = [train_dataloader]

    G = len(train_dataloader)

    f = model
    loss_history = []

    for epoch in range(epochs):
        iterators = [iter(loader) for loader in train_dataloader]
        batches = len(train_dataloader[G - 1])

        for i in range(batches):
            G_perm = np.random.permutation(G)
            for g in G_perm:
                if unsupervised:
                    y = next(iterators[g])
                else:
                    x, y = next(iterators[g])

                if type(x) is list or type(x) is tuple:
                    x = [s.to(device) for s in x]
                else:
                    x = x.to(device)

                y = y.to(device)

                x1 = f(y, physics[g])

                loss_total = 0
                for k, l in enumerate(loss_closure):
                    loss = 0
                    if l.name in ['mc']:
                        loss = l(y, x1, physics[g])
                    if l.name in ['ms']:
                        loss = l(y, physics[g], f)
                    if not unsupervised and l.name in ['sup']:
                        loss = l(x1, x)
                    if l.name in ['moi']:
                        loss = l(x1, physics, f)
                    if l.name.startswith('suremc'):
                        loss = l(y, x1, physics[g], f)
                    if l.name in ['ei', 'rei']:
                        loss = l(x1, physics[g], f)
                    loss_total += loss

                    if verbose:
                        losses_verbose[k].update(loss.item())

                losses.update(loss_total.item())

                if i == 0 and g == 0 and plot:
                    imgs = [physics[g].A_adjoint(y)[0, :, :, :].unsqueeze(0),
                            x1[0, :, :, :].unsqueeze(0)]
                    titles = ['Linear Inv.', 'Estimated']
                    if not unsupervised:
                        imgs.append(x[0, :, :, :].unsqueeze(0))
                        titles.append('Ground Truth')
                    plot_debug(imgs, titles=titles)

                if (not unsupervised) and verbose:
                    psnr_linear.update(cal_psnr(physics[g].A_adjoint(y), x))
                    psnr_net.update(cal_psnr(x1, x))

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        if scheduler:
            scheduler.step()

        loss_history.append(loss_total.detach().cpu().numpy())

        progress.display(epoch + 1)
        save_model(epoch, model, optimizer, ckp_interval, epochs, loss_history, save_path)
    return model


def test(model, test_dataloader,
          physics,
          dtype=torch.float,
          device=torch.device(f"cuda:0"),
          plot=False,
          save_img_path=None):

    f = model
    psnr_linear = []
    psnr_net = []

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)
    imgs = []

    show_operators = 5

    for g in range(G):
        dataloader = test_dataloader[g]
        print(f'Processing data of operator {g+1} out of {G}')
        for i, (x, y) in enumerate(tqdm(dataloader)):

            if type(x) is list or type(x) is tuple:
                x = [s.to(device) for s in x]
            else:
                x = x.type(dtype).to(device)

            y = y.type(dtype).to(device)

            x1 = f(y, physics[g])

            if g < show_operators and i == 0 and plot:
                xlin = physics[g].A_adjoint(y)
                imgs.append(torch2cpu(xlin[0, :, :, :].unsqueeze(0)))
                imgs.append(torch2cpu(x1[0, :, :, :].unsqueeze(0)))
                imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

            psnr_linear.append(cal_psnr(physics[g].A_adjoint(y), x))
            psnr_net.append(cal_psnr(x1, x))

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    pinv_psnr = np.mean(psnr_linear)
    pinv_std_psnr = np.std(psnr_linear)
    print(f'Test PSNR: Linear Inv: {pinv_psnr:.2f}+-{pinv_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. ')

    if plot:
        titles = ['Linear', 'Network', 'Ground Truth']
        plot_debug(imgs, shape=(min(show_operators, G), 3), titles=titles,
                   row_order=True, save_dir=save_img_path)

    return test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr
