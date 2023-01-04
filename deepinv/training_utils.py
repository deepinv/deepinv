from utils.nn import adjust_learning_rate, save_model
from utils.logger import AverageMeter, ProgressMeter, get_timestamp
from utils.metric import cal_psnr
from utils.plotting import plot_debug
import numpy as np
import torch


def load_checkpoint(model, path_checkpoint, device):
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def data_parallel(model, ngpu=1):
    if ngpu > 1:
        model = torch.nn.DataParallel(model, list(range(ngpu)))
    return model


def train(model,
          train_dataloader,
          learning_rate,
          epochs,
          schedule,
          loss_closure=None,  # list
          optimizer=None,
          physics=None,
          dtype=torch.float,
          device=torch.device(f"cuda:0"),
          ckp_interval=100,
          save_path=None,
          verbose=False,
          unsupervised=False,
          plot=False,
          save_dir='.'):

    losses = AverageMeter('loss', ':.3e')
    meters = [losses]
    losses_verbose = []
    if verbose:
        losses_verbose = [AverageMeter('loss_' + l.name, ':.3e') for l in loss_closure]
        psnr_net = AverageMeter('psnr_net', ':.2f')
        psnr_fbp = AverageMeter('psnr_fbp', ':.2f')

        for loss in losses_verbose:
            meters.append(loss)
        meters.append(psnr_fbp)
        meters.append(psnr_net)

    progress = ProgressMeter(epochs, meters, surfix=f"[{save_path}]")

    save_path = save_dir + '/ckp/{}'.format('_'.join([get_timestamp(), save_path]))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params} trainable parameters')

    if type(physics) is not list:
        physics = [physics]

    if type(train_dataloader) is not list:
        train_dataloader = [train_dataloader]

    G = len(train_dataloader)

    f = model

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate, cos=False, epochs=epochs, schedule=schedule)
        iterators = [iter(loader) for loader in train_dataloader]
        batches = len(train_dataloader[G - 1])


        for i in range(batches):
            G_perm = np.random.permutation(G)
            for g in G_perm:
                if unsupervised:
                    y = next(iterators[g])
                else:
                    x, y = next(iterators[g])

                #x = x[0] if isinstance(x, list) else x
                #y = physics(x)  # generate noisy measurement input y
                x = x.type(dtype).to(device)
                y = y.type(dtype).to(device)

                x1 = f(y, physics[g])

                loss_total = 0
                for l in loss_closure:
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
                        for loss_verbose in losses_verbose:
                            loss_verbose.update(loss.item())

                losses.update(loss_total.item())

                if i == 0 and plot:
                    imgs = [physics[g].A_adjoint(y), x1]
                    titles = ['Linear Inv.', 'Estimated']
                    if not unsupervised:
                        imgs.append(x)
                        titles.append('Supervised')
                    plot_debug(imgs, titles)

                if (not unsupervised) and verbose:
                    psnr_fbp.update(cal_psnr(physics[g].A_adjoint(y), x, normalize=True))
                    psnr_net.update(cal_psnr(x1, x, normalize=True))

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        progress.display(epoch + 1)
        save_model(epoch, model, optimizer, ckp_interval, epochs, save_path)

    return model



def test(model,
          test_dataloader,
          physics,
          dtype=torch.float,
          device=torch.device(f"cuda:0"),
          plot=True):

    f = model
    psnr_fbp = []
    psnr_net = []

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)

    for g in range(G):
        dataloader = test_dataloader[g]
        for i, (x, y) in enumerate(dataloader):
            #x = x[0] if isinstance(x, list) else x
            x = x.type(dtype).to(device)
            y = y.type(dtype).to(device)
            #y0 = physics(x)  # generate measurement input y

            x1 = f(y, physics[g])

            if i == 0 and plot:
                plot_debug([physics[g].A_adjoint(y), x1, x], ['Linear Inv.', 'Estimated', 'Ground Truth'])

            psnr_fbp.append(cal_psnr(physics[g].A_adjoint(y), x, normalize=True))
            psnr_net.append(cal_psnr(x1, x, normalize=True))

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    pinv_psnr = np.mean(psnr_fbp)
    pinv_std_psnr = np.std(psnr_fbp)
    print(f'Test PSNR: Linear Inv: {pinv_psnr:.2f}+-{pinv_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. ')

    return test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr
