from deepinv.utils import save_model, AverageMeter, ProgressMeter, get_timestamp, cal_psnr
from deepinv.utils.plotting import plot_debug, torch2cpu
import numpy as np
from tqdm import tqdm
import torch
import wandb

def train(model,
          train_dataloader,
          epochs,
          loss_closure,
          eval_dataloader=None,
          physics=None,
          scheduler=None,
          optimizer=None,
          device=torch.device(f"cuda:0"),
          ckp_interval=100,
          eval_interval=1, 
          save_path='.',
          verbose=False,
          unsupervised=False,
          plot=False,
          plot_input=False,
          wandb_vis=False):
    """
    Trains a reconstruction model with the train dataloader.
    ----------
    train_dataloader
        A string indicating the name of the person.
    learning_rate
        learning rate of the optimizer
    """

    if wandb_vis:
        wandb.watch(model)

    losses = AverageMeter('loss', ':.2e')
    meters = [losses]
    losses_verbose = []
    train_psnr_net = []
    train_psnr_linear = []
    eval_psnr_net = []
    eval_psnr_linear = []

    if verbose:
        losses_verbose = [AverageMeter('loss_' + l.name, ':.2e') for l in loss_closure]
        train_psnr_net = AverageMeter('train_psnr_net', ':.2f')
        train_psnr_linear = AverageMeter('train_psnr_linear', ':.2f')
        eval_psnr_net = AverageMeter('eval_psnr_net', ':.2f')
        eval_psnr_linear = AverageMeter('eval_psnr_linear', ':.2f')

        for loss in losses_verbose:
            meters.append(loss)
        meters.append(train_psnr_linear)
        meters.append(train_psnr_net)
        if eval_dataloader : 
            meters.append(eval_psnr_linear)
            meters.append(eval_psnr_net)

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
    
    if eval_dataloader and type(eval_dataloader) is not list:
        eval_dataloader = [eval_dataloader]

    G = len(train_dataloader)

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

                x1 = model(y, physics[g])   # Requires grad ok



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
                    train_psnr_linear.update(cal_psnr(physics[g].A_adjoint(y), x))
                    train_psnr_net.update(cal_psnr(x1, x))

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        if (not unsupervised) and eval_dataloader and (epoch+1) % eval_interval == 0:
            test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr = test(model, eval_dataloader, physics, device, verbose=False, wandb_vis=wandb_vis, plot_input=plot_input)
            if verbose : 
                eval_psnr_linear.update(test_psnr)
                eval_psnr_net.update(pinv_psnr)

        if scheduler:
            scheduler.step()

        loss_history.append(loss_total.detach().cpu().numpy())
        if wandb_vis :
            wandb.log({"training loss": loss_total})

        progress.display(epoch + 1)
        save_model(epoch, model, optimizer, ckp_interval, epochs, loss_history, save_path)
        
    if wandb_vis :
        wandb.save('model.h5')

    return model


def test(model, test_dataloader,
          physics,
          device=torch.device(f"cuda:0"),
          plot=False,
          plot_input=False,
          save_img_path=None,
          verbose=True,
          wandb_vis=False,
          **kwargs):

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
        if verbose : 
            print(f'Processing data of operator {g+1} out of {G}')
        for i, (x, y) in enumerate(tqdm(dataloader)):

            if type(x) is list or type(x) is tuple:
                x = [s.to(device) for s in x]
            else:
                x = x.to(device)

            y = y.to(device)

            with torch.no_grad():
                x1 = model(y, physics[g], **kwargs)

            if g < show_operators and i == 0 :
                xlin = physics[g].A_adjoint(y)
                if plot :
                    if plot_input : 
                        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
                    imgs.append(torch2cpu(xlin[0, :, :, :].unsqueeze(0)))
                    imgs.append(torch2cpu(x1[0, :, :, :].unsqueeze(0)))
                    imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))
                if wandb_vis :
                    n_plot = min(8,len(x))
                    imgs = []
                    if plot_input : 
                        imgs.append(wandb.Image(y[:n_plot], caption="Input"))
                    imgs.append(wandb.Image(xlin[:n_plot], caption="Linear"))
                    imgs.append(wandb.Image(x1[:n_plot], caption="Estimated"))
                    imgs.append(wandb.Image(x[:n_plot], caption="Ground Truth"))
                    wandb.log({ "images" : imgs})

            psnr_linear.append(cal_psnr(physics[g].A_adjoint(y), x))
            psnr_net.append(cal_psnr(x1, x))

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    pinv_psnr = np.mean(psnr_linear)
    pinv_std_psnr = np.std(psnr_linear)
    if verbose : 
        print(f'Test PSNR: Linear Inv: {pinv_psnr:.2f}+-{pinv_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. ')
    if wandb_vis : 
         wandb.log({
            "Test linear PSNR": pinv_psnr,
            "Test model PSNR": test_psnr})

    if plot:
        titles = ['Linear', 'Network', 'Ground Truth']
        num_im = 3
        if plot_input:
            titles = ['Input'] + titles
            num_im = 4
        plot_debug(imgs, shape=(min(show_operators, G), num_im), titles=titles,
                   row_order=True, save_dir=save_img_path)

    return test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr
