import sys
sys.path.append('../deepinv')
import deepinv as dinv


# dataloader = dinv.datasets.mnist_dataloader(mode='train', batch_size=128, num_workers=4, shuffle=True)
dataloader = dinv.datasets.mnist_dataloader(mode='test', batch_size=64, num_workers=4, shuffle=True)
save_dir = '..'
# physics = dinv.physics.CompressedSensing(m=100, img_shape=(1,28,28), save=True, save_dir=save_dir).to(dinv.device)

physics = dinv.physics.Inpainting(tensor_size=(28,28), save=True, device=dinv.device)
physics.noise_model = dinv.physics.GaussianNoise(std=0)

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         residual=True,
                         scales=3).to(dinv.device)

model = dinv.models.ar(backbone, pinv=True)

loss_mc = dinv.loss.MCLoss(metric=dinv.metric.mse(dinv.device))
loss_sup = dinv.loss.SupLoss(metric=dinv.metric.mse(dinv.device))

loss_ei = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                           metric=dinv.metric.mse(dinv.device))

loss_mcsure = dinv.loss.SureMCLoss(sigma=.3)

loss_ms = dinv.loss.MeaSplitLoss(metric=dinv.metric.mse(dinv.device),
                                 split_ratio=0.9)

optimizer = dinv.optim.Adam(model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-8)

dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=1e-4,
           physics=physics,
           epochs=500,
           schedule=[400],
           # loss_closure=[loss_ms],
           loss_closure=[loss_mc,loss_ei],
           # loss_closure=[loss_sup],
           # loss_closure=[loss_mc],
           # loss_weight=[1],
           loss_weight=[1,1],
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_ei',
           verbose=True)

# CS matrix has been LOADED from ../saved_forward/CS/forw_cs_784x100_G1/forw_g0.pt
# [dinv_ms]	22-12-30-08:49:19	[  1/500]	loss=5.024e-01	loss_ms=5.024e-01	psnr_fbp=10.11	psnr_net=1.24
# [dinv_ms]	22-12-30-08:49:35	[  2/500]	loss=3.819e-01	loss_ms=3.819e-01	psnr_fbp=10.11	psnr_net=1.23
# [dinv_ms]	22-12-30-08:49:51	[  3/500]	loss=3.286e-01	loss_ms=3.286e-01	psnr_fbp=10.11	psnr_net=1.24
# [dinv_ms]	22-12-30-08:50:09	[  4/500]	loss=2.971e-01	loss_ms=2.971e-01	psnr_fbp=10.11	psnr_net=1.25
# [dinv_ms]	22-12-30-08:50:25	[  5/500]	loss=2.766e-01	loss_ms=2.766e-01	psnr_fbp=10.11	psnr_net=1.26
# [dinv_ms]	22-12-30-08:50:40	[  6/500]	loss=2.618e-01	loss_ms=2.618e-01	psnr_fbp=10.11	psnr_net=1.27
# [dinv_ms]	22-12-30-08:50:55	[  7/500]	loss=2.490e-01	loss_ms=2.490e-01	psnr_fbp=10.11	psnr_net=1.27
# [dinv_ms]	22-12-30-08:51:11	[  8/500]	loss=2.396e-01	loss_ms=2.396e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:51:26	[  9/500]	loss=2.314e-01	loss_ms=2.314e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:51:41	[ 10/500]	loss=2.243e-01	loss_ms=2.243e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:51:57	[ 11/500]	loss=2.184e-01	loss_ms=2.184e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:52:12	[ 12/500]	loss=2.140e-01	loss_ms=2.140e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:52:28	[ 13/500]	loss=2.096e-01	loss_ms=2.096e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:52:44	[ 14/500]	loss=2.057e-01	loss_ms=2.057e-01	psnr_fbp=10.11	psnr_net=1.28
# [dinv_ms]	22-12-30-08:53:00	[ 15/500]	loss=2.021e-01	loss_ms=2.021e-01	psnr_fbp=10.11	psnr_net=1.28