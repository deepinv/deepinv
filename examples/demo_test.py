import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch


save_dir = '..'
folder = '22-11-17-14:26:08_dinv_sup' #'22-11-17-14:21:57_dinv_ms' #'22-11-17-08:53:04_dinv_sure_ei' # '22-11-16-13:51:50_dinv_mc_ei' #'22-11-16-13:45:50_dinv_ms_ei' #'22-11-17-08:53:04_dinv_sure_ei' #  '22-11-16-13:45:50_dinv_ms_ei'
ckp = 499

ckp_path = save_dir + '/ckp/' + folder + '/ckp_' + str(ckp) + '.pth.tar'

dataloader = dinv.datasets.mnist_dataloader(mode='test', batch_size=128, num_workers=4, shuffle=False)

physics = dinv.physics.compressed_sensing(m=100, img_shape=(1,28,28), save=True, save_dir=save_dir).to(dinv.device)

model = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         compact=3).to(dinv.device)

model.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
model.eval()

model = dinv.models.ar(model)

dinv.test(model=model,
          test_dataloader=dataloader,
          physics=physics,
          device=dinv.device)