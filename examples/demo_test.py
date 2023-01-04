import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch


save_dir = '..'
folder = ''
ckp = 499

ckp_path = save_dir + '/ckp/' + folder + '/ckp_' + str(ckp) + '.pth.tar'

dataloader = dinv.datasets.mnist_dataloader(mode='test', batch_size=128, num_workers=4, shuffle=False)

physics = dinv.physics.CompressedSensing(m=100, img_shape=(1, 28, 28), save=True, save_dir=save_dir).to(dinv.device)

model = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
model.eval()

model = dinv.models.ar(model)

dinv.test(model=model,
          test_dataloader=dataloader,
          physics=physics,
          device=dinv.device)