import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch
from torch.utils.data import DataLoader


G = 1
m = 100
physics = []
dataloader = []
dir = f'../datasets/MNIST/G_{G}_m{m}/'

for g in range(G):
    p = dinv.physics.CompressedSensing(m=m, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=False)
    dataloader.append(DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False))

folder = '23-01-26-16:43:34_dinv_moi_demo'
ckp = 0
ckp_path = 'ckp/' + folder + '/ckp_' + str(ckp) + '.pth.tar'

# FNE DnCNN from Terris et al.
denoiser = dinv.models.dncnn(in_channels=1,
                         out_channels=1).to(dinv.device)
ckp_path = '/Users/matthieuterris/Documents/work/research/codes/checkpoints/DnCNN/DNCNN_nch_1_sigma_2.0_ljr_0.001.ckpt'

# DRUNet (not FNE) from Zhang et al.
# denoiser = dinv.models.drunet(in_channels=2,
#                          out_channels=1).to(dinv.device)
# ckp_path = '/Users/matthieuterris/Documents/work/research/codes/checkpoints/zhang_modelzoo/drunet_gray.pth'


# Testing the adjointness
# (it will be important for what follows that our implementation of the adjoint is correct!
# To check this, we check that <Hu,v>=<u,H'v> for random vectors u and v.)

def pow_it(x0, A, At, max_iter=100, tol=1e-3, verbose=True):
    x = torch.randn_like(x0)
    x /= torch.norm(x)
    zold = torch.zeros_like(x)
    for it in range(max_iter):
        y = A(x)
        y = At(y)
        z = torch.matmul(x.reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

        rel_var = torch.norm(z - zold)
        if rel_var < tol and verbose:
            print("Power iteration converged at iteration: ", it, ", val: ", z)
            break
        zold = z
        x = y / torch.norm(y)

    return z

u = torch.randn(1, 28, 28)
Au = physics[0].A(u)

v = torch.randn_like(Au)
Atv = physics[0].A_adjoint(v)

s1 = v.flatten().T @ Au.flatten()
s2 = Atv.flatten().T @ u.flatten()

print("adjointness test: (should be small) ", s1-s2)

def A(x): return physics[0].A(x)
def At(x): return physics[0].A_adjoint(x)

lip = pow_it(u, A, At)

print('Lip cte = ', lip)

denoiser.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
denoiser = denoiser.eval()

pnp_algo = dinv.pnp.ProximalGradient(denoiser, denoise_level=None, gamma = 1/lip, max_iter=500)  # Remove pinv
# Pnp algo has a forward function

dinv.test(model=pnp_algo,  # Safe because it has forward
          test_dataloader=dataloader,
          physics=physics,
          device=dinv.device,
          save_dir='results/results_pnp_CS_MNIST.png')
