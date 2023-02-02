import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch
from torch.utils.data import DataLoader

G = 1  # number of operators
epochs = 2  # number of training epochs
num_workers = 4  # set to 0 if using small cpu
batch_size = 128  # choose if using small cpu/gpu
plot = True
dataset = 'MNIST'
problem = 'deblur'
dir = f'../datasets/MNIST/{problem}/G{G}/'

physics = []
dataloader = []
for g in range(G):
    if problem == 'CS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device)
    elif problem == 'onebitCS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device)
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == 'inpainting':
        p = dinv.physics.Inpainting(tensor_size=(1, 28, 28), mask=.5, device=dinv.device)
    elif problem == 'denoising':
        p = dinv.physics.Denoising(sigma=.2)
    elif problem == 'blind_deblur':
        p = dinv.physics.BlindBlur(kernel_size=11)
    elif problem == 'deblur':
        p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(1, .5)), device=dinv.device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=False)
    dataloader.append(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False))

# FNE DnCNN from Terris et al.
denoiser = dinv.models.dncnn(in_channels=1,
                         out_channels=1).to(dinv.device)

ckp_path = '../saved_models/DNCNN_nch_1_sigma_2.0_ljr_0.001.ckpt'
# https://drive.google.com/file/d/1VSVwDvoEPHOEIiM1Dc37sonTu7pMQ4Xa
# DRUNet (not FNE) from Zhang et al.
# denoiser = dinv.models.drunet(in_channels=2,
#                          out_channels=1).to(dinv.device)
# ckp_path = '/Users/matthieuterris/Documents/work/research/codes/checkpoints/zhang_modelzoo/drunet_gray.pth'


# Testing the adjointness
# (it will be important for what follows that our implementation of the adjoint is correct!
# To check this, we check that <Hu,v>=<u,H'v> for random vectors u and v.)

# def pow_it(x0, A, At, max_iter=100, tol=1e-5, verbose=True):
#     x = torch.randn_like(x0)
#     x /= torch.norm(x)
#     zold = torch.zeros_like(x)
#     for it in range(max_iter):
#         y = A(x)
#         y = At(y)
#         z = torch.matmul(x.reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2
#
#         rel_var = torch.norm(z - zold)
#         if rel_var < tol and verbose:
#             print("Power iteration converged at iteration: ", it, ", val: ", z)
#             break
#         zold = z
#         x = y / torch.norm(y)
#
#     return z

u = torch.randn((1, 28, 28), device=dinv.device) #.type(dtype)
# Au = physics[0].A(u)
#
# v = torch.randn_like(Au)
# Atv = physics[0].A_adjoint(v)
#
# s1 = v.flatten().T @ Au.flatten()
# s2 = Atv.flatten().T @ u.flatten()
#
# print("adjointness test: (should be small) ", s1-s2)
#
# def A(x): return physics[0].A(x)
# def At(x): return physics[0].A_adjoint(x)

# lip = pow_it(u, A, At)
#
# print('Lip cte = ', lip)
#
lip = physics[0].power_method(u.unsqueeze(0), tol=1e-5)
print('Lip cte = ', lip)

testadj = physics[0].adjointness_test(u.unsqueeze(0))
print('Adj test : ', testadj)

denoiser.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
denoiser = denoiser.eval()

pnp_algo = dinv.pnp.ProximalGradient(denoiser, denoise_level=None, gamma=0.15/lip, max_iter=20, verbose=False)  # Remove pinv
# Pnp algo has a forward function

dinv.test(model=pnp_algo,  # Safe because it has forward
          test_dataloader=dataloader,
          physics=physics,
          device=dinv.device,
          plot=plot,
          save_img_path='../results/results_pnp_CS_MNIST.png')
