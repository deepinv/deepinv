import torch
import deepinv as dinv
device = dinv.utils.get_device()

def open_m4raw(fname):
    x = dinv.io.load_ismrmd(fname, data_slice=8).unsqueeze(0) #12NHW
    x = dinv.utils.MRIMixin().kspace_to_im(x)
    x = dinv.utils.MRIMixin().rss(x, multicoil=True) #11HW # this would be better with virtual coil comb, but we follow the M4Raw paper and use RSS
    x = dinv.utils.normalize_signal(x, mode="min_max")
    return x

DATA_DIR = dinv.utils.get_data_home() / "m4raw" / "motion"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Note: these are T1 MRI
# Load y (raw image) and x (average of 3 repetitions; there is inter scan motion so this is blurry)

dinv.utils.download_example("demo_m4raw_inter-scan_motion_0.h5", DATA_DIR)
dinv.utils.download_example("demo_m4raw_inter-scan_motion_1.h5", DATA_DIR)
dinv.utils.download_example("demo_m4raw_inter-scan_motion_2.h5", DATA_DIR)

y = open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_0.h5")

x = torch.cat([
    open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_0.h5"),
    open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_1.h5"),
    open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_2.h5"),
]).mean(dim=0, keepdim=True)

noise_estimator = dinv.models.PatchCovarianceNoiseEstimator()

dinv.utils.plot({"Noisy scan": y, "3x rep, averaged": x}, subtitles=[f"sigma: {noise_estimator(y).item():.4f}", f"sigma: {noise_estimator(x).item():.4f}"])

sigma = noise_estimator(y)
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=sigma))

model = dinv.models.RAM(device=device)
with torch.no_grad():
    x_net = model(y.to(device), physics)

dataset = dinv.datasets.TensorDataset(y=y, params={"sigma": sigma})

# Self-supervised fine-tuning with Recorrupted2Recorrupted loss :footcite:p:`pang2021recorrupted`

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=1),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
    epochs=1 if str(device) == "cpu" else 50,
    losses=dinv.loss.R2RLoss(noise_model=None),
    metrics=None,
    device=device,
    save_path=None,
)
model = trainer.train()

with torch.no_grad():
    x_ft = model(y.to(device), physics)


dinv.utils.plot({
    "Noisy scan": y,
    "3x rep, averaged": x,
    "Zero-shot RAM": x_net,
    "Fine-tuned RAM": x_ft
},)