
import deepinv as dinv

# Load the input (clean) image
url = "..."
x = dinv.utils.demo.load_url_image(url=url)

# Define the forward operator
physics = dinv.physics.BlurFFT(
    filter=dinv.physics.blur.gaussian_blur(5,5),
    noise_model=dinv.physics.GaussianNoise(sigma=0.03),
)

# Generate the measurement
y = physics(x)

# Define the data fidelity term
data_fidelity = dinv.optim.data_fidelity.L2()

# Define the prior term
prior = dinv.optim.prior.PnP(denoiser=dinv.models.GSDRUNet(pretrained="download", train=False))

# Set up the optimization algorithm
model = dinv.optim.optim_builder(iteration="PGD",
                                prior=prior, 
                                data_fidelity=data_fidelity,
                                params_algo={"g_param": 0.05, "lambda": 1.}, # sigma and lambda regularization parameters 
                                backtracking=True, # stepsize line-search backtracking 
                                )()

# Run the algorithm
x_hat = model(y, physics)

# Plot the results
dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"])



