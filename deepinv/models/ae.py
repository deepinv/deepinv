import torch


class AutoEncoder(torch.nn.Module):
    r"""
    Simple fully connected autoencoder network.

    Simple architecture that can be used for debugging or fast prototyping.

    :param int dim_input: total number of elements (pixels) of the input.
    :param int dim_hid: number of features in intermediate layer.
    :param int dim_hid: latent space dimension.
    :param int residual: use a residual connection between input and output.

    """

    def __init__(self, dim_input, dim_mid=1000, dim_hid=32, residual=True):
        super().__init__()
        self.residual = residual

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_mid, dim_hid),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim_hid, dim_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_mid, dim_input),
        )

    def forward(self, x, sigma=None):
        N, C, H, W = x.shape
        x = x.view(N, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.residual:
            decoded = decoded + x

        decoded = decoded.view(N, C, H, W)
        return decoded
