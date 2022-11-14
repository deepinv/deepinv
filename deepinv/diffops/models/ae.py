import torch

class AE(torch.nn.Module):
    def __init__(self, residual=False, dim_input=28*28, dim_hid=32):
        super().__init__()
        self.residual = residual

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_input, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, dim_hid),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim_hid, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, dim_input)
        )

    def forward(self, x):
        N,C,H,W = x.shape
        x = x.view(N, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.residual:
            decoded = decoded + x

        decoded = decoded.view(N,C,H,W)
        return decoded