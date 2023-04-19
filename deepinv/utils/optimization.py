import torch.nn as nn


class NeuralIteration(nn.Module):
    def __init__(self):
        super(NeuralIteration, self).__init__()

    def init(self, backbone_blocks, step_size=1.0, iterations=1):
        self.iterations = iterations
        self.n_blocks = len(backbone_blocks) if isinstance(backbone_blocks, list) else 1
        self.blocks = backbone_blocks

        self.register_parameter(
            name="step_size",
            param=torch.nn.Parameter(
                step_size * torch.ones(self.iterations), requires_grad=True
            ),
        )
        if self.n_blocks > 1:  # weight_tied=False (many blocks)
            assert (
                self.n_blocks == iterations
            ), "'# blocks' does not equal to 'iterations'"
            self.blocks = torch.nn.ModuleList(
                [backbone_blocks[_] for _ in range(iterations)]
            )
        else:  # weight_tied=True (only one block)
            self.blocks = torch.nn.ModuleList([backbone_blocks])

    def forward(self, y, physics, x_init=None):
        return physics.A_adjoint(y)

    @staticmethod
    def measurement_consistency_grad(physics, x, y):
        # grad(||y-Ax||) = A^T(y-Ax)
        return physics.A_adjoint(
            y.to(physics.device) - physics.A(x.to(physics.device))
        ).to(x.device)


class GradientDescent(NeuralIteration):
    def __init__(self, backbone_blocks, step_size=1.0, iterations=1):
        super(GradientDescent, self).__init__()
        self.name = "gd"
        self.init(backbone_blocks, step_size, iterations)

    def forward(self, y, physics, x_init=None):
        x = x_init.clone() if x_init is not None else physics.A_adjoint(y)
        for t in range(self.iterations):
            if self.n_blocks == 1:
                t = 0
            x = (
                self.blocks[t](x)
                + x
                + self.step_size[t]
                * NeuralIteration.measurement_consistency_grad(physics, x, y)
            )
        return x


class ProximalGradientDescent(NeuralIteration):
    def __init__(self, backbone_blocks, step_size=1.0, iterations=1):
        super(ProximalGradientDescent, self).__init__()
        self.name = "pgd"
        self.init(backbone_blocks, step_size, iterations)
        self.mc_grad = NeuralIteration.measurement_consistency_grad

    def forward(self, y, physics, x_init=None):
        x = x_init.clone() if x_init is not None else physics.A_adjoint(y)
        for t in range(self.iterations):
            if self.n_blocks == 1:
                t = 0
            x = self.blocks[t](
                x
                + self.step_size[t]
                * NeuralIteration.measurement_consistency_grad(physics, x, y)
            )
        return x


if __name__ == "__main__":
    import torch
    import deepinv as dinv

    net = dinv.models.unet().to(dinv.device)
    physics = dinv.physics.Inpainting([32, 32], device=dinv.device)

    x = torch.randn(10, 1, 32, 32).to(dinv.device)
    y = physics.A(x)
    fbp = physics.A_dagger(y)
    x_rec = net(fbp)

    unroll = ProximalGradientDescent(net, step_size=1.0, iterations=1)
    x_unroll = unroll(y, physics, x_init=fbp)

    print("iterations=3")
    iterations = 3
    step_size = 1.0
    blocks = [dinv.models.unet().to(dinv.device) for _ in range(iterations)]

    physics = dinv.physics.Inpainting([32, 32], device=dinv.device)
    x = torch.randn(10, 1, 32, 32).to(dinv.device)
    y = physics.A(x)
    fbp = physics.A_dagger(y)

    unroll = ProximalGradientDescent(blocks, step_size=step_size, iterations=iterations)
    x_unroll = unroll(y, physics, x_init=fbp)

    print(f"iterations={iterations}", x.shape, y.shape, fbp.shape, x_unroll.shape)
