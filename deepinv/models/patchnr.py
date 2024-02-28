import torch
import torch.nn as nn

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except:
    Ff = ImportError("The FrEIA package is not installed.")
    Fm = ImportError("The FrEIA package is not installed.")


class PatchNR(nn.Module):
    r"""
    Patch prior via normalizing flows.

    The forward method evaluates its negative log likelihood.

    :param torch.nn.Module normalizing_flow: describes the normalizing flow of the model. Generally it can be any torch.nn.Module
        supporting backprobagation. It takes a (batched) tensor of flattened patches and the boolean rev (default False)
        as input and provides the value and the log-determinant of the Jacobian of the normalizing flow as an output
        If rev=True, it considers the inverse of the normalizing flow.
        When set to None it is set to a dense invertible neural network built with the FrEIA library, where the number of
        invertible blocks and the size of the subnetworks is determined by the parameters num_lyers and sub_net_size.
    :param str pretrained_weights: Define pretrained weights by its path to a ".pt" file
    :param int patch_size: size of patches
    :param int channels: number of channels for the underlying images/patches.
    :param int num_layers: defines the number of blocks of the generated normalizing flow if normalizing_flow is None.
    :param int sub_net_size: defines the number of hidden neurons in the subnetworks of the generated normalizing flow if normalizing_flow is None.
    :param str device: used device
    """

    def __init__(
        self,
        normalizing_flow=None,
        pretrained_weights=None,
        patch_size=6,
        channels=1,
        num_layers=5,
        sub_net_size=256,
        device="cpu",
    ):
        super(PatchNR, self).__init__()
        if isinstance(Ff, ImportError):
            raise ImportError(
                "FrEIA is needed to use the PatchNR class. "
                "It should be installed with `pip install FrEIA`."
            ) from Ff
        if normalizing_flow is None:
            # Create Normalizing Flow with FrEIA
            dimension = patch_size**2 * channels

            def subnet_fc(c_in, c_out):
                return nn.Sequential(
                    nn.Linear(c_in, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, c_out),
                )

            nodes = [Ff.InputNode(dimension, name="input")]
            for k in range(num_layers):
                nodes.append(
                    Ff.Node(
                        nodes[-1],
                        Fm.GLOWCouplingBlock,
                        {"subnet_constructor": subnet_fc, "clamp": 1.6},
                        name=f"coupling_{k}",
                    )
                )
            nodes.append(Ff.OutputNode(nodes[-1], name="output"))

            self.normalizing_flow = Ff.ReversibleGraphNet(nodes, verbose=False).to(
                device
            )
        else:
            self.normalizing_flow = normalizing_flow
        if pretrained_weights:
            if pretrained_weights[-3:] == ".pt":
                weights = torch.load(pretrained_weights, map_location=device)
            else:
                if pretrained_weights == "PatchNR_lodopab_small":
                    assert patch_size == 3
                    assert channels == 1
                    file_name = "PatchNR_lodopab_small.pt"
                    url = "https://drive.google.com/uc?export=download&id=1Z2us9ZHjDGOlU6r1Jee0s2BBej2XV5-i"
                else:
                    raise ValueError("Pretrained weights not found!")
                weights = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
            self.normalizing_flow.load_state_dict(weights)

    def forward(self, x):
        r"""
        Evaluates the negative log likelihood function of th PatchNR.
        """
        B, n_patches = x.shape[0:2]
        latent_x, logdet = self.normalizing_flow(x.view(B * n_patches, -1))
        logpz = 0.5 * torch.sum(latent_x.view(B, n_patches, -1) ** 2, -1)
        return logpz - logdet.view(B, n_patches)
