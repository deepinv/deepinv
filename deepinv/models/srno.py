# code borrowed from https://github.com/2y7c3/Super-Resolution-Neural-Operator

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR(nn.Module):
    """EDSR encoder (baseline version)"""

    def __init__(
        self,
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        scale=2,
        no_upsampling=False,
        rgb_range=1,
        n_colors=3,
    ):
        super(EDSR, self).__init__()
        self.no_upsampling = no_upsampling
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv

        # Head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # Body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = n_colors

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    """RDN encoder"""

    def __init__(
        self, G0=64, RDNkSize=3, RDNconfig="B", scale=2, no_upsampling=False, n_colors=3
    ):
        super(RDN, self).__init__()
        self.no_upsampling = no_upsampling
        kSize = RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            "A": (20, 6, 32),
            "B": (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1
        )
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

        if no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = n_colors

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class SimpleAttention(nn.Module):
    """Galerkin attention module"""

    def __init__(self, midc, heads):
        super().__init__()
        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name="0"):
        B, C, H, W = x.shape
        bias = x

        qkv = (
            self.qkv_proj(x)
            .permute(0, 2, 3, 1)
            .reshape(B, H * W, self.heads, 3 * self.headc)
        )
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias

        return bias


class SRNO(nn.Module):
    """
    TODO Super-Resolution Neural Operator with EDSR encoder

    Architecture:
    - Encoder: EDSR-baseline (16 residual blocks, 64 features)
    - Processing: Two Galerkin attention layers (256 width, 16 heads)
    - Decoder: Two 1x1 convolutions
    """

    def __init__(self, encoder_type="rdn", encoder_n_feats=64, width=256, blocks=16):
        super().__init__()
        self.width = width

        if encoder_type == "rdn":
            encoder = RDN(
                G0=encoder_n_feats,
                RDNkSize=3,
                RDNconfig="B",
                scale=2,
                no_upsampling=True,
            )
        elif encoder_type == "edsr":
            encoder = EDSR(
                n_resblocks=16,
                n_feats=encoder_n_feats,
                res_scale=1,
                scale=2,
                no_upsampling=True,
                rgb_range=1,
            )
        else:
            raise NotImplementedError

        self.encoder = encoder

        # Initial convolution: (encoder_n_feats + 2)*4 + 2
        self.conv00 = nn.Conv2d((encoder_n_feats + 2) * 4 + 2, self.width, 1)

        # Galerkin attention blocks
        self.conv0 = SimpleAttention(self.width, blocks)
        self.conv1 = SimpleAttention(self.width, blocks)

        # Output layers
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)

    def gen_feat(self, inp):
        """Generate features from input using encoder"""
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell):
        """
        Query RGB values at arbitrary coordinates

        Args:
            coord: Target coordinates, shape (B, H, W, 2)
            cell: Cell size, shape (B, 2)
        """
        feat = self.feat

        # Generate low-res position grid
        pos_lr = make_coord(feat.shape[-2:], flatten=False).to(feat.device)
        pos_lr = (
            pos_lr.permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        # Calculate offsets for 4-neighbor sampling
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []

        # Sample from 4 nearest neighbors
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # Sample feature
                feat_ = F.grid_sample(
                    feat, coord_.flip(-1), mode="nearest", align_corners=False
                )

                # Calculate relative coordinates
                old_coord = F.grid_sample(
                    pos_lr, coord_.flip(-1), mode="nearest", align_corners=False
                )
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # Calculate area for weighted interpolation
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)

        # Prepare cell information
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        # Weight features by area (bilinear interpolation weights)
        tot_area = torch.stack(areas).sum(dim=0)
        # Swap areas for correct weighting
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        # Concatenate all features: 4 rel_coords + 4 feat_s + rel_cell
        grid = torch.cat(
            [
                *rel_coords,
                *feat_s,
                rel_cell.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, coord.shape[1], coord.shape[2]),
            ],
            dim=1,
        )

        # Process through attention and output layers
        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))

        # Add bilinear upsampled input as residual
        ret = ret + F.grid_sample(
            self.inp,
            coord.flip(-1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return ret

    def forward(self, inp, coord, cell):
        """
        Forward pass

        Args:
            inp: Input image, shape (B, 3, H_lr, W_lr), normalized to [-1, 1]
            coord: Target coordinates, shape (B, H_hr, W_hr, 2), range [-1, 1]
            cell: Cell size, shape (B, 2)

        Returns:
            Super-resolved image, shape (B, 3, H_hr, W_hr)
        """
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


def load_model(checkpoint_path, device="cuda", encoder_type="auto"):
    """
    Load SRNO model from checkpoint

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        encoder_type: 'edsr', 'rdn', or 'auto' to detect from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    model_spec = checkpoint["model"]
    state_dict = model_spec["sd"]

    # Auto-detect encoder type from state dict keys
    if encoder_type == "auto":
        if any("RDBs" in key for key in state_dict.keys()):
            encoder_type = "rdn"
            print("Detected RDN encoder")
        else:
            encoder_type = "edsr"
            print("Detected EDSR encoder")

    # Determine encoder output dimension from state dict
    if encoder_type == "rdn":
        # For RDN, check GFF output
        encoder_out_dim = state_dict["encoder.GFF.0.weight"].shape[0]
    else:
        # For EDSR, check body output
        encoder_out_dim = state_dict["encoder.body.16.weight"].shape[0]

    # Determine width from conv00 output
    width = state_dict["conv00.weight"].shape[0]

    # Determine blocks (heads) - default to 16
    blocks = 16

    print(
        f"Model config: encoder_out_dim={encoder_out_dim}, width={width}, blocks={blocks}"
    )

    # Create model
    model = SRNO(
        encoder_type=encoder_type,
        encoder_n_feats=encoder_out_dim,
        width=width,
        blocks=blocks,
    )

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
