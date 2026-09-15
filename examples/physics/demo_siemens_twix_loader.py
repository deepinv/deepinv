r"""
Siemens TWIX -> deepinv multi-coil MRI
======================================

Raw Siemens ``.dat`` -> labelled k-space grid -> ESPIRiT coil sensitivities ->
:class:`deepinv.physics.MultiCoilMRI` forward operator, then compare the adjoint
against a SENSE reconstruction on a single slice.

Authors: George Hammerton (St George's NHS), Margaret Duff (STFC-UKRI),
Emily Redpath (St George's NHS), Ben Hawkes (University of Edinburgh),
Matthieu Terris (Blur Labs).
"""

# %%
# Imports and helper functions
# ----------------------------

import io
import urllib.request
import zipfile

import numpy as np
import torch
import matplotlib.pyplot as plt
from twixtools import map_twix
import deepinv as dinv

device = dinv.utils.get_device()


def as_image(x):
    """deepinv tensor (1, 2, H, W) -> 2D magnitude array."""
    x = x.detach().cpu()
    if x.shape[1] == 2:  # real/imag channels
        x = torch.view_as_complex(x.moveaxis(1, -1).contiguous()).abs()
    return x.squeeze().numpy()


def show(img, ax, title="", cmap="gray", vmin=0, p=99.75, cbar=True):
    """imshow with percentile windowing, no ticks/spines, axis-matched colorbar."""
    img = np.asarray(img)
    vmax = np.percentile(img, p) if p is not None else img.max()
    m = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        h, w = img.shape[-2:]
        ax.figure.colorbar(m, ax=ax, fraction=0.047 * h / w, pad=0.02)
    return m


# %%
# Datasets
# --------
#
# We use the Siemens spin-echo scan ``meas_MID03485_FID51095_se.dat`` from the
# `ISMRMRD raw data record on Zenodo <https://zenodo.org/records/33166>`.
#
# The file is cached in :func:`deepinv.utils.get_cache_home`
# (``~/.cache/deepinv`` by default, override with ``DEEPINV_CACHE_DIR``), so the
# download only happens once.
#
# Other public Siemens raw data you could run this on: ``epi.dat`` and ``gre.dat``
# from the `twixtools example data
# <https://github.com/pehses/twixtools/tree/master/example_data>`_.

ZENODO_URL = "https://zenodo.org/api/records/33166/files/ismrmrd_data.zip/content"
ZIP_MEMBER = "ismrmrd_data/Siemens/meas_MID03485_FID51095_se.dat"


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP byte ranges (server must honour them)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head).headers["Content-Length"])

    readable = seekable = lambda self: True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buf):
        end = min(self.pos + len(buf), self.size)
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}
        )
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


# BufferedReader turns zipfile's small reads into a few large range requests.
data_dir = dinv.utils.get_cache_home() / "twix"
path = data_dir / ZIP_MEMBER
if not path.exists():
    with zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(ZENODO_URL), 1 << 22)) as z:
        z.extract(ZIP_MEMBER, data_dir)

# %%
# 1. Load raw k-space
# -------------------
#
# ``map_twix`` sorts the raw acquisition lines into a labelled 16-D grid. We keep
# ``Cha``/``Lin``/``Col`` (and ``Sli`` unless a single slice is requested) and take
# index 0 of anything else, warning if a dropped dimension was non-singleton.


def load_siemanns_dat(path, extract_slice=None, remove_os=True, create_mask=False):
    r"""Load a Siemens .dat file as a deepinv-style k-space tensor.

    Args:
        path (str): path to the Siemens .dat file.
        extract_slice (int, optional): extract only this slice index. Defaults to None (all slices).
        remove_os (bool): remove 2x readout oversampling. Defaults to True.
        create_mask (bool): also return a sampling mask from the non-zero k-space lines.

    Returns:
        torch_kspace (torch.Tensor): k-space, shape (1, 2, Cha, [Sli,] Lin, Col).
        info (dict): ``n_coils`` and the image ``shape``.
        mask (torch.Tensor): sampling mask, only if ``create_mask``.
    """
    mapped = map_twix(str(path))  # twixtools reads a non-str argument as a measID
    im = mapped[-1]["image"]  # last scan; earlier ones are noise/adjustment scans
    im.flags["remove_os"] = remove_os
    im.flags["zf_missing_lines"] = True

    dim_order = im.dim_order
    arr_size = {d: int(im.size[d]) for d in dim_order}
    print(f"  found array of size {arr_size}")

    index, kept, extra = [], [], {}
    for d in dim_order:
        if d in ("Lin", "Cha", "Col") or (d == "Sli" and extract_slice is None):
            index.append(slice(None))
            kept.append(d)
        elif d == "Sli":
            index.append(extract_slice)
        else:
            if arr_size[d] > 1:
                extra[d] = arr_size[d]
            index.append(0)
    if extra:
        print(f"  note: taking index 0 of extra non-singleton dims {extra}")

    kspace = im[tuple(index)]
    # to (Cha, [Sli,] Lin, Col); permutation derived from dim_order so it holds
    # whatever order the file stores its dimensions in
    target = [d for d in ("Cha", "Sli", "Lin", "Col") if d in kept]
    kspace = np.transpose(kspace, [kept.index(d) for d in target])
    print(f"  kept {target} -> {kspace.shape}")

    torch_kspace = torch.from_numpy(kspace)
    if create_mask:
        mask = torch.squeeze(torch.abs(torch_kspace).sum(dim=0) > 0)

    shape = kspace.shape
    torch_kspace = dinv.utils.MRIMixin().from_torch_complex(torch_kspace.unsqueeze(0))
    info = {"n_coils": arr_size["Cha"], "shape": shape[1:]}

    return (torch_kspace, info, mask) if create_mask else (torch_kspace, info)


SLICE = 0
ks, info, mask = load_siemanns_dat(
    path, extract_slice=SLICE, remove_os=True, create_mask=True
)
print(ks.shape, info)

# %%
# 2. Sampling pattern and raw k-space
# -----------------------------------
#
# Sanity check before anything else: energy should be concentrated at the centre of
# k-space, and the mask should show which phase-encode lines were actually acquired.

kc = dinv.utils.MRIMixin().to_torch_complex(ks).squeeze(0)  # (Cha, Lin, Col) complex
rss_k = kc.abs().pow(2).sum(0).sqrt().numpy()  # coil-combined k-space

frac = mask.float().mean().item()
fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
show(np.log1p(rss_k), axes[0], "k-space magnitude, RSS over coils (log)", p=99.5)
show(
    mask.numpy().astype(float),
    axes[1],
    f"Sampling mask - {frac:.0%} of lines acquired",
    p=None,
)
fig.suptitle(
    f"Slice {SLICE}, {info['n_coils']} coils, {info['shape'][0]}x{info['shape'][1]}",
    fontsize=11,
)
plt.show()

# %%
# 3. Coil sensitivity maps
# ------------------------
#
# Note: only works on single slices.

coil_maps = dinv.physics.MultiCoilMRI.estimate_coil_maps(
    ks, calib_size=24, use_cupy=False
)
print(coil_maps.shape, coil_maps.dtype)

cm = coil_maps.squeeze(0).cpu()  # (Cha, Lin, Col) complex
ncol = 6
nrow = int(np.ceil(cm.shape[0] / ncol))
fig, axes = plt.subplots(
    nrow, ncol, figsize=(1.7 * ncol, 2.6 * nrow), constrained_layout=True
)
for i, ax in enumerate(axes.ravel()):
    if i < cm.shape[0]:
        show(cm[i].abs().numpy(), ax, f"coil {i}", cbar=False)
    else:
        ax.axis("off")
fig.suptitle("ESPIRiT coil sensitivity magnitudes", fontsize=11)
plt.show()

# %%
# 4. Forward operator
# -------------------

forward = dinv.physics.MultiCoilMRI(
    mask=mask,
    coil_maps=coil_maps,
    img_size=info["shape"],
    device=device,
    three_d=False,
)

# %%
# 5. Reconstruction
# -----------------
#
# Adjoint :math:`A^{H}y` (coil-combined, no de-aliasing) against the SENSE
# least-squares solution :math:`A^{\dagger}y`.

y = ks.to(device)
adjoint = forward.A_adjoint(y)
sense = forward.A_dagger(y)

fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
show(as_image(adjoint), axes[0], r"Adjoint  $A^{H}y$")
show(as_image(sense), axes[1], r"SENSE  $A^{\dagger}y$")
plt.show()
