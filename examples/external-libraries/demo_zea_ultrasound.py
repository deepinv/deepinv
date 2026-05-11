"""
Ultrasound image reconstruction with zea and deepinv
=====================================================================

This example demonstrates ultrasound image reconstruction (beamforming),
combining the `zea <https://zea.readthedocs.io/>`_
ultrasound toolbox with `deepinv <https://deepinv.github.io/>`_.

.. note::

    Work in progress!  This example is currently being developed and
    is not be fully functional yet.

:class:`deepinv.physics.UltrasoundWithZea` requires the ``zea`` package and
the PyTorch Keras backend, which can be installed with:
``pip install zea``

Set ``KERAS_BACKEND=torch`` **before** importing ``zea`` or ``deepinv``.
"""

# %%
# Set up the Keras / zea backend **first** (before any other imports)
# -----------------------------------------------------------------------
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import importlib

import deepinv as dinv

if importlib.util.find_spec("zea") is not None:
    from zea.data import load_file

else:
    raise ModuleNotFoundError(
        "This example requires the zea package.\n"
        "Install with:  pip install zea\n"
        "Then set:      os.environ['KERAS_BACKEND'] = 'torch'"
    )

device = dinv.utils.get_device()

# %%
# Load PICMUS experimental data
# --------------------------------------------------------------------------------
# The PICMUS contrast-speckle phantom dataset is hosted on Hugging Face.
# ``load_file`` returns raw IQ channel data together with ``zea.Probe``
# and ``zea.Scan`` objects that describe the acquisition geometry.


HF_PATH = (
    "hf://zeahub/picmus/database/experiments/"
    "contrast_speckle/contrast_speckle_expe_dataset_iq/"
    "contrast_speckle_expe_dataset_iq.hdf5"
)

data, scan, probe = load_file(
    path=HF_PATH,
    indices=[0],  # load only the first frame
    data_type="raw_data",
)

rf_frame_raw = data[0]
N_TX = rf_frame_raw.shape[0]
print(f"RF data shape : {rf_frame_raw.shape}  (n_tx, n_ax, n_el, n_ch)")
print(f"Total transmits : {N_TX}")
