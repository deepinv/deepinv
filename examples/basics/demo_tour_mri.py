r"""
Tour of MRI functionality in DeepInverse
========================================

This example presents the various datasets, forward physics and models
available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:

-  Physics: :class:`deepinv.physics.MRI`,
   :class:`deepinv.physics.MulticoilMRI`,
   :class:`deepinv.physics.DynamicMRI`
-  Datasets: :class:`deepinv.datasets.FastMRISliceDataset`,
   :class:`deepinv.datasets.SimpleFastMRISliceDataset`
-  Models: :class:`deepinv.models.VarNet` (both VarNet and
   E2E-VarNet), :class:`deepinv.utils.demo.demo_mri_model` (a simple
   MoDL unrolled model)

Contents: 1. Get started with FastMRI (singlecoil + multicoil) 2. Train
an accelerated MRI with neural networks 3. Load raw FastMRI data
(singlecoil + multicoil) 4. Train using raw data 5. Explore 3D MRI 6.
Explore dynamic MRI

"""

# %%
# 1. Get started with FastMRI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can get started with our simple FastMRI slice subsets which provide
# quick, easy-to-use, in-memory datasets which can be used for simulation
# experiments.
#
# Load knee and brain datasets:
#

pass


# %%
# Generate an accelerated single-coil MRI measurement dataset. We can
# generate a dataset either using a constant mask, or random mask
# per-image:
#

pass


# %%
# You can also generate multicoil MRI data:
#

pass


# %%
# 2. Train an accelerated MRI problem with neural networks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Train a neural network to solve the inverse problem.
#
# We provide various models�
#

pass


# %%
# 3. Load raw FastMRI data
# ~~~~~~~~~~~~~~~~~~~~~~~~
#

pass


# %%
# 4. Train using raw data
# ~~~~~~~~~~~~~~~~~~~~~~~
#

pass


# %%
# 5. Explore 3D MRI
# ~~~~~~~~~~~~~~~~~
#

pass


# %%
# 6. Explore dynamic MRI
# ~~~~~~~~~~~~~~~~~~~~~~
#

pass
