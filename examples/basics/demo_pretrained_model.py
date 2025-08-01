r"""
Use a pretrained model
====================================================================================================

Follow this example to 

# Demo pretrained model: RAM simple inference + finetuning
# TODO reconstructor pretrained docs = simple table + reference this example
# TODO Simple inference/FT RAM (include: Want a better model or want help? get in touch!);  (online, offline) (ref custom physics example)

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`


"""

import deepinv as dinv

model = dinv.models.MedianFilter()
# just single image here, not dataset
