.. _training:

Training and Testing
====================


Training a reconstruction model can be done using the Trainer class, which can be easily customized
to fit your needs.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.Trainer
        deepinv.training.AdversarialTrainer

We also provide train and test functions that can be used to train and test a model with a single call.


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.train
        deepinv.test

