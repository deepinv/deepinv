.. _trainer:

Trainer
=======

Training a reconstruction model can be done using the :class:`deepinv.Trainer` class, which can be easily customized
to fit your needs. A trainer can be used for both training and testing a model, and can be used to save and load models.

See :ref:`sphx_glr_auto_examples_basics_demo_train_inpainting` for a simple example of how to use the trainer.


.. _adversarial-networks:

Adversarial Training
--------------------
Adversarial training can be done using the :class:`deepinv.training.AdversarialTrainer` class,
which is a subclass of :class:`deepinv.Trainer`. This class can be used to train a model using adversarial losses
which can be found :ref:`here <adversarial-losses>`.

