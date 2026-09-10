<a id="training"></a>

# deepinv.training

This module contains the training and testing functions.
Please refer to the [user guide](https://deepinv.org/user_guide/training/trainer.html.md#trainer) for more information.

| [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer)   | Trainer class for training a reconstruction network.   |
|------------------------------------------------------------------------------------|--------------------------------------------------------|

| [`deepinv.test`](https://deepinv.org/api/stubs/deepinv.test.html.md#deepinv.test)   | Tests a reconstruction model (algorithm or network).   |
|------------------------------------------------------------------------------|--------------------------------------------------------|

## Adversarial Training

| [`deepinv.training.AdversarialTrainer`](https://deepinv.org/api/stubs/deepinv.training.AdversarialTrainer.html.md#deepinv.training.AdversarialTrainer)     | Trainer class for training a reconstruction network using adversarial learning.                     |
|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`deepinv.training.AdversarialOptimizer`](https://deepinv.org/api/stubs/deepinv.training.AdversarialOptimizer.html.md#deepinv.training.AdversarialOptimizer) | Optimizer for adversarial training that encapsulates both generator and discriminator's optimizers. |
