from dataclasses import dataclass
from deepinv.training.trainer import Trainer


class AdversarialOptimizer:
    def __init__(*args, **kwargs):
        raise DeprecationWarning(
            "AdversarialOptimizer is deprecated. See https://deepinv.github.io/deepinv/auto_examples/adversarial-learning/demo_gan_imaging.html for how to train adversarial networks in DeepInverse."
        )


class AdversarialScheduler(AdversarialOptimizer):
    def __init__(*args, **kwargs):
        raise DeprecationWarning(
            "AdversarialScheduler is deprecated. See https://deepinv.github.io/deepinv/auto_examples/adversarial-learning/demo_gan_imaging.html for how to train adversarial networks in DeepInverse."
        )


@dataclass
class AdversarialTrainer(Trainer):
    def __init__(*args, **kwargs):
        raise DeprecationWarning(
            "AdversarialTrainer is deprecated. See https://deepinv.github.io/deepinv/auto_examples/adversarial-learning/demo_gan_imaging.html for how to train adversarial networks in DeepInverse."
        )
