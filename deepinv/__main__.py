# Usage: python -m deepinv train --config ./config.yaml

from __future__ import annotations

import deepinv as dinv
import torch
import torch.utils.data
import torchvision.transforms as transforms

import yaml
from typing import Any
from dataclasses import dataclass
import argparse


class CommandLineTrainer:
    @classmethod
    def run(cls, *, config_path: str) -> None:
        parsed_config: cls._ParsedConfig = cls._parse_config(config_path)

        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        model = parsed_config.model
        model.to(device)
        physics = parsed_config.physics
        physics.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-8)
        scheduler = None
        train_dataloader = torch.utils.data.DataLoader(
            parsed_config.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
        )
        trainer = dinv.training.Trainer(
            epochs=10,
            physics=physics,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            device=device,
            online_measurements=True,
        )
        trained_model = trainer.train()
        _ = trained_model

    @dataclass
    class _ParsedConfig:
        dataset: dinv.datasets.ImageDataset
        physics: dinv.physics.Physics
        model: dinv.models.Reconstructor

    @classmethod
    def _parse_config(cls, path: str, /) -> _ParsedConfig:
        with open(path, "r") as f:
            config: Any = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                f"The YAML file does not parse to a dictionary, got: {type(config)}. Make sure it follows the expected format."
            )

        try:
            dataset_config: dict = config.pop("dataset")
            if not isinstance(dataset_config, dict):
                raise ValueError(
                    f"The 'dataset' entry must be a dictionary, got: {type(dataset_config)}."
                )

            dataset_cls_name: str = dataset_config.pop("cls_name")
            dataset_cls: type = getattr(dinv.datasets, dataset_cls_name)
            if not issubclass(dataset_cls, dinv.datasets.ImageDataset):
                raise ValueError(
                    f"Could not find dataset class '{dataset_cls_name}' in 'deepinv.datasets'."
                )

            dataset_args: list = dataset_config.pop("args", [])
            if not isinstance(dataset_args, list):
                raise ValueError(
                    f"The 'dataset.args' entry must be a list, got: {type(dataset_args)}."
                )

            dataset_kwargs: dict = dataset_config.pop("kwargs", {})
            if not isinstance(dataset_kwargs, dict):
                raise ValueError(
                    f"The 'dataset.kwargs' entry must be a dictionary, got: {type(dataset_kwargs)}."
                )

            if dataset_config:
                raise ValueError(
                    f"The 'dataset' entry contains unexpected keys: {list(dataset_config.keys())}."
                )

            dataset_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ]
            )

            dataset = dataset_cls(
                *dataset_args, transform=dataset_transform, **dataset_kwargs
            )

            physics_config: dict = config.pop("physics")
            if not isinstance(physics_config, dict):
                raise ValueError(
                    f"The 'physics' entry must be a dictionary, got: {type(physics_config)}."
                )

            physics_cls_name: str = physics_config.pop("cls_name")
            physics_cls: type = getattr(dinv.physics, physics_cls_name)
            if not issubclass(physics_cls, dinv.physics.Physics):
                raise ValueError(
                    f"Could not find physics class '{physics_cls_name}' in 'deepinv.physics'."
                )

            physics_args: list = physics_config.pop("args", [])
            if not isinstance(physics_args, list):
                raise ValueError(
                    f"The 'physics.args' entry must be a list, got: {type(physics_args)}."
                )

            physics_kwargs: dict = physics_config.pop("kwargs", {})
            if not isinstance(physics_kwargs, dict):
                raise ValueError(
                    f"The 'physics.kwargs' entry must be a dictionary, got: {type(physics_kwargs)}."
                )

            noise_model_config: dict | None = physics_config.pop("noise_model", None)

            if noise_model_config is not None:
                if not isinstance(noise_model_config, dict):
                    raise ValueError(
                        f"The 'physics.noise_model' entry must be a dictionary, got: {type(noise_model_config)}."
                    )

                noise_model_cls_name: str = noise_model_config.pop("cls_name")
                noise_model_cls: type = getattr(dinv.physics, noise_model_cls_name)
                if not issubclass(noise_model_cls, dinv.physics.NoiseModel):
                    raise ValueError(
                        f"Could not find noise model class '{noise_model_cls_name}' in 'deepinv.physics'."
                    )

                noise_model_args: list = noise_model_config.pop("args", [])
                if not isinstance(noise_model_args, list):
                    raise ValueError(
                        f"The 'physics.noise_model.args' entry must be a list, got: {type(noise_model_args)}."
                    )

                noise_model_kwargs: dict = noise_model_config.pop("kwargs", {})
                if not isinstance(noise_model_kwargs, dict):
                    raise ValueError(
                        f"The 'physics.noise_model.kwargs' entry must be a dictionary, got: {type(noise_model_kwargs)}."
                    )

                if noise_model_config:
                    raise ValueError(
                        f"The 'physics.noise_model' entry contains unexpected keys: {list(noise_model_config.keys())}."
                    )

                noise_model: dinv.physics.NoiseModel = noise_model_cls(
                    *noise_model_args, **noise_model_kwargs
                )
            else:
                noise_model = None

            if physics_config:
                raise ValueError(
                    f"The 'physics' entry contains unexpected keys: {list(physics_config.keys())}."
                )

            physics: dinv.physics.Physics = physics_cls(*physics_args, **physics_kwargs)
            if noise_model is not None:
                physics.noise_model = noise_model

            model_config: dict = config.pop("model")
            if not isinstance(model_config, dict):
                raise ValueError(
                    f"The 'model' entry must be a dictionary, got: {type(model_config)}."
                )

            model_cls_name: str = model_config.pop("cls_name")
            model_cls: type = getattr(dinv.models, model_cls_name)
            if not issubclass(model_cls, dinv.models.Reconstructor):
                raise ValueError(
                    f"Could not find model class '{model_cls_name}' in 'deepinv.models'."
                )

            model_args: list = model_config.pop("args", [])
            if not isinstance(model_args, list):
                raise ValueError(
                    f"The 'model.args' entry must be a list, got: {type(model_args)}."
                )

            model_kwargs: dict = model_config.pop("kwargs", {})
            if not isinstance(model_kwargs, dict):
                raise ValueError(
                    f"The 'model.kwargs' entry must be a dictionary, got: {type(model_kwargs)}."
                )

            if issubclass(model_cls, dinv.models.ArtifactRemoval):
                backbonet_net_config: dict = model_config.pop("backbone_net")
                if not isinstance(backbonet_net_config, dict):
                    raise ValueError(
                        f"The 'model.kwargs.backbonet_net' entry must be a dictionary, got: {type(backbonet_net_config)}."
                    )

                backbonet_net_cls_name: str = backbonet_net_config.pop("cls_name")
                backbonet_net_cls: type = getattr(dinv.models, backbonet_net_cls_name)
                if not issubclass(backbonet_net_cls, dinv.models.Denoiser):
                    raise ValueError(
                        f"Could not find denoiser class '{backbonet_net_cls_name}' in 'deepinv.models'."
                    )

                backbonet_net_args: list = backbonet_net_config.pop("args", [])
                if not isinstance(backbonet_net_args, list):
                    raise ValueError(
                        f"The 'model.kwargs.backbonet_net.args' entry must be a list, got: {type(backbonet_net_args)}."
                    )

                backbonet_net_kwargs: dict = backbonet_net_config.pop("kwargs", {})
                if not isinstance(backbonet_net_kwargs, dict):
                    raise ValueError(
                        f"The 'model.kwargs.backbonet_net.kwargs' entry must be a dictionary, got: {type(backbonet_net_kwargs)}."
                    )

                if backbonet_net_config:
                    raise ValueError(
                        f"The 'model.kwargs.backbonet_net' entry contains unexpected keys: {list(backbonet_net_config.keys())}."
                    )

                backbone_net: dinv.models.Denoiser = backbonet_net_cls(
                    *backbonet_net_args, **backbonet_net_kwargs
                )
                if "backbone_net" in model_kwargs:
                    raise ValueError(
                        "The 'model.kwargs' entry contains a 'backbonet_net' key, which is reserved."
                    )
                model_kwargs["backbone_net"] = backbone_net

            if model_config:
                raise ValueError(
                    f"The 'model' entry contains unexpected keys: {list(model_config.keys())}."
                )

            model: dinv.models.Reconstructor = model_cls(*model_args, **model_kwargs)

        except KeyError as e:
            (key,) = e.args
            raise ValueError(f"The mandatory entry for key '{key}' is missing.")

        if config:
            raise ValueError(
                f"The configuration dictionary contains unexpected entries. Keys: {list(config.keys())}."
            )

        return cls._ParsedConfig(
            dataset=dataset,
            physics=physics,
            model=model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using DeepInverse.")

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args: argparse.Namespace = parser.parse_args()

    if args.command == "train":
        trainer = CommandLineTrainer()
        trainer.run(config_path=args.config)
    else:
        raise RuntimeError(f"Unexpected command '{args.command}'.")
