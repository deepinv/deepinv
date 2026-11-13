# deepinv.unfolded

This module provides networks architectures based on unfolding optimization algorithms.
Please refer to [user guide](https://deepinv.org/user_guide/reconstruction/unfolded.md#unfolded) for more details.

## Unfolded

**User Guide:** refer to [Unfolded Algorithms](https://deepinv.org/user_guide/reconstruction/unfolded.md#unfolded) for more information.

| [`deepinv.unfolded.unfolded_builder`](https://deepinv.org/api/stubs/deepinv.unfolded.unfolded_builder.md#deepinv.unfolded.unfolded_builder)   | Helper function for building an unfolded architecture.   |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|

| [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.md#deepinv.unfolded.BaseUnfold)   | Base class for unfolded algorithms.   |
|------------------------------------------------------------------------------------------------------------|---------------------------------------|

## Deep Equilibrium

**User Guide:** refer to [Deep Equilibrium](https://deepinv.org/user_guide/reconstruction/unfolded.md#deep-equilibrium) for more information.

| [`deepinv.unfolded.DEQ_builder`](https://deepinv.org/api/stubs/deepinv.unfolded.DEQ_builder.md#deepinv.unfolded.DEQ_builder)   | Helper function for building an instance of the [`deepinv.unfolded.BaseDEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseDEQ.md#deepinv.unfolded.BaseDEQ) class.   |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.unfolded.BaseDEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseDEQ.md#deepinv.unfolded.BaseDEQ)   | Base class for deep equilibrium (DEQ) algorithms.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------|

## Custom Unfolded Blocks

**User Guide:** refer to [Predefined Unfolded Blocks](https://deepinv.org/user_guide/reconstruction/unfolded.md#custom-unfolded-blocks) for more information.

| [`deepinv.models.PDNet_PrimalBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_PrimalBlock.md#deepinv.models.PDNet_PrimalBlock)   | Primal block for the Primal-Dual unfolding model.   |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`deepinv.models.PDNet_DualBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_DualBlock.md#deepinv.models.PDNet_DualBlock)       | Dual block for the Primal-Dual unfolding model.     |
