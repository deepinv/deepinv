# deepinv.unfolded

This module provides networks architectures based on unfolding optimization algorithms.
Please refer to [user guide](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#unfolded) for more details.

## Unfolded

**User Guide:** refer to [Unfolded Algorithms](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#unfolded) for more information.

| [`deepinv.unfolded.unfolded_builder`](https://deepinv.org/api/stubs/deepinv.unfolded.unfolded_builder.html.md#deepinv.unfolded.unfolded_builder)   | Helper function for building an unfolded architecture.   |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|

| [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold)   | Base class for unfolded algorithms.   |
|------------------------------------------------------------------------------------------------------------|---------------------------------------|

## Deep Equilibrium

**User Guide:** refer to [Deep Equilibrium](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#deep-equilibrium) for more information.

| [`deepinv.unfolded.DEQ_builder`](https://deepinv.org/api/stubs/deepinv.unfolded.DEQ_builder.html.md#deepinv.unfolded.DEQ_builder)   | Helper function for building an instance of the [`deepinv.unfolded.BaseDEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseDEQ.html.md#deepinv.unfolded.BaseDEQ) class.   |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.unfolded.BaseDEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseDEQ.html.md#deepinv.unfolded.BaseDEQ)   | Base class for deep equilibrium (DEQ) algorithms.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------|

## Custom Unfolded Blocks

**User Guide:** refer to [Predefined Unfolded Blocks](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#custom-unfolded-blocks) for more information.

| [`deepinv.models.PDNet_PrimalBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_PrimalBlock.html.md#deepinv.models.PDNet_PrimalBlock)   | Primal block for the Primal-Dual unfolding model.   |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`deepinv.models.PDNet_DualBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_DualBlock.html.md#deepinv.models.PDNet_DualBlock)       | Dual block for the Primal-Dual unfolding model.     |
