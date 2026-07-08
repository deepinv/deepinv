try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Failed to import torch. Please install PyTorch first.')

import sirf.SIRF as sirf


# This module provides a PyTorch interface for SIRF operators and objective functions.
# It wraps SIRF objects to enable their use within PyTorch's autograd system.
# The core idea is to use torch.autograd.Function to define custom forward and backward
# passes that interact with SIRF's C++ backend.  This allows SIRF operations to be
# seamlessly integrated into PyTorch computational graphs.
#
# based on
# https://github.com/educating-dip/pet_deep_image_prior/blob/main/src/deep_image_prior/torch_wrapper.py


def sirf_to_torch(
        sirf_src: sirf.DataContainer | float,
        device: torch.device,
        requires_grad: bool = False
        ) -> torch.Tensor:
    """
    Converts a SIRF object to a PyTorch tensor.

    Args:
        sirf_src: The SIRF object to convert.  This can be a SIRF
          `DataContainer` such as `AcquisitionData`, `ImageData`, or a scalar (float).
        device: The PyTorch device to place the tensor on (e.g., 'cpu' or
          'cuda').
        requires_grad:  Whether the resulting tensor should track gradients.

    Returns:
        A PyTorch tensor representing the SIRF object's data.

    Raises:
        TypeError: If `sirf_src` is not a supported SIRF object or a float.
    """

    if isinstance(sirf_src, float):
        # Handle float separately for efficiency and type consistency.
        return torch.tensor(sirf_src, requires_grad=requires_grad, \
            device=device
            )
    elif hasattr(sirf_src, 'as_array'):
        # Check for as_array method (ImageData, AcquisitionData, etc.)
        return torch.tensor(sirf_src.as_array(), requires_grad=requires_grad,
            device=device
            )
    else:
        raise TypeError(f"Unsupported SIRF object type: {type(sirf_src)}")

def torch_to_sirf_(
        torch_src: torch.Tensor,
        sirf_dest: sirf.DataContainer,
        ) -> sirf.DataContainer:
    """
    Copies data from a PyTorch tensor to a SIRF object in-place.

    This function *modifies* the `sirf_dest` object. It is crucial that
    `sirf_dest` is pre-allocated with the correct shape and data type. This
    function is primarily intended for use within `torch.autograd.Function`
    where in-place operations are more appropriate.

    Args:
        torch_src: The source PyTorch tensor.  It will be detached from the
            computational graph and moved to the CPU before copying.
        sirf_dest: The destination SIRF object. This object will be modified
            in-place.

    Returns:
        The modified `sirf_dest` object.

    Raises:
        TypeError: if `sirf_dest` is not a supported SIRF object
    """

    if hasattr(sirf_dest, 'fill'):
        sirf_dest.fill(torch_src.detach().cpu().numpy())
        return sirf_dest
    else:
        raise TypeError(f"Unsupported SIRF object type for in-place fill: \
            {type(sirf_dest)}"
            )


class _Operator(torch.autograd.Function):
    """
    A PyTorch autograd Function wrapper for SIRF operators.

    This class allows SIRF operators (e.g., projectors, transformations) to be
    used as part of a PyTorch computational graph.  It handles the forward and
    backward passes, converting between SIRF objects and PyTorch tensors.
    """
    @staticmethod
    def forward(ctx,
            torch_src: torch.Tensor,
            sirf_src_template: sirf.DataContainer,
            sirf_operator
            ) -> torch.Tensor:
        """
        Performs the forward pass of the SIRF operator.

        Args:
            ctx: The PyTorch context object for storing information needed for
              the backward pass.
            torch_src: The input PyTorch tensor.
            sirf_src_template: A SIRF object that serves as a template for the
              input to the SIRF operator. It will have its data replaced by the
              content of `torch_src`.
            sirf_operator: The SIRF operator to apply (e.g., a projector).

        Returns:
            A PyTorch tensor representing the result of the SIRF operator
              applied to the input.
        """

        device = torch_src.device
        sirf_src_template = torch_to_sirf_(torch_src, sirf_src_template)
        sirf_dest = sirf_operator.forward(sirf_src_template)
        if torch_src.requires_grad:
            ctx.device = device
            ctx.sirf_dest = sirf_dest
            ctx.sirf_operator = sirf_operator
            return sirf_to_torch(sirf_dest, device, requires_grad=True)
        else:
            return sirf_to_torch(sirf_dest, device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,
            grad_output: torch.Tensor
            ) -> tuple[torch.Tensor | None, None, None]:

        sirf_src = ctx.sirf_operator.backward(torch_to_sirf_(grad_output,
                ctx.sirf_dest
                )
            )
        grad = sirf_to_torch(sirf_src, ctx.device, requires_grad=False)
        return grad, None, None

class _ObjectiveFunction(torch.autograd.Function):
    """
    A PyTorch autograd Function wrapper for SIRF objective functions.

    This class enables the use of SIRF objective functions within PyTorch
    optimisation loops. It handles calculating the objective function value and
    its gradient.
    """
    @staticmethod
    def forward(ctx,
            torch_image: torch.Tensor,
            sirf_image_template: sirf.ImageData,
            sirf_obj_func
            ) -> torch.Tensor:
        """
        Calculates the value of the SIRF objective function.

        Args:
            ctx: The PyTorch context object.
            torch_image: The input PyTorch tensor (representing an image).
            sirf_image_template: A SIRF image object that serves as a template.
                Its data will be replaced by `torch_image`.
            sirf_obj_func: The SIRF objective function.

        Returns:
            A PyTorch scalar tensor representing the value of the objective
            function.
        """

        device = torch_image.device
        sirf_image = torch_to_sirf_(torch_image, sirf_image_template)
        if torch_image.requires_grad:
            ctx.device = device
            ctx.sirf_image = sirf_image
            ctx.sirf_obj_func = sirf_obj_func
            # ensure value is a tensor with requires_grad=True
            return sirf_to_torch(sirf_obj_func(sirf_image), device,
                requires_grad=True)
        else:
            return sirf_to_torch(sirf_obj_func(sirf_image), device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,
            grad_output: torch.Tensor
            ) -> tuple[torch.Tensor | None, None, None]:
        """
        Calculates the gradient of the SIRF objective function.

        Args:
            ctx: The PyTorch context object.
            grad_output: The gradient of the loss with respect to the output
              of the forward pass (which is the objective function value).
              This will normally be a tensor containing the scalar value 1.

        Returns:
            A tuple containing the gradient of the loss with respect to the
            input image, and None for the other inputs.  The gradient is
            scaled by `grad_output`.
        """

        sirf_obj_func = ctx.sirf_obj_func
        sirf_image = ctx.sirf_image
        device = ctx.device
        sirf_grad = sirf_obj_func.get_gradient(sirf_image)
        grad = sirf_to_torch(sirf_grad, device, requires_grad=False)
        return grad_output*grad, None, None


class _ObjectiveFunctionGradient(torch.autograd.Function):
    """
    A PyTorch autograd Function wrapper for the *gradient* of SIRF objective
    functions. Returns the gradient (not the objective value) in the forward
    pass, and computes the Hessian-vector product in the backward pass.
    """
    @staticmethod
    def forward(ctx,
            torch_image: torch.Tensor,
            sirf_image_template: sirf.ImageData,
            sirf_obj_func
            ) -> torch.Tensor:
        """
        Calculates the *gradient* of the SIRF objective function.

        Args:
            ctx: The PyTorch context object.
            torch_image: The input PyTorch tensor (representing an image).
            sirf_image_template:  A SIRF image object used as a template. Its
              data will be replaced by the content of `torch_image`.
            sirf_obj_func: The SIRF objective function.

        Returns:
            A PyTorch tensor representing the gradient of the objective function.
        """

        device = torch_image.device
        sirf_image = torch_to_sirf_(torch_image, sirf_image_template)
        if torch_image.requires_grad:
            ctx.device = device
            ctx.sirf_image = sirf_image
            ctx.sirf_obj_func = sirf_obj_func
            return sirf_to_torch(sirf_obj_func.get_gradient(sirf_image),
                device, requires_grad=True
                )
        else:
            return sirf_to_torch(sirf_obj_func.get_gradient(sirf_image), device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,
            grad_output: torch.Tensor
            ) -> tuple[torch.Tensor | None, None, None]:
        """
        Calculates the Hessian-vector product (HVP) for the SIRF objective
        function.

        Args:
            ctx: The PyTorch context object.
            grad_output: The gradient of the loss with respect to the output of
              the forward pass (which is the *gradient* of the objective
              function).
              Represents the "vector" in the HVP.

        Returns:
            A tuple containing the HVP, and None for the other inputs. The HVP
            represents the gradient of the loss with respect to the input image,
            accounting for the second-order derivatives.
        """

        sirf_obj_func = ctx.sirf_obj_func
        sirf_image = ctx.sirf_image
        device = ctx.device

        sirf_grad = torch_to_sirf_(grad_output, sirf_image.clone())
        # arguments current estimate and input_ (i.e. the vector)
        sirf_HVP = sirf_obj_func.multiply_with_Hessian(sirf_image, sirf_grad)

        torch_HVP = sirf_to_torch(sirf_HVP, device, requires_grad=False)
        return torch_HVP, None, None

def check_shapes(torch_shape, sirf_shape):
    """
    Checks if the PyTorch and SIRF shapes are compatible.

    Args:
        torch_shape: The shape of the PyTorch tensor (excluding batch and
          channel dimensions).
        sirf_shape: The shape of the SIRF object.

    Raises:
        ValueError: If the shapes are not compatible.
    """
    if torch_shape != sirf_shape:
        raise ValueError(f"Invalid shape. Expected sirf shape {sirf_shape} but \
            got torch shape {torch_shape}")


def apply_wrapped_sirf(wrapped_sirf_func, torch_src, sirf_src_shape):
    """
    Applies a wrapped SIRF function to a batched PyTorch tensor.

    This function handles the batch and channel dimensions of the input tensor,
    applying the wrapped SIRF function to each element of the batch and channel.

    Args:
        wrapped_sirf_func:  A function that takes a single PyTorch tensor
          (representing a single SIRF object's data) and applies the wrapped
          SIRF operation (either an Operator or an ObjectiveFunction).
        torch_src: The input PyTorch tensor. Dimensions (batch, channel,
          *sirf_object_shape).  If the channel dimension is not present, it's
          added temporarily.
        sirf_src_shape: The expected shape of the SIRF object.

    Returns:
        A PyTorch tensor representing the result of applying the wrapped SIRF
        function to each element of the batch and channel.

    Raises:
        ValueError: If the input tensor shape is invalid.
    """
    torch_src_shape = torch_src.shape
    if len(torch_src_shape) == len(sirf_src_shape):
        raise ValueError(f"Invalid shape of src. Expected a batch dim. Such \
            that the dims are [batch, {sirf_src_shape}]")
    elif len(torch_src_shape) == len(sirf_src_shape) + 1:
        check_shapes(torch_src_shape[1:], sirf_src_shape)
        if sirf_src_shape == torch_src_shape[1:]:
            torch_src = torch_src.unsqueeze(1) # add channel dimension
            channel = False
    elif len(torch_src_shape) == len(sirf_src_shape) + 2:
        check_shapes(torch_src_shape[2:], sirf_src_shape)
        channel = True
    else:
        raise ValueError(f"Invalid shape of src. Expected batch (+ channel)\
            dim, and {sirf_src_shape}, got {torch_src_shape}")

    n_batch = torch_src.shape[0]
    n_channel = torch_src.shape[1]

    # This looks horrible, but PyTorch will be able to trace.
    batch_values = []
    for batch in range(n_batch):
        channel_values = []
        for channel in range(n_channel):
            channel_values.append(wrapped_sirf_func(torch_src[batch, channel]))
        batch_values.append(torch.stack(channel_values, dim=0))

    # [batch, channel, *value.shape]
    out = torch.stack(batch_values, dim=0)
    if channel:
        # [batch, channel, *value.shape]
        return out
    else:
        # [batch, *value.shape]
        return out.squeeze(1)

class Operator(torch.nn.Module):
    """
    A PyTorch Module that wraps a SIRF operator for use in a neural network.

    This class allows a SIRF operator (like a projector) to be used as a
    layer within a PyTorch `nn.Module`. It handles the conversion between
    PyTorch tensors and SIRF objects and manages batch and channel dimensions.
    """
    def __init__(self,
            operator,
            sirf_src_template: sirf.DataContainer
            ):
        """
        Initializes the Operator.

        Args:
            operator: The SIRF operator to wrap.
            sirf_src_template: A SIRF object (e.g., `sirf.ImageData`) that
              serves as a template for the input to the operator.  Its shape
              will be used to validate input tensors.  The data in this object
              is *not* used during the forward pass, only its geometry.
        """
        super(Operator, self).__init__()
        # get the shape of src
        self.wrapped_sirf_operator = lambda x: _Operator.apply(x,
            sirf_src_template,
            operator
            )

        self.sirf_src_shape = sirf_src_template.shape

    def forward(self, torch_src: torch.Tensor) -> torch.Tensor:
        """
        Applies the wrapped SIRF operator to the input tensor.

        Args:
            torch_src: The input PyTorch tensor.  Expected dimensions are
              `[batch, channel, *sirf_src_shape]` or `[batch, *sirf_src_shape]`.

        Returns:
            A PyTorch tensor representing the result of the SIRF operator.
            The output dimensions will be `[batch, channel, *sirf_dest_shape]`
            or `[batch, *sirf_dest_shape]`, where `sirf_dest_shape` is the
            natural output shape of the SIRF operator.

        Raises:
            ValueError:  If the input tensor has an invalid shape.  See
                `apply_wrapped_sirf` for details.
        """

        return apply_wrapped_sirf(self.wrapped_sirf_operator, torch_src,
            self.sirf_src_shape
            )


class ObjectiveFunction(torch.nn.Module):
    """
    A PyTorch Module that wraps a SIRF objective function.

    This class allows a SIRF objective function to be evaluated within a
    PyTorch training loop. It handles the necessary conversions and provides
    the objective function value. The gradient is handled by the autograd
    Function `_ObjectiveFunction`.
    """
    def __init__(self,
            sirf_obj_func,
            sirf_image_template: sirf.ImageData
            ):
        """
        Initializes the ObjectiveFunction.

        Args:
            sirf_obj_func: The SIRF objective function to wrap.
            sirf_image_template: A SIRF image object that serves as a template
              for the input to the objective function.  Its shape is used for
              validation, and during the forward pass, its data will be
              temporarily replaced by the input tensor's data.
        """
        super(ObjectiveFunction, self).__init__()
        self.wrapped_sirf_obj_func = lambda x: _ObjectiveFunction.apply(x,
            sirf_image_template, sirf_obj_func
            )

        self.sirf_image_shape = sirf_image_template.shape

    def forward(self, torch_image: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the wrapped SIRF objective function.

        Args:
            torch_image: The input PyTorch tensor (representing an image).
              Expected dimensions are `[batch, channel, *sirf_image_shape]` or
              `[batch, *sirf_image_shape]`.

        Returns:
            A PyTorch tensor representing the value of the objective function.
            The output dimensions will be `[batch, channel]` or `[batch]`.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        return apply_wrapped_sirf(self.wrapped_sirf_obj_func, torch_image,
            self.sirf_image_shape
            )

class ObjectiveFunctionGradient(torch.nn.Module):
    """
    A PyTorch Module that wraps the *gradient* of a SIRF objective function.

    This class provides access to the gradient of a SIRF objective function,
    and computes the Hessian-vector product during the backward pass. This
    is useful for optimization methods that require second-order information
    (e.g., Newton-based methods) or for analyzing the curvature of the
    objective function.
    """
    def __init__(self,
            sirf_obj_func,
            sirf_image_template: sirf.ImageData
            ):
        """
        Initializes the ObjectiveFunctionGradient.

        Args:
            sirf_obj_func: The SIRF objective function.
            sirf_image_template: A SIRF image object that serves as a template
              for the input to the objective function.  Its shape is used for
              validation, and during the forward pass, its data will be
              temporarily replaced by the input tensor's data.
        """
        super(ObjectiveFunctionGradient, self).__init__()
        self.wrapper_sirf_obj_func = lambda x: \
            _ObjectiveFunctionGradient.apply(x, sirf_image_template,
            sirf_obj_func
            )

        self.sirf_image_shape = sirf_image_template.shape

    def forward(self, torch_image):
        """
        Calculates the *gradient* of the wrapped SIRF objective function.

        Args:
            torch_image:  The input image as a PyTorch tensor.  Dimensions:
              `[batch, channel, *sirf_image_shape]` or
              `[batch, *sirf_image_shape]`.

        Returns:
            A PyTorch tensor representing the gradient of the objective function.
            Output dimensions match the input dimensions: `[batch, channel,
            *sirf_image_shape]` or `[batch, *sirf_image_shape]`.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        return apply_wrapped_sirf(self.wrapper_sirf_obj_func, torch_image,
            self.sirf_image_shape
            )
