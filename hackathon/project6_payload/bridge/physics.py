from __future__ import annotations

from typing import Any

import torch

from .containers import ensure_batch_channel, infer_tensor_shape, sirf_to_torch, tensor_to_sirf_like

try:
    from deepinv.physics import LinearPhysics as _DeepInvLinearPhysics
except Exception:  # pragma: no cover - host machine does not have deepinv installed
    class _DeepInvLinearPhysics(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()


class _SIRFLinearApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, physics: "EmissionTomographyWithSIRF") -> torch.Tensor:
        ctx.physics = physics
        with torch.no_grad():
            return physics._apply_operator_impl(x, mode="forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with torch.no_grad():
            grad_input = ctx.physics._apply_operator_impl(grad_output, mode="adjoint")
        return grad_input, None


class EmissionTomographyWithSIRF(_DeepInvLinearPhysics):
    """Bridge a linear SIRF emission-tomography operator into DeepInverse."""

    def __init__(
        self,
        operator: Any,
        image_template: Any,
        measurement_template: Any,
        *,
        device: str | torch.device = "cpu",
        enable_autograd: bool = True,
        operator_norm: float | torch.Tensor | None = None,
        normalize: bool = False,
        norm_max_iter: int = 12,
        norm_tol: float = 1e-3,
        norm_seed: int = 0,
        **kwargs,
    ):
        self.operator = operator
        self.image_template = image_template
        self.measurement_template = measurement_template
        self.image_tensor_shape = infer_tensor_shape(image_template)
        self.measurement_tensor_shape = infer_tensor_shape(measurement_template)
        self.enable_autograd = enable_autograd
        self._operator_norm: float | None = None
        try:
            super().__init__(
                A=self._call_forward,
                A_adjoint=self._call_adjoint,
                img_size=self.image_tensor_shape,
                **kwargs,
            )
        except TypeError:
            super().__init__()
        self._device = torch.device(device)
        if operator_norm is not None:
            self.set_operator_norm(operator_norm)
        elif normalize:
            self.calibrate_operator_norm(
                max_iter=norm_max_iter,
                tol=norm_tol,
                seed=norm_seed,
                verbose=False,
            )

    def _call_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs or not self.enable_autograd:
            return self._apply_operator_impl(x, mode="forward", **kwargs)
        return _SIRFLinearApply.apply(x, self)

    def _call_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply_operator_impl(y, mode="adjoint", **kwargs)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._call_forward(x, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._call_adjoint(y, **kwargs)

    def A_raw(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply_operator_impl(x, mode="forward", apply_normalization=False, **kwargs)

    def A_adjoint_raw(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply_operator_impl(y, mode="adjoint", apply_normalization=False, **kwargs)

    def image_tensor_from_sirf(self, image) -> torch.Tensor:
        return sirf_to_torch(image, device=self._device, add_batch=True, add_channel=True)

    def measurement_tensor_from_sirf(self, data) -> torch.Tensor:
        return sirf_to_torch(data, device=self._device, add_batch=True, add_channel=True)

    def image_from_tensor(self, tensor: torch.Tensor):
        return tensor_to_sirf_like(self.image_template, tensor[0] if tensor.ndim >= 1 else tensor)

    def measurement_from_tensor(self, tensor: torch.Tensor):
        return tensor_to_sirf_like(
            self.measurement_template,
            tensor[0] if tensor.ndim >= 1 else tensor,
        )

    @property
    def operator_norm(self) -> float | None:
        return self._operator_norm

    def set_operator_norm(self, value: float | torch.Tensor | None) -> None:
        if value is None:
            self._operator_norm = None
            return
        scale = float(torch.as_tensor(value).detach().cpu().item())
        if scale <= 0.0:
            raise ValueError(f"operator_norm must be positive, got {scale}.")
        self._operator_norm = scale

    def clear_operator_norm(self) -> None:
        self._operator_norm = None

    def _power_iteration(
        self,
        *,
        max_iter: int = 12,
        tol: float = 1e-3,
        seed: int = 0,
        use_normalization: bool = False,
        squared: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        shape = (1,) + self.image_tensor_shape
        generator = torch.Generator(device=self._device)
        generator.manual_seed(seed)
        with torch.no_grad():
            x = torch.randn(shape, generator=generator, device=self._device, dtype=torch.float32)
            x = x / torch.linalg.vector_norm(x)
            last = None
            eig = None
            for iteration in range(max_iter):
                y = self._apply_operator_impl(
                    x,
                    mode="forward",
                    apply_normalization=use_normalization,
                    **kwargs,
                )
                z = self._apply_operator_impl(
                    y,
                    mode="adjoint",
                    apply_normalization=use_normalization,
                    **kwargs,
                )
                eig = torch.real(torch.vdot(x.reshape(-1), z.reshape(-1)))
                norm_z = torch.linalg.vector_norm(z)
                if float(norm_z) <= 0.0:
                    raise RuntimeError("Encountered zero norm during power iteration.")
                if last is not None:
                    rel = torch.abs(eig - last) / max(torch.abs(eig).item(), 1e-12)
                    if float(rel) < tol:
                        if verbose:
                            print(
                                f"Power iteration converged at iteration {iteration}, sqnorm={float(eig):.6e}"
                            )
                        break
                x = z / norm_z
                last = eig
        if eig is None:
            raise RuntimeError("Power iteration did not run.")
        eig = torch.clamp(eig, min=0.0)
        return eig if squared else torch.sqrt(eig)

    def estimate_operator_norm(
        self,
        *,
        max_iter: int = 12,
        tol: float = 1e-3,
        seed: int = 0,
        use_normalization: bool = False,
        squared: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> float:
        value = self._power_iteration(
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            use_normalization=use_normalization,
            squared=squared,
            verbose=verbose,
            **kwargs,
        )
        return float(value.detach().cpu().item())

    def calibrate_operator_norm(
        self,
        *,
        max_iter: int = 12,
        tol: float = 1e-3,
        seed: int = 0,
        verbose: bool = False,
        **kwargs,
    ) -> float:
        norm = self.estimate_operator_norm(
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            use_normalization=False,
            squared=False,
            verbose=verbose,
            **kwargs,
        )
        self.set_operator_norm(norm)
        return norm

    def _apply_operator_impl(
        self,
        tensor: torch.Tensor,
        *,
        mode: str,
        apply_normalization: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        raw_shape = (
            self.image_tensor_shape[1:]
            if mode == "forward"
            else self.measurement_tensor_shape[1:]
        )
        template = self.image_template if mode == "forward" else self.measurement_template
        batched = ensure_batch_channel(tensor, raw_shape)
        outputs = []
        for sample in batched:
            sirf_input = tensor_to_sirf_like(template, sample, squeeze_channel=True)
            if mode == "forward":
                sirf_output = self.operator.forward(sirf_input, **kwargs)
            else:
                sirf_output = self.operator.backward(sirf_input, **kwargs)
            outputs.append(
                sirf_to_torch(
                    sirf_output,
                    device=tensor.device,
                    copy=True,
                    add_batch=False,
                    add_channel=True,
                )
            )
        out = torch.stack(outputs, dim=0)
        if apply_normalization and self._operator_norm is not None:
            out = out / out.new_tensor(self._operator_norm)
        return out


SIRFLinearPhysics = EmissionTomographyWithSIRF


def adjointness_error(
    physics: EmissionTomographyWithSIRF,
    *,
    trials: int = 3,
    seed: int = 0,
) -> float:
    """Monte-Carlo adjointness check."""
    generator = torch.Generator(device=physics._device)
    generator.manual_seed(seed)
    worst = 0.0
    for _ in range(trials):
        x = torch.randn((1,) + physics.image_tensor_shape, generator=generator, device=physics._device)
        y = torch.randn((1,) + physics.measurement_tensor_shape, generator=generator, device=physics._device)
        lhs = torch.sum(torch.conj(physics.A(x)) * y)
        rhs = torch.sum(torch.conj(x) * physics.A_adjoint(y))
        rel = torch.abs(lhs - rhs) / max(torch.abs(lhs).item(), torch.abs(rhs).item(), 1e-12)
        worst = max(worst, float(rel))
    return worst
