import torch


class MetricsRecorder:
    r"""
    Records per-iteration convergence metrics of an optimization algorithm.

    Layout: ``metrics[name][i][j]`` is the metric ``name`` for batch element ``i``
    at iteration ``j`` (same layout as ``BaseOptim`` metrics in ``deepinv.optim``).
    Keys: ``residual`` and ``psnr`` always, ``cost`` as soon as a cost is passed to
    :meth:`update`, plus one key per custom metric.

    A no-op when ``enabled=False``, so the ``metrics.update(...)`` line inside the
    algorithm loop costs nothing at inference.

    :param torch.Tensor x_init: initial iterate, used for the batch size and the
        initial PSNR entry.
    :param torch.Tensor x_gt: ground truth image, enables PSNR. Default: ``None``.
    :param dict custom_metrics: dictionary ``{name: fn}`` where
        ``fn(history, x_prev_i, x_i) -> float`` and ``history`` is the list of past
        values of this metric for this batch element. Default: ``None``.
    :param bool enabled: if ``False``, all methods are no-ops and :meth:`as_dict`
        returns ``None``. Default: ``False``.
    """

    def __init__(self, x_init, x_gt=None, custom_metrics=None, enabled=False):
        self.enabled = enabled
        if not enabled:
            return
        from deepinv.loss.metric.distortion import PSNR

        self.x_gt = x_gt
        self.custom_metrics = custom_metrics if custom_metrics is not None else {}
        self.psnr_fn = PSNR()
        B = self.batch_size = x_init.shape[0]

        self.metrics = {"residual": [[] for _ in range(B)]}
        if x_gt is not None:
            # the PSNR of the initialization is the first entry
            self.metrics["psnr"] = [
                [self.psnr_fn(x_init[i : i + 1], x_gt[i : i + 1]).cpu().item()]
                for i in range(B)
            ]
        else:
            self.metrics["psnr"] = [[] for _ in range(B)]
        for name in self.custom_metrics:
            self.metrics[name] = [[] for _ in range(B)]
        # "cost" is created lazily on the first update() that receives one

    def update(self, x_prev, x, cost=None):
        r"""
        Records the metrics of one iteration.

        :param torch.Tensor x_prev: iterate before the step.
        :param torch.Tensor x: iterate after the step.
        :param torch.Tensor cost: per-batch objective value at ``x`` (shape ``(B,)``
            or broadcastable), recorded under the ``cost`` key. Default: ``None``.
        """
        if not self.enabled:
            return
        if cost is not None and "cost" not in self.metrics:
            self.metrics["cost"] = [[] for _ in range(self.batch_size)]
        cost = cost.reshape(-1) if isinstance(cost, torch.Tensor) else cost
        for i in range(self.batch_size):
            res = (
                ((x_prev[i] - x[i]).norm() / (x[i].norm() + 1e-6)).detach().cpu().item()
            )
            self.metrics["residual"][i].append(res)
            if self.x_gt is not None:
                psnr = self.psnr_fn(x[i : i + 1], self.x_gt[i : i + 1])
                self.metrics["psnr"][i].append(psnr.cpu().item())
            if cost is not None:
                c = cost[i] if hasattr(cost, "__getitem__") else cost
                self.metrics["cost"][i].append(
                    c.detach().cpu().item() if isinstance(c, torch.Tensor) else float(c)
                )
            for name, fn in self.custom_metrics.items():
                self.metrics[name][i].append(fn(self.metrics[name], x_prev[i], x[i]))

    def as_dict(self):
        r"""Returns the recorded metrics, or ``None`` if the recorder is disabled."""
        return self.metrics if self.enabled else None
