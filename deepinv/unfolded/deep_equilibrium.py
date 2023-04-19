import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.utils import str_to_class
from deepinv.optim.data_fidelity import L2


class BaseDEQ(BaseUnfold):
    def __init__(self, *args, max_iter_backward=50, crit_conv_backward=1e-5, **kwargs):
        super(BaseDEQ, self).__init__(*args, **kwargs)
        self.max_iter_backward = max_iter_backward

    def get_params_it(self, it):
        cur_params_dict = {
            key: value[0]
            for key, value in zip(self.params_algo.keys(), self.params_algo.values())
        }
        return cur_params_dict

    def update_prior_fn(self, it):
        prior_cur = {
            key: value[0] for key, value in zip(self.prior.keys(), self.prior.values())
        }
        return prior_cur

    def forward(self, y, physics):
        init_params = self.get_params_it(0)
        x = self.get_init(init_params, y, physics)
        with torch.no_grad():
            x = self.fixed_point(x, y, physics)
        cur_prior = self.update_prior_fn(0)
        cur_params = self.update_params_fn_pre(0, x, None)
        x = self.fixed_point.iterator(x, cur_prior, cur_params, y, physics)["est"][0]
        x0 = x.clone().detach().requires_grad_()
        f0 = self.fixed_point.iterator(
            {"est": (x0,)}, cur_prior, cur_params, y, physics
        )["est"][0]

        def backward_hook(grad):
            backward_iterator = lambda y, *args: {
                "est": (
                    torch.autograd.grad(f0, x0, y["est"][0], retain_graph=True)[0]
                    + grad,
                )
            }
            if self.anderson_acceleration:
                backward_FP = AndersonAcceleration(
                    backward_iterator,
                    update_params_fn_pre=self.update_params_fn_pre,
                    update_prior_fn=self.update_prior_fn,
                    max_iter=self.max_iter_backward,
                    history_size=self.anderson_history_size,
                    beta=self.anderson_beta,
                    early_stop=self.early_stop,
                    crit_conv=self.crit_conv_backward,
                    verbose=self.verbose,
                )
            else:
                backward_FP = FixedPoint(
                    backward_iterator,
                    update_params_fn_pre=self.update_params_fn_pre,
                    update_prior_fn=self.update_prior_fn,
                    max_iter=self.max_iter_backward,
                    early_stop=False,
                    verbose=self.verbose,
                )
            g = backward_FP({"est": (grad,)}, None)["est"][0]
            return g

        if x.requires_grad:
            x.register_hook(backward_hook)
        return x


def DEQ(
    algo_name,
    params_algo,
    trainable_params=[],
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    max_iter_backward=50,
    beta=1.0,
    **kwargs
):
    iterator_fn = str_to_class(algo_name + "Iteration")
    iterator = iterator_fn(data_fidelity=data_fidelity, g_first=g_first, beta=beta)
    return BaseDEQ(
        iterator,
        params_algo=params_algo,
        trainable_params=trainable_params,
        F_fn=F_fn,
        max_iter_backward=max_iter_backward,
        **kwargs
    )
