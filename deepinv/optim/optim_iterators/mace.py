# -*- coding: utf-8 -*-
import torch
from .optim_iterator import OptimIterator
import warnings


class MACEIteration(OptimIterator):
    r"""Iterator for Multi-Agent Consensus Equilibrium (MACE).

    This iterator implements the MACE algorithm. The update is based on Mann iterations:

    .. math::
        v_i^{(k+1)} = (1-\rho)v_i^{(k)} + \rho \left( (2G-I)(2F-I)v^{(k)} \right)_i

    where :math:`v^{(k)}` is a list of variables, one for each agent,
    :math:`F` applies each agent's proximal operator to its corresponding :math:`v_i`,
    and :math:`G` is an averaging operator based on weights :math:`\mu`.

    This iterator expects the following in ``cur_params`` from ``params_algo``:
    - ``rho``: Mann iteration relaxation parameter (float).
    - ``mu``: List of weights for averaging agent outputs, summing to 1 (list of floats).
    - ``stepsize``: Base stepsize (gamma) for data fidelity agents (float or list).
    - ``lambda``: Regularization parameter for prior agents (float or list).
    - ``g_param``: Parameter for prior agents, e.g., noise level for PnP (any type, or list).

    The agents themselves (data fidelity and prior objects) are configured at initialization.
    """

    def __init__(self, data_fidelity_list, prior_list, **kwargs):
        super().__init__(F_fn=None, has_cost=False)
        self.v_list = None
        self.mu = None

        self.agents_ops = []
        self.agents_gammas_keys = []
        self.agents_sigmas_keys = []
        self._y_ref = None
        self._physics_ref = None

        self._configure_agents(data_fidelity_list, prior_list)
        self.num_agents = len(self.agents_ops)
        if self.num_agents == 0:
            raise ValueError(
                "MACEIteration requires at least one data_fidelity or prior agent."
            )

        self.requires_prox_g = any(ptype == "prior" for ptype, _ in self.agents_ops)
        
        # Build the cost function and check if it's valid
        self.F_fn = self._build_cost_function()
 
    def _build_cost_function(self):
        """
        Builds a cost function that sums the data fidelity and all explicit prior costs.
        """
        def F_fn(x, data_fidelity, prior, cur_params, y, physics):
            total_cost = 0
            prior_idx_counter = 0
            # We need to iterate over the configured agents
            for agent_type, agent_obj in self.agents_ops:
                if agent_type == 'data_fidelity':
                    total_cost += agent_obj(x, y, physics)
                elif agent_type == 'prior':
                    if hasattr(agent_obj, 'explicit_prior') and agent_obj.explicit_prior:
                        lambda_key = f"lambda_{prior_idx_counter}"
                        lambda_val = cur_params.get(lambda_key, cur_params.get("lambda", 1.0))
                        g_param = cur_params.get("g_param")
                        prior_cost = agent_obj(x, g_param)
                        total_cost += (lambda_val * prior_cost).sum()
                    prior_idx_counter += 1
            return total_cost
        return F_fn

    def _configure_agents(self, data_fidelity_list, prior_list):
        """Collects and configures all agent operations from initial lists."""
        df_list = (
            data_fidelity_list
            if isinstance(data_fidelity_list, list)
            else [data_fidelity_list]
        )
        p_list = prior_list if isinstance(prior_list, list) else [prior_list]

        for df_agent in df_list:
            if df_agent is not None and hasattr(df_agent, "prox"):
                self.agents_ops.append(("data_fidelity", df_agent))
                self.agents_gammas_keys.append("stepsize")
                self.agents_sigmas_keys.append(None) 

        prior_idx = 0
        for p_agent in p_list:
            if p_agent is not None and hasattr(p_agent, "prox"):
                self.agents_ops.append(("prior", p_agent))
                self.agents_gammas_keys.append((f"lambda_{prior_idx}", "stepsize"))
                self.agents_sigmas_keys.append("g_param")
                prior_idx += 1

    def _get_agent_params(self, agent_idx, cur_params):
        """Fetches dynamic parameters for a specific agent."""
        gamma_keys = self.agents_gammas_keys[agent_idx]
        sigma_key = self.agents_sigmas_keys[agent_idx]

        gamma = 1.
        if isinstance(gamma_keys, tuple):  # (lambda_key, stepsize_key)
            lambda_key = gamma_keys[0]
            lambda_val = cur_params.get(lambda_key, cur_params.get("lambda", 1.0))
            stepsize_val = cur_params.get(gamma_keys[1], 1.0)
            gamma = lambda_val * stepsize_val
        else:
            gamma = cur_params.get(gamma_keys, 1.0)

        sigma = cur_params.get(sigma_key, None) if sigma_key else None
        return gamma, sigma

    def _F_operator(self, v_list_k, cur_params):
        """Applies all agent proximal operators to their respective v_i."""
        F_v_k = []
        for i, (agent_type, agent_obj) in enumerate(self.agents_ops):
            gamma, sigma = self._get_agent_params(i, cur_params)
            if agent_type == "data_fidelity":
                F_v_k.append(agent_obj.prox(v_list_k[i], self._y_ref, self._physics_ref, gamma=gamma))
            else:  # prior
                F_v_k.append(agent_obj.prox(v_list_k[i], sigma, gamma=gamma))
        return F_v_k

    def _G_operator(self, input_list_k):
        """Computes weighted average and redistributes."""
        avg = torch.zeros_like(input_list_k[0])
        for i in range(self.num_agents):
            avg += self.mu[i] * input_list_k[i]
        return [avg.clone() for _ in range(self.num_agents)]

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
  
        self._y_ref = y
        self._physics_ref = physics

        x_consensus_k = X["est"][0]

        if self.v_list is None:
            self.v_list = [x_consensus_k.clone() for _ in range(self.num_agents)]
            mu_param = cur_params.get("mu")
            if len(mu_param) != self.num_agents:
                raise ValueError(
                    f"MACEIteration: Length of 'mu' ({len(mu_param)}) must match number of agents ({self.num_agents})."
                )
            else:
                self.mu = mu_param

        rho = cur_params.get("rho", 0.5)
        if not (0.0 < rho <= 1.0):
            warnings.warn(
                f"MACEIteration: rho ({rho}) is outside (0, 1]. Clipping to range [1e-6, 1.0]."
            )
            rho = max(1e-6, min(rho, 1.0))

        # MACE Algorithm steps:
        # 1. F_v_k = F(v_k)
        F_v_k = self._F_operator(self.v_list, cur_params)

        # 2. TF_v_k = 2F-I
        TF_v_k = [2 * f_out - v_k_i for f_out, v_k_i in zip(F_v_k, self.v_list)]

        # 3. G_TF_v_k = 2G-I
        G_TF_v_k = self._G_operator(TF_v_k)

        # 4. TFG_v_k =  (2G - I) (2F - I)
        TFG_v_k = [
            2 * g_tf_out - tf_out for g_tf_out, tf_out in zip(G_TF_v_k, TF_v_k)
        ]

        # 5. Mann iteration for v_list: v_{k+1} = (1 - rho) I + rho * TFG_v_k
        self.v_list = [
            (1 - rho) * v_k_i + rho * tfg_out
            for v_k_i, tfg_out in zip(self.v_list, TFG_v_k)
        ]

        x_consensus_next = torch.zeros_like(x_consensus_k)
        for i in range(self.num_agents):
            x_consensus_next += self.mu[i] * self.v_list[i]

        cost = self.F_fn(x_consensus_next, cur_data_fidelity, cur_prior, cur_params, y, physics) if self.has_cost else None

        return {
            "est": (x_consensus_next, ),
            "cost": cost,
        } 