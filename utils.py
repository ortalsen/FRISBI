from torch.utils.data import DataLoader, random_split
import argparse
import torch
import torch.nn as nn
import ot
import torch.nn.functional as F
import numpy as np
import corner
import matplotlib.pyplot as plt
import numpy as np


def compute_semi_balanced_ot(real_embeddings, simulated_embeddings, epsilon=0.01, tau_a=1.0, tau_b=1.0):
    """
    Compute the semi-balanced entropic optimal transport matrix.

    Parameters:
    - real_embeddings: np.array of shape (n_real, d), embeddings from real data
    - simulated_embeddings: np.array of shape (n_sim, d), embeddings from simulated data
    - epsilon: float, entropic regularization parameter
    - tau_a: float, marginal relaxation for the real distribution
    - tau_b: float, marginal relaxation for the simulated distribution

    Returns:
    - transport_matrix: np.array of shape (n_real, n_sim), optimal transport matrix
    - ot_cost: float, optimal transport cost
    """
    n_real, n_sim = len(real_embeddings), len(simulated_embeddings)

    real_embeddings = np.array(real_embeddings.tolist(), dtype=np.float32)
    simulated_embeddings = np.array(simulated_embeddings.tolist(), dtype=np.float32)

    # normalise the embeddings
    real_embeddings = (real_embeddings - np.mean(real_embeddings, axis=0)) / np.std(real_embeddings, axis=0)
    simulated_embeddings = (simulated_embeddings - np.mean(simulated_embeddings, axis=0)) / np.std(simulated_embeddings, axis=0)

    # Compute the cost matrix (Euclidean distance squared)
    cost_matrix = ot.dist(real_embeddings, simulated_embeddings, metric='sqeuclidean')

    cost_matrix = cost_matrix / real_embeddings.shape[1]

    # Define non-uniform marginals
    a, b = np.ones(n_real) / n_real, np.ones(n_sim) / n_sim
    tau_a = float(tau_a)
    tau_b = float(tau_b)

    if tau_a < 1e4 or tau_b < 1e4:
    # Compute the semi-balanced OT matrix using KL relaxation
        transport_matrix = ot.sinkhorn_unbalanced(
        a=a,
        b=b,
        M=cost_matrix,
        reg=epsilon,
        reg_m=tau_a,
        reg_n=tau_b,
        method="sinkhorn"
    )
    else:
        # Compute the balanced OT matrix using Sinkhorn algorithm
        transport_matrix = ot.sinkhorn(
        a=a,
        b=b,
        M=cost_matrix,
        reg=epsilon)
    # Compute the total transport cost
    ot_cost = np.sum(transport_matrix * cost_matrix)
    transport_matrix = np.array(transport_matrix, dtype=np.float32, copy=True)

    return transport_matrix, ot_cost


def compute_acauc_for_samples(posterior_samples, true_parameters, num_alpha=100):
    n_o, n_samples, d = posterior_samples.shape
    # Ensure true_parameters are on the same device as posterior_samples.
    true_parameters = true_parameters.to(posterior_samples.device)

    # Generate the grid of nominal alpha levels.
    alpha_levels = torch.linspace(0.01, 0.99, num_alpha, device=posterior_samples.device)

    # Pre-sort the samples along the n_samples axis.
    sorted_samples, _ = torch.sort(posterior_samples, dim=1)  # shape: (n_o, n_samples, d)

    # Compute the lower and upper indices for each alpha level.
    lower_idx = (((1 - alpha_levels) / 2) * n_samples).long()  # shape: (num_alpha,)
    upper_idx = (((1 - (1 - alpha_levels) / 2) * n_samples).long()) - 1  # shape: (num_alpha,)

    # Expand indices for gathering: shape becomes (n_o, num_alpha, d)
    lower_idx_exp = lower_idx.view(1, num_alpha, 1).expand(n_o, num_alpha, d)
    upper_idx_exp = upper_idx.view(1, num_alpha, 1).expand(n_o, num_alpha, d)

    # Gather the corresponding lower and upper bounds from the sorted samples.
    lower_bounds = torch.gather(sorted_samples, dim=1, index=lower_idx_exp)
    upper_bounds = torch.gather(sorted_samples, dim=1, index=upper_idx_exp)

    # Expand true_parameters for comparison: shape (n_o, 1, d)
    true_params_exp = true_parameters.unsqueeze(1)

    # Determine if each true parameter lies within the credible interval.
    observed_coverage = ((true_params_exp >= lower_bounds) & (true_params_exp <= upper_bounds)).float()

    # Compute the difference between the nominal level and observed coverage.
    coverage_diff = alpha_levels.view(1, num_alpha, 1) - observed_coverage

    # Average over test examples, alpha levels, and dimensions.
    acauc = coverage_diff.mean().item()
    return acauc


def compute_prior_lpp(true_parameters, params_args):
    log_probs = torch.zeros(true_parameters.shape[0], device=true_parameters.device)
    for param in params_args:
        param = params_args[param]
        x = true_parameters[:, param['index']]
        if param['scalers']['type'] == 'uniform':
            scaler_min = param['scalers']['params']['min']
            scaler_max = param['scalers']['params']['max']
            mask = (x >= scaler_min) & (x <= scaler_max)
            lp = torch.full_like(x, -torch.log(torch.tensor(scaler_max - scaler_min)))
            lp = torch.where(mask, lp, torch.tensor(-float('inf')))
        elif param['scalers']['type'] == 'normal':
            scaler_mean = param['scalers']['params']['mean']
            scaler_std = param['scalers']['params']['std']
            lp = -0.5 * ((x - scaler_mean) / scaler_std)**2 - torch.log(scaler_std * torch.sqrt(2 * torch.tensor(np.pi)))
        log_probs += lp
    log_probs_no_inf = log_probs[torch.isfinite(log_probs)]
    lpp = log_probs_no_inf.mean().item()
    return lpp, log_probs



def compute_lpp(nflow_model, true_parameters, context_simulated, transport_plan=None):
    """
    Compute the average log posterior probability (LPP).

    This function works in two modes:
      - If transport_plan is None, it assumes that context_simulated is of shape (n_o, c)
        and directly computes LPP as the average log probability over the test set.
      - If transport_plan is provided, it assumes that context_simulated is of shape (n_s, c),
        and that transport_plan is of shape (n_o, n_s). In this case, it computes a weighted
        log probability via a log-sum-exp over the simulated contexts.

    Parameters:
      nflow_model: a trained conditional normalizing flow model that provides a method
                   log_prob(theta, context=...) returning a tensor of shape (n_o,).
      true_parameters: torch.Tensor of shape (n_o, d) with ground-truth parameter values.
      context_simulated: if transport_plan is None, a tensor of shape (n_o, c) containing
                         the context/embedding for each test observation; otherwise, a tensor of
                         shape (n_s, c) for the simulated contexts.
      transport_plan: (optional) torch.Tensor of shape (n_o, n_s) with OT weights (P_ij*).
                      If provided, the OT-based posterior is computed as a weighted combination
                      over simulated contexts.
      Returns:
      lpp: float, the average log posterior probability.
    """
    true_parameters = true_parameters.to(context_simulated.device)
    with torch.no_grad():
        n_o = true_parameters.shape[0]

        # Case 1: NF-based posterior
        if transport_plan is None:
            # context_simulated should be of shape (n_o, c)
            log_probs = nflow_model(true_parameters, context=context_simulated)  # shape (n_o,)
            lpp = log_probs.mean().item()
        else:
            # Case 2: OT-based posterior
            # Here, context_simulated is assumed to be of shape (n_s, c)
            n_s = context_simulated.shape[0]
            # Normalize the probabilities
            alpha = transport_plan * n_o  # shape (n_o, n_s)
            sum_rows = alpha.sum(dim=1, keepdim=True)
            zero_sum = torch.where(sum_rows == 0)
            for i in zero_sum[0]:
                alpha[i, :] = 1.0 / n_s
                sum_rows[i, :] = 1.0
            # Initialize tensor to hold log probabilities computed for each simulated context.
            # We assume nflow_model.log_prob returns a tensor of shape (n_o,) per call.
            log_probs_sim = torch.zeros(n_o, n_s, device=context_simulated.device)
            for j in range(n_s):
                # For each simulated context, expand it to all test examples.
                sim_context = context_simulated[j].expand(n_o, -1)
                # Compute log probability for each test example with this simulated context.
                log_probs_sim[:, j] = nflow_model(true_parameters, context=sim_context)
            
            # Here, transport_plan is a tensor of shape (n_o, n_s)
            # We scale the OT weights by n_o (as in the original code: α_ij = n_o P_ij*)
            
            # Now compute the weighted log posterior:
            # For each observation, we want log(sum_j α_ij * p(θ | x_s^j)).
            # We do this in log space using logsumexp.
            # (Assuming log(alpha) is well-defined; note that transport_plan should be positive.)
            #log_probs = torch.logsumexp(torch.log(alpha) + log_probs_sim, dim=1)  # shape (n_o,)
            log_probs = torch.log(torch.sum(alpha*torch.exp(log_probs_sim), dim=1))
            log_probs_no_inf = log_probs[torch.isfinite(log_probs) ]
            lpp = log_probs_no_inf.mean().item()
    return lpp, log_probs


def unbalanced_ot_wasserstine_loss(g_xo, h_xs, gamma, eps=1e-8):
    """
    Compute an OT loss using softmax coupling with additional entropy
    and marginal relaxation penalties.
    
    Args:
        g_xo (torch.Tensor): Tensor of shape (B, d) for real observation representations.
        h_xs (torch.Tensor): Tensor of shape (M, d) for simulated observation representations.
        gamma (float): Regularization parameter (used in the softmax and entropy penalty).
        eps (float): Small constant to avoid log(0).
        
    Returns:
        loss_ot (torch.Tensor): the OT loss with the closed form plan
        loss_entropy: the weights entropy
        alpha (torch.Tensor): (Closed-form) coupling weights of shape (B, M).
    """
    B, d = g_xo.shape
    M = h_xs.shape[0]
    
    # Compute the cost matrix (squared Euclidean distances)
    diff = g_xo.unsqueeze(1) - h_xs.unsqueeze(0)  # shape: (B, M, d)
    sq_dist = torch.sum(diff ** 2, dim=2)/d           # shape: (B, M)
    
    # Compute softmax coupling weights: alpha_ij = exp(-C_ij/gamma) / sum_j exp(-C_ij/gamma)
    alpha = torch.softmax(-sq_dist / gamma, dim=1)  # shape: (B, M)
    
    # OT loss: average weighted cost over the batch
    loss_ot = (alpha * sq_dist).sum() / B

    loss_entropy = (alpha * torch.log(alpha + eps)).sum() / B
    
    
    return loss_ot, loss_entropy, alpha


def draw_posterior_corner(posterior_samples, true_parameters, log_probs, param_names, method, save_path=None):
    """
    Draws a single corner plot with multiple test points overlaid in different colors.

    Parameters:
        posterior_samples (np.ndarray): Shape (N, num_samples, d)
        true_parameters (np.ndarray): Shape (N, d)
        log_probs (np.ndarray): Shape (N,)
        param_names (list of str): List of length d
        method (str): Label for the method to annotate the plot
        save_path (str, optional): Path to save the figure. If None, the figure is only displayed.
    """
    posterior_samples = np.array(posterior_samples)
    true_parameters = np.array(true_parameters)
    N, num_samples, d = posterior_samples.shape
    assert true_parameters.shape == (N, d), "Shape of true_parameters should match (N, d)"

    # Select a few random test points with a fixed seed
    np.random.seed(42)
    test_indices = np.random.choice(N, 3, replace=False)

    # Define a color palette
    colors = ['red', 'blue', 'green', 'purple']
    figure = None

    for i, test_index in enumerate(test_indices):
        samples = posterior_samples[test_index, :, :]  # (num_samples, d)
        true_vals = true_parameters[test_index, :]     # (d,)
        log_probs_i = log_probs[test_index]            # Scalar

        fig_arg = None if (i == 0 or d == 1) else figure

        # Omit truths if d == 1 to avoid corner bug
        kwargs = dict(
            labels=param_names,
            color=colors[i % len(colors)],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            fig=fig_arg,
            smooth=1.2
        )

        if d > 1:
            kwargs["truths"] = true_vals
            kwargs["truth_color"] = colors[i % len(colors)]

        corner_plot = corner.corner(samples, **kwargs)

        # Always get the figure
        figure = corner_plot.figure if isinstance(corner_plot, plt.Axes) else corner_plot

        # Manual truth marker if d == 1
        if d == 1:
            ax = figure.axes[0]
            ax.axvline(true_vals[0], color=colors[i % len(colors)], linestyle="--")
            ax.annotate(
                f"True: {true_vals[0]:.2f}",
                xy=(0.95, 0.9 - i * 0.1),
                xycoords='axes fraction',
                ha='right',
                fontsize=10,
                color=colors[i % len(colors)]
            )

        # Log probability annotation (non-overlapping)
        ax0 = figure.axes[0]
        ax0.annotate(
            f"log p = {log_probs_i:.2f}",
            xy=(0.05, 0.95 - i * 0.15),
            xycoords='axes fraction',
            ha='left',
            va='top',
            fontsize=10,
            color=colors[i % len(colors)],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.7)
        )

    # Annotate "True" values for each diagonal axis (d > 1)
    if d > 1:
        axes = np.array(figure.axes).reshape((d, d))
        for i in range(d):
            ax = axes[i, i]
            for j, test_index in enumerate(test_indices):
                true_val = true_parameters[test_index, i]
                ax.annotate(
                    f"True: {true_val:.2f}",
                    xy=(0.95, 0.9 - j * 0.1),
                    xycoords='axes fraction',
                    ha='right',
                    fontsize=10,
                    color=colors[j % len(colors)]
                )

    title = f"Posterior Corner Plot with {len(test_indices)} Test Point(s), {method}"
    figure.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        figure.savefig(f"{save_path}.png", dpi=300)
    
    return figure

def safe_draw_corner(posterior_samples, true_parameters, log_probs, param_names, method, save_path=None):
    try:
        return draw_posterior_corner(posterior_samples, true_parameters, log_probs, param_names, method, save_path)
    except Exception as e:
        print(f"[WARNING] Failed to draw corner plot for '{method}': {e}")
        fig = plt.figure()
        fig.suptitle(f"{method} — Failed to plot", fontsize=14, color='red')
        return fig


def get_ot_weighted_posterior_samples(nflow_model, output_dim, context_simulated, alpha_ij, num_samples):
        """
        Given a set of simulated contexts and an OT coupling (transport_plan*n_o),
        return a tensor of weighted posterior samples for each real observation.
        
        Args:
        nflow_model: your trained conditional NF with a sample(num_samples, context=...) method.
        context_simulated: Tensor of shape (n_s, c) for simulated contexts.
        alpha_ij: Tensor of shape (n_o, n_s) containing OT weights P_{ij}*n_o.
        num_samples: Number of samples to draw per simulated context.
        
        Returns:
        posterior_samples_weighted: Tensor of shape (n_o, num_samples, d) representing
                                    the OT-based posterior for each real observation.
        """
        n_o, n_s = alpha_ij.size()
        # Get samples for each simulated context.

        with torch.no_grad():
            posterior_samples_sim = nflow_model.sample(5000, context=context_simulated)
        
        # Normalize the probabilities
        sum_rows = alpha_ij.sum(dim=1, keepdim=True)
        zero_sum = torch.where(sum_rows == 0)
        for i in zero_sum[0]:
            alpha_ij[i, :] = 1.0 / n_s
            sum_rows[i, :] = 1.0

        alpha_ij_prob = alpha_ij / sum_rows

        # Sample indices `j` for each i (shape: [n_o, num_samples])
        js = torch.distributions.Categorical(probs=alpha_ij_prob).sample((num_samples,)).T  # shape: [n_o, num_samples]

        # Sample indices for selecting from the second dimension of posterior_samples_sim
        rand_idx = torch.randint(0, 5000, (n_o, num_samples))  # shape: [n_o, num_samples]

        # Gather posterior samples
        # posterior_samples_sim is assumed to have shape [n_j, n_samples_j, output_dim]
        # We need to gather samples at js[i, idx], rand_idx[i, idx]
        posterior_samples_weighted = posterior_samples_sim[js, rand_idx]

        
        return posterior_samples_weighted