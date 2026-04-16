import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import FuncScale
from itertools import cycle
import wandb
import json
api = wandb.Api()


mean_baselines_str = ['mean_test_OT_align_{}_sbi', 'mean_test_NF_align_{}', 'mean_test_OT_align_{}', 'mean_test_OT_align_{}_finetune_only', 'mean_test_OT_align_{}_npe', 'mean_test_WassOT_NF_align_{}', 'mean_test_WassOT_NF_align_{}_wass', 'mean_test_OT_align_{}_OT_only', 'mean_test_OT_align_{}_OT_only_transductive', 'mean_test_OT_align_{}_transductive']
std_baselines_str = ['std_test_OT_align_{}_sbi', 'std_test_NF_align_{}', 'std_test_OT_align_{}', 'std_test_OT_align_{}_finetune_only', 'std_test_OT_align_{}_npe', 'std_test_WassOT_NF_align_{}', 'std_test_WassOT_NF_align_{}_wass', 'std_test_OT_align_{}_OT_only',  'std_test_OT_align_{}_OT_only_transductive', 'std_test_OT_align_{}_transductive']


def import_results(list_of_runs, x=None):
    results = {}
    for i, run_path in enumerate(list_of_runs):
        run = api.run(run_path)

        # load the wandb-metadata.json
        fobj = run.file("wandb-metadata.json").download(replace=True)
        meta = json.load(fobj)
        args = meta["args"]

        if "--num_samples" in args and x is None:
            idx = args.index("--num_samples")
            num_samples = int(args[idx + 1])
        else:
            num_samples = x[i]

        summary = run.summary._json_dict

        # 1) Ensure the top‐level key exists as a dict
        grp = results.setdefault(str(num_samples), {})

        # 2) Ensure sub‐dicts for lpp and acauc exist
        lpp   = grp.setdefault('lpp', {'mean': {}, 'std': {}})
        acauc = grp.setdefault('acauc', {'mean': {}, 'std': {}})

        # 3) Fill in the mean/std for lpp
        lpp['mean'].update({
            fmt.format('lpp')[5:]: summary.get(fmt.format('lpp'))
            for fmt in mean_baselines_str
        })
        lpp['std'].update({
            fmt.format('lpp')[4:]: summary.get(fmt.format('lpp'))
            for fmt in std_baselines_str
        })

        # 4) Fill in the mean/std for acauc
        acauc['mean'].update({
            fmt.format('acauc')[5:]: summary.get(fmt.format('acauc'))
            for fmt in mean_baselines_str
        })
        acauc['std'].update({
            fmt.format('acauc')[4:]: summary.get(fmt.format('acauc'))
            for fmt in std_baselines_str
        })
    return results

def get_mean_stds_matrix(names, metric_name, x, results, prior):
    means = np.zeros([len(names), x.shape[0]])
    stds = np.zeros([len(names), x.shape[0]])
    num_samples_fixed = str(x[0])
    for i, name in enumerate(names):
        if name=='Prior':
            if metric_name=='lpp':
                means[i, 0] = prior
            elif metric_name=='acauc':
                means[i, 0] = prior
            means[i, 1:] = np.nan
            continue
        if name=='SBI':
            means[i, 0] = results[num_samples_fixed][metric_name]['mean']['test_OT_align_{}_sbi'.format(metric_name)]
            means[i, 1:] = np.nan
            continue
        if name=='NPE':
            means[i, 0] = results[num_samples_fixed][metric_name]['mean']['test_OT_align_{}_npe'.format(metric_name)]
            means[i, 1:] = np.nan
            continue
        if name=='OT-only(full test)':
            means[i, 0] = results[num_samples_fixed][metric_name]['mean']['test_OT_align_{}_OT_only_transductive'.format(metric_name)]
            means[i, 1:] = np.nan
            continue
        if name=='OT-only(single sample)':
            means[i, 0] = results[num_samples_fixed][metric_name]['mean']['test_OT_align_{}_OT_only'.format(metric_name)]
            means[i, 1:] = np.nan
            continue
            
        for j, num_samples in enumerate(x):
            if name=='finetune-only':
                name = 'test_OT_align_{}_finetune_only'.format(metric_name)
            if name=='RoPE(single sample)':
                name = 'test_OT_align_{}'.format(metric_name)
            if name == 'ours':
                name = 'test_WassOT_NF_align_{}'.format(metric_name)
            if name == 'RoPE (full test)':
                name = 'test_OT_align_{}_transductive'.format(metric_name)
            means[i, j] = results[str(num_samples)][metric_name]['mean'][name]
            stds[i, j] = results[str(num_samples)][metric_name]['std'][name]
    return means, stds

def plot_metric_vs_x_linear_y(
    means: np.ndarray,
    stds:  np.ndarray,
    names:  list[str],
    x:     np.ndarray,
    metric_name: str,
    x_label:     str,
    is_log: bool = True,
):
    fig, ax = plt.subplots(figsize=(7.5, 4)) 
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # color cycle for baselines
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    baseline_colors = iter(base_colors)

    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':']
    m_i = ls_i = 0

    # plot each series
    for i, label in enumerate(names):
        mean = means[i]
        std  = stds[i]

        if np.count_nonzero(~np.isnan(mean)) == 1:
            val   = float(np.nanmax(mean))
            color = next(baseline_colors)
            ax.hlines(val, x.min(), x.max(),
                      color=color,
                      linestyle='dashdot',
                      label=label)
        else:
            low  = mean - std
            high = mean + std
            ax.plot(x, mean,
                    marker=markers[m_i % len(markers)],
                    linestyle=linestyles[ls_i % len(linestyles)],
                    label=label)
            ax.fill_between(x, low, high, alpha=0.2)
            m_i += 1
            if m_i % len(markers) == 0:
                ls_i += 1

    if is_log:
        ax.set_xscale('log')
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylim(-0.5, 0.5)


    ax.set_yscale('linear')
    ax.tick_params(axis='both', labelsize=14)


    ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()

    plt.tight_layout()
    plt.show()

def make_piecewise_scale(prior: float, offset: float = 1e-6):
    """
    Returns a (forward, inverse) pair for a FuncScale
    that maps:
      y ≤ prior    →  –log(prior – y + offset)
      y >  prior   →   y – prior
    so that small deviations below prior get log-compressed,
    and above-prior grows linearly.
    """

    def forward(y):
        y = np.array(y, copy=False)
        out = np.empty_like(y, dtype=float)
        mask = y <= prior
        # log‐compress below or equal to prior
        out[mask]    = -np.log(prior - y[mask] + offset)
        # linear above prior
        out[~mask]   = y[~mask] - prior
        return out

    def inverse(x):
        x = np.array(x, copy=False)
        out = np.empty_like(x, dtype=float)
        mask = x <= 0
        # invert log‐compress
        out[mask]    = prior - (np.exp(-x[mask]) - offset)
        # invert linear
        out[~mask]   = x[~mask] + prior
        return out

    return forward, inverse


def plot_metric_vs_x_piecewise(
    means: np.ndarray,
    stds: np.ndarray,
    names: list[str],
    x: np.ndarray,
    metric_name: str,
    x_label: str, 
    prior_value: float,
    is_log: bool = True,
    is_legend: bool = False,
):

    # set up the single axes
    fig, ax = plt.subplots(figsize=(7.5, 4)) #7.5
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # install our custom piecewise scale
    fwd, inv = make_piecewise_scale(prior_value)
    ax.set_yscale('function', functions=(fwd, inv))


    # plot each series exactly as before
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':']
    m_i = ls_i = 0

    # get the default color cycle
    base_colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    baseline_colors = cycle(base_colors)   # infinite iterator over colors


    y_ticks = []
    for i, label in enumerate(names):
        row_mean = means[i, :]
        row_std  = stds[i, :]

        # fixed baseline
        if np.count_nonzero(~np.isnan(row_mean)) == 1:
            val = float(np.nanmax(row_mean))
            color = next(baseline_colors)
            ax.hlines(val, x.min(), x.max(),
                      linestyles='dashdot', label=label, color=color)
            if label == 'Prior':
                y_ticks.append(val)
            if label == 'SBI':
                y_ticks.append(val)
                upper_bound = val + 0.2
            if label == 'NPE':
                y_ticks.append(val)
        else:
            low  = row_mean - row_std
            high = row_mean + row_std
            ax.plot(x, row_mean,
                    marker=markers[m_i % len(markers)],
                    linestyle=linestyles[ls_i % len(linestyles)],
                    label=label)
            print(label, row_mean[0], low[1], high[1])
            ax.fill_between(x, low, high, alpha=0.2)
            m_i += 1
            if m_i % len(markers) == 0:
                ls_i += 1

    # tidy up ticks and labels
    if is_log:
        ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_yticks([0]+y_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel(x_label)
    #ax.set_ylabel(metric_name)

    if 'upper_bound' in locals():
        ax.set_ylim(top=upper_bound)

    if is_legend:
        # Add transparent, horizontal legend above plot
        legend = ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),  # Centered above axes
            ncol=5,
            frameon=False,
            fontsize='large'
        )

        # Adjust layout to make space for the legend
        plt.subplots_adjust(top=0.8)  # Push plot down to make space on top
    #ax.legend(loc='best')
    #ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.grid(which='both', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.show()