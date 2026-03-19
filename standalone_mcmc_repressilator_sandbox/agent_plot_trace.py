import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def plot_trace(samples, n_percentiles=None, parameter_names=None, ref_parameters=None, filename=None):
    """Creates trace plots for MCMC samples."""
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape
    
    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]
    
    fig, axes = plt.subplots(n_param, 2, figsize=(12, 2 * n_param), squeeze=False)
    
    stacked_chains = np.vstack(samples)
    if n_percentiles is None:
        xmin = np.min(stacked_chains, axis=0)
        xmax = np.max(stacked_chains, axis=0)
    else:
        xmin = np.percentile(stacked_chains, 50 - n_percentiles / 2., axis=0)
        xmax = np.percentile(stacked_chains, 50 + n_percentiles / 2., axis=0)
    xbins = np.linspace(xmin, xmax, bins)
    
    for i in range(n_param):
        ymin_all, ymax_all = np.inf, -np.inf
        for j_list, samples_j in enumerate(samples):
            axes[i, 0].set_xlabel(parameter_names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].hist(samples_j[:, i], bins=xbins[:, i], alpha=alpha, label='Samples ' + str(1 + j_list))
            
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel(parameter_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=alpha)
            
            ymin_all = min(ymin_all, xmin[i])
            ymax_all = max(ymax_all, xmax[i])
        axes[i, 1].set_ylim([ymin_all, ymax_all])
        
        if ref_parameters is not None:
            ymin_tv, ymax_tv = axes[i, 0].get_ylim()
            axes[i, 0].plot([ref_parameters[i], ref_parameters[i]], [0.0, ymax_tv], '--', c='k')
            xmin_tv, xmax_tv = axes[i, 1].get_xlim()
            axes[i, 1].plot([0.0, xmax_tv], [ref_parameters[i], ref_parameters[i]], '--', c='k')
    
    if n_list > 1:
        axes[0, 0].legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f'Saved: {filename}')
    
    return fig, axes
