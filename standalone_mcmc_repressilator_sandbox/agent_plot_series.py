import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def plot_series(samples, problem, ref_parameters=None, thinning=None, filename=None):
    """Creates predicted time series plots."""
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('samples must be of shape (n_sample, n_parameters).')
    
    n_parameters = problem.n_parameters()
    n_outputs = problem.n_outputs()
    
    if ref_parameters is not None:
        ref_series = problem.evaluate(ref_parameters[:n_parameters])
    
    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning must be > 0.')
    
    times = problem.times()
    
    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)
    
    alpha = min(1, max(0.05 * (1000 / (n_sample / thinning)), 0.5))
    
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, np.sqrt(n_outputs) * 3), sharex=True)
    
    if n_outputs == 1:
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.plot(times, problem.values(), 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Original data')
        plt.plot(times, predicted_values[0], color='#1f77b4', label='Inferred series')
        for v in predicted_values[1:]:
            plt.plot(times, v, color='#1f77b4', alpha=alpha)
        plt.plot(times, mean_values, 'k:', lw=2, label='Mean of inferred series')
        if ref_parameters is not None:
            plt.plot(times, ref_series, color='#d62728', ls='--', label='Reference series')
        plt.legend()
    else:
        fig.subplots_adjust(hspace=0)
        axes[-1].set_xlabel('Time')
        for i_output in range(n_outputs):
            axes[i_output].set_ylabel('Output %d' % (i_output + 1))
            axes[i_output].plot(times, problem.values()[:, i_output], 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Original data')
            axes[i_output].plot(times, predicted_values[0][:, i_output], color='#1f77b4', label='Inferred series')
            for v in predicted_values[1:]:
                axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
            axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2, label='Mean of inferred series')
            if ref_parameters is not None:
                axes[i_output].plot(times, ref_series[:, i_output], color='#d62728', ls='--', label='Reference series')
        axes[0].legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f'Saved: {filename}')
    
    return fig, axes
