import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def plot_series(samples, problem, save_path='posterior_predictive.png', thinning=None, state_names=None):
    """
    Creates and saves posterior predictive plots.
    """
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('samples must be of shape (n_sample, n_parameters).')

    n_parameters = problem.n_parameters()
    n_outputs = problem.n_outputs()

    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be None or an integer > 0.')

    times = problem.times()

    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)

    alpha = min(1, max(0.05 * (1000 / (n_sample / thinning)), 0.5))

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3 * n_outputs), sharex=True)
    if n_outputs == 1:
        axes = [axes]

    if state_names is None:
        state_names = ['mRNA (x)', 'Protein (y)', 'Protein (z)']

    for i_output in range(n_outputs):
        axes[i_output].set_ylabel(state_names[i_output] if i_output < len(state_names) else f'Output {i_output + 1}')
        axes[i_output].plot(
            times, problem.values()[:, i_output], 'x', color='#7f7f7f',
            ms=6.5, alpha=0.5, label='Observed data')
        axes[i_output].plot(
            times, predicted_values[0][:, i_output], color='#1f77b4',
            label='Inferred series')
        for v in predicted_values[1:]:
            axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
        axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2,
                            label='Mean of inferred series')
    axes[0].legend()
    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Posterior predictive plot saved to: {save_path}')
    return fig, axes
