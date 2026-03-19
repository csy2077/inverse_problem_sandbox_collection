# =====================================================================================
# Standalone SIR Model with MCMC Inference
# =====================================================================================
# This is a self-contained Python script for Bayesian inference on the SIR
# (Susceptible-Infected-Recovered) epidemic model using Markov Chain Monte Carlo (MCMC).
#
# Problem: SIR epidemiological model parameter estimation
# Algorithm: Haario-Bardenet Adaptive Covariance MCMC
#
# File Dependencies:
# ------------------
# Input:
#   - data/standalone_mcmc_sir.json: JSON data file containing model parameters and
#     common cold outbreak data from Tristan da Cunha island. This file must be
#     located in the ./data subdirectory relative to this script.
#   - No model weights required
#
# Output:
#   - simulation_results.png: Plot of model simulation with suggested parameters
#   - real_data_plot.png: Plot of observed real data (common cold outbreak)
#   - mcmc_trace.png: MCMC trace plots and histograms for each parameter
#   - posterior_predictive.png: Posterior predictive time series plots
#   - Console output: MCMC summary statistics (mean, std, quantiles, rhat, ESS)
#
# Change Log:
# -----------
#   - Data (suggested_parameters, suggested_times, suggested_values, initial_state,
#     and MCMC initial positions) moved to external JSON file (data/standalone_mcmc_sir.json).
#   - Added DataLoader class to read and parse JSON data file.
#   - SIRModel class now accepts data from DataLoader instead of hard-coded values.
#
# =====================================================================================

import os
import json

# Set CUDA device (not used in this CPU-based computation, but set as requested)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import timeit
import warnings

# Use tabulate for nice summary tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    warnings.warn("tabulate not installed. Summary will be printed in basic format.")


# =====================================================================================
# Data Loading
# =====================================================================================

class DataLoader:
    """
    Loads model data from a JSON file.
    
    The JSON file should contain:
        - initial_state: dict with S, I, R values
        - suggested_parameters: dict with gamma, v, S0 values
        - suggested_times: list of time points
        - suggested_values: dict with 'data' key containing [Infected, Recovered] pairs
        - mcmc_settings: dict with initial_positions and parameter_names
    """
    
    def __init__(self, json_path=None):
        if json_path is None:
            # Default to JSON file with same name as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, 'data', 'standalone_mcmc_sir.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            self._data = json.load(f)
        
        self._parse_data()
    
    def _parse_data(self):
        """Parse the loaded JSON data into numpy arrays."""
        # Initial state [S, I, R]
        init = self._data['initial_state']
        self._initial_state = np.array([init['S'], init['I'], init['R']], dtype=float)
        
        # Suggested parameters [gamma, v, S0]
        params = self._data['suggested_parameters']
        self._suggested_parameters = [params['gamma'], params['v'], params['S0']]
        
        # Suggested times
        self._suggested_times = np.array(self._data['suggested_times'], dtype=float)
        
        # Suggested values (observed data)
        self._suggested_values = np.array(
            self._data['suggested_values']['data'], dtype=float
        )
        
        # MCMC settings
        mcmc = self._data.get('mcmc_settings', {})
        self._mcmc_initial_positions = mcmc.get('initial_positions', None)
        self._mcmc_parameter_names = mcmc.get('parameter_names', None)
    
    def initial_state(self):
        """Returns the initial state [S, I, R]."""
        return self._initial_state.copy()
    
    def suggested_parameters(self):
        """Returns suggested parameters [gamma, v, S0]."""
        return self._suggested_parameters.copy()
    
    def suggested_times(self):
        """Returns suggested simulation times."""
        return self._suggested_times.copy()
    
    def suggested_values(self):
        """Returns observed data array with shape (n_times, 2) for [Infected, Recovered]."""
        return self._suggested_values.copy()
    
    def mcmc_initial_positions(self):
        """Returns initial positions for MCMC chains."""
        return self._mcmc_initial_positions
    
    def mcmc_parameter_names(self):
        """Returns parameter names for MCMC."""
        return self._mcmc_parameter_names


# =====================================================================================
# Utility Functions
# =====================================================================================

def vector(x):
    """
    Copies x and returns a 1d read-only NumPy array of floats with shape (n,).
    """
    if np.isscalar(x):
        x = np.array([float(x)])
    else:
        x = np.array(x, copy=True, dtype=float)
    x.setflags(write=False)
    if x.ndim != 1:
        n = np.max(x.shape)
        if np.prod(x.shape) != n:
            raise ValueError('Unable to convert to 1d vector of scalar values.')
        x = x.reshape((n,))
    return x


def matrix2d(x):
    """
    Copies x and returns a 2d read-only NumPy array of floats with shape (m, n).
    """
    x = np.array(x, copy=True, dtype=float)
    if x.ndim == 1:
        x = x.reshape((len(x), 1))
    elif x.ndim != 2:
        raise ValueError('Unable to convert to 2d matrix.')
    x.setflags(write=False)
    return x


class Timer:
    """Provides accurate timing."""
    def __init__(self):
        self._start = timeit.default_timer()

    def format(self, time=None):
        """Formats a number of seconds into a readable string."""
        if time is None:
            time = self.time()
        if time < 1e-2:
            return str(time) + ' seconds'
        elif time < 60:
            return str(round(time, 2)) + ' seconds'
        output = []
        time = int(round(time))
        units = [(604800, 'week'), (86400, 'day'), (3600, 'hour'), (60, 'minute')]
        for k, name in units:
            f = time // k
            if f > 0 or output:
                output.append(str(f) + ' ' + (name if f == 1 else name + 's'))
            time -= f * k
        output.append('1 second' if time == 1 else str(time) + ' seconds')
        return ', '.join(output)

    def reset(self):
        """Resets this timer's start time."""
        self._start = timeit.default_timer()

    def time(self):
        """Returns the time (in seconds) since this timer was created."""
        return timeit.default_timer() - self._start


# =====================================================================================
# SIR Model
# =====================================================================================

class SIRModel:
    r"""
    The SIR model of infectious disease models the number of susceptible (S),
    infected (I), and recovered (R) people in a population.

    ODE system:
        dS/dt = -gamma * S * I
        dI/dt = gamma * S * I - v * I
        dR/dt = v * I

    Parameters: (gamma, v, S0) - infection rate, recovery rate, initial susceptibles

    Outputs: (I, R) - Infected and Recovered people (2 outputs)
    """

    def __init__(self, data_loader, y0=None):
        self._data_loader = data_loader
        if y0 is None:
            self._y0 = data_loader.initial_state()
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError('Initial value must have size 3.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """Returns the number of outputs (2: Infected, Recovered)."""
        return 2

    def n_parameters(self):
        """Returns the number of parameters (3: gamma, v, S0)."""
        return 3

    def _rhs(self, y, t, gamma, v):
        """Calculates the model RHS."""
        dS = -gamma * y[0] * y[1]
        dI = gamma * y[0] * y[1] - v * y[1]
        dR = v * y[1]
        return np.array([dS, dI, dR])

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        gamma, v, S0 = parameters
        y0 = np.array(self._y0, copy=True)
        y0[0] = S0
        y = odeint(self._rhs, y0, times, (gamma, v))
        return y[:, 1:]  # Return I and R columns only

    def suggested_parameters(self):
        """Returns a suggested set of parameters from the data file."""
        return self._data_loader.suggested_parameters()

    def suggested_times(self):
        """Returns a suggested set of simulation times from the data file."""
        return self._data_loader.suggested_times()

    def suggested_values(self):
        """
        Returns the data from a common-cold outbreak on Tristan da Cunha
        (loaded from the data file).
        """
        return self._data_loader.suggested_values()


# =====================================================================================
# Problem Classes
# =====================================================================================

class MultiOutputProblem:
    """
    Represents an inference problem where a model is fit to a multi-valued time series.
    """

    def __init__(self, model, times, values):
        self._model = model
        self._times = vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        self._values = matrix2d(values)
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError('Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """Runs a simulation using the given parameters."""
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def n_outputs(self):
        return self._n_outputs

    def n_parameters(self):
        return self._n_parameters

    def n_times(self):
        return self._n_times

    def times(self):
        return self._times

    def values(self):
        return self._values


# =====================================================================================
# Error Measures and Log-Likelihoods
# =====================================================================================

class SumOfSquaresError:
    """Calculates the sum of squares error."""

    def __init__(self, problem, weights=None):
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._n_outputs = problem.n_outputs()
        self._n_parameters = problem.n_parameters()

        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError('Number of weights must match number of problem outputs.')
        self._weights = np.asarray([float(w) for w in weights])

    def __call__(self, x):
        return np.sum((np.sum(((self._problem.evaluate(x) - self._values)**2),
                              axis=0) * self._weights), axis=0)

    def n_parameters(self):
        return self._n_parameters


class GaussianLogLikelihood:
    r"""
    Calculates a log-likelihood assuming independent Gaussian noise.

    For multi-output problems, adds n_outputs sigma parameters.

    Log-likelihood:
        log L = -0.5*n_t*log(2*pi) - n_t*log(sigma) - SSE/(2*sigma^2)
    """

    def __init__(self, problem):
        self._problem = problem
        self._values = problem.values()
        self._times = problem.times()
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._n_parameters = problem.n_parameters() + self._no
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))

    def n_parameters(self):
        return self._n_parameters


# =====================================================================================
# MCMC Diagnostics
# =====================================================================================

def autocorrelation(x):
    """Calculates autocorrelation for a vector x."""
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


def _autocorrelate_negative(autocorrelation):
    """Returns the index of the first negative entry in autocorrelation."""
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)


def effective_sample_size_single_parameter(x):
    """Calculates effective sample size (ESS) for a single parameter."""
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess


def effective_sample_size(samples):
    """Calculates effective sample size (ESS) for n-dimensional samples."""
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')
    return [effective_sample_size_single_parameter(samples[:, i])
            for i in range(0, n_params)]


def _within(chains):
    """Calculates mean within-chain variance."""
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)


def _between(chains):
    """Calculates mean between-chain variance."""
    n = chains.shape[1]
    within_chain_means = np.mean(chains, axis=1)
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)
    return n * between_chain_var


def rhat(chains, warm_up=0.0):
    """
    Returns the convergence measure R-hat for MCMC chains.
    """
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError('Dimension of chains must be 2 or 3.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError('warm_up only takes values in [0,1].')

    n = chains.shape[1]
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]
    n = n // 2
    if n < 1:
        raise ValueError('Number of samples per chain after warm-up is too small.')
    chains = np.vstack([chains[:, :n], chains[:, -n:]])
    w = _within(chains)
    b = _between(chains)
    return np.sqrt((n - 1.0) / n + b / (w * n))


# =====================================================================================
# MCMC Summary
# =====================================================================================

class MCMCSummary:
    """
    Calculates and prints key summaries of posterior samples from MCMC chains.
    """

    def __init__(self, chains, time=None, parameter_names=None):
        self._chains = chains
        self._chains_unmodified = chains

        if len(chains) == 1:
            warnings.warn('Summaries calculated with one chain may be unreliable.')

        self._n_parameters = chains[0].shape[1]

        if time is not None and float(time) <= 0:
            raise ValueError('Elapsed time must be positive.')
        self._time = time

        if parameter_names is None:
            parameter_names = ['param ' + str(i + 1) for i in range(self._n_parameters)]
        elif self._n_parameters != len(parameter_names):
            raise ValueError('Parameter names list must be same length as number of sampled parameters')
        self._parameter_names = parameter_names

        self._ess = None
        self._ess_per_second = None
        self._mean = None
        self._quantiles = None
        self._rhat = None
        self._std = None
        self._summary_list = []
        self._summary_str = None

        self._make_summary()

    def __str__(self):
        if self._summary_str is None:
            headers = ['param', 'mean', 'std.', '2.5%', '25%', '50%', '75%', '97.5%', 'rhat', 'ess']
            if self._time is not None:
                headers.append('ess per sec.')

            if HAS_TABULATE:
                self._summary_str = tabulate(
                    self._summary_list, headers=headers,
                    numalign='left', floatfmt='.2f')
            else:
                # Basic formatting without tabulate
                lines = [' | '.join(headers)]
                for row in self._summary_list:
                    lines.append(' | '.join(['{:.2f}'.format(x) if isinstance(x, float) else str(x) for x in row]))
                self._summary_str = '\n'.join(lines)

        return self._summary_str

    def _make_summary(self):
        """Calculates posterior summaries for all parameters."""
        stacked = np.vstack(self._chains)
        self._mean = np.mean(stacked, axis=0)
        self._std = np.std(stacked, axis=0)
        self._quantiles = np.percentile(stacked, [2.5, 25, 50, 75, 97.5], axis=0)
        self._rhat = rhat(self._chains)
        self._ess = np.zeros(self._n_parameters)
        for i, chain in enumerate(self._chains):
            self._ess += effective_sample_size(chain)

        if self._time is not None:
            self._ess_per_second = np.array(self._ess) / self._time

        for i in range(0, self._n_parameters):
            row = [
                self._parameter_names[i],
                self._mean[i], self._std[i],
                self._quantiles[0, i], self._quantiles[1, i],
                self._quantiles[2, i], self._quantiles[3, i],
                self._quantiles[4, i],
                self._rhat[i], self._ess[i],
            ]
            if self._time is not None:
                row.append(self._ess_per_second[i])
            self._summary_list.append(row)

    def mean(self):
        return self._mean

    def std(self):
        return self._std

    def ess(self):
        return self._ess


# =====================================================================================
# MCMC Sampler - Haario-Bardenet Adaptive Covariance
# =====================================================================================

class HaarioBardenetACMC:
    """
    Haario-Bardenet Adaptive Covariance MCMC method.
    A single-chain MCMC method with adaptive proposal covariance.
    """

    def __init__(self, x0, sigma0=None):
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)

        if sigma0 is None:
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.prod(self._sigma0.shape) == self._n_parameters:
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                self._sigma0 = self._sigma0.reshape((self._n_parameters, self._n_parameters))

        self._running = False
        self._adaptive = False
        self._current = None
        self._current_log_pdf = None
        self._proposed = None
        self._iterations = 0
        self._adaptations = 1
        self._target_acceptance = 0.234
        self._acceptance_count = 0
        self._acceptance_rate = 0
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)
        self._eta = 0.6
        self._gamma = 1
        self._log_lambda = 0

    def name(self):
        return 'Haario-Bardenet adaptive covariance MCMC'

    def in_initial_phase(self):
        return not self._adaptive

    def needs_initial_phase(self):
        return True

    def set_initial_phase(self, initial_phase):
        self._adaptive = not bool(initial_phase)

    def acceptance_rate(self):
        return self._acceptance_rate

    def ask(self):
        if not self._running:
            self._running = True
            self._proposed = self._x0
            self._proposed.setflags(write=False)

        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(
                self._current, self._sigma * np.exp(self._log_lambda))
            self._proposed.setflags(write=False)

        return self._proposed

    def tell(self, fx):
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        fx = float(fx)
        self._iterations += 1

        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError('Initial point for MCMC must have finite logpdf.')
            self._current = self._proposed
            self._current_log_pdf = fx
            self._proposed = None
            return self._current, self._current_log_pdf, True

        log_ratio = fx - self._current_log_pdf
        accepted = False
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < log_ratio:
                accepted = True
                self._acceptance_count += 1
                self._current = self._proposed
                self._current_log_pdf = fx

        self._acceptance_rate = self._acceptance_count / self._iterations
        self._proposed = None

        if self._adaptive:
            self._gamma = (self._adaptations + 1) ** -self._eta
            self._adaptations += 1
            # Adapt mu
            self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current
            # Adapt sigma
            dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
            self._sigma = (1 - self._gamma) * self._sigma + self._gamma * np.dot(dsigm, dsigm.T)
            # Adapt lambda
            p = 1 if accepted else 0
            self._log_lambda += self._gamma * (p - self._target_acceptance)

        return self._current, self._current_log_pdf, accepted


# =====================================================================================
# MCMC Controller
# =====================================================================================

class MCMCController:
    """
    Samples from a LogPDF using MCMC.
    """

    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()
        self._n_chains = int(chains)

        if self._n_chains < 1:
            raise ValueError('Number of chains must be at least 1.')
        if len(x0) != chains:
            raise ValueError('Number of initial positions must equal number of chains.')
        if not all([len(x) == self._n_parameters for x in x0]):
            raise ValueError('All initial positions must have same dimension as the LogPDF.')

        if method is None:
            method = HaarioBardenetACMC

        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()

        self._initial_phase_iterations = 200 if self._needs_initial_phase else None
        self._log_to_screen = True
        self._max_iterations = 10000
        self._samples = None
        self._time = None

    def set_max_iterations(self, iterations=10000):
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_log_to_screen(self, enabled):
        self._log_to_screen = True if enabled else False

    def time(self):
        return self._time

    def run(self):
        """Runs the MCMC sampler and returns the chains."""
        if self._max_iterations is None:
            raise ValueError('At least one stopping criterion must be set.')

        iteration = 0

        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

        if self._log_to_screen:
            print('Using ' + str(self._samplers[0].name()))
            print('Generating ' + str(self._n_chains) + ' chains.')
            print('Running in sequential mode.')

        samples = np.zeros((self._n_chains, self._max_iterations, self._n_parameters))
        n_samples = [0] * self._n_chains
        active = list(range(self._n_chains))

        timer = Timer()
        running = True

        while running:
            if iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')

            xs = [self._samplers[i].ask() for i in active]
            fxs = [self._log_pdf(x) for x in xs]

            fxs_iterator = iter(fxs)
            for i in list(active):
                reply = self._samplers[i].tell(next(fxs_iterator))
                if reply is not None:
                    y, fy, accepted = reply
                    samples[i][n_samples[i]] = y
                    n_samples[i] += 1
                    if n_samples[i] == self._max_iterations:
                        active.remove(i)

            intermediate_step = min(n_samples) <= iteration
            if intermediate_step:
                continue

            iteration += 1

            if iteration % 500 == 0 and self._log_to_screen:
                print(f'Iteration {iteration}/{self._max_iterations}')

            if self._max_iterations is not None and iteration >= self._max_iterations:
                running = False

        self._time = timer.time()

        if self._log_to_screen:
            print(f'Halting: Maximum number of iterations ({iteration}) reached.')
            print(f'Time: {timer.format(self._time)}')

        self._samples = samples
        return samples


# =====================================================================================
# Plotting Functions
# =====================================================================================

def plot_trace(samples, parameter_names=None, save_path='mcmc_trace.png'):
    """
    Creates and saves trace plots and histograms for MCMC samples.
    """
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]

    fig, axes = plt.subplots(n_param, 2, figsize=(12, 2 * n_param), squeeze=False)

    stacked_chains = np.vstack(samples)
    xmin = np.min(stacked_chains, axis=0)
    xmax = np.max(stacked_chains, axis=0)
    xbins = np.linspace(xmin, xmax, bins)

    for i in range(n_param):
        ymin_all, ymax_all = np.inf, -np.inf
        for j_list, samples_j in enumerate(samples):
            axes[i, 0].set_xlabel(parameter_names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].hist(samples_j[:, i], bins=xbins[:, i], alpha=alpha,
                            label='Chain ' + str(1 + j_list))
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel(parameter_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=alpha)
            ymin_all = min(ymin_all, xmin[i])
            ymax_all = max(ymax_all, xmax[i])
        axes[i, 1].set_ylim([ymin_all, ymax_all])

    if n_list > 1:
        axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Trace plot saved to: {save_path}')
    return fig, axes


def plot_series(samples, problem, save_path='posterior_predictive.png', thinning=None):
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

    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, np.sqrt(n_outputs) * 3), sharex=True)

    if n_outputs == 1:
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.plot(times, problem.values(), 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Original data')
        plt.plot(times, predicted_values[0], color='#1f77b4', label='Inferred series')
        for v in predicted_values[1:]:
            plt.plot(times, v, color='#1f77b4', alpha=alpha)
        plt.plot(times, mean_values, 'k:', lw=2, label='Mean of inferred series')
        plt.legend()
    elif n_outputs > 1:
        fig.subplots_adjust(hspace=0)
        axes[-1].set_xlabel('Time')
        for i_output in range(n_outputs):
            axes[i_output].set_ylabel('Output %d' % (i_output + 1))
            axes[i_output].plot(
                times, problem.values()[:, i_output], 'x', color='#7f7f7f',
                ms=6.5, alpha=0.5, label='Original data')
            axes[i_output].plot(
                times, predicted_values[0][:, i_output], color='#1f77b4',
                label='Inferred series')
            for v in predicted_values[1:]:
                axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
            axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2,
                                label='Mean of inferred series')
        axes[0].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Posterior predictive plot saved to: {save_path}')
    return fig, axes


# =====================================================================================
# Main Script
# =====================================================================================

if __name__ == '__main__':
    print('=' * 70)
    print('SIR Model - MCMC Inference')
    print('=' * 70)

    # Load data from JSON file
    print('Loading data from data/standalone_mcmc_sir.json...')
    data_loader = DataLoader()

    # Create a model with loaded data
    model = SIRModel(data_loader)

    # Run a simulation with suggested parameters
    parameters = model.suggested_parameters()
    times = model.suggested_times()
    values = model.simulate(parameters, times)

    print('Suggested Parameters:')
    print(f'  gamma (infection rate): {parameters[0]}')
    print(f'  v (recovery rate): {parameters[1]}')
    print(f'  S0 (initial susceptibles): {parameters[2]}')

    # Plot the simulated results
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.plot(times, values)
    plt.legend(['Infected', 'Recovered'])
    plt.title('SIR Model Simulation')
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Simulation plot saved to: simulation_results.png')

    # Get real data
    real_values = model.suggested_values()

    # Plot the real data
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.plot(times, real_values)
    plt.legend(['Infected', 'Recovered'])
    plt.title('Real Data: Common Cold Outbreak on Tristan da Cunha')
    plt.savefig('real_data_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Real data plot saved to: real_data_plot.png')

    # Create problem
    problem = MultiOutputProblem(model, times, real_values)

    # Create error measure (for reference)
    score = SumOfSquaresError(problem)

    # Create log-likelihood
    loglikelihood = GaussianLogLikelihood(problem)

    print('\nSetting up MCMC...')
    print(f'Number of model parameters: {model.n_parameters()}')
    print(f'Number of likelihood parameters: {loglikelihood.n_parameters()}')
    print('  (3 model params + 2 noise sigma params)')

    # Starting points for 3 chains (loaded from data file)
    x0 = data_loader.mcmc_initial_positions()

    # Create MCMC routine
    mcmc = MCMCController(loglikelihood, 3, x0)
    mcmc.set_max_iterations(3000)
    mcmc.set_log_to_screen(False)

    print('\nRunning MCMC (3000 iterations, 3 chains)...')
    chains = mcmc.run()

    # Check convergence and other properties of chains
    print('\n' + '=' * 70)
    print('MCMC Summary')
    print('=' * 70)
    # Get parameter names from data file
    parameter_names = data_loader.mcmc_parameter_names()
    
    results = MCMCSummary(
        chains=chains,
        time=mcmc.time(),
        parameter_names=parameter_names
    )
    print(results)

    # Create trace plots
    print('\nGenerating trace plots...')
    plot_trace(
        chains,
        parameter_names=parameter_names,
        save_path='mcmc_trace.png'
    )

    # Extract samples from first chain (last 1000 samples as burn-in removal)
    samples = chains[0, -1000:]

    # Create posterior predictive plots
    print('Generating posterior predictive plots...')
    fig, axes = plot_series(samples, problem, save_path='posterior_predictive.png')

    # Overlay the real data on the posterior predictive plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0)
    axes[-1].set_xlabel('Time (days)')

    # Get predictions for plotting
    thinning = max(1, int(len(samples) / 200))
    predicted_values = []
    for params in samples[::thinning, :3]:
        predicted_values.append(problem.evaluate(params))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)
    alpha = min(1, max(0.05 * (1000 / (len(samples) / thinning)), 0.5))

    output_names = ['Infected', 'Recovered']
    for i_output in range(2):
        axes[i_output].set_ylabel(output_names[i_output])
        axes[i_output].plot(
            times, problem.values()[:, i_output], 'x', color='#7f7f7f',
            ms=6.5, alpha=0.5, label='Observed data')
        for v in predicted_values:
            axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
        axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2,
                            label='Mean prediction')
        axes[i_output].plot(times, real_values[:, i_output], 'o', color='#ff7f0e',
                            ms=4, label='True data')

    axes[0].legend()
    plt.suptitle('SIR Model Posterior Predictive')
    plt.tight_layout()
    plt.savefig('posterior_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Posterior predictive plot saved to: posterior_predictive.png')

    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)
    print('\nOutput files:')
    print('  - simulation_results.png')
    print('  - real_data_plot.png')
    print('  - mcmc_trace.png')
    print('  - posterior_predictive.png')

