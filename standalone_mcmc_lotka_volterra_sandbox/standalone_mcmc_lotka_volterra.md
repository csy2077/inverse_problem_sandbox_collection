"""python
# =====================================================================================
# Standalone Lotka-Volterra Predator-Prey Model with MCMC Inference
# =====================================================================================
# This is a self-contained Python script for Bayesian inference on the Lotka-Volterra
# predator-prey model using Markov Chain Monte Carlo (MCMC) methods.
#
# Problem: Lotka-Volterra predator-prey model parameter estimation
# Algorithm: Haario-Bardenet Adaptive Covariance MCMC and Hamiltonian Monte Carlo (HMC)
#
# File Dependencies:
# ------------------
# Input:
#   - data/standalone_mcmc_lotka_volterra.json: JSON file containing Hudson's Bay Company
#     lynx-hare pelt count data (1900-1920), model parameters, MCMC configuration,
#     and prior settings
#   - No model weights required
#
# Output:
#   - lotka_volterra_data.png: Plot of observed real data (hare-lynx population)
#   - lotka_volterra_phase.png: Phase plot of hare vs lynx populations
#   - lotka_volterra_trace_adaptive.png: MCMC trace plots from adaptive covariance MCMC
#   - lotka_volterra_posterior_predictive.png: Posterior predictive time series plots
#   - lotka_volterra_trace_hmc.png: MCMC trace plots from Hamiltonian Monte Carlo
#   - Console output: MCMC summary statistics (mean, std, quantiles, rhat, ESS)
#
# Changes from Original (model-lotka-volterra.py):
# ------------------------------------------------
#   1. Self-contained: All pints classes (LotkaVolterraModel, MultiOutputProblem,
#      UniformLogPrior, GaussianLogPrior, ComposedLogPrior, GaussianLogLikelihood,
#      LogPosterior, MCMCController, MCMCSummary, HaarioBardenetACMC, HamiltonianMCMC)
#      are re-implemented directly - no "import pints" dependency required.
#   2. Plots saved to PNG: All plt.show() calls replaced with plt.savefig() to save
#      plots as PNG files instead of displaying interactively.
#   3. CUDA device set: os.environ['CUDA_VISIBLE_DEVICES'] = '0' added at startup.
#   4. Reduced iterations for faster execution (~10x speedup):
#      - Adaptive Covariance MCMC: 400 iterations (originally 4000)
#      - Hamiltonian Monte Carlo: 20 iterations (originally 200)
#      - Warm-up removal: 50 samples (originally 500)
#   5. Non-interactive matplotlib backend: matplotlib.use('Agg') for headless execution.
#   6. Data loaded from JSON: All model data (suggested_values, suggested_times,
#      suggested_parameters, MCMC config) are loaded from data/standalone_mcmc_lotka_volterra.json
#      instead of being hard-coded in the script.
#
# =====================================================================================

import os

# Set CUDA device
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
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

def load_data(json_file):
    """
    Load model data from JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing model data.
    
    Returns
    -------
    dict
        Dictionary containing:
        - model_parameters: dict with initial_conditions
        - suggested_parameters: list of model parameters [a, b, c, d]
        - suggested_times: list of time points
        - suggested_values: list of [hare, lynx] population data
        - mcmc_config: MCMC configuration settings
        - prior_config: prior distribution settings
        - hmc_config: HMC-specific settings
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


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
# Lotka-Volterra Model
# =====================================================================================

class LotkaVolterraModel:
    r"""
    Lotka-Volterra model of Predatory-Prey relationships.

    This model describes cyclical fluctuations in the populations of two
    interacting species.

    ODE system:
        dx/dt = ax - bxy  (prey)
        dy/dt = -cy + dxy (predator)

    where x is the number of prey (hare), and y is the number of predators (lynx).

    Parameters: (a, b, c, d) - model parameters
    Outputs: (x, y) - prey and predator populations (2 outputs)

    Real data is included via suggested_values(), which contains hare and lynx
    pelt count data collected by the Hudson's Bay Company in Canada in the
    early twentieth century.
    """

    def __init__(self, y0=None):
        if y0 is None:
            self.set_initial_conditions(np.log([30, 4]))
        else:
            self.set_initial_conditions(y0)

    def set_initial_conditions(self, y0):
        """Changes the initial conditions for this model."""
        a, b = y0
        if a < 0 or b < 0:
            raise ValueError('Initial populations cannot be negative.')
        self._y0 = [a, b]

    def initial_conditions(self):
        """Returns the current initial conditions."""
        return np.array(self._y0, copy=True)

    def n_outputs(self):
        """Returns the number of outputs (2: prey and predator)."""
        return 2

    def n_parameters(self):
        """Returns the number of parameters (4: a, b, c, d)."""
        return 4

    def _rhs(self, state, time, parameters):
        """Right-hand side equation of the ODE to solve."""
        x, y = state
        a, b, c, d = parameters
        return np.array([a * x - b * x * y, -c * y + d * x * y])

    def jacobian(self, z, t, p):
        """Returns the Jacobian of the RHS."""
        x, y = z
        a, b, c, d = [float(param) for param in p]
        ret = np.empty((2, 2))
        ret[0, 0] = a - b * y
        ret[0, 1] = -b * x
        ret[1, 0] = d * y
        ret[1, 1] = -c + d * x
        return ret

    def _dfdp(self, z, t, p):
        """Returns the derivative of the ODE RHS with respect to model parameters."""
        x, y = z
        a, b, c, d = [float(param) for param in p]
        ret = np.empty((2, 4))
        ret[0, 0] = x
        ret[0, 1] = -x * y
        ret[0, 2] = 0
        ret[0, 3] = 0
        ret[1, 0] = 0
        ret[1, 1] = 0
        ret[1, 2] = -y
        ret[1, 3] = x * y
        return ret

    def _rhs_S1(self, y_and_dydp, t, p):
        """
        Forms the RHS of ODE for numerical integration to obtain both outputs
        and sensitivities.
        """
        n_states = 2
        n_params = 4
        y = y_and_dydp[0:n_states]
        dydp = y_and_dydp[n_states:].reshape((n_params, n_states))
        dydt = self._rhs(y, t, p)
        d_dydp_dt = (
            np.matmul(dydp, np.transpose(self.jacobian(y, t, p))) +
            np.transpose(self._dfdp(y, t, p)))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        return self._simulate(parameters, times, False)

    def simulateS1(self, parameters, times):
        """Runs a forward simulation with sensitivities."""
        values, dvalues_dp = self._simulate(parameters, times, True)
        n_outputs = self.n_outputs()
        return values[:, :n_outputs].squeeze(), dvalues_dp[:, :n_outputs, :]

    def _simulate(self, parameters, times, sensitivities):
        """
        Private helper function that uses scipy.integrate.odeint to
        simulate a model (with or without sensitivities).
        """
        times = vector(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        offset = 0
        if len(times) < 1 or times[0] != 0:
            times = np.concatenate(([0], times))
            offset = 1

        if sensitivities:
            n_params = self.n_parameters()
            n_outputs = 2
            y0 = np.zeros(n_params * n_outputs + n_outputs)
            y0[0:n_outputs] = self._y0
            result = odeint(self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = (result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params), order="F"))
            return values[offset:], dvalues_dp[offset:]
        else:
            values = odeint(self._rhs, self._y0, times, (parameters,))
            return values[offset:, :self.n_outputs()].squeeze()

    def suggested_parameters(self):
        """Returns a suggested set of parameters."""
        return np.array([0.5, 0.15, 1.0, 0.3])

    def suggested_times(self):
        """Returns a suggested set of simulation times."""
        return np.linspace(0, 20, 21)

    def suggested_values(self):
        """
        Returns hare-lynx pelt count data collected by the Hudson's Bay Company
        in Canada in the early twentieth century (1900-1920).
        """
        return np.array([
            [30.0, 4.0],   # 1900
            [47.2, 6.1],
            [70.2, 9.8],
            [77.4, 35.2],
            [36.3, 59.4],  # 1904
            [20.6, 41.7],
            [18.1, 19.0],
            [21.4, 13.0],
            [22.0, 8.3],
            [25.4, 9.1],   # 1909
            [27.1, 7.4],
            [40.3, 8.0],
            [57.0, 12.3],
            [76.6, 19.5],
            [52.3, 45.7],  # 1914
            [19.5, 51.1],
            [11.2, 29.7],
            [7.6, 15.8],
            [14.6, 9.7],
            [16.2, 10.1],
            [24.7, 8.1],   # 1920
        ])


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

    def evaluateS1(self, parameters):
        """Runs a simulation with first-order sensitivity calculation."""
        y, dy = self._model.simulateS1(parameters, self._times)
        return (
            np.asarray(y).reshape(self._n_times, self._n_outputs),
            np.asarray(dy).reshape(self._n_times, self._n_outputs, self._n_parameters)
        )

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
# Log-PDFs, Priors, and Likelihoods
# =====================================================================================

class RectangularBoundaries:
    """Represents a set of lower and upper boundaries for model parameters."""

    def __init__(self, lower, upper):
        self._lower = vector(lower)
        self._upper = vector(upper)
        self._n_parameters = len(self._lower)
        if len(self._upper) != self._n_parameters:
            raise ValueError('Lower and upper bounds must have same length.')
        if self._n_parameters < 1:
            raise ValueError('The parameter space must have a dimension > 0')
        if not np.all(self._upper > self._lower):
            raise ValueError('Upper bounds must exceed lower bounds.')

    def check(self, parameters):
        """Returns True if the point is within the boundaries."""
        if np.any(parameters < self._lower):
            return False
        if np.any(parameters >= self._upper):
            return False
        return True

    def n_parameters(self):
        return self._n_parameters

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper

    def range(self):
        return self._upper - self._lower

    def sample(self, n=1):
        return np.random.uniform(self._lower, self._upper, size=(n, self._n_parameters))


class UniformLogPrior:
    """
    Defines a uniform prior over a given range.
    """

    def __init__(self, lower_or_boundaries, upper=None):
        if upper is None:
            if not isinstance(lower_or_boundaries, RectangularBoundaries):
                raise ValueError('UniformLogPrior requires a lower and an upper bound.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = RectangularBoundaries(lower_or_boundaries, upper)
        self._n_parameters = self._boundaries.n_parameters()
        self._value = -np.log(np.prod(self._boundaries.range()))

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else -np.inf

    def evaluateS1(self, x):
        """Returns the log-prior and its gradient."""
        return self(x), np.zeros(self._n_parameters)

    def n_parameters(self):
        return self._n_parameters

    def sample(self, n=1):
        return self._boundaries.sample(n)


class GaussianLogPrior:
    """
    Defines a 1-d Gaussian (log) prior with a given mean and standard deviation.
    """

    def __init__(self, mean, sd):
        self._mean = float(mean)
        if sd <= 0:
            raise ValueError('sd parameter must be positive')
        self._sd = float(sd)
        self._offset = np.log(1 / np.sqrt(2 * np.pi * self._sd ** 2))
        self._factor = 1 / (2 * self._sd ** 2)
        self._factor2 = 1 / self._sd**2

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def evaluateS1(self, x):
        """Returns the log-prior and its gradient."""
        return self(x), self._factor2 * (self._mean - np.asarray(x))

    def n_parameters(self):
        return 1

    def sample(self, n=1):
        return np.random.normal(self._mean, self._sd, size=(n, 1))


class ComposedLogPrior:
    """
    N-dimensional LogPrior composed of one or more other LogPriors.
    """

    def __init__(self, *priors):
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')
        self._n_parameters = 0
        for prior in priors:
            self._n_parameters += prior.n_parameters()
        self._priors = priors

    def __call__(self, x):
        output = 0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output += prior(x[lo:hi])
        return output

    def evaluateS1(self, x):
        """Returns the log-prior and its gradient."""
        output = 0
        doutput = np.zeros(self._n_parameters)
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            p, dp = prior.evaluateS1(x[lo:hi])
            output += p
            doutput[lo:hi] = np.asarray(dp)
        return output, doutput

    def n_parameters(self):
        return self._n_parameters

    def sample(self, n=1):
        output = np.zeros((n, self._n_parameters))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output[:, lo:hi] = prior.sample(n)
        return output


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

    def evaluateS1(self, x):
        """Returns the log-likelihood and its gradient."""
        sigma = np.asarray(x[-self._no:])
        L = self.__call__(x)
        if np.isneginf(L):
            return L, np.tile(np.nan, self._n_parameters)

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x[:-self._no])
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)
        r = self._values - y

        # Calculate derivatives in the model parameters
        dL = np.sum((sigma**(-2.0) * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)

        # Calculate derivative wrt sigma
        dsigma = -self._nt / sigma + sigma**(-3.0) * np.sum(r**2, axis=0)
        dL = np.concatenate((dL, np.array(list(dsigma))))

        return L, dL

    def n_parameters(self):
        return self._n_parameters


class LogPosterior:
    """
    Represents the sum of a LogLikelihood and a LogPrior.
    """

    def __init__(self, log_likelihood, log_prior):
        self._n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != self._n_parameters:
            raise ValueError('Given log_prior and log_likelihood must have same dimension.')
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood
        self._minf = -np.inf

    def __call__(self, x):
        log_prior = self._log_prior(x)
        if log_prior == self._minf:
            return self._minf
        return log_prior + self._log_likelihood(x)

    def evaluateS1(self, x):
        """Returns the log-posterior and its gradient."""
        a, da = self._log_prior.evaluateS1(x)
        b, db = self._log_likelihood.evaluateS1(x)
        return a + b, da + db

    def n_parameters(self):
        return self._n_parameters

    def log_prior(self):
        return self._log_prior


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

    def needs_sensitivities(self):
        return False

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

        # Handle sensitivities (fx may be tuple for HMC, scalar for this method)
        if isinstance(fx, tuple):
            fx = fx[0]
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
# MCMC Sampler - Hamiltonian Monte Carlo
# =====================================================================================

class HamiltonianMCMC:
    """
    Hamiltonian Monte Carlo (HMC) sampler.

    Uses a physical analogy of a particle moving across a landscape under
    Hamiltonian dynamics to aid efficient exploration of parameter space.
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
        self._ready_for_tell = False

        self._current = None
        self._current_energy = None
        self._current_gradient = None
        self._current_momentum = None

        self._momentum = None
        self._position = None
        self._gradient = None

        self._mcmc_iteration = 0
        self._mcmc_acceptance = 0
        self._frog_iteration = 0

        self._n_frog_iterations = 20
        self._epsilon = 0.1
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))
        self._hamiltonian_threshold = 10**3

    def name(self):
        return 'Hamiltonian Monte Carlo'

    def needs_initial_phase(self):
        return False

    def needs_sensitivities(self):
        return True

    def set_leapfrog_steps(self, steps):
        """Sets the number of leapfrog steps to carry out for each iteration."""
        steps = int(steps)
        if steps < 1:
            raise ValueError('Number of steps must exceed 0.')
        self._n_frog_iterations = steps

    def set_leapfrog_step_size(self, step_size):
        """Sets the step size for the leapfrog algorithm."""
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError('Step size for leapfrog algorithm must be greater than zero.')
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError('Step size should either be of length 1 or equal to the number of parameters')
        self._step_size = step_size
        self._set_scaled_epsilon()

    def _set_scaled_epsilon(self):
        """Rescales epsilon along the dimensions of step_size."""
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def ask(self):
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        if not self._running:
            self._running = True

        if self._current is None:
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        if self._frog_iteration == 0:
            self._current_momentum = np.random.multivariate_normal(
                np.zeros(self._n_parameters), np.eye(self._n_parameters))
            self._position = np.array(self._current, copy=True)
            self._gradient = np.array(self._current_gradient, copy=True)
            self._momentum = np.array(self._current_momentum, copy=True)
            self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        self._position += self._scaled_epsilon * self._momentum

        self._ready_for_tell = True
        return np.array(self._position, copy=True)

    def tell(self, reply):
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        energy, gradient = reply
        energy = float(energy)
        gradient = vector(gradient)
        assert gradient.shape == (self._n_parameters,)

        # Energy = -log_pdf, so flip both signs!
        energy = -energy
        gradient = -gradient

        if self._current is None:
            if not np.isfinite(energy):
                raise ValueError('Initial point for MCMC must have finite logpdf.')
            self._current = self._x0
            self._current_energy = energy
            self._current_gradient = gradient
            self._mcmc_iteration += 1
            self._current.setflags(write=False)
            return (self._current, (-self._current_energy, -self._current_gradient), False)

        self._gradient = gradient
        self._frog_iteration += 1

        if self._frog_iteration < self._n_frog_iterations:
            self._momentum -= self._scaled_epsilon * self._gradient
            return None

        # Final leapfrog iteration: only do half a step
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        accept = 0
        if np.isfinite(energy) and np.all(np.isfinite(self._momentum)):
            current_U = self._current_energy
            current_K = np.sum(self._current_momentum**2 / 2)
            proposed_U = energy
            proposed_K = np.sum(self._momentum**2 / 2)

            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:
                self._momentum = self._position = self._gradient = None
                self._frog_iteration = 0
                self._mcmc_iteration += 1
                self._mcmc_acceptance = (
                    (self._mcmc_iteration * self._mcmc_acceptance + accept) /
                    (self._mcmc_iteration + 1))
                return (self._current, (-self._current_energy, -self._current_gradient), False)
            else:
                r = np.exp(current_U - proposed_U + current_K - proposed_K)
                if np.random.uniform(0, 1) < r:
                    accept = 1
                    self._current = self._position
                    self._current_energy = energy
                    self._current_gradient = gradient
                    self._current.setflags(write=False)

        self._momentum = self._position = self._gradient = None
        self._frog_iteration = 0
        self._mcmc_iteration += 1
        self._mcmc_acceptance = (
            (self._mcmc_iteration * self._mcmc_acceptance + accept) /
            (self._mcmc_iteration + 1))

        return (self._current, (-self._current_energy, -self._current_gradient), accept > 0)


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
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()

        self._initial_phase_iterations = 200 if self._needs_initial_phase else None
        self._log_to_screen = True
        self._log_interval = 20
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

    def set_log_interval(self, interval):
        self._log_interval = int(interval)

    def samplers(self):
        return self._samplers

    def time(self):
        return self._time

    def run(self):
        """Runs the MCMC sampler and returns the chains."""
        if self._max_iterations is None:
            raise ValueError('At least one stopping criterion must be set.')

        iteration = 0

        # Use evaluateS1 if sensitivities are needed
        if self._needs_sensitivities:
            f = self._log_pdf.evaluateS1
        else:
            f = self._log_pdf

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
            fxs = [f(x) for x in xs]

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

            if iteration % self._log_interval == 0 and self._log_to_screen:
                print(f'Iter. {iteration}/{self._max_iterations}')

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


# =====================================================================================
# Main Script
# =====================================================================================

if __name__ == '__main__':
    print('=' * 70)
    print('Lotka-Volterra Predator-Prey Model - MCMC Inference')
    print('=' * 70)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'data', 'standalone_mcmc_lotka_volterra.json')
    
    # Load data from JSON file
    print(f'\nLoading data from: {json_file}')
    data = load_data(json_file)
    
    # Extract data from JSON
    times = np.array(data['suggested_times'])
    values = np.array(data['suggested_values'])
    initial_conditions = data['model_parameters']['initial_conditions']
    mcmc_config = data['mcmc_config']
    prior_config = data['prior_config']
    hmc_config = data['hmc_config']
    
    print(f'Loaded {len(times)} time points')
    print(f'Loaded {len(values)} data points (hare-lynx populations)')

    # Create model with initial conditions from JSON
    model = LotkaVolterraModel(y0=initial_conditions)

    print('Outputs: ' + str(model.n_outputs()))
    print('Parameters: ' + str(model.n_parameters()))

    # Plot the real data
    plt.figure()
    plt.xlabel('Years since 1900')
    plt.ylabel('Population size')
    plt.plot(times, values)
    plt.legend(['hare', 'lynx'])
    plt.title('Hudson\'s Bay Company Pelt Data (1900-1920)')
    plt.savefig('lotka_volterra_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Data plot saved to: lotka_volterra_data.png')

    # Plot phase diagram
    plt.figure()
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.xlabel('hare')
    plt.ylabel('lynx')
    plt.plot(values[:, 0], values[:, 1])
    plt.title('Phase Plot: Hare vs Lynx Populations')
    plt.savefig('lotka_volterra_phase.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Phase plot saved to: lotka_volterra_phase.png')

    # Create problem with log-transformed values (multiplicative noise model)
    problem = MultiOutputProblem(model, times, np.log(values))

    # Create a log posterior using prior config from JSON
    log_prior_theta = UniformLogPrior(
        lower_or_boundaries=prior_config['theta_lower'], 
        upper=prior_config['theta_upper'])
    log_prior_sigma = GaussianLogPrior(
        mean=prior_config['sigma_mean'], 
        sd=prior_config['sigma_sd'])
    log_prior = ComposedLogPrior(
        log_prior_theta, log_prior_theta, log_prior_theta, log_prior_theta,
        log_prior_sigma, log_prior_sigma)
    log_likelihood = GaussianLogLikelihood(problem)
    log_posterior = LogPosterior(log_likelihood, log_prior)

    print('\nSetting up MCMC with Adaptive Covariance...')
    print(f'Number of model parameters: {model.n_parameters()}')
    print(f'Number of total parameters: {log_posterior.n_parameters()}')
    print('  (4 model params + 2 noise sigma params)')

    # =========================================================================
    # Part 1: Haario-Bardenet Adaptive Covariance MCMC
    # =========================================================================
    print('\n' + '=' * 70)
    print('Part 1: Adaptive Covariance MCMC')
    print('=' * 70)

    # Starting points for chains (from JSON config)
    n_chains = mcmc_config['n_chains']
    x0 = [mcmc_config['x0']] * n_chains
    adaptive_iterations = mcmc_config['adaptive_iterations']

    # Create MCMC routine
    mcmc = MCMCController(log_posterior, n_chains, x0)
    mcmc.set_max_iterations(adaptive_iterations)
    mcmc.set_log_interval(50)

    print(f'Running MCMC ({adaptive_iterations} iterations, {n_chains} chains)...')
    chains = mcmc.run()

    # Create trace plots
    print('\nGenerating trace plots...')
    plot_trace(
        chains,
        parameter_names=['a', 'b', 'c', 'd', 'sigma_1', 'sigma_2'],
        save_path='lotka_volterra_trace_adaptive.png'
    )

    # Print summary
    print('\n' + '=' * 70)
    print('MCMC Summary (Adaptive Covariance)')
    print('=' * 70)
    results = MCMCSummary(
        chains=chains,
        parameter_names=["a", "b", "c", "d", "sigma_1", "sigma_2"],
        time=mcmc.time())
    print(results)

    # Select first chain and remove warm up period (from JSON config)
    warmup_samples = mcmc_config['warmup_samples']
    chain1 = chains[0]
    chain1 = chain1[warmup_samples:]

    # Create fine-grained list of times for plotting
    n_fine = 1000
    times_fine = np.linspace(min(times), max(times), n_fine)

    # Plot posterior predictive
    print('\nGenerating posterior predictive plot...')
    num_lines = 100
    hare = np.zeros((n_fine, num_lines))
    lynx = np.zeros((n_fine, num_lines))
    for i in range(num_lines):
        temp = np.exp(model.simulate(times=times_fine, parameters=chain1[i, :4]))
        hare[:, i] = temp[:, 0]
        lynx[:, i] = temp[:, 1]

    plt.figure(figsize=(15, 7))
    plt.xlabel('Years since 1900')
    plt.ylabel('Populations')
    plt.plot(times_fine, hare, color='blue', alpha=0.01)
    plt.plot(times_fine, lynx, color='orange', alpha=0.01)
    plt.plot(times, values, 'o-')
    plt.legend(['Hare predictions', 'Lynx predictions', 'Hare data', 'Lynx data'])
    plt.title('Posterior Predictive - Adaptive Covariance MCMC')
    plt.savefig('lotka_volterra_posterior_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Posterior predictive plot saved to: lotka_volterra_posterior_predictive.png')

    # =========================================================================
    # Part 2: Hamiltonian Monte Carlo
    # =========================================================================
    print('\n' + '=' * 70)
    print('Part 2: Hamiltonian Monte Carlo')
    print('=' * 70)

    # Starting points for chains (from JSON config)
    x0 = [mcmc_config['x0']] * n_chains
    hmc_iterations = mcmc_config['hmc_iterations']
    mcmc_hmc = MCMCController(log_posterior, n_chains, x0, method=HamiltonianMCMC)
    mcmc_hmc.set_max_iterations(hmc_iterations)
    mcmc_hmc.set_log_interval(5)

    # Set leapfrog parameters from JSON config
    for sampler in mcmc_hmc.samplers():
        sampler.set_leapfrog_step_size(hmc_config['leapfrog_step_size'])
        sampler.set_leapfrog_steps(hmc_config['leapfrog_steps'])

    print(f'Running HMC ({hmc_iterations} iterations, {n_chains} chains)...')
    chains_hmc = mcmc_hmc.run()

    # Create trace plots for HMC
    print('\nGenerating HMC trace plots...')
    plot_trace(
        chains_hmc,
        parameter_names=['a', 'b', 'c', 'd', 'sigma_1', 'sigma_2'],
        save_path='lotka_volterra_trace_hmc.png'
    )

    # Print summary
    print('\n' + '=' * 70)
    print('MCMC Summary (Hamiltonian Monte Carlo)')
    print('=' * 70)
    results_hmc = MCMCSummary(
        chains=chains_hmc,
        parameter_names=["a", "b", "c", "d", "sigma_1", "sigma_2"],
        time=mcmc_hmc.time())
    print(results_hmc)

    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)
    print('\nOutput files:')
    print('  - lotka_volterra_data.png')
    print('  - lotka_volterra_phase.png')
    print('  - lotka_volterra_trace_adaptive.png')
    print('  - lotka_volterra_posterior_predictive.png')
    print('  - lotka_volterra_trace_hmc.png')

"""
