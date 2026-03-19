# ==============================================================================
# Standalone FitzHugh-Nagumo Model with Optimization and MCMC Inference
# ==============================================================================
# This is a self-contained script that performs parameter inference on the
# FitzHugh-Nagumo model using optimization (CMAES) and MCMC sampling
# (Monomial-Gamma Hamiltonian Monte Carlo).
#
# File Dependencies:
# ------------------
# Inputs:
#   - data/standalone_mcmc_fitzhugh_nagumo.json: JSON file containing model data
#     (times, noisy observations, true parameters, configuration settings)
#
# Outputs:
#   - fitzhugh_nagumo_simulation.png: Original simulation plot
#   - fitzhugh_nagumo_noisy.png: Noisy data plot  
#   - fitzhugh_nagumo_optimization.png: Optimization result comparison plot
#   - fitzhugh_nagumo_trace.png: MCMC trace plot
#   - fitzhugh_nagumo_series.png: MCMC series prediction plot
#
# Algorithm: CMAES (optimization), MonomialGammaHamiltonianMCMC (sampling)
# Model: FitzHugh-Nagumo neuron action potential model
#
# Changes from Original:
# ----------------------
#   - Extracted all pints dependencies into this standalone file (no import pints)
#   - Replaced plt.show() with plt.savefig() to save plots as PNG files
#   - Reduced MCMC max_iterations from 200 to 20 for ~10x faster execution
#   - Modified to read data from external JSON file instead of generating
#     synthetic data internally. Data file: data/standalone_mcmc_fitzhugh_nagumo.json
# ==============================================================================

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import numpy as np
import scipy.integrate
from scipy import integrate
from scipy import interpolate
import scipy.special
import scipy.stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import timeit
import warnings
from tabulate import tabulate

# ==============================================================================
# Utility Functions
# ==============================================================================

FLOAT_FORMAT = '{: .17e}'

def strfloat(x):
    """Converts a float to a string, with maximum precision."""
    return FLOAT_FORMAT.format(float(x))

def vector(x):
    """Copies x and returns a 1d read-only NumPy array of floats with shape (n,)."""
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
    """Copies x and returns a 2d read-only NumPy array of floats."""
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

    def time(self):
        return timeit.default_timer() - self._start

    def reset(self):
        self._start = timeit.default_timer()

# ==============================================================================
# Data Loading Module
# ==============================================================================

def load_data(json_path):
    """
    Load model data from a JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON data file.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'true_parameters': list of true model parameters [a, b, c]
        - 'times': numpy array of time points
        - 'noisy_values': numpy array of noisy observations (shape: n_times x n_outputs)
        - 'sigma': noise standard deviation
        - 'optimization': dict with 'x0', 'boundaries_lower', 'boundaries_upper'
        - 'mcmc': dict with MCMC configuration settings
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['times'] = np.array(data['times'])
    data['noisy_values'] = np.array(data['noisy_values'])
    data['true_parameters'] = list(data['true_parameters'])
    
    return data

# ==============================================================================
# FitzHugh-Nagumo Model
# ==============================================================================

class FitzhughNagumoModel:
    """
    FitzHugh-Nagumo model of the action potential.
    
    Has two states (V, R), and three phenomenological parameters: a, b, c.
    
    dV/dt = c * (R - V^3/3 + V)
    dR/dt = -(V - a + b*R) / c
    """
    def __init__(self, y0=None):
        if y0 is None:
            self._y0 = np.array([-1, 1], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_outputs(self):
        return 2

    def n_parameters(self):
        return 3

    def _rhs(self, y, t, p):
        V, R = y
        a, b, c = [float(x) for x in p]
        # Protect against division by zero
        if c == 0:
            c = 1e-10
        dV_dt = (V - V**3 / 3 + R) * c
        dR_dt = (V - a + b * R) / -c
        return dV_dt, dR_dt

    def jacobian(self, y, t, p):
        V, R = y
        a, b, c = [float(param) for param in p]
        # Protect against division by zero
        if c == 0:
            c = 1e-10
        ret = np.empty((2, 2))
        ret[0, 0] = c * (1 - V**2)
        ret[0, 1] = c
        ret[1, 0] = -1 / c
        ret[1, 1] = -b / c
        return ret

    def _dfdp(self, y, t, p):
        V, R = y
        a, b, c = [float(param) for param in p]
        # Protect against division by zero
        if c == 0:
            c = 1e-10
        ret = np.empty((2, 3))
        ret[0, 0] = 0
        ret[0, 1] = 0
        ret[0, 2] = R - V**3 / 3 + V
        ret[1, 0] = 1 / c
        ret[1, 1] = -R / c
        ret[1, 2] = (R * b + V - a) / c**2
        return ret

    def _rhs_S1(self, y_and_dydp, t, p):
        n_outputs = 2
        n_params = 3
        y = y_and_dydp[0:n_outputs]
        dydp = y_and_dydp[n_outputs:].reshape((n_params, n_outputs))
        dydt = self._rhs(y, t, p)
        d_dydp_dt = (
            np.matmul(dydp, np.transpose(self.jacobian(y, t, p))) +
            np.transpose(self._dfdp(y, t, p)))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def simulate(self, parameters, times):
        return self._simulate(parameters, times, False)

    def _simulate(self, parameters, times, sensitivities):
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
            result = scipy.integrate.odeint(self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params), order="F")
            return values[offset:], dvalues_dp[offset:]
        else:
            values = scipy.integrate.odeint(self._rhs, self._y0, times, (parameters,))
            return values[offset:, :self.n_outputs()].squeeze()

    def simulateS1(self, parameters, times):
        values, dvalues_dp = self._simulate(parameters, times, True)
        n_outputs = self.n_outputs()
        return values[:, :n_outputs].squeeze(), dvalues_dp[:, :n_outputs, :]

# ==============================================================================
# Problem Classes
# ==============================================================================

class MultiOutputProblem:
    """Represents an inference problem with multiple outputs."""
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
            raise ValueError('Values array must have shape (n_times, n_outputs).')

    def evaluate(self, parameters):
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def evaluateS1(self, parameters):
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

# ==============================================================================
# Error Measures
# ==============================================================================

class ErrorMeasure:
    """Abstract base class for error measures."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

class SumOfSquaresError(ErrorMeasure):
    """Calculates the sum of squares error."""
    def __init__(self, problem, weights=None):
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._n_outputs = problem.n_outputs()
        self._n_parameters = problem.n_parameters()
        self._n_times = len(self._times)
        
        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError('Number of weights must match number of problem outputs.')
        self._weights = np.asarray([float(w) for w in weights])

    def __call__(self, x):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._problem.evaluate(x)
            if not np.all(np.isfinite(result)):
                return np.inf
            return np.sum((np.sum(((result - self._values)**2),
                                  axis=0) * self._weights), axis=0)
        except Exception:
            return np.inf

    def evaluateS1(self, x):
        y, dy = self._problem.evaluateS1(x)
        dy = dy.reshape((self._n_times, self._n_outputs, self._n_parameters))
        r = y - self._values
        e = np.sum(np.sum(r**2, axis=0) * self._weights, axis=0)
        de = 2 * np.sum(np.sum((r.T * dy.T), axis=2) * self._weights, axis=1)
        return e, de

    def n_parameters(self):
        return self._n_parameters

class ProbabilityBasedError(ErrorMeasure):
    """Changes the sign of a LogPDF to use it as an error."""
    def __init__(self, log_pdf):
        self._log_pdf = log_pdf

    def __call__(self, x):
        return -self._log_pdf(x)

    def evaluateS1(self, x):
        y, dy = self._log_pdf.evaluateS1(x)
        return -y, -np.asarray(dy)

    def n_parameters(self):
        return self._log_pdf.n_parameters()

# ==============================================================================
# Boundaries
# ==============================================================================

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

# ==============================================================================
# Log PDFs and Priors
# ==============================================================================

class LogPDF:
    """Represents the natural logarithm of a probability density function."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

class LogPrior(LogPDF):
    """Represents the natural logarithm of a known probability density function."""
    def sample(self, n=1):
        raise NotImplementedError

class UniformLogPrior(LogPrior):
    """Defines a uniform prior over a given range."""
    def __init__(self, lower, upper):
        self._boundaries = RectangularBoundaries(lower, upper)
        self._n_parameters = self._boundaries.n_parameters()
        self._value = -np.log(np.prod(self._boundaries.range()))

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else -np.inf

    def evaluateS1(self, x):
        return self(x), np.zeros(self._n_parameters)

    def n_parameters(self):
        return self._n_parameters

    def sample(self, n=1):
        return self._boundaries.sample(n)

class GaussianLogLikelihood(LogPDF):
    """Calculates log-likelihood assuming independent Gaussian noise."""
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
        sigma = np.asarray(x[-self._no:])
        L = self.__call__(x)
        if np.isneginf(L):
            return L, np.tile(np.nan, self._n_parameters)
        
        y, dy = self._problem.evaluateS1(x[:-self._no])
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)
        r = self._values - y
        dL = np.sum((sigma**(-2.0) * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)
        dsigma = -self._nt / sigma + sigma**(-3.0) * np.sum(r**2, axis=0)
        dL = np.concatenate((dL, np.array(list(dsigma))))
        return L, dL

    def n_parameters(self):
        return self._n_parameters

class LogPosterior(LogPDF):
    """Represents the sum of a LogPDF and a LogPrior."""
    def __init__(self, log_likelihood, log_prior):
        if not isinstance(log_prior, LogPrior):
            raise ValueError('Given prior must extend LogPrior.')
        if not isinstance(log_likelihood, LogPDF):
            raise ValueError('Given log_likelihood must extend LogPDF.')
        
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
        a, da = self._log_prior.evaluateS1(x)
        b, db = self._log_likelihood.evaluateS1(x)
        return a + b, da + db

    def log_likelihood(self):
        return self._log_likelihood

    def log_prior(self):
        return self._log_prior

    def n_parameters(self):
        return self._n_parameters

# ==============================================================================
# Parallelisation / Evaluation
# ==============================================================================

class SequentialEvaluator:
    """Evaluates function sequentially."""
    def __init__(self, function):
        self._function = function

    def evaluate(self, positions):
        return [self._function(x) for x in positions]

class ParallelEvaluator:
    """Evaluates function in parallel."""
    def __init__(self, function, n_workers=None):
        import multiprocessing
        self._function = function
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        self._n_workers = n_workers

    def evaluate(self, positions):
        import multiprocessing
        with multiprocessing.Pool(self._n_workers) as pool:
            return pool.map(self._function, positions)

    @staticmethod
    def cpu_count():
        import multiprocessing
        return multiprocessing.cpu_count()

# ==============================================================================
# Logger
# ==============================================================================

class Logger:
    """Basic logging class."""
    def __init__(self):
        self._stream = True
        self._filename = None
        self._csv = False
        self._fields = []
        self._line = []
        self._header_logged = False

    def add_counter(self, name, max_value=None):
        self._fields.append(('counter', name, max_value))

    def add_float(self, name):
        self._fields.append(('float', name))

    def add_time(self, name):
        self._fields.append(('time', name))

    def log(self, *args):
        for arg in args:
            self._line.append(arg)
        if len(self._line) >= len(self._fields):
            self._write_line()
            self._line = []

    def set_stream(self, stream):
        self._stream = stream

    def set_filename(self, filename, csv=False):
        self._filename = filename
        self._csv = csv

    def _write_line(self):
        if not self._header_logged:
            if self._stream:
                header = ' '.join([f[1] for f in self._fields])
                print(header)
            self._header_logged = True
        
        if self._stream:
            parts = []
            for i, (ftype, name, *rest) in enumerate(self._fields):
                val = self._line[i]
                if ftype == 'counter':
                    parts.append(f'{val:>8}')
                elif ftype == 'float':
                    parts.append(f'{val:>12.4e}')
                elif ftype == 'time':
                    parts.append(f'{val:>10.2f}')
            print(' '.join(parts))

# ==============================================================================
# CMAES Optimizer
# ==============================================================================

class CMAES:
    """CMA-ES Optimizer (Covariance Matrix Adaptation Evolution Strategy)."""
    def __init__(self, x0, sigma0=None, boundaries=None):
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)
        self._boundaries = boundaries
        
        if sigma0 is None:
            if boundaries is not None:
                self._sigma0 = (1 / 6) * boundaries.range()
            else:
                self._sigma0 = (1 / 3) * np.abs(self._x0)
                self._sigma0 += (self._sigma0 == 0)
        elif np.isscalar(sigma0):
            self._sigma0 = np.ones(self._n_parameters) * float(sigma0)
        else:
            self._sigma0 = vector(sigma0)

        # CMA-ES specific initialization
        self._running = False
        self._ready_for_tell = False
        self._xbest = self._x0.copy()
        self._fbest = np.inf
        
        # Population size
        self._population_size = 4 + int(3 * np.log(self._n_parameters))
        
        # Initialize CMA-ES parameters
        self._mean = self._x0.copy()
        self._sigma = np.min(self._sigma0)
        self._C = np.eye(self._n_parameters)  # Covariance matrix
        self._pc = np.zeros(self._n_parameters)  # Evolution path for C
        self._ps = np.zeros(self._n_parameters)  # Evolution path for sigma
        
        # Strategy parameters
        self._mu = self._population_size // 2
        self._weights = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        self._weights /= np.sum(self._weights)
        self._mueff = 1 / np.sum(self._weights**2)
        
        # Adaptation parameters
        self._cc = 4 / (self._n_parameters + 4)
        self._cs = (self._mueff + 2) / (self._n_parameters + self._mueff + 3)
        self._c1 = 2 / ((self._n_parameters + 1.3)**2 + self._mueff)
        self._cmu = min(1 - self._c1, 2 * (self._mueff - 2 + 1/self._mueff) / ((self._n_parameters + 2)**2 + self._mueff))
        self._damps = 1 + 2 * max(0, np.sqrt((self._mueff - 1) / (self._n_parameters + 1)) - 1) + self._cs
        
        self._chiN = np.sqrt(self._n_parameters) * (1 - 1/(4*self._n_parameters) + 1/(21*self._n_parameters**2))

    def ask(self):
        if self._ready_for_tell:
            raise RuntimeError('Ask called when expecting tell.')
        self._ready_for_tell = True
        
        # Generate offspring
        try:
            self._B = np.linalg.cholesky(self._C)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self._C)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            self._B = eigvecs @ np.diag(np.sqrt(eigvals))
        
        self._population = []
        for _ in range(self._population_size):
            z = np.random.randn(self._n_parameters)
            x = self._mean + self._sigma * np.dot(self._B, z)
            if self._boundaries is not None:
                # Ensure parameters are strictly within bounds (not on boundary)
                lower = self._boundaries.lower() + 1e-6
                upper = self._boundaries.upper() - 1e-6
                x = np.clip(x, lower, upper)
            self._population.append(x)
        return self._population

    def tell(self, fx):
        if not self._ready_for_tell:
            raise RuntimeError('Tell called without ask.')
        self._ready_for_tell = False
        self._running = True
        
        # Sort by fitness
        idx = np.argsort(fx)
        fx = np.array(fx)[idx]
        self._population = [self._population[i] for i in idx]
        
        # Update best
        if fx[0] < self._fbest:
            self._fbest = fx[0]
            self._xbest = self._population[0].copy()
        
        # Select elite
        elite = np.array(self._population[:self._mu])
        
        # Compute new mean
        old_mean = self._mean.copy()
        self._mean = np.dot(self._weights, elite)
        
        # Update evolution paths
        self._ps = (1 - self._cs) * self._ps + np.sqrt(self._cs * (2 - self._cs) * self._mueff) * \
                   np.linalg.solve(self._B, (self._mean - old_mean) / self._sigma)
        hsig = np.linalg.norm(self._ps) / np.sqrt(1 - (1 - self._cs)**(2 * (len(fx) + 1))) / self._chiN < 1.4 + 2 / (self._n_parameters + 1)
        
        self._pc = (1 - self._cc) * self._pc + hsig * np.sqrt(self._cc * (2 - self._cc) * self._mueff) * \
                   (self._mean - old_mean) / self._sigma
        
        # Update covariance matrix
        artmp = (1/self._sigma) * (elite - old_mean).T
        self._C = (1 - self._c1 - self._cmu) * self._C + \
                  self._c1 * np.outer(self._pc, self._pc) + \
                  self._cmu * np.dot(artmp, np.dot(np.diag(self._weights), artmp.T))
        
        # Update sigma
        self._sigma = self._sigma * np.exp((self._cs / self._damps) * (np.linalg.norm(self._ps) / self._chiN - 1))

    def f_best(self):
        return self._fbest

    def x_best(self):
        return self._xbest

    def f_guessed(self):
        return self._fbest

    def x_guessed(self):
        return self._mean

    def running(self):
        return self._running

    def stop(self):
        return False

    def name(self):
        return 'CMA-ES'

    def population_size(self):
        return self._population_size

    def _log_init(self, logger):
        pass

    def _log_write(self, logger):
        pass

# ==============================================================================
# Optimization Controller
# ==============================================================================

def optimise(function, x0, sigma0=None, boundaries=None, method=None):
    """Finds parameter values that minimize an ErrorMeasure or maximize a LogPDF."""
    # Check if minimizing or maximizing
    minimising = not isinstance(function, LogPDF)
    
    if not minimising:
        function = ProbabilityBasedError(function)
    
    if method is None:
        method = CMAES
    
    optimiser = method(x0, sigma0, boundaries)
    
    # Run optimization
    max_iterations = 10000
    unchanged_max = 200
    unchanged_threshold = 1e-11
    
    iteration = 0
    evaluations = 0
    fb = np.inf
    f_sig = np.inf
    unchanged_iterations = 0
    
    print('Minimising error measure')
    print('Using ' + str(optimiser.name()))
    print('Running in sequential mode.')
    print('Population size: ' + str(optimiser.population_size()))
    
    timer = Timer()
    
    while iteration < max_iterations:
        xs = optimiser.ask()
        fs = [function(x) for x in xs]
        optimiser.tell(fs)
        evaluations += len(fs)
        
        fb = optimiser.f_best()
        
        if np.abs(fb - f_sig) >= unchanged_threshold:
            unchanged_iterations = 0
            f_sig = fb
        else:
            unchanged_iterations += 1
        
        if iteration % 20 == 0 or iteration < 3:
            print(f'Iter: {iteration:6d}  Eval: {evaluations:8d}  Best: {fb:12.4e}  Time: {timer.time():8.2f}')
        
        iteration += 1
        
        if unchanged_iterations >= unchanged_max:
            print(f'Halting: No significant change for {unchanged_iterations} iterations.')
            break
    
    x = optimiser.x_best()
    f = optimiser.f_best()
    return x, f if minimising else -f

# ==============================================================================
# MCMC Classes
# ==============================================================================

class MonomialGammaHamiltonianMCMC:
    """Monomial Gamma Hamiltonian MCMC sampler."""
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
        self._divergent = np.asarray([], dtype='int')
        self._hamiltonian_threshold = 10**3
        self._a = 1.0
        self._c = 0.2
        self._m = 1.0
        self._f = None

    def name(self):
        return 'Monomial-Gamma Hamiltonian Monte Carlo'

    def needs_sensitivities(self):
        return True

    def needs_initial_phase(self):
        return False

    def set_leapfrog_step_size(self, step_size):
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError('Step size must be greater than zero.')
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError('Step size should equal number of parameters.')
        self._step_size = step_size
        self._set_scaled_epsilon()

    def set_leapfrog_steps(self, steps):
        steps = int(steps)
        if steps < 1:
            raise ValueError('Number of steps must exceed 0.')
        self._n_frog_iterations = steps

    def _set_scaled_epsilon(self):
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def _g(self, p, a, m):
        return (1 / m) * np.sign(p) * np.abs(p)**(1 / a)

    def _K_indiv(self, p, a, c, m):
        return -self._g(p, a, m) + (2.0 / c) * np.log(1.0 + np.exp(c * self._g(p, a, m)))

    def _K(self, v_p, a, c, m):
        return np.sum([self._K_indiv(p, a, c, m) for p in v_p])

    def _K_deriv_indiv(self, p, a, c, m):
        abs_p = np.abs(p)
        sign_p = np.sign(p)
        tanh = np.tanh(0.5 * c * abs_p**(1.0 / a) * sign_p / m)
        return abs_p**(-2 + 1.0 / a) * p * sign_p * tanh / (a * m)

    def _K_deriv(self, v_p, a, c, m):
        return np.array([self._K_deriv_indiv(p, a, c, m) for p in v_p])

    def _pdf(self, p, a, c, m, z):
        return (1.0 / z) * np.exp(-self._K_indiv(p, a, c, m))

    def _cdf(self, p, a, c, m, z):
        return integrate.quad(lambda p1: self._pdf(p1, a, c, m, z), -np.inf, p)[0]

    def _inverse_cdf_calculator(self, a, c, m, z, pmax=100):
        p = np.linspace(-pmax, pmax, 1000)
        lcdf = [self._cdf(p[i], a, c, m, z) for i in range(1000)]
        f = interpolate.interp1d(lcdf, p, fill_value="extrapolate")
        return f

    def _initialise_ke(self):
        z = integrate.quad(lambda p: np.exp(-self._K_indiv(p, self._a, self._c, self._m)), -np.inf, np.inf)[0]
        self._f = self._inverse_cdf_calculator(self._a, self._c, self._m, z)

    def _sample_momentum(self):
        us = np.random.uniform(size=self._n_parameters)
        return np.array([self._f([u])[0] for u in us])

    def ask(self):
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')
        
        if not self._running:
            self._running = True
            self._initialise_ke()
        
        if self._current is None:
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)
        
        if self._frog_iteration == 0:
            self._current_momentum = self._sample_momentum()
            self._position = np.array(self._current, copy=True)
            self._gradient = np.array(self._current_gradient, copy=True)
            self._momentum = np.array(self._current_momentum, copy=True)
            self._momentum -= self._scaled_epsilon * self._gradient * 0.5
        
        self._position += self._scaled_epsilon * self._K_deriv(self._momentum, self._a, self._c, self._m)
        self._ready_for_tell = True
        return np.array(self._position, copy=True)

    def tell(self, reply):
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False
        
        energy, gradient = reply
        energy = float(energy)
        gradient = vector(gradient)
        energy = -energy
        gradient = -gradient
        
        if self._current is None:
            if not np.isfinite(energy):
                raise ValueError('Initial point must have finite logpdf.')
            self._current = self._x0
            self._current_energy = energy
            self._current_gradient = gradient
            self._mcmc_iteration += 1
            self._current.setflags(write=False)
            return (self._current, (-self._current_energy, -self._current_gradient), True)
        
        self._gradient = gradient
        self._frog_iteration += 1
        
        if self._frog_iteration < self._n_frog_iterations:
            self._momentum -= self._scaled_epsilon * self._gradient
            return None
        
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5
        
        accept = 0
        if np.isfinite(energy) and np.all(np.isfinite(self._momentum)):
            current_U = self._current_energy
            current_K = self._K(self._current_momentum, self._a, self._c, self._m)
            proposed_U = energy
            proposed_K = self._K(self._momentum, self._a, self._c, self._m)
            
            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:
                self._divergent = np.append(self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None
                self._frog_iteration = 0
                self._mcmc_iteration += 1
                self._mcmc_acceptance = ((self._mcmc_iteration * self._mcmc_acceptance + accept) / (self._mcmc_iteration + 1))
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
        self._mcmc_acceptance = ((self._mcmc_iteration * self._mcmc_acceptance + accept) / (self._mcmc_iteration + 1))
        return (self._current, (-self._current_energy, -self._current_gradient), accept != 0)

    def _log_init(self, logger):
        logger.add_float('Accept.')

    def _log_write(self, logger):
        logger.log(self._mcmc_acceptance)

# ==============================================================================
# MCMC Controller
# ==============================================================================

class MCMCController:
    """Controls MCMC sampling."""
    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()
        self._n_chains = int(chains)
        
        if len(x0) != chains:
            raise ValueError('Number of initial positions must equal number of chains.')
        
        if method is None:
            method = MonomialGammaHamiltonianMCMC
        
        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._log_to_screen = True
        self._max_iterations = 10000
        self._parallel = False
        self._n_workers = 1
        self._message_interval = 20
        self._message_warm_up = 3
        self._has_run = False
        self._samples = None
        self._time = None

    def set_max_iterations(self, iterations):
        self._max_iterations = int(iterations)

    def set_log_interval(self, iters=20, warm_up=3):
        self._message_interval = int(iters)
        self._message_warm_up = max(0, int(warm_up))

    def set_parallel(self, parallel=False):
        if parallel is True:
            self._parallel = True
            self._n_workers = ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def samplers(self):
        return self._samplers

    def time(self):
        return self._time

    def run(self):
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True
        
        f = self._log_pdf
        if self._needs_sensitivities:
            f = f.evaluateS1
        
        if self._parallel:
            evaluator = ParallelEvaluator(f, n_workers=min(self._n_workers, self._n_chains))
        else:
            evaluator = SequentialEvaluator(f)
        
        samples = np.zeros((self._n_chains, self._max_iterations, self._n_parameters))
        
        active = list(range(self._n_chains))
        n_samples = [0] * self._n_chains
        
        print('Using ' + str(self._samplers[0].name()))
        print('Generating ' + str(self._n_chains) + ' chains.')
        if self._parallel:
            print('Running in parallel.')
        else:
            print('Running in sequential mode.')
        
        timer = Timer()
        iteration = 0
        next_message = 0
        
        while active:
            xs = [self._samplers[i].ask() for i in active]
            fxs = evaluator.evaluate(xs)
            
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
            
            if iteration >= next_message:
                print(f'Iter: {iteration:6d}  Time: {timer.time():8.2f}')
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (1 + iteration // self._message_interval)
            
            iteration += 1
        
        self._time = timer.time()
        self._samples = samples
        print(f'Halting: Maximum number of iterations ({self._max_iterations}) reached.')
        return samples

# ==============================================================================
# Diagnostics
# ==============================================================================

def autocorrelation(x):
    """Calculates autocorrelation for a vector x."""
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]

def _autocorrelate_negative(autocorrelation):
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)

def effective_sample_size_single_parameter(x):
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess

def effective_sample_size(samples):
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')
    return [effective_sample_size_single_parameter(samples[:, i]) for i in range(n_params)]

def _within(chains):
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)

def _between(chains):
    n = chains.shape[1]
    within_chain_means = np.mean(chains, axis=1)
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)
    return n * between_chain_var

def rhat(chains, warm_up=0.0):
    """Returns the convergence measure R-hat."""
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError('Chains must be 2 or 3 dimensional.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError('warm_up must be in [0,1].')
    
    n = chains.shape[1]
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]
    n = n // 2
    if n < 1:
        raise ValueError('Not enough samples.')
    chains = np.vstack([chains[:, :n], chains[:, -n:]])
    
    w = _within(chains)
    b = _between(chains)
    return np.sqrt((n - 1.0) / n + b / (w * n))

# ==============================================================================
# MCMC Summary
# ==============================================================================

class MCMCSummary:
    """Calculates and prints key summaries of posterior samples."""
    def __init__(self, chains, time=None, parameter_names=None):
        self._chains = chains
        self._chains_unmodified = chains
        
        if len(chains) == 1:
            warnings.warn('Summaries with one chain may be unreliable.')
        
        self._n_parameters = chains[0].shape[1]
        
        if time is not None and float(time) <= 0:
            raise ValueError('Elapsed time must be positive.')
        self._time = time
        
        if parameter_names is None:
            parameter_names = ['param ' + str(i + 1) for i in range(self._n_parameters)]
        elif self._n_parameters != len(parameter_names):
            raise ValueError('Parameter names list must match number of parameters.')
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
            self._summary_str = tabulate(self._summary_list, headers=headers, numalign='left', floatfmt='.2f')
        return self._summary_str

    def _make_summary(self):
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
        
        for i in range(self._n_parameters):
            row = [
                self._parameter_names[i], self._mean[i], self._std[i],
                self._quantiles[0, i], self._quantiles[1, i], self._quantiles[2, i],
                self._quantiles[3, i], self._quantiles[4, i], self._rhat[i], self._ess[i],
            ]
            if self._time is not None:
                row.append(self._ess_per_second[i])
            self._summary_list.append(row)

# ==============================================================================
# Plotting Functions
# ==============================================================================

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

# ==============================================================================
# Main Script
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FitzHugh-Nagumo Model: Optimization and MCMC Inference")
    print("=" * 60)
    
    # Load data from JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'data', 'standalone_mcmc_fitzhugh_nagumo.json')
    print(f"\nLoading data from: {json_path}")
    data = load_data(json_path)
    
    # Extract data
    parameters = data['true_parameters']
    times = data['times']
    noisy = data['noisy_values']
    sigma = data['sigma']
    
    # Create a model
    model = FitzhughNagumoModel()
    
    # Simulate with true parameters for comparison plots
    values = model.simulate(parameters, times)
    
    print("\n--- Data Loaded ---")
    print(f"True parameters: a={parameters[0]}, b={parameters[1]}, c={parameters[2]}")
    print(f"Number of time points: {len(times)}")
    print(f"Noise sigma: {sigma}")
    
    # Plot the results
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(times, values)
    plt.legend(['Voltage', 'Recovery'])
    plt.title('FitzHugh-Nagumo Model Simulation')
    plt.savefig('fitzhugh_nagumo_simulation.png', dpi=150)
    plt.close()
    print("Saved: fitzhugh_nagumo_simulation.png")
    
    # Plot noisy data
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Noisy values')
    plt.plot(times, noisy)
    plt.title('Noisy Observations')
    plt.savefig('fitzhugh_nagumo_noisy.png', dpi=150)
    plt.close()
    print("Saved: fitzhugh_nagumo_noisy.png")
    
    # Create optimization problem
    problem = MultiOutputProblem(model, times, noisy)
    score = SumOfSquaresError(problem)
    
    # Select boundaries (from JSON config)
    opt_config = data['optimization']
    boundaries = RectangularBoundaries(
        opt_config['boundaries_lower'], 
        opt_config['boundaries_upper']
    )
    
    # Select a starting point (from JSON config)
    x0 = opt_config['x0']
    
    # Perform optimization
    print("\n--- Optimization ---")
    found_parameters, found_value = optimise(score, x0, boundaries=boundaries)
    
    print('\nScore at true solution:')
    print(score(parameters))
    
    print('\nFound solution:          True parameters:')
    for k, x in enumerate(found_parameters):
        print(strfloat(x) + '    ' + strfloat(parameters[k]))
    
    # Plot optimization results
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.plot(times, noisy, '-', alpha=0.25, label='noisy signal')
    plt.plot(times, values, alpha=0.4, lw=5, label='original signal')
    plt.plot(times, problem.evaluate(found_parameters), 'k--', label='recovered signal')
    plt.legend()
    plt.title('Optimization Result')
    plt.savefig('fitzhugh_nagumo_optimization.png', dpi=150)
    plt.close()
    print("Saved: fitzhugh_nagumo_optimization.png")
    
    # Create Bayesian inference problem
    problem = MultiOutputProblem(model, times, noisy)
    
    # Create a log-likelihood function (adds extra parameters for noise)
    log_likelihood = GaussianLogLikelihood(problem)
    
    # Create a uniform prior over both parameters and noise variables (from JSON config)
    mcmc_config = data['mcmc']
    log_prior = UniformLogPrior(
        mcmc_config['prior_lower'],
        mcmc_config['prior_upper']
    )
    
    # Create posterior
    log_posterior = LogPosterior(log_likelihood, log_prior)
    
    # Choose starting points for MCMC chains (from JSON config)
    real_parameters1 = np.array(parameters + [sigma, sigma])
    xs = [
        real_parameters1 * factor 
        for factor in mcmc_config['starting_point_factors']
    ]
    
    # Create MCMC routine
    print("\n--- MCMC Sampling ---")
    n_chains = mcmc_config['n_chains']
    mcmc = MCMCController(log_posterior, n_chains, xs, method=MonomialGammaHamiltonianMCMC)
    
    # Add stopping criterion (from JSON config)
    mcmc.set_max_iterations(mcmc_config['max_iterations'])
    mcmc.set_log_interval(1)
    
    # Configure parallel execution
    mcmc.set_parallel(True)
    
    # Configure samplers (from JSON config)
    for sampler in mcmc.samplers():
        sampler.set_leapfrog_step_size(mcmc_config['leapfrog_step_size'])
        sampler.set_leapfrog_steps(mcmc_config['leapfrog_steps'])
    
    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')
    
    # Plot trace
    plot_trace(chains, parameter_names=['a', 'b', 'c', 'sigma_V', 'sigma_R'], 
               filename='fitzhugh_nagumo_trace.png')
    
    # Print summary
    results = MCMCSummary(
        chains=chains,
        time=mcmc.time(),
        parameter_names=['a', 'b', 'c', 'sigma_V', 'sigma_R'],
    )
    print("\n--- MCMC Summary ---")
    print(results)
    
    # Plot series
    plot_series(np.vstack(chains), problem, filename='fitzhugh_nagumo_series.png')
    
    print("\n" + "=" * 60)
    print("All outputs saved successfully!")
    print("=" * 60)

