# ==============================================================================
# Standalone Repressilator Model with MCMC Inference
# ==============================================================================
# This is a self-contained script that performs parameter inference on the
# Repressilator model using MCMC sampling (Haario-Bardenet Adaptive Covariance).
#
# The Repressilator model describes oscillations in a network of proteins
# that suppress their own creation. It has three mRNA states (m_lacI, m_tetR, m_cl)
# and three protein states (p_lacI, p_tetR, p_cl).
#
# File Dependencies:
# ------------------
# Inputs:
#   - data/standalone_mcmc_repressilator.json: JSON file containing model parameters,
#     initial conditions, time points, noise level, and MCMC settings
#
# Outputs:
#   - repressilator_simulation.png: Original simulation plot
#   - repressilator_noisy.png: Noisy data plot  
#   - repressilator_trace.png: MCMC trace plot
#   - repressilator_series.png: MCMC series prediction plot
#
# Algorithm: HaarioBardenetACMC (Haario-Bardenet Adaptive Covariance MCMC)
# Model: Repressilator oscillatory gene regulatory network model
#
# Changes from Original:
# ----------------------
#   - Extracted all pints dependencies into this standalone file (no import pints)
#   - Replaced plt.show() with plt.savefig() to save plots as PNG files
#   - Reduced MCMC max_iterations from 6000 to 600 for ~10x faster execution
#   - Added data loading from JSON file (data/standalone_mcmc_repressilator.json)
# ==============================================================================

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import timeit
import warnings
from tabulate import tabulate

# ==============================================================================
# Data Loading Module
# ==============================================================================

def load_data(json_file):
    """
    Load model data and parameters from a JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing model data.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['model_initial_conditions'] = np.array(data['model_initial_conditions'])
    data['suggested_parameters'] = np.array(data['suggested_parameters'])
    data['suggested_times'] = np.array(data['suggested_times'])
    data['mcmc_initial_guesses'] = [list(x) for x in data['mcmc_initial_guesses']]
    
    return data

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
# Repressilator Model
# ==============================================================================

class RepressilatorModel:
    """
    The Repressilator model describes oscillations in a network of proteins
    that suppress their own creation.
    
    The formulation has three protein states (p_i) and three mRNA states (m_i):
    
    dm_0/dt = -m_0 + alpha/(1 + p_2^n) + alpha_0
    dm_1/dt = -m_1 + alpha/(1 + p_0^n) + alpha_0
    dm_2/dt = -m_2 + alpha/(1 + p_1^n) + alpha_0
    dp_0/dt = -beta * (p_0 - m_0)
    dp_1/dt = -beta * (p_1 - m_1)
    dp_2/dt = -beta * (p_2 - m_2)
    
    With parameters: alpha_0, alpha, beta, n
    Only the mRNA states are visible as output.
    """
    def __init__(self, y0=None):
        # Check initial values
        if y0 is None:
            # Toni et al. initial conditions
            self._y0 = np.array([0, 0, 0, 2, 1, 3])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 6:
                raise ValueError('Initial value must have size 6.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """Returns the number of outputs (3 mRNA states)."""
        return 3

    def n_parameters(self):
        """Returns the number of parameters (4: alpha_0, alpha, beta, n)."""
        return 4

    def _rhs(self, y, t, alpha_0, alpha, beta, n):
        """Calculates the model RHS."""
        dy = np.zeros(6)
        dy[0] = -y[0] + alpha / (1 + y[5]**n) + alpha_0
        dy[1] = -y[1] + alpha / (1 + y[3]**n) + alpha_0
        dy[2] = -y[2] + alpha / (1 + y[4]**n) + alpha_0
        dy[3] = -beta * (y[3] - y[0])
        dy[4] = -beta * (y[4] - y[1])
        dy[5] = -beta * (y[5] - y[2])
        return dy

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        alpha_0, alpha, beta, n = parameters
        y = odeint(self._rhs, self._y0, times, (alpha_0, alpha, beta, n))
        return y[:, :3]  # Return only mRNA states

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
# Log PDFs and Likelihoods
# ==============================================================================

class LogPDF:
    """Represents the natural logarithm of a probability density function."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

class GaussianKnownSigmaLogLikelihood(LogPDF):
    """
    Calculates a log-likelihood assuming independent Gaussian noise at each
    time point, using a known value for the standard deviation (sigma).
    """
    def __init__(self, problem, sigma):
        self._problem = problem
        self._values = problem.values()
        self._times = problem.times()
        
        # Store counts
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()

        # Check sigma
        if np.isscalar(sigma):
            sigma = np.ones(self._no) * float(sigma)
        else:
            sigma = vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)
        
        # Number of parameters is same as problem parameters
        self._n_parameters = self._np

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

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
# MCMC Sampler: Haario-Bardenet Adaptive Covariance MCMC
# ==============================================================================

class HaarioBardenetACMC:
    """
    Haario-Bardenet Adaptive Covariance MCMC sampler.
    
    This is an adaptive Metropolis algorithm that adapts the proposal
    covariance matrix based on the history of samples.
    """
    def __init__(self, x0, sigma0=None):
        # Check initial position
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.prod(self._sigma0.shape) == self._n_parameters:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._n_parameters, self._n_parameters))

        # Current running status
        self._running = False
        self._adaptive = False

        # Current point and its log PDF
        self._current = None
        self._current_log_pdf = None

        # Proposed point
        self._proposed = None

        # Acceptance rate monitoring
        self._iterations = 0
        self._adaptations = 1

        # Target acceptance rate
        self._target_acceptance = 0.234

        # Measured acceptance rate
        self._acceptance_count = 0
        self._acceptance_rate = 0

        # Parameters used in setting the proposal distributions
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # Determines decay rate in adaptation
        self._eta = 0.6

        # Initial decay rate in adaptation
        self._gamma = 1

        # Initial log lambda is zero (for Haario-Bardenet adaptation)
        self._log_lambda = 0

    def name(self):
        return 'Haario-Bardenet adaptive covariance MCMC'

    def needs_sensitivities(self):
        return False

    def needs_initial_phase(self):
        return True

    def in_initial_phase(self):
        return not self._adaptive

    def set_initial_phase(self, initial_phase):
        self._adaptive = not bool(initial_phase)

    def ask(self):
        """Returns a parameter vector to evaluate the LogPDF for."""
        # Initialise on first call
        if not self._running:
            self._running = True
            self._proposed = self._x0
            self._proposed.setflags(write=False)

        # Propose new point
        if self._proposed is None:
            # Generate proposal using multivariate normal
            self._proposed = np.random.multivariate_normal(
                self._current, self._sigma * np.exp(self._log_lambda))
            self._proposed.setflags(write=False)

        return self._proposed

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm.
        Returns (current position, current log pdf, accepted).
        """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # Increase iteration count
        self._iterations += 1

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return self._current, self._current_log_pdf, True

        # Calculate log of the ratio of proposed and current log pdf
        log_ratio = fx - self._current_log_pdf

        # Accept or reject the point
        accepted = False
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < log_ratio:
                accepted = True
                self._acceptance_count += 1

                # Update current point
                self._current = self._proposed
                self._current_log_pdf = fx

        # Calculate acceptance rate
        self._acceptance_rate = self._acceptance_count / self._iterations

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma = (self._adaptations + 1) ** -self._eta

            # Update the number of adaptations
            self._adaptations += 1

            # Update mu
            self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current

            # Update sigma
            dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
            self._sigma = ((1 - self._gamma) * self._sigma +
                           self._gamma * np.dot(dsigm, dsigm.T))

            # Update log_lambda (Haario-Bardenet adaptation)
            p = 1 if accepted else 0
            self._log_lambda += self._gamma * (p - self._target_acceptance)

        # Return current sample
        return self._current, self._current_log_pdf, accepted

    def _log_init(self, logger):
        logger.add_float('Accept.')

    def _log_write(self, logger):
        logger.log(self._acceptance_rate)

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
            method = HaarioBardenetACMC
        
        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        
        self._log_to_screen = True
        self._max_iterations = 10000
        self._parallel = False
        self._n_workers = 1
        self._message_interval = 20
        self._message_warm_up = 3
        self._initial_phase_iterations = 200
        self._has_run = False
        self._samples = None
        self._time = None

    def set_max_iterations(self, iterations):
        self._max_iterations = int(iterations)

    def set_log_interval(self, iters=20, warm_up=3):
        self._message_interval = int(iters)
        self._message_warm_up = max(0, int(warm_up))

    def set_log_to_screen(self, enabled):
        self._log_to_screen = True if enabled else False

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

    def set_initial_phase_iterations(self, iterations=200):
        self._initial_phase_iterations = int(iterations)

    def samplers(self):
        return self._samplers

    def time(self):
        return self._time

    def run(self):
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True
        
        f = self._log_pdf
        
        if self._parallel:
            n_workers = min(self._n_workers, self._n_chains)
            evaluator = ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = SequentialEvaluator(f)
        
        # Set initial phase
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)
        
        samples = np.zeros((self._n_chains, self._max_iterations, self._n_parameters))
        
        active = list(range(self._n_chains))
        n_samples = [0] * self._n_chains
        
        # Set up logging
        if self._log_to_screen:
            print('Using ' + str(self._samplers[0].name()))
            print('Generating ' + str(self._n_chains) + ' chains.')
            if self._parallel:
                print('Running in parallel.')
            else:
                print('Running in sequential mode.')
        
        logger = Logger()
        if not self._log_to_screen:
            logger.set_stream(None)
        
        logger.add_counter('Iter.', max_value=self._max_iterations)
        logger.add_counter('Eval.', max_value=self._max_iterations * self._n_chains)
        for sampler in self._samplers:
            sampler._log_init(logger)
        logger.add_time('Time')
        
        timer = Timer()
        iteration = 0
        n_evaluations = 0
        next_message = 0
        
        while active:
            # Check for initial phase end
            if self._needs_initial_phase and iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')
            
            xs = [self._samplers[i].ask() for i in active]
            fxs = evaluator.evaluate(xs)
            n_evaluations += len(fxs)
            
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
                logger.log(iteration, n_evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
                logger.log(timer.time())
                
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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'data', 'standalone_mcmc_repressilator.json')
    
    # Load data from JSON file
    print("=" * 60)
    print("Repressilator Model: MCMC Inference")
    print("=" * 60)
    print(f"\nLoading data from: {json_file}")
    data = load_data(json_file)
    
    # Set random seed for reproducibility
    np.random.seed(data['random_seed'])
    
    # Extract data from loaded JSON
    parameters = data['suggested_parameters']
    times = data['suggested_times']
    initial_conditions = data['model_initial_conditions']
    sigma = data['noise_sigma']
    x0 = data['mcmc_initial_guesses']
    max_iterations = data['mcmc_max_iterations']
    parameter_names = data['parameter_names']
    output_names = data['output_names']
    
    # Create a model with initial conditions from JSON
    model = RepressilatorModel(y0=initial_conditions)
    
    # Run a simulation with suggested parameters
    values = model.simulate(parameters, times)
    
    print("\n--- Simulation ---")
    print('Parameters:')
    for i, name in enumerate(parameter_names):
        print(f'  {name} = {parameters[i]}')
    
    # Plot the results
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.plot(times, values)
    plt.legend(output_names)
    plt.title('Repressilator Model Simulation')
    plt.savefig('repressilator_simulation.png', dpi=150)
    plt.close()
    print("Saved: repressilator_simulation.png")
    
    # Add noise
    noisy = values + np.random.normal(0, sigma, values.shape)
    
    # Plot noisy data
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.plot(times, noisy)
    plt.legend(output_names)
    plt.title('Noisy Observations')
    plt.savefig('repressilator_noisy.png', dpi=150)
    plt.close()
    print("Saved: repressilator_noisy.png")
    
    # Create inference problem
    problem = MultiOutputProblem(model, times, noisy)
    loglikelihood = GaussianKnownSigmaLogLikelihood(problem, sigma)
    
    # Create MCMC routine
    print("\n--- MCMC Sampling ---")
    mcmc = MCMCController(loglikelihood, 3, x0)
    mcmc.set_log_to_screen(False)
    mcmc.set_max_iterations(max_iterations)
    
    print(f'Running MCMC with {max_iterations} iterations (reduced from 6000 for ~10x speedup)...')
    chains = mcmc.run()
    print('Done!')
    
    # Check convergence and other properties of chains
    print("\n--- MCMC Summary ---")
    results = MCMCSummary(chains=chains, time=mcmc.time(), parameter_names=parameter_names)
    print(results)
    
    # Plot trace
    plt.figure()
    plot_trace(chains, ref_parameters=parameters, 
               parameter_names=parameter_names,
               filename='repressilator_trace.png')
    plt.close()
    
    # Get samples from one chain for series plot (last 100 samples)
    samples = chains[1][-100:]
    
    # Plot series
    plt.figure(figsize=(12, 6))
    plot_series(samples, problem, filename='repressilator_series.png')
    plt.close()
    
    print("\n" + "=" * 60)
    print("All outputs saved successfully!")
    print("=" * 60)
