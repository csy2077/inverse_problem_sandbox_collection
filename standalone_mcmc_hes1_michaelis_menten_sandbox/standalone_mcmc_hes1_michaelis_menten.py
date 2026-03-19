# ==============================================================================
# Standalone Hes1 Michaelis-Menten Model with MCMC Inference
# ==============================================================================
# This is a self-contained script that performs parameter inference on the
# Hes1 Michaelis-Menten model of regulatory dynamics using MCMC sampling
# (both Adaptive Covariance MCMC and Hamiltonian Monte Carlo).
#
# File Dependencies:
# ------------------
# Inputs:
#   - data/standalone_mcmc_hes1_michaelis_menten.json: JSON data file containing:
#       * model_config: m0, fixed_parameters, initial_conditions
#       * suggested_parameters: [P0, v, k1, h]
#       * suggested_times: observation time points
#       * suggested_values: observed Hes1 concentrations
#       * smooth_times: parameters for smooth plotting
#       * exploration: initial condition multipliers and trial parameters
#       * inference: MCMC settings (priors, chains, iterations)
#
# Outputs:
#   - hes1_simulation.png: Initial simulation plot
#   - hes1_3d_phase.png: 3D phase portrait of the system
#   - hes1_3d_initial_conditions.png: 3D phase portrait with varying initial conditions
#   - hes1_3d_parameters.png: 3D phase portrait with varying parameters
#   - hes1_suggested_values.png: Model fit to suggested values
#   - hes1_trace_adaptive.png: MCMC trace plot (Adaptive Covariance)
#   - hes1_posterior_predictive.png: Posterior predictive plot
#   - hes1_trace_hmc.png: MCMC trace plot (Hamiltonian Monte Carlo)
#
# Algorithm: AdaptiveCovarianceMCMC, HamiltonianMCMC
# Model: Hes1 Michaelis-Menten model of transcription factor dynamics
#
# Changes from Original:
# ----------------------
#   - Extracted all pints dependencies into this standalone file (no import pints)
#   - Replaced plt.show() with plt.savefig() to save plots as PNG files
#   - Reduced MCMC max_iterations from 10000 to 1000 for Adaptive MCMC (~10x faster)
#   - Reduced MCMC max_iterations from 100 to 10 for HamiltonianMCMC (~10x faster)
#   - Reduced burn-in from 5000 to 500 accordingly
#   - Added JSON data file for external data loading (data/standalone_mcmc_hes1_michaelis_menten.json)
#   - Model data (suggested_parameters, suggested_times, suggested_values) now loaded from JSON
# ==============================================================================

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import numpy as np
import scipy.integrate
import scipy.special
import scipy.stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import warnings
from tabulate import tabulate
from scipy.special import logit, expit

# ==============================================================================
# Data Loading Module
# ==============================================================================

def load_data(json_path):
    """
    Load model data from JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON data file.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['suggested_parameters'] = np.array(data['suggested_parameters'])
    data['suggested_times'] = np.array(data['suggested_times'])
    data['suggested_values'] = np.array(data['suggested_values'])
    data['model_config']['fixed_parameters'] = np.array(data['model_config']['fixed_parameters'])
    data['model_config']['initial_conditions'] = np.array(data['model_config']['initial_conditions'])
    
    # Generate smooth times from config
    st_config = data['smooth_times']
    data['smooth_times_array'] = np.linspace(
        st_config['start'], 
        st_config['end'], 
        st_config['num_points']
    )
    
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
# Hes1 Michaelis-Menten Model
# ==============================================================================

class Hes1Model:
    """
    Hes1 Michaelis-Menten model of regulatory dynamics.
    
    This model describes the expression level of the transcription factor Hes1.
    
    dm/dt = -k_deg * m + 1/(1 + (p2/P0)^h)
    dp1/dt = -k_deg * p1 + v * m - k1 * p1
    dp2/dt = -k_deg * p2 + k1 * p1
    
    The system has 3 state variables m, p1, p2. Only m is observable.
    Input order of parameters: {P0, v, k1, h}
    """
    def __init__(self, m0=None, fixed_parameters=None, data_config=None):
        # Store data config for suggested_* methods
        self._data_config = data_config
        
        if fixed_parameters is None:
            if data_config is not None and 'model_config' in data_config:
                self.set_fixed_parameters(list(data_config['model_config']['fixed_parameters']))
            else:
                self.set_fixed_parameters([5., 3., 0.03])
        else:
            self.set_fixed_parameters(fixed_parameters)
        if m0 is None:
            if data_config is not None and 'model_config' in data_config:
                self.set_m0(data_config['model_config']['m0'])
            else:
                self.set_m0(2)
        else:
            self.set_m0(m0)

    def _dfdp(self, state, time, parameters):
        """Derivative with respect to parameters."""
        m, p1, p2 = state
        P0, v, k1, h = parameters
        p2_over_p0 = p2 / P0
        p2_over_p0_h = p2_over_p0**h
        one_plus_p2_expression_sq = (1 + p2_over_p0_h)**2
        ret = np.empty((self.n_states(), self.n_parameters()))
        ret[0, 0] = h * p2 * p2_over_p0**(h - 1) / (P0**2 * one_plus_p2_expression_sq)
        ret[0, 1] = 0
        ret[0, 2] = 0
        ret[0, 3] = - (p2_over_p0_h * np.log(p2_over_p0)) / one_plus_p2_expression_sq
        ret[1, 0] = 0
        ret[1, 1] = m
        ret[1, 2] = -p1
        ret[1, 3] = 0
        ret[2, 0] = 0
        ret[2, 1] = 0
        ret[2, 2] = p1
        ret[2, 3] = 0
        return ret

    def m0(self):
        """Returns the initial conditions of the m variable."""
        return self._y0[0]

    def fixed_parameters(self):
        """Returns the fixed parameters [p1_0, p2_0, k_deg]."""
        return [self._p0[0], self._p0[1], self._kdeg]

    def jacobian(self, state, time, parameters):
        """Returns the Jacobian of the ODE."""
        m, p1, p2 = state
        P0, v, k1, h = parameters
        k_deg = self._kdeg
        p2_over_p0 = p2 / P0
        p2_over_p0_h = p2_over_p0**h
        one_plus_p2_expression_sq = (1 + p2_over_p0_h)**2
        ret = np.zeros((self.n_states(), self.n_states()))
        ret[0, 0] = -k_deg
        ret[0, 1] = 0
        ret[0, 2] = -h * p2_over_p0**(h - 1) / (P0 * one_plus_p2_expression_sq)
        ret[1, 0] = v
        ret[1, 1] = -k1 - k_deg
        ret[1, 2] = 0
        ret[2, 0] = 0
        ret[2, 1] = k1
        ret[2, 2] = -k_deg
        return ret

    def n_states(self):
        """Returns number of states (3)."""
        return 3

    def n_outputs(self):
        """Returns number of outputs (1 - only m is observable)."""
        return 1

    def n_parameters(self):
        """Returns number of parameters (4)."""
        return 4

    def _rhs(self, state, time, parameters):
        """Right-hand side of the ODE."""
        m, p1, p2 = state
        P0, v, k1, h = parameters
        output = np.array([
            - self._kdeg * m + 1. / (1. + (p2 / P0)**h),
            - self._kdeg * p1 + v * m - k1 * p1,
            - self._kdeg * p2 + k1 * p1
        ])
        return output

    def _rhs_S1(self, y_and_dydp, t, p):
        """RHS for sensitivity calculation."""
        n_outputs = self.n_states()
        n_params = self.n_parameters()
        y = y_and_dydp[0:n_outputs]
        dydp = y_and_dydp[n_outputs:].reshape((n_params, n_outputs))
        dydt = self._rhs(y, t, p)
        d_dydp_dt = (
            np.matmul(dydp, np.transpose(self.jacobian(y, t, p))) +
            np.transpose(self._dfdp(y, t, p)))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def set_m0(self, m0):
        """Sets the initial conditions of the m variable."""
        if m0 < 0:
            raise ValueError('Initial condition cannot be negative.')
        self._y0 = [m0, self._p0[0], self._p0[1]]

    def set_initial_conditions(self, y0):
        """Sets the initial conditions of the model."""
        self._y0 = y0

    def initial_conditions(self):
        """Returns the initial conditions of the model."""
        return self._y0

    def set_fixed_parameters(self, k):
        """Sets the implicit parameters [p1_0, p2_0, k_deg]."""
        a, b, c = k
        if a < 0 or b < 0 or c < 0:
            raise ValueError('Implicit parameters cannot be negative.')
        self._p0 = [a, b]
        self._kdeg = c

    def simulate_all_states(self, parameters, times):
        """Returns all state variables."""
        solved_states = scipy.integrate.odeint(
            self._rhs, self._y0, times, args=(parameters,))
        return solved_states

    def simulate(self, parameters, times):
        """Simulates the model and returns observable (m only)."""
        return self._simulate(parameters, times, False)

    def _simulate(self, parameters, times, sensitivities):
        """Internal simulation method."""
        times = vector(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        
        offset = 0
        if len(times) < 1 or times[0] != 0:
            times = np.concatenate(([0], times))
            offset = 1

        if sensitivities:
            n_params = self.n_parameters()
            n_outputs = self.n_states()
            y0 = np.zeros(n_params * n_outputs + n_outputs)
            y0[0:n_outputs] = self._y0
            result = scipy.integrate.odeint(
                self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params), order="F")
            return values[offset:], dvalues_dp[offset:]
        else:
            values = scipy.integrate.odeint(
                self._rhs, self._y0, times, (parameters,))
            return values[offset:, :self.n_outputs()].squeeze()

    def simulateS1(self, parameters, times):
        """Simulates with sensitivities."""
        values, dvalues_dp = self._simulate(parameters, times, True)
        n_outputs = self.n_outputs()
        return values[:, :n_outputs].squeeze(), dvalues_dp[:, :n_outputs, :]

    def suggested_parameters(self):
        """Returns suggested parameter values."""
        if self._data_config is not None and 'suggested_parameters' in self._data_config:
            return np.array(self._data_config['suggested_parameters'])
        return np.array([2.4, 0.025, 0.11, 6.9])

    def suggested_times(self):
        """Returns suggested time points."""
        if self._data_config is not None and 'suggested_times' in self._data_config:
            return np.array(self._data_config['suggested_times'])
        return np.arange(0, 270, 30)

    def suggested_values(self):
        """Returns suggested values matching suggested_times()."""
        if self._data_config is not None and 'suggested_values' in self._data_config:
            return np.array(self._data_config['suggested_values'])
        return np.array([2, 1.20, 5.90, 4.58, 2.64, 5.38, 6.42, 5.60, 4.48])

# ==============================================================================
# Problem Classes
# ==============================================================================

class SingleOutputProblem:
    """Represents an inference problem with a single output."""
    def __init__(self, model, times, values):
        self._model = model
        if model.n_outputs() != 1:
            raise ValueError('Only single-output models can be used.')
        
        self._times = vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] >= self._times[1:]):
            raise ValueError('Times must be increasing.')
        
        self._values = vector(values)
        self._n_parameters = int(model.n_parameters())
        self._n_times = len(self._times)
        
        if len(self._values) != self._n_times:
            raise ValueError('Times and values arrays must have same length.')

    def evaluate(self, parameters):
        """Runs a simulation using the given parameters."""
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape((self._n_times,))

    def evaluateS1(self, parameters):
        """Runs a simulation with first-order sensitivity calculation."""
        y, dy = self._model.simulateS1(parameters, self._times)
        return (
            np.asarray(y).reshape((self._n_times,)),
            np.asarray(dy).reshape((self._n_times, self._n_parameters))
        )

    def n_outputs(self):
        return 1

    def n_parameters(self):
        return self._n_parameters

    def n_times(self):
        return self._n_times

    def times(self):
        return self._times

    def values(self):
        return self._values

# ==============================================================================
# Boundaries
# ==============================================================================

class RectangularBoundaries:
    """Represents a set of lower and upper boundaries."""
    def __init__(self, lower, upper):
        self._lower = vector(lower)
        self._upper = vector(upper)
        self._n_parameters = len(self._lower)
        if len(self._upper) != self._n_parameters:
            raise ValueError('Lower and upper bounds must have same length.')
        if self._n_parameters < 1:
            raise ValueError('The parameter space must have dimension > 0.')
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
# Transformation
# ==============================================================================

class RectangularBoundariesTransformation:
    """Transforms parameters from bounded to unbounded space."""
    def __init__(self, lower_or_boundaries, upper=None):
        if upper is None:
            if not isinstance(lower_or_boundaries, RectangularBoundaries):
                raise ValueError('Need lower and upper bounds.')
            boundaries = lower_or_boundaries
        else:
            boundaries = RectangularBoundaries(lower_or_boundaries, upper)
        
        self._a = boundaries.lower()
        self._b = boundaries.upper()
        self._n_parameters = boundaries.n_parameters()

    def elementwise(self):
        return True

    def n_parameters(self):
        return self._n_parameters

    def to_model(self, q):
        """Transform from search space to model space."""
        q = vector(q)
        return (self._b - self._a) * expit(q) + self._a

    def to_search(self, p):
        """Transform from model space to search space."""
        p = vector(p)
        return np.log(p - self._a) - np.log(self._b - p)

    def jacobian(self, q):
        """Returns the Jacobian matrix."""
        q = vector(q)
        diag = (self._b - self._a) / (np.exp(q) * (1. + np.exp(-q)) ** 2)
        return np.diag(diag)

    def log_jacobian_det(self, q):
        """Returns log|det(J)|."""
        q = vector(q)
        s = np.log(1. + np.exp(-q))
        return np.sum(np.log(self._b - self._a) - 2. * s - q)

    def log_jacobian_det_S1(self, q):
        """Returns log|det(J)| and its gradient."""
        q = vector(q)
        logjacdet = self.log_jacobian_det(q)
        dlogjacdet = 2. * np.exp(-q) * expit(q) - 1.
        return logjacdet, dlogjacdet

# ==============================================================================
# Log PDFs and Priors
# ==============================================================================

class LogPDF:
    """Represents a log probability density function."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

class LogPrior(LogPDF):
    """Represents a log prior distribution."""
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
            raise ValueError('log_prior and log_likelihood must have same dimension.')
        
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

class TransformedLogPDF(LogPDF):
    """A LogPDF that accepts parameters in a transformed search space."""
    def __init__(self, log_pdf, transformation):
        self._log_pdf = log_pdf
        self._transform = transformation
        self._n_parameters = self._log_pdf.n_parameters()

    def __call__(self, q):
        p = self._transform.to_model(q)
        logpdf_nojac = self._log_pdf(p)
        log_jacobian_det = self._transform.log_jacobian_det(q)
        return logpdf_nojac + log_jacobian_det

    def evaluateS1(self, q):
        p = self._transform.to_model(q)
        logpdf_nojac, dlogpdf_nojac = self._log_pdf.evaluateS1(p)
        logjacdet, dlogjacdet = self._transform.log_jacobian_det_S1(q)
        logpdf = logpdf_nojac + logjacdet
        jacobian = self._transform.jacobian(q)
        dlogpdf = np.matmul(dlogpdf_nojac, jacobian)
        dlogpdf += vector(dlogjacdet)
        return logpdf, dlogpdf

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
# MCMC Samplers
# ==============================================================================

class AdaptiveCovarianceMCMC:
    """Adaptive Covariance MCMC sampler (Haario-Bardenet)."""
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
        self._current_log_pdf = None
        self._proposed = None
        self._acceptance = 0
        self._mu = np.zeros(self._n_parameters)
        self._sigma = np.array(self._sigma0)
        self._iteration = 0
        self._initial_phase = True
        self._eta = 0.6
        self._gamma = 1.01

    def name(self):
        return 'Haario-Bardenet adaptive covariance MCMC'

    def needs_sensitivities(self):
        return False

    def needs_initial_phase(self):
        return True

    def set_initial_phase(self, in_initial_phase):
        self._initial_phase = in_initial_phase

    def in_initial_phase(self):
        return self._initial_phase

    def ask(self):
        if self._ready_for_tell:
            raise RuntimeError('Ask called when expecting tell.')
        
        if not self._running:
            self._running = True
        
        if self._current is None:
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)
        
        self._proposed = np.random.multivariate_normal(self._current, self._sigma)
        self._ready_for_tell = True
        return np.array(self._proposed, copy=True)

    def tell(self, fx):
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before ask.')
        self._ready_for_tell = False
        
        fx = float(fx)
        
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError('Initial point must have finite logpdf.')
            self._current = self._x0
            self._current_log_pdf = fx
            self._current.setflags(write=False)
            return (self._current, self._current_log_pdf, True)
        
        accepted = False
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < fx - self._current_log_pdf:
                accepted = True
                self._current = self._proposed
                self._current_log_pdf = fx
                self._current.setflags(write=False)
        
        # Update acceptance rate
        self._acceptance = (self._iteration * self._acceptance + accepted) / (self._iteration + 1)
        
        # Adapt covariance (unless in initial phase)
        if not self._initial_phase:
            gamma = (self._iteration + 1) ** (-self._eta)
            self._mu = (1 - gamma) * self._mu + gamma * self._current
            delta = self._current - self._mu
            self._sigma = (1 - gamma) * self._sigma + gamma * np.outer(delta, delta)
            
            # Scale to maintain acceptance rate
            self._sigma *= self._gamma if self._acceptance < 0.234 else (1 / self._gamma)
        
        self._iteration += 1
        return (self._current, self._current_log_pdf, accepted)

    def _log_init(self, logger):
        logger.add_float('Accept.')

    def _log_write(self, logger):
        logger.log(self._acceptance)

class HamiltonianMCMC:
    """Hamiltonian MCMC sampler."""
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

    def name(self):
        return 'Hamiltonian Monte Carlo'

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
        elif len(step_size) != self._n_parameters:
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

    def ask(self):
        if self._ready_for_tell:
            raise RuntimeError('Ask called when expecting tell.')
        
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
            raise RuntimeError('Tell called before ask.')
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
            current_K = np.sum(self._current_momentum**2 / 2)
            proposed_U = energy
            proposed_K = np.sum(self._momentum**2 / 2)
            
            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:
                self._divergent = np.append(self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None
                self._frog_iteration = 0
                self._mcmc_iteration += 1
                self._mcmc_acceptance = ((self._mcmc_iteration * self._mcmc_acceptance + accept) /
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
        self._mcmc_acceptance = ((self._mcmc_iteration * self._mcmc_acceptance + accept) /
                                 (self._mcmc_iteration + 1))
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
    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None, transformation=None):
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()
        self._n_chains = int(chains)
        self._transformation = transformation
        
        if transformation is not None:
            self._log_pdf = TransformedLogPDF(log_pdf, transformation)
            x0 = [transformation.to_search(x) for x in x0]
        
        if len(x0) != chains:
            raise ValueError('Number of initial positions must equal number of chains.')
        
        if method is None:
            method = AdaptiveCovarianceMCMC
        
        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        self._log_to_screen = True
        self._max_iterations = 10000
        self._initial_phase_iterations = 200 if self._needs_initial_phase else None
        self._parallel = False
        self._n_workers = 1
        self._message_interval = 20
        self._message_warm_up = 3
        self._has_run = False
        self._samples = None
        self._time = None

    def set_max_iterations(self, iterations):
        self._max_iterations = int(iterations)

    def set_initial_phase_iterations(self, iterations=200):
        if not self._needs_initial_phase:
            raise NotImplementedError
        self._initial_phase_iterations = int(iterations)

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
        
        # Initial phase setup
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)
        
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
            # Initial phase toggle
            if self._needs_initial_phase and iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                print('Initial phase completed.')
            
            xs = [self._samplers[i].ask() for i in active]
            fxs = evaluator.evaluate(xs)
            
            fxs_iterator = iter(fxs)
            for i in list(active):
                reply = self._samplers[i].tell(next(fxs_iterator))
                
                if reply is not None:
                    y, fy, accepted = reply
                    
                    # Transform back to model space if needed
                    if self._transformation:
                        y_store = self._transformation.to_model(y)
                    else:
                        y_store = y
                    
                    samples[i][n_samples[i]] = y_store
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

# ==============================================================================
# Main Script
# ==============================================================================

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 70)
    print("Hes1 Michaelis-Menten Model: Exploration and MCMC Inference")
    print("=" * 70)
    
    # Load data from JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'data', 'standalone_mcmc_hes1_michaelis_menten.json')
    print(f'\nLoading data from: {json_path}')
    data = load_data(json_path)
    
    # Create model with loaded data
    model = Hes1Model(data_config=data)
    
    print('\nOutputs: ' + str(model.n_outputs()))
    print('Parameters: ' + str(model.n_parameters()))
    
    times = model.suggested_times()
    smooth_times = data['smooth_times_array']
    parameters = model.suggested_parameters()
    print('Suggested parameters:', parameters)
    
    # Plot 1: Initial simulation
    plt.figure()
    plt.xlabel('Time [minute]')
    plt.ylabel('Hes1 concentration')
    plt.plot(times, model.simulate(parameters, times), 'o', label='Sparse sampling')
    plt.plot(smooth_times, model.simulate(parameters, smooth_times), '--', label='Underlying model')
    plt.legend()
    plt.savefig('hes1_simulation.png', dpi=150)
    plt.close()
    print('Saved: hes1_simulation.png')
    
    # Plot 2: 3D phase portrait
    all_states = model.simulate_all_states(parameters, smooth_times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('m')
    ax.set_ylabel('p1')
    ax.set_zlabel('p2')
    ax.plot(all_states[:, 0], all_states[:, 1], all_states[:, 2])
    plt.savefig('hes1_3d_phase.png', dpi=150)
    plt.close()
    print('Saved: hes1_3d_phase.png')
    
    # Plot 3: Varying initial conditions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('m')
    ax.set_ylabel('p1')
    ax.set_zlabel('p2')
    
    suggested_init = np.array(model.initial_conditions())
    suggested_hidden_param = np.array(model.fixed_parameters())
    init_multipliers = data['exploration']['initial_condition_multipliers']
    for x in init_multipliers:
        model.set_initial_conditions(list(suggested_init * x))
        model.set_fixed_parameters(list(suggested_hidden_param[0:-1] / x) + [suggested_hidden_param[-1]])
        all_states = model.simulate_all_states(parameters, smooth_times)
        ax.plot(all_states[:, 0], all_states[:, 1], all_states[:, 2])
    plt.savefig('hes1_3d_initial_conditions.png', dpi=150)
    plt.close()
    print('Saved: hes1_3d_initial_conditions.png')
    
    # Reset initial conditions
    model.set_initial_conditions(list(suggested_init))
    model.set_fixed_parameters(list(suggested_hidden_param))
    
    # Plot 4: Varying parameters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('m')
    ax.set_ylabel('p1')
    ax.set_zlabel('p2')
    
    # Try some different parameter sets {P0, v, k1, h} from JSON data
    Try_params = data['exploration']['try_params']
    
    # Compare with suggested parameters
    all_states = model.simulate_all_states(parameters, smooth_times)
    ax.plot(all_states[:, 0], all_states[:, 1], all_states[:, 2], label='Suggested parameters')
    for i, try_param in enumerate(Try_params):
        all_states = model.simulate_all_states(try_param, smooth_times)
        ax.plot(all_states[:, 0], all_states[:, 1], all_states[:, 2], label='Trial parameters %d' % (i+1))
    ax.legend()
    plt.savefig('hes1_3d_parameters.png', dpi=150)
    plt.close()
    print('Saved: hes1_3d_parameters.png')
    
    # Plot 5: Model fit to suggested values
    values = model.suggested_values()
    
    plt.figure()
    plt.xlabel('Time [minute]')
    plt.ylabel('Hes1 concentration')
    plt.plot(smooth_times, model.simulate(parameters, smooth_times), label='Literature model')
    plt.plot(times, values, '*', label='Suggested values')
    plt.legend()
    plt.savefig('hes1_suggested_values.png', dpi=150)
    plt.close()
    print('Saved: hes1_suggested_values.png')
    
    # Create inference problem
    problem = SingleOutputProblem(model, times, values)
    
    # Create a log posterior using bounds from JSON config
    inference_config = data['inference']
    prior_bounds = inference_config['prior_bounds']
    lower = list(parameters * prior_bounds['lower_multiplier']) + [prior_bounds['sigma_lower']]
    upper = list(parameters * prior_bounds['upper_multiplier']) + [prior_bounds['sigma_upper']]
    log_prior = UniformLogPrior(lower, upper)
    log_likelihood = GaussianLogLikelihood(problem)
    log_posterior = LogPosterior(log_likelihood, log_prior)
    
    # Run MCMC (Adaptive Covariance) on the noisy data
    # Reduced from 10000 to 1000 iterations for ~10x speedup
    print("\n" + "=" * 70)
    print("MCMC Sampling (Adaptive Covariance)")
    print("=" * 70)
    
    mcmc_adaptive_config = inference_config['mcmc_adaptive']
    n_chains = mcmc_adaptive_config['n_chains']
    x0 = [list(parameters * mcmc_adaptive_config['x0_param_multiplier']) + 
          [mcmc_adaptive_config['x0_sigma']]] * n_chains
    mcmc = MCMCController(log_posterior, n_chains, x0)
    mcmc.set_max_iterations(mcmc_adaptive_config['max_iterations'])  # Reduced from 10000 for ~10x speedup
    
    chains = mcmc.run()
    
    results = MCMCSummary(
        chains=chains,
        time=mcmc.time(),
        parameter_names=["P0", "nu", "k_1", "h", "sigma"])
    print(results)
    
    # Plot trace
    ref_sigma = inference_config['reference_sigma']
    plot_trace(chains, ref_parameters=list(parameters)+[ref_sigma], filename='hes1_trace_adaptive.png')
    
    # Select first chain
    chain1 = chains[0]
    
    # Remove burn-in (reduced from 5000 to 500)
    burn_in = mcmc_adaptive_config['burn_in']
    chain1 = chain1[burn_in:]
    
    # Plot posterior predictions
    plt.figure()
    plt.xlabel('Time [minute]')
    plt.ylabel('Hes1 concentration')
    thinning = mcmc_adaptive_config['thinning']
    for posterior_param in chain1[::thinning]:  # Thinning from config
        model_prediction = model.simulate(posterior_param[:-1], smooth_times)
        plt.plot(smooth_times, model_prediction, c='Gray', alpha=0.3)
    plt.plot(smooth_times, model_prediction, c='Gray', alpha=0.3, label='Model prediction')
    plt.plot(times, values, 'kx', label='Original data')
    plt.plot(smooth_times, model.simulate(parameters, smooth_times), label='Literature model')
    plt.legend()
    plt.savefig('hes1_posterior_predictive.png', dpi=150)
    plt.close()
    print('Saved: hes1_posterior_predictive.png')
    
    # Run MCMC with Hamiltonian Monte Carlo
    print("\n" + "=" * 70)
    print("MCMC Sampling (Hamiltonian Monte Carlo)")
    print("=" * 70)
    
    transformation = RectangularBoundariesTransformation(lower, upper)
    
    mcmc_hmc_config = inference_config['mcmc_hmc']
    n_chains_hmc = mcmc_hmc_config['n_chains']
    x0 = [list(parameters) + [mcmc_hmc_config['x0_sigma']]] * n_chains_hmc
    mcmc = MCMCController(log_posterior, n_chains_hmc, x0, method=HamiltonianMCMC,
                          transformation=transformation)
    mcmc.set_max_iterations(mcmc_hmc_config['max_iterations'])  # Reduced from 100 for ~10x speedup
    
    chains = mcmc.run()
    
    results = MCMCSummary(
        chains=chains, 
        time=mcmc.time(), 
        parameter_names=["P0", "nu", "k_1", "h", "sigma"])
    print(results)
    
    plot_trace(chains, ref_parameters=list(parameters)+[ref_sigma], filename='hes1_trace_hmc.png')
    
    print("\n" + "=" * 70)
    print("All outputs saved successfully!")
    print("=" * 70)

