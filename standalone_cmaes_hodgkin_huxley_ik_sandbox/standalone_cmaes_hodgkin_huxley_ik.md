"""python
# =====================================================================================
# Standalone Hodgkin-Huxley IK Model with CMA-ES Optimization
# =====================================================================================
# This is a self-contained Python script for parameter optimization of the 
# Hodgkin-Huxley potassium current (IK) model using the CMA-ES algorithm.
#
# Problem: Hodgkin-Huxley potassium current (IK) model parameter estimation
# Algorithm: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
#
# File Dependencies:
# ------------------
# Input:
#   - data/standalone_cmaes_hodgkin_huxley_ik.json: JSON file containing model configuration,
#     protocol parameters, suggested parameters, times, and pre-generated noisy data
#   - No model weights required
#
# Output:
#   - hodgkin_huxley_ik_simulation.png: Plot of true model simulation (potassium current vs time)
#   - hodgkin_huxley_ik_folded.png: Folded voltage-step current traces
#   - hodgkin_huxley_ik_fit.png: Comparison of noisy data with optimized fit
#   - Console output: Optimization progress and final parameter comparison
#
# Changes from Original (model-hodgkin-huxley-ik.py):
# ------------------------------------------------
#   1. Self-contained: All pints classes (HodgkinHuxleyIKModel, SingleOutputProblem,
#      SumOfSquaresError, RectangularBoundaries, OptimisationController, CMAES)
#      are re-implemented directly - no "import pints" dependency required.
#   2. Plots saved to PNG: All plt.show() calls replaced with plt.savefig() to save
#      plots as PNG files instead of displaying interactively.
#   3. CUDA device set: os.environ['CUDA_VISIBLE_DEVICES'] = '0' added at startup.
#   4. Reduced iterations for faster execution (~10x speedup):
#      - max_unchanged_iterations: 10 (originally 100)
#   5. Non-interactive matplotlib backend: matplotlib.use('Agg') for headless execution.
#   6. Data loaded from JSON: All model data (times, noisy_values, parameters, protocol)
#      are loaded from data/standalone_cmaes_hodgkin_huxley_ik.json instead of being generated.
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
import timeit
import warnings
import sys


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
        - model_parameters: dict with initial_condition, E_k, g_max
        - protocol: dict with t_hold, t_step, v_hold, v_step
        - suggested_parameters: list of true model parameters
        - suggested_duration: duration of protocol in ms
        - noise_std: standard deviation of noise
        - random_seed: seed used for noise generation
        - times: list of time points
        - noisy_values: list of noisy measurement values
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# =====================================================================================
# Constants and Utility Functions
# =====================================================================================

# Float format for maximum precision output
FLOAT_FORMAT = '{: .17e}'


def strfloat(x):
    """Converts a float to a string, with maximum precision."""
    return FLOAT_FORMAT.format(float(x))


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
# Logger Classes
# =====================================================================================

# Value types for Logger
_COUNTER = 0
_FLOAT = 1
_INT = 2
_TIME = 3
_TEXT = 4


class Loggable:
    """Interface for classes that can log to a Logger."""
    def _log_init(self, logger):
        """Adds this Loggable's fields to a Logger."""
        pass

    def _log_write(self, logger):
        """Logs data for each of the fields specified in _log_init()."""
        pass


class Logger:
    """Logs numbers to screen and/or a file."""
    
    def __init__(self):
        super(Logger, self).__init__()
        self._stream = sys.stdout
        self._filename = None
        self._csv_mode = False
        self._have_logged = False
        self._field_names = []
        self._field_formats = []
        self._stream_fields = []
        import collections
        self._buffer = collections.deque()

    def add_counter(self, name, width=5, max_value=None, file_only=False):
        """Adds a field for positive integers."""
        if self._have_logged:
            raise RuntimeError('Cannot add fields after logging has started.')
        name = str(name)
        width = int(width)
        width = max(width, len(name), 1)
        if max_value is not None:
            max_value = float(max_value)
            width = max(width, int(np.ceil(np.log10(max_value))))
        f1 = f2 = '{:<' + str(width) + 'd}'
        self._field_names.append(name)
        self._field_formats.append((width, _COUNTER, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)
        return self

    def add_float(self, name, width=9, file_only=False):
        """Adds a field for floating point number."""
        if self._have_logged:
            raise RuntimeError('Cannot add fields after logging has started.')
        name = str(name)
        width = int(width)
        width = max(width, len(name), 7)
        f1 = '{: .' + str(width - 2) + 'g}'
        f2 = '{: .' + str(width - 6) + 'g}'
        self._field_names.append(name)
        self._field_formats.append((width, _FLOAT, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)
        return self

    def add_time(self, name, file_only=False):
        """Adds a field showing a formatted time (given in seconds)."""
        if self._have_logged:
            raise RuntimeError('Cannot add fields after logging has started.')
        name = str(name)
        width = max(len(name), 8)
        f1 = f2 = None
        self._field_names.append(name)
        self._field_formats.append((width, _TIME, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)
        return self

    def log(self, *data):
        """Logs a new row of data."""
        if self._stream is None and self._filename is None:
            return
        nfields = len(self._field_names)
        if nfields < 1:
            raise ValueError('Unable to log: No fields specified.')
        rows = []
        if len(self._buffer) == 0 and len(data) == nfields:
            rows.append(data)
        else:
            self._buffer.extend(data)
            while len(self._buffer) >= nfields:
                rows.append([self._buffer.popleft() for i in range(nfields)])
            if not rows:
                return

        # CSV format
        if self._csv_mode and self._filename is not None:
            mode = 'a' if self._have_logged else 'w'
            with open(self._filename, mode) as f:
                if not self._have_logged:
                    f.write(','.join(['"' + x + '"' for x in self._field_names]) + '\n')
                for row in rows:
                    line = []
                    column = iter(row)
                    for width, dtype, f1, f2 in self._field_formats:
                        v = next(column)
                        if v is None:
                            x = ''
                        elif dtype == _FLOAT:
                            x = '{:.17e}'.format(v)
                        elif dtype == _TIME:
                            x = str(v)
                        elif dtype == _TEXT:
                            x = '"' + str(v) + '"'
                        else:
                            x = str(int(v))
                        line.append(x)
                    f.write(','.join(line) + '\n')
            if not self._stream:
                self._have_logged = True
                return

        # Format fields
        formatted_rows = []
        if not self._have_logged:
            headers = []
            for i, name in enumerate(self._field_names):
                width = self._field_formats[i][0]
                headers.append(name + ' ' * (width - len(name)))
            formatted_rows.append(headers)

        for row in rows:
            column = iter(row)
            formatted_row = []
            for width, dtype, f1, f2 in self._field_formats:
                v = next(column)
                if v is None:
                    x = ' ' * width
                elif dtype == _FLOAT:
                    x = f1.format(v)
                    if len(x) > width:
                        x = f2.format(v)
                    x += ' ' * (width - len(x))
                elif dtype == _TIME:
                    x = self._format_time(v)
                elif dtype == _TEXT:
                    x = str(v)[:width]
                    x += ' ' * (width - len(x))
                else:
                    x = f1.format(int(v))
                formatted_row.append(x)
            formatted_rows.append(formatted_row)

        if self._stream is not None:
            lines = []
            for row in formatted_rows:
                lines.append(' '.join([row[i] for i in self._stream_fields]))
            self._stream.write('\n'.join(lines) + '\n')

        if self._filename is not None and not self._csv_mode:
            lines = []
            for row in formatted_rows:
                lines.append(' '.join([x for x in row]))
            with open(self._filename, 'a' if self._have_logged else 'w') as f:
                f.write('\n'.join(lines) + '\n')

        self._have_logged = True

    def set_filename(self, filename=None, csv=False):
        """Enables logging to a file if a filename is passed in."""
        if self._have_logged:
            raise RuntimeError('Cannot configure after logging has started.')
        if filename is None:
            self._filename = None
        else:
            self._filename = str(filename)
        self._csv_mode = True if csv else False

    def set_stream(self, stream=sys.stdout):
        """Enables logging to screen if an output stream is passed in."""
        if self._have_logged:
            raise RuntimeError('Cannot configure after logging has started.')
        self._stream = stream

    def _format_time(self, seconds):
        """Formats a time in seconds to the format 'mmm:ss.s'."""
        minutes = int(seconds // 60)
        seconds -= 60 * minutes
        if seconds >= 59.95:
            minutes += 1
            seconds = 0
        return '{:>3d}:{:0>4.1f}'.format(minutes, seconds)


# =====================================================================================
# Forward Model Base Class
# =====================================================================================

class ForwardModel:
    """Defines an interface for user-supplied forward models."""
    
    def __init__(self):
        super(ForwardModel, self).__init__()

    def n_parameters(self):
        """Returns the dimension of the parameter space."""
        raise NotImplementedError

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        raise NotImplementedError

    def n_outputs(self):
        """Returns the number of outputs this model has. The default is 1."""
        return 1


class ToyModel:
    """Mixin interface for toy models with suggested parameters and times."""
    
    def suggested_parameters(self):
        """Returns suggested parameters for this model."""
        raise NotImplementedError

    def suggested_times(self):
        """Returns suggested times for simulation."""
        raise NotImplementedError


# =====================================================================================
# Hodgkin-Huxley IK Model
# =====================================================================================

class HodgkinHuxleyIKModel(ForwardModel, ToyModel):
    r"""
    Toy model based on the potassium current experiments used for Hodgkin and
    Huxley's 1952 model of the action potential of a squid's giant axon.

    A voltage-step protocol is created and applied to an axon, and the elicited
    potassium current (I_K) is given as model output.

    The model equations are:
        alpha = p1 * (-V - 75 + p2) / (exp((-V - 75 + p2) / p3) - 1)
        beta = p4 * exp((-V - 75) / p5)
        dn/dt = alpha * (1 - n) - beta * n
        E_K = -88 mV
        g_max = 36 mS/cm^2
        I_K = g_max * n^4 * (V - E_K)

    Where p1, p2, ..., p5 are the parameters varied in this toy model.

    During simulation, the membrane potential V is varied by holding it
    at -75mV for 90ms, then at a "step potential" for 10ms. The step potentials
    are based on the values used in the original paper.
    """

    def __init__(self, initial_condition=0.3):
        super(HodgkinHuxleyIKModel, self).__init__()

        # Initial conditions
        self._n0 = float(initial_condition)
        if self._n0 <= 0 or self._n0 >= 1:
            raise ValueError('Initial condition must be > 0 and < 1.')

        # Reversal potential, in mV
        self._E_k = -88

        # Maximum conductance, in mS/cm^2
        self._g_max = 36

        # Voltage step protocol
        self._prepare_protocol()

    def fold(self, times, values):
        """
        Takes a set of times and values as return by this model, and "folds"
        the individual currents over each other, to create a very common plot
        in electrophysiology.

        Returns a list of tuples (times, values) for each different voltage step.
        """
        # Get modulus of times
        times = np.mod(times, self._t_both)

        # Remove all points during t_hold
        selection = times >= self._t_hold
        times = times[selection]
        values = values[selection]

        # Use the start of the step as t=0
        times -= self._t_hold

        # Find points to split arrays
        split = 1 + np.argwhere(times[1:] < times[:-1])
        split = split.reshape((len(split),))

        # Split arrays
        traces = []
        i = 0
        for j in split:
            traces.append((times[i:j], values[i:j]))
            i = j
        traces.append((times[i:], values[i:]))
        return traces

    def n_parameters(self):
        """Returns the number of parameters (5)."""
        return 5

    def _prepare_protocol(self):
        """
        Sets up a voltage step protocol for use with this model.

        The protocol consists of multiple steps, each starting with 90ms at a
        fixed holding potential, followed by 10ms at a varying step potential.
        """
        self._t_hold = 90         # 90ms at v_hold
        self._t_step = 10         # 10ms at v_step
        self._t_both = self._t_hold + self._t_step
        self._v_hold = -(0 + 75)
        self._v_step = np.array([
            -(-6 + 75),
            -(-11 + 75),
            -(-19 + 75),
            -(-26 + 75),
            -(-32 + 75),
            -(-38 + 75),
            -(-51 + 75),
            -(-63 + 75),
            -(-76 + 75),
            -(-88 + 75),
            -(-100 + 75),
            -(-109 + 75),
        ])
        self._n_steps = len(self._v_step)

        # Protocol duration
        self._duration = len(self._v_step) * (self._t_hold + self._t_step)

        # Create list of times when V changes (not including t=0)
        self._events = np.concatenate((
            self._t_both * (1 + np.arange(self._n_steps)),
            self._t_both * np.arange(self._n_steps) + self._t_hold))
        self._events.sort()

        # List of voltages (not including V(t=0))
        self._voltages = np.repeat(self._v_step, 2)
        self._voltages[1::2] = self._v_hold

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        if times[0] < 0:
            raise ValueError('All times must be positive.')
        times = np.asarray(times)

        # Unpack parameters
        p1, p2, p3, p4, p5 = parameters

        # Analytically calculate n, during a fixed-voltage step
        def calculate_n(v, n0, t0, times):
            a = p1 * (-(v + 75) + p2) / (np.exp((-(v + 75) + p2) / p3) - 1)
            b = p4 * np.exp((-v - 75) / p5)
            tau = 1 / (a + b)
            inf = a * tau
            return inf - (inf - n0) * np.exp(-(times - t0) / tau)

        # Output arrays
        ns = np.zeros(times.shape)
        vs = np.zeros(times.shape)

        # Iterate over the step, fill in the output arrays
        v = self._v_hold
        t_last = 0
        n_last = self._n0
        for i, t_next in enumerate(self._events):
            index = (t_last <= times) * (times < t_next)
            vs[index] = v
            ns[index] = calculate_n(v, n_last, t_last, times[index])
            n_last = calculate_n(v, n_last, t_last, t_next)
            t_last = t_next
            v = self._voltages[i]
        index = times >= t_next
        vs[index] = v
        ns[index] = calculate_n(v, n_last, t_last, times[index])
        n_last = calculate_n(v, n_last, t_last, t_next)

        # Calculate and return current
        return self._g_max * ns**4 * (vs - self._E_k)

    def suggested_duration(self):
        """Returns the duration of the experimental protocol."""
        return self._duration

    def suggested_parameters(self):
        """
        Returns an array with the original model parameters used by Hodgkin
        and Huxley.
        """
        p1 = 0.01
        p2 = 10
        p3 = 10
        p4 = 0.125
        p5 = 80
        return p1, p2, p3, p4, p5

    def suggested_times(self):
        """Returns suggested times for simulation."""
        fs = 4
        return np.arange(self._duration * fs) / fs


# =====================================================================================
# Problem Classes
# =====================================================================================

class SingleOutputProblem:
    """
    Represents an inference problem where a model is fit to a single time
    series, such as measured from a system with a single output.
    """

    def __init__(self, model, times, values):
        # Check model
        self._model = model
        if model.n_outputs() != 1:
            raise ValueError(
                'Only single-output models can be used for a SingleOutputProblem.')

        # Check times
        self._times = vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times can not be negative.')
        if np.any(self._times[:-1] >= self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values
        self._values = vector(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_times = len(self._times)

        if len(self._values) != self._n_times:
            raise ValueError('Times and values arrays must have same length.')

    def evaluate(self, parameters):
        """Runs a simulation using the given parameters."""
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape((self._n_times,))

    def n_outputs(self):
        """Returns the number of outputs for this problem (always 1)."""
        return 1

    def n_parameters(self):
        """Returns the dimension of this problem."""
        return self._n_parameters

    def n_times(self):
        """Returns the number of sampling points."""
        return self._n_times

    def times(self):
        """Returns this problem's times."""
        return self._times

    def values(self):
        """Returns this problem's values."""
        return self._values


# =====================================================================================
# Error Measures
# =====================================================================================

class ErrorMeasure:
    """Abstract base class for error measures."""
    
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        """Returns the dimension of the parameter space."""
        raise NotImplementedError


class ProblemErrorMeasure(ErrorMeasure):
    """Base class for ErrorMeasures defined for problems."""
    
    def __init__(self, problem):
        super(ProblemErrorMeasure, self).__init__()
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._n_outputs = problem.n_outputs()
        self._n_parameters = problem.n_parameters()
        self._n_times = len(self._times)

    def n_parameters(self):
        return self._n_parameters


class SumOfSquaresError(ProblemErrorMeasure):
    r"""
    Calculates the sum of squares error:
        f = sum_i (y_i - x_i)^2
    where y is the data, x the model output.
    """

    def __init__(self, problem, weights=None):
        super(SumOfSquaresError, self).__init__(problem)

        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError('Number of weights must match number of problem outputs.')
        self._weights = np.asarray([float(w) for w in weights])

    def __call__(self, x):
        return np.sum((np.sum(((self._problem.evaluate(x) - self._values)**2),
                              axis=0) * self._weights), axis=0)


class ProbabilityBasedError(ErrorMeasure):
    """Changes the sign of a LogPDF to use it as an error."""
    
    def __init__(self, log_pdf):
        super(ProbabilityBasedError, self).__init__()
        self._log_pdf = log_pdf

    def __call__(self, x):
        return -self._log_pdf(x)

    def n_parameters(self):
        return self._log_pdf.n_parameters()


# =====================================================================================
# Boundaries
# =====================================================================================

class Boundaries:
    """Abstract class representing boundaries on a parameter space."""
    
    def check(self, parameters):
        """Returns True if the given point is within the boundaries."""
        raise NotImplementedError

    def n_parameters(self):
        """Returns the dimension of the parameter space."""
        raise NotImplementedError

    def sample(self, n=1):
        """Returns n random samples from within the boundaries."""
        raise NotImplementedError


class RectangularBoundaries(Boundaries):
    """
    Represents a set of lower and upper boundaries for model parameters.
    A point x is considered within the boundaries if lower <= x < upper.
    """

    def __init__(self, lower, upper):
        super(RectangularBoundaries, self).__init__()

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
        """Returns the lower boundaries."""
        return self._lower

    def range(self):
        """Returns the size of the parameter space (upper - lower)."""
        return self._upper - self._lower

    def sample(self, n=1):
        return np.random.uniform(
            self._lower, self._upper, size=(n, self._n_parameters))

    def upper(self):
        """Returns the upper boundary."""
        return self._upper


# =====================================================================================
# Optimiser Base Classes
# =====================================================================================

class TunableMethod:
    """Interface for a numerical method with hyper-parameters."""
    
    def n_hyper_parameters(self):
        """Returns the number of hyper-parameters."""
        return 0

    def set_hyper_parameters(self, x):
        """Sets the hyper-parameters."""
        pass


class Optimiser(Loggable, TunableMethod):
    """Base class for optimisers implementing an ask-and-tell interface."""

    def __init__(self, x0, sigma0=None, boundaries=None):
        # Convert and store initial position
        self._x0 = vector(x0)

        # Get dimension
        self._n_parameters = len(self._x0)
        if self._n_parameters < 1:
            raise ValueError('Problem dimension must be greater than zero.')

        # Store boundaries
        self._boundaries = boundaries
        if self._boundaries:
            if self._boundaries.n_parameters() != self._n_parameters:
                raise ValueError(
                    'Boundaries must have same dimension as starting point.')

        # Check initial position is within boundaries
        if self._boundaries:
            if not self._boundaries.check(self._x0):
                raise ValueError('Initial position must lie within given boundaries.')

        # Check initial standard deviation
        if sigma0 is None:
            try:
                self._sigma0 = (1 / 6) * self._boundaries.range()
            except AttributeError:
                self._sigma0 = (1 / 3) * np.abs(self._x0)
                self._sigma0 += (self._sigma0 == 0)
            self._sigma0.setflags(write=False)
        elif np.isscalar(sigma0):
            sigma0 = float(sigma0)
            if sigma0 <= 0:
                raise ValueError('Initial standard deviation must be greater than zero.')
            self._sigma0 = np.ones(self._n_parameters) * sigma0
            self._sigma0.setflags(write=False)
        else:
            self._sigma0 = vector(sigma0)
            if len(self._sigma0) != self._n_parameters:
                raise ValueError(
                    'Initial standard deviation must be None, scalar, or have'
                    ' dimension ' + str(self._n_parameters) + '.')
            if np.any(self._sigma0 <= 0):
                raise ValueError('Initial standard deviations must be greater than zero.')

    def ask(self):
        """Returns a list of positions to evaluate."""
        raise NotImplementedError

    def f_best(self):
        """Returns the best objective function evaluation seen."""
        raise NotImplementedError

    def f_guessed(self):
        """Returns an estimate of the objective function value at x_guessed."""
        return self.f_best()

    def name(self):
        """Returns this method's full name."""
        raise NotImplementedError

    def needs_sensitivities(self):
        """Returns True if this method needs sensitivities."""
        return False

    def running(self):
        """Returns True if an optimisation is in progress."""
        raise NotImplementedError

    def stop(self):
        """Checks if this method should terminate."""
        return False

    def tell(self, fx):
        """Performs an iteration using the evaluations fx."""
        raise NotImplementedError

    def x_best(self):
        """Returns the best position seen during optimisation."""
        raise NotImplementedError

    def x_guessed(self):
        """Returns the optimiser's current best estimate of where the optimum is."""
        return self.x_best()


class PopulationBasedOptimiser(Optimiser):
    """Base class for optimisers that work with multiple points."""

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(PopulationBasedOptimiser, self).__init__(x0, sigma0, boundaries)
        self._population_size = self._suggested_population_size()

    def population_size(self):
        """Returns this optimiser's population size."""
        return self._population_size

    def set_population_size(self, population_size=None):
        """Sets a population size to use."""
        if self.running():
            raise Exception('Cannot change population size during run.')

        if population_size is not None:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')

        self._population_size = population_size

    def _suggested_population_size(self):
        """Returns a suggested population size."""
        raise NotImplementedError

    def n_hyper_parameters(self):
        return 1

    def set_hyper_parameters(self, x):
        self.set_population_size(x[0])


# =====================================================================================
# CMA-ES Optimiser
# =====================================================================================

class CMAES(PopulationBasedOptimiser):
    """
    Finds the best parameters using the CMA-ES method.
    
    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy,
    designed for non-linear derivative-free optimization problems.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(CMAES, self).__init__(x0, sigma0, boundaries)

        if len(x0) < 2:
            raise ValueError('1-dimensional optimisation is not supported by CMA-ES.')

        self._running = False
        self._ready_for_tell = False
        self._f_guessed = np.inf

    def ask(self):
        if not self._running:
            self._initialise()

        self._ready_for_tell = True
        self._user_xs = self._xs = np.array(self._es.ask())

        if self._manual_boundaries:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:
                warnings.warn(
                    'All points requested by CMA-ES are outside the boundaries.')

        self._user_xs.setflags(write=False)
        return self._user_xs

    def f_best(self):
        f = self._es.result.fbest if self._running else None
        return np.inf if f is None else f

    def f_guessed(self):
        return self._f_guessed

    def _initialise(self):
        assert not self._running

        # Import cma
        import cma

        options = cma.CMAOptions()

        # Set boundaries
        self._manual_boundaries = False
        if isinstance(self._boundaries, RectangularBoundaries):
            options.set(
                'bounds',
                [list(self._boundaries._lower), list(self._boundaries._upper)]
            )
        elif self._boundaries is not None:
            self._manual_boundaries = True

        # CMA-ES wants a single standard deviation
        self._sigma0 = np.min(self._sigma0)

        # Tell cma-es to be quiet
        options.set('verbose', -9)

        # Set population size
        options.set('popsize', self._population_size)

        # Seed for reproducibility
        options.set('seed', np.random.randint(2**31))

        # Create the optimizer
        self._es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, options)

        self._running = True

    def name(self):
        return 'Covariance Matrix Adaptation Evolution Strategy (CMA-ES)'

    def running(self):
        return self._running

    def stop(self):
        if not self._running:
            return False

        stop = self._es.stop()
        if stop:
            if 'tolconditioncov' in stop:
                return 'Ill-conditioned covariance matrix.'
        return False

    def _suggested_population_size(self):
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        if self._manual_boundaries and len(fx) < self._population_size:
            user_fx = fx
            fx = np.ones((self._population_size, )) * np.inf
            fx[self._user_ids] = user_fx

        self._es.tell(self._xs, fx)
        self._f_guessed = np.min(fx)

    def x_best(self):
        x = self._es.result.xbest if self._running else None
        return np.array(self._x0 if x is None else x)

    def x_guessed(self):
        x = self._es.result.xfavorite if self._running else None
        return np.array(self._x0 if x is None else x)


# =====================================================================================
# Evaluators
# =====================================================================================

class SequentialEvaluator:
    """Evaluates a function sequentially for a list of input values."""

    def __init__(self, function, args=None):
        if not callable(function):
            raise ValueError('The given function must be callable.')
        self._function = function
        self._args = () if args is None else args

    def evaluate(self, positions):
        scores = [0] * len(positions)
        for k, x in enumerate(positions):
            scores[k] = self._function(x, *self._args)
        return scores


# =====================================================================================
# Optimisation Controller
# =====================================================================================

class OptimisationController:
    """
    Finds the parameter values that minimise an ErrorMeasure.
    """

    def __init__(self, function, x0, sigma0=None, boundaries=None,
                 transformation=None, method=None):
        x0 = vector(x0)

        if function.n_parameters() != len(x0):
            raise ValueError(
                'Starting point must have same dimension as function.')

        # Check if minimising
        self._minimising = True

        # Store function
        self._function = function

        # Create optimiser
        if method is None:
            method = CMAES
        self._optimiser = method(x0, sigma0, boundaries)

        self._needs_sensitivities = self._optimiser.needs_sensitivities()

        # Track optimiser's f_best or f_guessed
        self._use_f_guessed = False

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self._message_interval = 20
        self._message_warm_up = 3

        # Parallelisation
        self._parallel = False
        self._n_workers = 1

        # User callback
        self._callback = None

        # Run once
        self._has_run = False

        # Stopping criteria
        self._max_iterations = 10000
        self._unchanged_max_iterations = 200
        self._unchanged_threshold = 1e-11
        self._max_evaluations = None
        self._threshold = None

        # Post-run statistics
        self._evaluations = None
        self._iterations = None
        self._time = None

    def evaluations(self):
        """Returns the number of evaluations performed."""
        return self._evaluations

    def iterations(self):
        """Returns the number of iterations performed."""
        return self._iterations

    def optimiser(self):
        """Returns the underlying optimiser object."""
        return self._optimiser

    def run(self):
        """Runs the optimisation, returns (x_best, f_best)."""
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        has_stopping_criterion |= (self._unchanged_max_iterations is not None)
        has_stopping_criterion |= (self._max_evaluations is not None)
        has_stopping_criterion |= (self._threshold is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        iteration = 0
        evaluations = 0
        unchanged_iterations = 0

        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        evaluator = SequentialEvaluator(f)

        fb = fg = np.inf
        fb_user, fg_user = fb, fg
        f_sig = np.inf
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Minimising error measure')
                print('Using ' + str(self._optimiser.name()))
                print('Running in sequential mode.')

            pop_size = 1
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                pop_size = self._optimiser.population_size()
                if self._log_to_screen:
                    print('Population size: ' + str(pop_size))

            logger = Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max(
                self._max_evaluations or 0, max_iter_guess * pop_size)
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Best')
            logger.add_float('Current')
            self._optimiser._log_init(logger)
            logger.add_time('Time')

        timer = Timer()
        running = True
        halt_message = ''
        
        try:
            while running:
                xs = self._optimiser.ask()
                fs = evaluator.evaluate(xs)
                self._optimiser.tell(fs)

                fb = self._optimiser.f_best()
                fg = self._optimiser.f_guessed()
                fb_user, fg_user = fb, fg

                f_new = fg if self._use_f_guessed else fb
                if np.abs(f_new - f_sig) >= self._unchanged_threshold:
                    unchanged_iterations = 0
                    f_sig = f_new
                else:
                    unchanged_iterations += 1

                evaluations += len(fs)

                if logging and iteration >= next_message:
                    logger.log(iteration, evaluations, fb_user, fg_user)
                    self._optimiser._log_write(logger)
                    logger.log(timer.time())

                    if iteration < self._message_warm_up:
                        next_message = iteration + 1
                    else:
                        next_message = self._message_interval * (
                            1 + iteration // self._message_interval)

                iteration += 1

                # Check stopping criteria
                if (self._max_iterations is not None and
                        iteration >= self._max_iterations):
                    running = False
                    halt_message = ('Maximum number of iterations ('
                                    + str(iteration) + ') reached.')

                halt = (self._unchanged_max_iterations is not None and
                        unchanged_iterations >= self._unchanged_max_iterations)
                if running and halt:
                    running = False
                    halt_message = ('No significant change for ' +
                                    str(unchanged_iterations) + ' iterations.')

                if (self._max_evaluations is not None and
                        evaluations >= self._max_evaluations):
                    running = False
                    halt_message = (
                        'Maximum number of evaluations ('
                        + str(self._max_evaluations) + ') reached.')

                halt = (self._threshold is not None
                        and f_new < self._threshold)
                if running and halt:
                    running = False
                    halt_message = ('Objective function crossed threshold: '
                                    + str(self._threshold) + '.')

                error = self._optimiser.stop()
                if error:
                    running = False
                    halt_message = str(error)

                elif self._callback is not None:
                    self._callback(iteration - 1, self._optimiser)

        except (Exception, SystemExit, KeyboardInterrupt):
            print('\n' + '-' * 40)
            print('Unexpected termination.')
            print('Current score: ' + str(fg_user))
            print('Current position:')
            x_user = self._optimiser.x_guessed()
            for p in x_user:
                print(strfloat(p))
            print('-' * 40)
            raise

        self._time = timer.time()

        if logging:
            if iteration - 1 < next_message:
                logger.log(iteration, evaluations, fb_user, fg_user)
                self._optimiser._log_write(logger)
                logger.log(self._time)
            if self._log_to_screen:
                print('Halting: ' + halt_message)

        self._evaluations = evaluations
        self._iterations = iteration

        if self._use_f_guessed:
            x = self._optimiser.x_guessed()
            f = self._optimiser.f_guessed()
        else:
            x = self._optimiser.x_best()
            f = self._optimiser.f_best()

        return x, f

    def set_log_to_screen(self, enabled):
        """Enables or disables logging to screen."""
        self._log_to_screen = True if enabled else False

    def set_max_iterations(self, iterations=10000):
        """Sets the maximum number of iterations."""
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """Sets the maximum unchanged iterations stopping criterion."""
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        threshold = float(threshold)
        if threshold < 0:
            raise ValueError('Minimum significant change cannot be negative.')
        self._unchanged_max_iterations = iterations
        self._unchanged_threshold = threshold

    def time(self):
        """Returns the time needed for the last run."""
        return self._time


# =====================================================================================
# Main Script - Hodgkin-Huxley IK Model Optimization with CMA-ES
# =====================================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Hodgkin-Huxley IK Model Parameter Optimization using CMA-ES")
    print("=" * 70)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'data', 'standalone_cmaes_hodgkin_huxley_ik.json')
    
    # Load data from JSON file
    print(f"\nLoading data from: {json_file}")
    data = load_data(json_file)
    
    # Extract data from JSON
    times = np.array(data['times'])
    noisy_values = np.array(data['noisy_values'])
    x_true = np.array(data['suggested_parameters'])
    duration = data['suggested_duration']
    
    print("True parameters:", x_true)
    print("Protocol duration:", duration, "ms")
    print(f"Loaded {len(times)} time points")

    # Create model
    model = HodgkinHuxleyIKModel(initial_condition=data['model_parameters']['initial_condition'])

    # Simulate with true parameters for plotting
    values = model.simulate(x_true, times)

    # Plot simulation result
    plt.figure(figsize=(10, 6))
    plt.plot(times, values)
    plt.xlabel('Time (ms)')
    plt.ylabel('Potassium Current (mA/cm²)')
    plt.title('Hodgkin-Huxley IK Model - True Simulation')
    plt.tight_layout()
    plt.savefig('hodgkin_huxley_ik_simulation.png', dpi=150)
    plt.close()
    print("\nSaved: hodgkin_huxley_ik_simulation.png")

    # Plot folded traces
    plt.figure(figsize=(10, 6))
    for t, v in model.fold(times, values):
        plt.plot(t, v)
    plt.xlabel('Time (ms)')
    plt.ylabel('Potassium Current (mA/cm²)')
    plt.title('Hodgkin-Huxley IK Model - Folded Voltage Step Traces')
    plt.tight_layout()
    plt.savefig('hodgkin_huxley_ik_folded.png', dpi=150)
    plt.close()
    print("Saved: hodgkin_huxley_ik_folded.png")

    # Create an object with links to the model and time series (using loaded noisy data)
    problem = SingleOutputProblem(model, times, noisy_values)

    # Select a score function
    score = SumOfSquaresError(problem)

    # Select some boundaries above and below the true values
    lower = [x / 1.5 for x in x_true]
    upper = [x * 1.5 for x in x_true]
    boundaries = RectangularBoundaries(lower, upper)

    # Perform an optimization with boundaries and hints
    x0 = x_true * 0.98
    
    print("\n" + "-" * 70)
    print("Starting CMA-ES Optimization")
    print("-" * 70)
    
    optimiser = OptimisationController(score, x0, boundaries=boundaries, method=CMAES)
    
    # SPEEDUP: Reduced from 100 to 10 for faster execution (~10x speedup)
    optimiser.set_max_unchanged_iterations(10)
    optimiser.set_log_to_screen(True)
    
    found_parameters, found_score = optimiser.run()

    # Compare parameters with original
    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print('\nFound solution:          True parameters:')
    for k, x in enumerate(found_parameters):
        print(strfloat(x) + '    ' + strfloat(x_true[k]))
    
    print(f"\nFinal score: {found_score:.6f}")
    print(f"Total iterations: {optimiser.iterations()}")
    print(f"Total evaluations: {optimiser.evaluations()}")
    print(f"Time: {optimiser.time():.2f} seconds")

    # Evaluate model at found parameters
    found_values = problem.evaluate(found_parameters)

    # Show quality of fit
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time (ms)')
    plt.ylabel('Potassium Current (mA/cm²)')
    
    # Plot folded noisy data
    for t, v in model.fold(times, noisy_values):
        plt.plot(t, v, c='b', alpha=0.5, label='Noisy data')
    
    # Plot folded fit
    for t, v in model.fold(times, found_values):
        plt.plot(t, v, c='r', linewidth=2, label='Fit')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title('Hodgkin-Huxley IK Model - Optimization Fit')
    plt.tight_layout()
    plt.savefig('hodgkin_huxley_ik_fit.png', dpi=150)
    plt.close()
    print("\nSaved: hodgkin_huxley_ik_fit.png")

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

"""
