import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_
import inspect as _inspect_
import json as _json_
_META_REGISTRY_ = set()
try:
    import numpy as _np_
except ImportError:
    _np_ = None
try:
    import torch as _torch_
except ImportError:
    _torch_ = None

def _fix_seeds_(seed=42):
    import random
    if _np_:
        _np_.random.seed(seed)
    random.seed(seed)
    if _torch_:
        _torch_.manual_seed(seed)
        if _torch_.cuda.is_available():
            _torch_.cuda.manual_seed_all(seed)
_fix_seeds_(42)

def _analyze_obj_(obj):
    if _torch_ and isinstance(obj, _torch_.Tensor):
        return {'type': 'torch.Tensor', 'shape': list(obj.shape), 'dtype': str(obj.dtype), 'device': str(obj.device)}
    if _np_ and isinstance(obj, _np_.ndarray):
        return {'type': 'numpy.ndarray', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return {'type': type(obj).__name__, 'length': len(obj), 'elements': [_analyze_obj_(item) for item in obj]}
    if hasattr(obj, '__dict__'):
        methods = []
        try:
            for m in dir(obj):
                if m.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, m)
                    if callable(attr):
                        methods.append(m)
                except Exception:
                    continue
        except Exception:
            pass
        return {'type': 'CustomObject', 'class_name': obj.__class__.__name__, 'public_methods': methods, 'attributes': list(obj.__dict__.keys())}
    try:
        val_str = str(obj)
    except:
        val_str = '<non-stringifiable>'
    return {'type': type(obj).__name__, 'value_sample': val_str}

def _record_io_decorator_(save_path='./'):

    def decorator(func, parent_function=None):

        @_functools_.wraps(func)
        def wrapper(*args, **kwargs):
            global _META_REGISTRY_
            func_name = func.__name__
            parent_key = str(parent_function)
            registry_key = (func_name, parent_key)
            should_record = False
            if registry_key not in _META_REGISTRY_:
                should_record = True
            result = None
            inputs_meta = {}
            if should_record:
                try:
                    sig = _inspect_.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for (name, value) in bound_args.arguments.items():
                        inputs_meta[name] = _analyze_obj_(value)
                except Exception as e:
                    inputs_meta = {'error': f'Analysis failed: {e}'}
            result = func(*args, **kwargs)
            if should_record:
                try:
                    output_meta = _analyze_obj_(result)
                except Exception:
                    output_meta = 'Analysis failed'
                try:
                    final_path = save_path
                    if not final_path.endswith('.json'):
                        if not _os_.path.exists(final_path):
                            _os_.makedirs(final_path, exist_ok=True)
                        if parent_function == None:
                            final_path = _os_.path.join(final_path, f'IO_meta_{func_name}.json')
                        else:
                            final_path = _os_.path.join(final_path, f'IO_meta_parent_function_{parent_function}_{func_name}.json')
                    dir_name = _os_.path.dirname(final_path)
                    if dir_name and (not _os_.path.exists(dir_name)):
                        _os_.makedirs(dir_name, exist_ok=True)
                    existing_data = []
                    file_exists = _os_.path.exists(final_path)
                    if file_exists:
                        try:
                            with open(final_path, 'r') as f:
                                existing_data = _json_.load(f)
                        except:
                            pass
                    already_in_file = False
                    for entry in existing_data:
                        if entry.get('function_name') == func_name:
                            already_in_file = True
                            break
                    if not already_in_file:
                        func_schema = {'function_name': func_name, 'inputs': inputs_meta, 'output': output_meta}
                        existing_data.append(func_schema)
                        with open(final_path, 'w') as f:
                            _json_.dump(existing_data, f, indent=4)
                        print(f'  [Metadata] Recorded schema for: {func_name}')
                    _META_REGISTRY_.add(registry_key)
                except Exception as e:
                    print(f'  [Metadata] Warning: {e}')
            if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
                return decorator(result, parent_function=func_name)
            return result
        return wrapper
    return decorator

def _data_capture_decorator_(func, parent_function=None):

    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/std_data'
            if not _os_.path.exists(out_dir):
                _os_.makedirs(out_dir, exist_ok=True)
            func_name = func.__name__
            if parent_function == None:
                save_path = _os_.path.join(out_dir, f'data_{func_name}.pkl')
            else:
                save_path = _os_.path.join(out_dir, f'data_parent_{parent_function}_{func_name}.pkl')

            def detach_recursive(obj):
                if hasattr(obj, 'detach'):
                    return obj.detach()
                if isinstance(obj, list):
                    return [detach_recursive(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple((detach_recursive(x) for x in obj))
                if isinstance(obj, dict):
                    return {k: detach_recursive(v) for (k, v) in obj.items()}
                return obj
            payload = {'func_name': func_name, 'args': detach_recursive(args), 'kwargs': detach_recursive(kwargs), 'output': detach_recursive(result)}
            with open(save_path, 'wb') as f:
                _dill_.dump(payload, f)
        except Exception as e:
            pass
        if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
            return _data_capture_decorator_(result, parent_function=func_name)
        return result
    return wrapper
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate
import timeit
import sys
import warnings
import json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'standalone_cmaes_beeler_reuter_ap.json')

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def load_data_from_json(filepath):
    """
    Load suggested parameters and times from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Dictionary containing:
            - 'suggested_parameters': numpy array of log-transformed conductances
            - 'suggested_times': numpy array of time points
    """
    print(f'Loading data from: {filepath}')
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {'suggested_parameters': np.array(data['suggested_parameters']), 'suggested_times': np.array(data['suggested_times'])}
FLOAT_FORMAT = '{: .17e}'

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def strfloat(x):
    """
    Converts a float to a string, with maximum precision.
    """
    return FLOAT_FORMAT.format(float(x))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
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
        if time < 0.01:
            return str(time) + ' seconds'
        elif time < 60:
            return str(round(time, 2)) + ' seconds'
        output = []
        time = int(round(time))
        units = [(604800, 'week'), (86400, 'day'), (3600, 'hour'), (60, 'minute')]
        for (k, name) in units:
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
        self._buffer = []

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

    def set_stream(self, stream=sys.stdout):
        """Enables or disables logging to screen."""
        if self._have_logged:
            raise RuntimeError('Cannot configure after logging has started.')
        self._stream = stream

    def set_filename(self, filename=None, csv=False):
        """Enables or disables logging to file."""
        if self._have_logged:
            raise RuntimeError('Cannot configure after logging has started.')
        if filename is None:
            self._filename = None
        else:
            self._filename = str(filename)
        self._csv_mode = True if csv else False

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
                rows.append([self._buffer.pop(0) for i in range(nfields)])
            if not rows:
                return
        formatted_rows = []
        if not self._have_logged:
            headers = []
            for (i, name) in enumerate(self._field_names):
                width = self._field_formats[i][0]
                headers.append(name + ' ' * (width - len(name)))
            formatted_rows.append(headers)
        for row in rows:
            column = iter(row)
            formatted_row = []
            for (width, dtype, f1, f2) in self._field_formats:
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
        self._have_logged = True

    def _format_time(self, seconds):
        """Formats a time in seconds to the format "mmm:ss.s"."""
        minutes = int(seconds // 60)
        seconds -= 60 * minutes
        if seconds >= 59.95:
            minutes += 1
            seconds = 0
        return '{:>3d}:{:0>4.1f}'.format(minutes, seconds)

class ForwardModel:
    """Defines an interface for user-supplied forward models."""

    def __init__(self):
        pass

    def n_parameters(self):
        """Returns the dimension of the parameter space."""
        raise NotImplementedError

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        raise NotImplementedError

    def n_outputs(self):
        """Returns the number of outputs this model has. Default is 1."""
        return 1

class ActionPotentialModel(ForwardModel):
    """
    The 1977 Beeler-Reuter model of the mammalian ventricular action potential (AP).

    This model is written as an ODE with 8 states and several intermediary variables.
    
    The model contains 5 ionic currents, each described by a sub-model with several 
    kinetic parameters, and a maximum conductance parameter that determines its magnitude.
    Only the 5 conductance parameters are varied, all other parameters are fixed.
    
    A parameter transformation is used: instead of specifying the maximum conductances 
    directly, their natural logarithm should be used.
    
    As outputs, we use the AP and the calcium transient, as these are the only two 
    states (out of the total of eight) with a physically observable counterpart.
    """

    def __init__(self, y0=None):
        super().__init__()
        if y0 is None:
            self.set_initial_conditions([-84.622, 2e-07])
        else:
            self.set_initial_conditions(y0)
        self._m0 = 0.01
        self._h0 = 0.99
        self._j0 = 0.98
        self._d0 = 0.003
        self._f0 = 0.99
        self._x10 = 0.0004
        self._C_m = 1.0
        self._E_Na = 50.0
        self._I_Stim_amp = 25.0
        self._I_Stim_period = 1000.0
        self._I_Stim_length = 2.0
        self.set_solver_tolerances()

    def initial_conditions(self):
        """Returns the initial conditions of this model."""
        return [self._v0, self._cai0]

    def n_outputs(self):
        """Returns 2 (membrane voltage and calcium concentration)."""
        return 2

    def n_parameters(self):
        """Returns 5 (conductance values)."""
        return 5

    def _rhs(self, states, time, parameters):
        """Right-hand side equation of the ode to solve."""
        (V, Cai, m, h, j, d, f, x1) = states
        (gNaBar, gNaC, gCaBar, gK1Bar, gx1Bar) = np.exp(parameters)
        INa = (gNaBar * m ** 3 * h * j + gNaC) * (V - self._E_Na)
        alpha = (V + 47) / (1 - np.exp(-0.1 * (V + 47)))
        beta = 40 * np.exp(-0.056 * (V + 72))
        dmdt = alpha * (1 - m) - beta * m
        alpha = 0.126 * np.exp(-0.25 * (V + 77))
        beta = 1.7 / (1 + np.exp(-0.082 * (V + 22.5)))
        dhdt = alpha * (1 - h) - beta * h
        alpha = 0.055 * np.exp(-0.25 * (V + 78)) / (1 + np.exp(-0.2 * (V + 78)))
        beta = 0.3 / (1 + np.exp(-0.1 * (V + 32)))
        djdt = alpha * (1 - j) - beta * j
        E_Ca = -82.3 - 13.0287 * np.log(Cai)
        ICa = gCaBar * d * f * (V - E_Ca)
        alpha = 0.095 * np.exp(-0.01 * (V + -5)) / (np.exp(-0.072 * (V + -5)) + 1)
        beta = 0.07 * np.exp(-0.017 * (V + 44)) / (np.exp(0.05 * (V + 44)) + 1)
        dddt = alpha * (1 - d) - beta * d
        alpha = 0.012 * np.exp(-0.008 * (V + 28)) / (np.exp(0.15 * (V + 28)) + 1)
        beta = 0.0065 * np.exp(-0.02 * (V + 30)) / (np.exp(-0.2 * (V + 30)) + 1)
        dfdt = alpha * (1 - f) - beta * f
        dCaidt = -1e-07 * ICa + 0.07 * (1e-07 - Cai)
        IK1 = gK1Bar * (4 * (np.exp(0.04 * (V + 85)) - 1) / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53))) + 0.2 * (V + 23) / (1 - np.exp(-0.04 * (V + 23))))
        Ix1 = gx1Bar * x1 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))
        alpha = 0.0005 * np.exp(0.083 * (V + 50)) / (np.exp(0.057 * (V + 50)) + 1)
        beta = 0.0013 * np.exp(-0.06 * (V + 20)) / (np.exp(-0.04 * (V + 333)) + 1)
        dx1dt = alpha * (1 - x1) - beta * x1
        if time % self._I_Stim_period < self._I_Stim_length:
            IStim = self._I_Stim_amp
        else:
            IStim = 0
        dVdt = -(1 / self._C_m) * (IK1 + Ix1 + INa + ICa - IStim)
        output = np.array([dVdt, dCaidt, dmdt, dhdt, djdt, dddt, dfdt, dx1dt])
        return output

    def set_initial_conditions(self, y0):
        """Changes the initial conditions for this model."""
        if y0[1] < 0:
            raise ValueError('Initial condition of ``cai`` cannot be negative.')
        self._v0 = y0[0]
        self._cai0 = y0[1]

    def set_solver_tolerances(self, rtol=0.0001, atol=1e-06):
        """Updates the solver tolerances."""
        self._rtol = float(rtol)
        self._atol = float(atol)

    def simulate(self, parameters, times):
        """Run forward simulation."""
        y0 = [self._v0, self._cai0, self._m0, self._h0, self._j0, self._d0, self._f0, self._x10]
        solved_states = scipy.integrate.odeint(self._rhs, y0, times, args=(parameters,), hmax=self._I_Stim_length, rtol=self._rtol, atol=self._atol)
        return solved_states[:, 0:2]

    def suggested_parameters(self):
        """
        Returns suggested parameters for this model.
        The returned vector is already log-transformed.
        """
        g_Na = 4.0
        g_NaC = 0.003
        g_Ca = 0.09
        g_K1 = 0.35
        g_x1 = 0.8
        return np.log([g_Na, g_NaC, g_Ca, g_K1, g_x1])

    def suggested_times(self):
        """Returns suggested time points for simulation."""
        return np.arange(0, 400, 0.5)

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
            raise ValueError('Values array must have shape (n_times, n_outputs).')

    def evaluate(self, parameters):
        """Runs a simulation using the given parameters."""
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def n_outputs(self):
        """Returns the number of outputs."""
        return self._n_outputs

    def n_parameters(self):
        """Returns the dimension of this problem."""
        return self._n_parameters

    def n_times(self):
        """Returns the number of time points."""
        return self._n_times

    def times(self):
        """Returns this problem's times."""
        return self._times

    def values(self):
        """Returns this problem's values."""
        return self._values

class ErrorMeasure:
    """Abstract base class for error measures."""

    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        """Returns the dimension of the parameter space."""
        raise NotImplementedError

class ProblemErrorMeasure(ErrorMeasure):
    """Base class for ErrorMeasures defined for problems."""

    def __init__(self, problem=None):
        super().__init__()
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._n_outputs = problem.n_outputs()
        self._n_parameters = problem.n_parameters()
        self._n_times = len(self._times)

    def n_parameters(self):
        """Returns the number of parameters."""
        return self._n_parameters

class SumOfSquaresError(ProblemErrorMeasure):
    """
    Calculates the sum of squares error:
    f = sum_i (y_i - x_i)^2
    """

    def __init__(self, problem, weights=None):
        super().__init__(problem)
        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError('Number of weights must match number of problem outputs.')
        self._weights = np.asarray([float(w) for w in weights])

    def __call__(self, x):
        return np.sum(np.sum((self._problem.evaluate(x) - self._values) ** 2, axis=0) * self._weights, axis=0)

class Boundaries:
    """Abstract class representing boundaries on a parameter space."""

    def check(self, parameters):
        """Returns True if the point is within the boundaries."""
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
    A point x is within bounds if lower <= x < upper.
    """

    def __init__(self, lower, upper):
        super().__init__()
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
        """Returns True if parameters are within bounds."""
        if np.any(parameters < self._lower):
            return False
        if np.any(parameters >= self._upper):
            return False
        return True

    def n_parameters(self):
        """Returns the dimension."""
        return self._n_parameters

    def lower(self):
        """Returns the lower boundaries."""
        return self._lower

    def upper(self):
        """Returns the upper boundary."""
        return self._upper

    def range(self):
        """Returns the size of the parameter space."""
        return self._upper - self._lower

    def sample(self, n=1):
        """Returns n random samples."""
        return np.random.uniform(self._lower, self._upper, size=(n, self._n_parameters))

class SequentialEvaluator:
    """Evaluates a function for a list of input values sequentially."""

    def __init__(self, function, args=None):
        if not callable(function):
            raise ValueError('The given function must be callable.')
        self._function = function
        if args is None:
            self._args = ()
        else:
            self._args = args

    def evaluate(self, positions):
        """Evaluate all positions."""
        scores = [0] * len(positions)
        for (k, x) in enumerate(positions):
            scores[k] = self._function(x, *self._args)
        return scores

class Optimiser(Loggable):
    """Base class for optimisers implementing an ask-and-tell interface."""

    def __init__(self, x0, sigma0=None, boundaries=None):
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)
        if self._n_parameters < 1:
            raise ValueError('Problem dimension must be greater than zero.')
        self._boundaries = boundaries
        if self._boundaries:
            if self._boundaries.n_parameters() != self._n_parameters:
                raise ValueError('Boundaries must have same dimension as starting point.')
        if self._boundaries:
            if not self._boundaries.check(self._x0):
                raise ValueError('Initial position must lie within given boundaries.')
        if sigma0 is None:
            try:
                self._sigma0 = 1 / 6 * self._boundaries.range()
            except AttributeError:
                self._sigma0 = 1 / 3 * np.abs(self._x0)
                self._sigma0 += self._sigma0 == 0
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
                raise ValueError('Initial standard deviation dimension mismatch.')
            if np.any(self._sigma0 <= 0):
                raise ValueError('Initial standard deviations must be greater than zero.')

    def ask(self):
        """Returns a list of positions in the search space to evaluate."""
        raise NotImplementedError

    def f_best(self):
        """Returns the best objective function evaluation seen."""
        raise NotImplementedError

    def f_guessed(self):
        """Returns estimate of objective at x_guessed."""
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
        """Returns the optimiser's best guess of where the optimum is."""
        return self.x_best()

class PopulationBasedOptimiser(Optimiser):
    """Base class for population-based optimisers."""

    def __init__(self, x0, sigma0=None, boundaries=None):
        super().__init__(x0, sigma0, boundaries)
        self._population_size = self._suggested_population_size()

    def population_size(self):
        """Returns the population size."""
        return self._population_size

    def set_population_size(self, population_size=None):
        """Sets the population size."""
        if self.running():
            raise Exception('Cannot change population size during run.')
        if population_size is not None:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')
        self._population_size = population_size

    def suggested_population_size(self, round_up_to_multiple_of=None):
        """Returns a suggested population size."""
        population_size = self._suggested_population_size()
        if round_up_to_multiple_of is not None:
            n = int(round_up_to_multiple_of)
            if n > 1:
                population_size = n * ((population_size - 1) // n + 1)
        return population_size

    def _suggested_population_size(self):
        """Returns suggested population size."""
        raise NotImplementedError

class CMAES(PopulationBasedOptimiser):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    Uses the Python cma module.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super().__init__(x0, sigma0, boundaries)
        if len(x0) < 2:
            raise ValueError('1-dimensional optimisation is not supported by CMA-ES.')
        self._running = False
        self._ready_for_tell = False
        self._f_guessed = np.inf

    def ask(self):
        """Returns positions to evaluate."""
        if not self._running:
            self._initialise()
        self._ready_for_tell = True
        self._user_xs = self._xs = np.array(self._es.ask())
        if self._manual_boundaries:
            self._user_ids = np.nonzero([self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:
                warnings.warn('All points requested by CMA-ES are outside boundaries.')
        self._user_xs.setflags(write=False)
        return self._user_xs

    def f_best(self):
        """Returns best function value seen."""
        f = self._es.result.fbest if self._running else None
        return np.inf if f is None else f

    def f_guessed(self):
        """Returns estimated function value at x_guessed."""
        return self._f_guessed

    def _initialise(self):
        """Initialises the optimiser for the first iteration."""
        assert not self._running
        import cma
        options = cma.CMAOptions()
        self._manual_boundaries = False
        if isinstance(self._boundaries, RectangularBoundaries):
            options.set('bounds', [list(self._boundaries._lower), list(self._boundaries._upper)])
        elif self._boundaries is not None:
            self._manual_boundaries = True
        self._sigma0 = np.min(self._sigma0)
        options.set('verbose', -9)
        options.set('popsize', self._population_size)
        options.set('seed', np.random.randint(2 ** 31))
        self._es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, options)
        self._running = True

    def name(self):
        """Returns method name."""
        return 'Covariance Matrix Adaptation Evolution Strategy (CMA-ES)'

    def running(self):
        """Returns True if running."""
        return self._running

    def stop(self):
        """Checks if should stop."""
        if not self._running:
            return False
        stop = self._es.stop()
        if stop:
            if 'tolconditioncov' in stop:
                return 'Ill-conditioned covariance matrix.'
        return False

    def _suggested_population_size(self):
        """Returns suggested population size."""
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """Updates optimizer with function evaluations."""
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False
        if self._manual_boundaries and len(fx) < self._population_size:
            user_fx = fx
            fx = np.ones((self._population_size,)) * np.inf
            fx[self._user_ids] = user_fx
        self._es.tell(self._xs, fx)
        self._f_guessed = np.min(fx)

    def x_best(self):
        """Returns best position seen."""
        x = self._es.result.xbest if self._running else None
        return np.array(self._x0 if x is None else x)

    def x_guessed(self):
        """Returns best guess of optimum position."""
        x = self._es.result.xfavorite if self._running else None
        return np.array(self._x0 if x is None else x)

class OptimisationController:
    """
    Finds parameter values that minimise an ErrorMeasure.
    """

    def __init__(self, function, x0, sigma0=None, boundaries=None, method=None):
        x0 = vector(x0)
        if function.n_parameters() != len(x0):
            raise ValueError('Starting point must have same dimension as function.')
        self._minimising = True
        self._function = function
        if method is None:
            method = CMAES
        self._optimiser = method(x0, sigma0, boundaries)
        self._needs_sensitivities = self._optimiser.needs_sensitivities()
        self._use_f_guessed = False
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self._message_interval = 20
        self._message_warm_up = 3
        self._max_iterations = 1000
        self._unchanged_max_iterations = 20
        self._unchanged_threshold = 1e-11
        self._max_evaluations = None
        self._threshold = None
        self._evaluations = None
        self._iterations = None
        self._time = None

    def run(self):
        """Runs the optimisation, returns (x_best, f_best)."""
        has_stopping_criterion = False
        has_stopping_criterion |= self._max_iterations is not None
        has_stopping_criterion |= self._unchanged_max_iterations is not None
        has_stopping_criterion |= self._max_evaluations is not None
        has_stopping_criterion |= self._threshold is not None
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
        (fb_user, fg_user) = (fb, fg)
        f_sig = np.inf
        next_message = 0
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
            max_eval_guess = max(self._max_evaluations or 0, max_iter_guess * pop_size)
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Best')
            logger.add_float('Current')
            self._optimiser._log_init(logger)
            logger.add_time('Time')
        timer = Timer()
        running = True
        try:
            while running:
                xs = self._optimiser.ask()
                fs = evaluator.evaluate(xs)
                self._optimiser.tell(fs)
                fb = self._optimiser.f_best()
                fg = self._optimiser.f_guessed()
                (fb_user, fg_user) = (fb, fg)
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
                        next_message = self._message_interval * (1 + iteration // self._message_interval)
                iteration += 1
                if self._max_iterations is not None and iteration >= self._max_iterations:
                    running = False
                    halt_message = 'Maximum number of iterations (' + str(iteration) + ') reached.'
                halt = self._unchanged_max_iterations is not None and unchanged_iterations >= self._unchanged_max_iterations
                if running and halt:
                    running = False
                    halt_message = 'No significant change for ' + str(unchanged_iterations) + ' iterations.'
                if self._max_evaluations is not None and evaluations >= self._max_evaluations:
                    running = False
                    halt_message = 'Maximum number of evaluations (' + str(self._max_evaluations) + ') reached.'
                halt = self._threshold is not None and f_new < self._threshold
                if running and halt:
                    running = False
                    halt_message = 'Objective function crossed threshold: ' + str(self._threshold) + '.'
                error = self._optimiser.stop()
                if error:
                    running = False
                    halt_message = str(error)
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
        return (x, f)

    def set_max_iterations(self, iterations=10000):
        """Sets maximum iterations stopping criterion."""
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """Sets unchanged iterations stopping criterion."""
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        threshold = float(threshold)
        if threshold < 0:
            raise ValueError('Minimum significant change cannot be negative.')
        self._unchanged_max_iterations = iterations
        self._unchanged_threshold = threshold
if __name__ == '__main__':
    print('=' * 70)
    print('Beeler-Reuter Action Potential Model - CMA-ES Optimization')
    print('=' * 70)
    print('\n--- Data Loading ---')
    data = load_data_from_json(DATA_FILE)
    x_true = data['suggested_parameters']
    times = data['suggested_times']
    print(f'  Loaded suggested_parameters: shape {x_true.shape}')
    print(f'  Loaded suggested_times: shape {times.shape}')
    model = ActionPotentialModel()
    print('\nSimulating action potential with true parameters...')
    values = model.simulate(x_true, times)
    (fig, axes) = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(times, values[:, 0])
    axes[1].plot(times, values[:, 1])
    axes[0].set_ylabel('Voltage [mV]')
    axes[1].set_ylabel('Calcium [mM]')
    axes[1].set_xlabel('Time [ms]')
    axes[0].set_title('Beeler-Reuter Action Potential Model - Simulation')
    plt.tight_layout()
    plt.savefig('beeler_reuter_simulation.png', dpi=150)
    plt.close()
    print('Saved: beeler_reuter_simulation.png')
    np.random.seed(42)
    values[:, 0] += np.random.normal(0, 1, values[:, 0].shape)
    values[:, 1] += np.random.normal(0, 5e-07, values[:, 1].shape)
    problem = MultiOutputProblem(model, times, values)
    weights = [1.0 / 70.0, 1.0 / 6e-06]
    score = SumOfSquaresError(problem, weights=weights)
    lower = x_true - 1
    upper = x_true + 1
    boundaries = RectangularBoundaries(lower, upper)
    x0 = x_true + 0.25
    print('\nRunning CMA-ES optimization...')
    print('(max_iterations=1000, max_unchanged_iterations=20)')
    optimiser = OptimisationController(score, x0, boundaries=boundaries, method=CMAES)
    print('Running...')
    (found_parameters, found_score) = optimiser.run()
    print('\n' + '=' * 50)
    print('Found solution:          True parameters:')
    for (k, x) in enumerate(found_parameters):
        print(strfloat(np.exp(x)) + '    ' + strfloat(np.exp(x0[k])))
    print('=' * 50)
    plt.figure(figsize=(8, 6))
    logGRatio = (found_parameters - x0) / np.log(10)
    x = range(len(logGRatio))
    plt.bar(x, logGRatio, color='steelblue', edgecolor='black')
    plt.xticks(x, ('$G_{Na}$', '$G_{NaC}$', '$G_{Ca}$', '$G_{K1}$', '$G_{x1}$'))
    plt.ylabel('$\\log(G_{found} / G_{true})$')
    plt.title('Beeler-Reuter Model - Parameter Recovery')
    plt.tight_layout()
    plt.savefig('beeler_reuter_optimization_result.png', dpi=150)
    plt.close()
    print('Saved: beeler_reuter_optimization_result.png')
    found_values = problem.evaluate(found_parameters)
    (fig, axes) = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(times, values[:, 0], 'b-', alpha=0.6, label='Noisy data')
    axes[0].plot(times, found_values[:, 0], 'r-', linewidth=1.5, label='Fit')
    axes[1].plot(times, values[:, 1], 'b-', alpha=0.6, label='Noisy data')
    axes[1].plot(times, found_values[:, 1], 'r-', linewidth=1.5, label='Fit')
    axes[0].set_ylabel('Voltage [mV]')
    axes[1].set_ylabel('Calcium [mM]')
    axes[1].set_xlabel('Time [ms]')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[0].set_title('Beeler-Reuter Model - Fit Quality')
    plt.tight_layout()
    plt.savefig('beeler_reuter_fit.png', dpi=150)
    plt.close()
    print('Saved: beeler_reuter_fit.png')
    print('\n' + '=' * 70)
    print('Optimization complete!')
    print('Final score:', strfloat(found_score))
    print('=' * 70)