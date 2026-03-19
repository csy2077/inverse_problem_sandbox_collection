import numpy as np
import timeit


# --- Extracted Dependencies ---

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

class Timer:
    """Provides accurate timing."""
    def __init__(self):
        self._start = timeit.default_timer()

    def time(self):
        return timeit.default_timer() - self._start

    def reset(self):
        self._start = timeit.default_timer()

class ErrorMeasure:
    """Abstract base class for error measures."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

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

class LogPDF:
    """Represents the natural logarithm of a probability density function."""
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

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
