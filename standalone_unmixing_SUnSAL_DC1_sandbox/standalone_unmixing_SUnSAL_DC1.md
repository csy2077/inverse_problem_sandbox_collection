"""python
# ==============================================================================
# STANDALONE UNMIXING SCRIPT - SUnSAL on DC1 Dataset
# ==============================================================================
#
# ORIGINAL SCRIPT PATH:
#   /fs-computility-new/UPDZ02_sunhe/shared/jiayx/HySupp/unmixing.py
#   with command: python unmixing.py data=DC1 model=SUnSAL SNR=30
#
# DESCRIPTION:
#   Semi-supervised unmixing using SUnSAL (Sparse Unmixing by variable Splitting
#   and Augmented Lagrangian) algorithm.
#
# DEPENDENCIES:
#   Inputs:
#     - ./data_standalone/DC1.mat (hyperspectral image data)
#     - ./data_standalone/standalone_unmixing_SUnSAL_DC1.json (configuration)
#   Outputs:
#     - ./exp_standalone_unmixing_SUnSAL_DC1/estimates.mat (estimated abundances)
#     - ./exp_standalone_unmixing_SUnSAL_DC1/metrics.json (SRE, aRMSE metrics)
#
# ITERATION COUNT:
#   Original: AL_iters = 1000 (unchanged - algorithm converges early via tolerance)
#
# INLINED CODE:
#   All high-level package calls from mlxp and src modules have been inlined:
#   - src.model.semisupervised.SUnSAL (SUnSAL algorithm)
#   - src.data.base.HSIWithGT (HSI data loading)
#   - src.data.noise.AdditiveWhiteGaussianNoise (noise generation)
#   - src.utils.metrics (SRE, aRMSE metrics)
#   - src.utils.aligners (AbundancesAligner using Hungarian algorithm)
#   - munkres library (Hungarian algorithm for optimal assignment)
#
# ==============================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import logging
import logging.config
import time
from typing import List, Tuple, Optional

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)
log = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "data_standalone", "standalone_unmixing_SUnSAL_DC1.json")

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")

# ==============================================================================
# LOAD CONFIGURATION
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ==============================================================================
# MUNKRES (HUNGARIAN ALGORITHM) - Inlined from munkres library
# ==============================================================================
# Source: https://github.com/bmc/munkres (BSD License)

class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance."""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def pad_matrix(self, matrix, pad_value=0):
        """Pad a possibly non-square matrix to make it square."""
        max_columns = 0
        total_rows = len(matrix)
        for row in matrix:
            max_columns = max(max_columns, len(row))
        total_rows = max(max_columns, total_rows)
        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix.append(new_row)
        while len(new_matrix) < total_rows:
            new_matrix.append([pad_value] * total_rows)
        return new_matrix

    def compute(self, cost_matrix):
        """
        Compute the indices for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.
        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0]) if cost_matrix else 0
        self.row_covered = [False for _ in range(self.n)]
        self.col_covered = [False for _ in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self._make_matrix(self.n * 2, 0)
        self.marked = self._make_matrix(self.n, 0)

        step = 1
        steps = {
            1: self._step1,
            2: self._step2,
            3: self._step3,
            4: self._step4,
            5: self._step5,
            6: self._step6,
        }

        while step < 7:
            func = steps[step]
            step = func()

        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results.append((i, j))
        return results

    def _make_matrix(self, n, val):
        """Create an n x n matrix, filled with the given value."""
        return [[val for _ in range(n)] for _ in range(n)]

    def _step1(self):
        """
        For each row of the matrix, find the smallest element and subtract
        it from every element in its row. Go to Step 2.
        """
        for i in range(self.n):
            minval = min(self.C[i])
            for j in range(self.n):
                self.C[i][j] -= minval
        return 2

    def _step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred zero
        in its row or column, star Z. Repeat for each element in the matrix.
        Go to Step 3.
        """
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] == 0 and not self.col_covered[j] and not self.row_covered[i]:
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break
        self._clear_covers()
        return 3

    def _step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        count = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 1 and not self.col_covered[j]:
                    self.col_covered[j] = True
                    count += 1
        if count >= self.n:
            return 7
        return 4

    def _step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        done = False
        row = -1
        col = -1
        star_col = -1

        while not done:
            row, col = self._find_a_zero()
            if row < 0:
                done = True
                return 6
            else:
                self.marked[row][col] = 2
                star_col = self._find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    self.Z0_r = row
                    self.Z0_c = col
                    done = True
                    return 5

    def _step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False

        while not done:
            row = self._find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count - 1][1]
            else:
                done = True
            if not done:
                col = self._find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]
                path[count][1] = col

        self._convert_path(path, count)
        self._clear_covers()
        self._erase_primes()
        return 3

    def _step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self._find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def _find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = float('inf')
        for i in range(self.n):
            for j in range(self.n):
                if not self.row_covered[i] and not self.col_covered[j]:
                    if self.C[i][j] < minval:
                        minval = self.C[i][j]
        return minval

    def _find_a_zero(self):
        """Find the first uncovered element with value 0."""
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] == 0 and not self.row_covered[i] and not self.col_covered[j]:
                    return i, j
        return -1, -1

    def _find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        for j in range(self.n):
            if self.marked[row][j] == 1:
                return j
        return -1

    def _find_star_in_col(self, col):
        """
        Find the first starred element in the specified column. Returns
        the row index, or -1 if no starred element was found.
        """
        for i in range(self.n):
            if self.marked[i][col] == 1:
                return i
        return -1

    def _find_prime_in_row(self, row):
        """
        Find the first primed element in the specified row. Returns
        the column index, or -1 if no primed element was found.
        """
        for j in range(self.n):
            if self.marked[row][j] == 2:
                return j
        return -1

    def _convert_path(self, path, count):
        """Convert path."""
        for i in range(count + 1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def _clear_covers(self):
        """Clear all covered matrix cells."""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def _erase_primes(self):
        """Erase all prime markings."""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0


# ==============================================================================
# METRICS - Inlined from src/utils/metrics.py
# ==============================================================================

class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class MSE(BaseMetric):
    """Mean Squared Error metric for alignment."""
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)
        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)
        return np.sqrt(normE.T**2 + normEref**2 - 2 * (E.T @ Eref))


class aRMSE(BaseMetric):
    """Abundance Root Mean Squared Error."""
    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        A, Aref = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())


class SRE(BaseMetric):
    """Signal to Reconstruction Error."""
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))


def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """Return individual and global metric."""
    d = {}
    d["Overall"] = round(float(metric(X_hat, X_gt)), 4)
    if detail:
        for ii, label in enumerate(labels):
            if on_endmembers:
                x_gt, x_hat = X_gt[:, ii][:, None], X_hat[:, ii][:, None]
                d[label] = round(float(metric(x_hat, x_gt)), 4)
            else:
                d[label] = round(float(metric(X_hat[ii], X_gt[ii])), 4)
    log.info(f"{metric} => {d}")
    return d


# ==============================================================================
# ALIGNERS - Inlined from src/utils/aligners.py
# ==============================================================================

class BaseAligner:
    def __init__(self, Aref, criterion):
        self.Aref = Aref
        self.criterion = criterion
        self.P = None
        self.dists = None

    def fit(self, A):
        raise NotImplementedError

    def transform(self, A):
        assert self.P is not None, "Must be fitted first"
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]
        return self.P @ A

    def transform_endmembers(self, E):
        assert self.P is not None, "Must be fitted first"
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]
        return E @ self.P.T

    def fit_transform(self, A):
        self.fit(A)
        res = self.transform(A)
        return res

    def __repr__(self):
        msg = f"{self.__class__.__name__}_crit{self.criterion}"
        return msg


class HungarianAligner(BaseAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, A):
        # Computing distance matrix
        self.dists = self.criterion(A.T, self.Aref.T)

        # Initialization
        p = A.shape[0]
        P = np.zeros((p, p))

        m = Munkres()
        indices = m.compute(self.dists.tolist())
        for row, col in indices:
            P[row, col] = 1.0

        self.P = P.T


class AbundancesAligner(HungarianAligner):
    def __init__(self, **kwargs):
        super().__init__(
            criterion=MSE(),
            **kwargs,
        )


# ==============================================================================
# NOISE - Inlined from src/data/noise.py
# ==============================================================================

class AdditiveWhiteGaussianNoise:
    """Additive White Gaussian Noise generator."""
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y.
        """
        log.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        L, N = Y.shape
        log.info(f"Desired SNR => {self.SNR}")

        if self.SNR is None:
            sigmas = np.zeros(L)
        else:
            assert self.SNR > 0, "SNR must be strictly positive"
            # Uniform across bands
            sigmas = np.ones(L)
            # Normalization
            sigmas /= np.linalg.norm(sigmas)
            log.debug(f"Sigmas after normalization: {sigmas[0]}")
            # Compute sigma mean based on SNR
            num = np.sum(Y**2) / N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            log.debug(f"Sigma mean based on SNR: {sigmas_mean}")
            # Noise variance
            sigmas *= sigmas_mean
            log.debug(f"Final sigmas value: {sigmas[0]}")

        # Transform
        noise = np.diag(sigmas) @ np.random.randn(L, N)

        # Return additive noise
        return Y + noise


# ==============================================================================
# DATA UTILS - Inlined from src/data/utils.py
# ==============================================================================

def SVD_projection(Y, p):
    """SVD-based projection for denoising."""
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)


# ==============================================================================
# HSI DATA LOADER - Inlined from src/data/base.py
# ==============================================================================

class HSI:
    """Hyperspectral Image data loader."""
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
        EPS: float = 1e-10,
    ) -> None:
        self.EPS = EPS

        # Populate with Null data
        self.H = 0
        self.W = 0
        self.M = 0
        self.L = 0
        self.p = 0
        self.N = 0
        # arrays
        self.Y = np.zeros((self.L, self.N))
        self.E = np.zeros((self.L, self.p))
        self.A = np.zeros((self.p, self.N))
        self.D = np.zeros((self.L, self.M))
        self.labels = []
        self.index = []

        # Locate and check data file
        self.name = dataset
        filename = f"{self.name}.mat"
        path = os.path.join(SCRIPT_DIR, data_dir, filename)
        log.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path), f"Data file not found: {path}"

        # Open data file
        data = sio.loadmat(path)
        log.debug(f"Data keys: {data.keys()}")

        # Populate attributes based on data file values
        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(
                key, data[key].item() if key in INTEGER_VALUES else data[key]
            )

        if "N" not in data.keys():
            self.N = self.H * self.W

        # Check data
        assert self.N == self.H * self.W
        assert self.Y.shape == (self.L, self.N)

        self.has_dict = False
        if "D" in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.L, self.M)

        if "index" in data.keys():
            self.index = list(self.index.squeeze())

        # Create output figures folder
        self.figs_dir = os.path.join(SCRIPT_DIR, figs_dir)
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_data(self):
        return (
            self.Y,
            self.p,
            self.D,
        )

    def get_HSI_dimensions(self):
        return {
            "bands": self.L,
            "pixels": self.N,
            "lines": self.H,
            "samples": self.W,
            "atoms": self.M,
        }

    def get_img_shape(self):
        return (
            self.H,
            self.W,
        )

    def get_labels(self):
        return self.labels

    def get_index(self):
        return self.index

    def __repr__(self) -> str:
        msg = f"HSI => {self.name}\n"
        msg += "------------------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({self.labels}),\n"
        msg += f"{self.M} atoms\n"
        msg += f"GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n"
        return msg

    def plot_endmembers(self, E0=None, run=0):
        """Display endmembers."""
        title = f"{self.name} - endmembers" + " (GT)" if E0 is None else ""
        ylabel = "Reflectance"
        xlabel = "# Bands"
        E = np.copy(self.E) if E0 is None else np.copy(E0)

        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            plt.plot(E[:, pp], label=self.labels[pp])
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        suffix = "-GT" if E0 is None else f"-{run}"
        figname = f"{self.name}-endmembers{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

    def plot_abundances(self, A0=None, run=0):
        nrows, ncols = (1, self.p)
        title = f"{self.name} - abundances" + " (GT)" if A0 is None else ""

        A = np.copy(self.A) if A0 is None else np.copy(A0)
        A = A.reshape(self.p, self.H, self.W)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 4 * nrows),
        )
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(A[kk], vmin=0.0, vmax=1.0)
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                fig.colorbar(mappable, ax=curr_ax, location="right", shrink=0.5)

                kk += 1
                if kk == self.p:
                    break

        plt.suptitle(title)
        suffix = "-GT" if A0 is None else f"-{run}"
        figname = f"{self.name}-abundances{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()


class HSIWithGT(HSI):
    """HSI data loader with ground truth."""
    def __init__(
        self,
        dataset,
        data_dir,
        figs_dir,
        EPS=1e-10,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            figs_dir=figs_dir,
            EPS=EPS,
        )

        # Sanity check on ground truth
        assert self.E.shape == (self.L, self.p)
        assert self.A.shape == (self.p, self.N)

        try:
            assert len(self.labels) == self.p
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]
        except Exception:
            # Create numeroted labels
            self.labels = [f"#{ii}" for ii in range(self.p)]

        # Check physical constraints
        # Abundance Sum-to-One Constraint (ASC)
        assert np.allclose(
            self.A.sum(0),
            np.ones(self.N),
            rtol=1e-3,
            atol=1e-3,
        )
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -self.EPS)
        # Endmembers Non-negative Constraint (ENC)
        assert np.all(self.E >= -self.EPS)

    def get_GT(self):
        return (
            self.E,
            self.A,
        )

    def has_GT(self):
        return True


# ==============================================================================
# UNMIXING MODEL BASE - Inlined from src/model/base.py
# ==============================================================================

class UnmixingModel:
    """Base class for unmixing models."""
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"


class SemiSupervisedUnmixingModel(UnmixingModel):
    """Base class for semi-supervised unmixing models."""
    def __init__(self):
        super().__init__()

    def compute_abundances(self, Y, D, *args, **kwargs):
        raise NotImplementedError(f"Solver is not implemented for {self}")


# ==============================================================================
# SUnSAL ALGORITHM - Inlined from src/model/semisupervised/SUnSAL.py
# ==============================================================================

class SUnSAL(SemiSupervisedUnmixingModel):
    """
    SUnSAL: Sparse Unmixing by variable Splitting and Augmented Lagrangian.
    
    Source: https://github.com/etienne-monier/lib-unmixing/blob/master/unmixing.py
    """
    def __init__(
        self,
        AL_iters=1000,
        lambd=0.0,
        verbose=True,
        positivity=False,
        addone=False,
        tol=1e-4,
        x0=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.positivity = positivity
        self.addone = addone
        self.tol = tol
        self.x0 = x0

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        # Lambda for all pixels
        lambd = self.lambd * np.ones((M, N))

        # Compute mean norm
        norm_d = np.sqrt(np.mean(D**2))
        log.debug(f"norm D => {norm_d:.3e}")
        # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        log.debug(f"lambda value: {np.mean(lambd):.3e}")

        # Least squares
        if np.sum(lambd == 0) and not self.addone and not self.positivity:
            log.debug("Least Squares")
            Ahat = LA.pinv(D) @ Y
            return Ahat

        # Constrained Least Squares (sum(x) = 1)
        SMALL = 1e-12
        B = np.ones((1, M))
        a = np.ones((1, N))

        if np.sum(lambd == 0) and self.addone and not self.positivity:
            log.debug("Constrained Least Squares (sum(x) = 1)")
            F = D.T @ D
            # Test if F is invertible
            if LA.cond(F) > SMALL:
                # Compute the solution explicitly
                IF = LA.inv(F)
                Ahat = IF @ D.T @ Y - IF @ B.T @ (LA.inv(B @ IF @ B.T)) @ (
                    B @ IF @ D.T @ Y - a
                )
                return Ahat

        # Constants and initialization
        mu_AL = 0.01
        mu = 10 * np.mean(lambd) + mu_AL
        log.debug(f"mu initial value: {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
        x_aux = Aux @ a
        IF1 = IF - Aux @ B @ IF
        yy = D.T @ Y

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        z = x
        # Scaled Lagrange Multipliers
        d = 0 * z

        # AL iterations
        tol1 = np.sqrt(N * M) * self.tol
        tol2 = np.sqrt(N * M) * self.tol
        log.debug(f"tolerance => {tol1:.3e}")
        i = 1
        res_p = float("inf")
        res_d = float("inf")
        mu_changed = 0

        # Constrained Least Squares (CLS) X >= 0
        if np.sum(lambd == 0) and not self.addone:
            log.debug("Constrained Least Squares (x >= 0)")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF @ (yy + mu * (z + d))
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        log.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        log.debug(f"mu changed ({i}) => {mu}")
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Fully Constraint Least Squares
        elif np.sum(lambd == 0) and self.addone:
            log.debug("Fully Constrained Least Squares")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF1 @ (yy + mu * (z + d)) + x_aux
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        log.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Generic SUnSAL
        else:
            log.debug("Generic SUnSAL")
            def softthresh(x, th):
                return np.sign(x) * np.maximum(np.abs(x) - th, 0)

            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = softthresh(x - d, lambd / mu)
                # Test for positivity
                if self.positivity:
                    z = np.maximum(z, 0)

                # Test of Sum-to-one
                if self.addone:
                    x = IF1 @ (yy + mu * (z + d)) + x_aux
                else:
                    x = IF @ (yy + mu * (z + d))

                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        log.info(
                            f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}"
                        )

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        log.debug(f"mu changed ({i}) => {mu}")
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        Ahat = z
        tac = time.time()
        self.time = tac - tic
        log.info(self.print_time())
        return Ahat


# ==============================================================================
# ESTIMATE ARTIFACT - Inlined from src/data/base.py
# ==============================================================================

def save_estimates(Ehat, Ahat, H, W, output_dir):
    """Save estimates to .mat file."""
    data = {"E": Ehat, "A": Ahat.reshape(-1, H, W)}
    filepath = os.path.join(output_dir, "estimates.mat")
    sio.savemat(filepath, data)
    log.info(f"Estimates saved to {filepath}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    log.info("Semi-Supervised Unmixing (SUnSAL) - [START]...")

    # Load configuration
    cfg = load_config(CONFIG_PATH)
    log.debug(f"Config loaded from: {CONFIG_PATH}")

    # Set random seed
    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    log.info(f"Random seed set to: {seed}")

    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, cfg["figs_dir"])
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # Initialize noise generator
    noise = AdditiveWhiteGaussianNoise(SNR=cfg["SNR"])

    # Load HSI data
    hsi = HSIWithGT(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        figs_dir=cfg["figs_dir"],
        EPS=cfg["EPS"],
    )
    log.info(hsi)

    # Get data
    Y, p, D = hsi.get_data()
    log.info(f"Data loaded: Y shape={Y.shape}, p={p}, D shape={D.shape}")

    # Get image dimensions
    H, W = hsi.get_img_shape()
    log.info(f"Image dimensions: H={H}, W={W}")

    # Apply noise
    Y = noise.apply(Y)

    # L2 normalization
    if cfg.get("l2_normalization", False):
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        log.info("L2 normalization applied")

    # Apply SVD projection
    if cfg.get("projection", False):
        Y = SVD_projection(Y, p)
        log.info("SVD projection applied")

    # Build model
    model_cfg = cfg["model"]
    model = SUnSAL(
        AL_iters=model_cfg["AL_iters"],
        lambd=model_cfg["lambd"],
        verbose=model_cfg["verbose"],
        positivity=model_cfg["positivity"],
        addone=model_cfg["addone"],
        tol=model_cfg["tol"],
        x0=model_cfg["x0"],
    )
    log.info(f"Model: {model}")

    # Solve unmixing
    A_hat = model.compute_abundances(Y, D, p=p, H=H, W=W)
    log.info(f"Abundances estimated: shape={A_hat.shape}")

    # Dummy endmembers (semi-supervised doesn't estimate endmembers)
    E_hat = np.zeros((Y.shape[0], p))

    # Save estimates
    save_estimates(E_hat, A_hat, H, W, output_dir)

    # Metrics dictionary to save
    metrics_results = {}

    if hsi.has_GT():
        # Get ground truth
        _, A_gt = hsi.get_GT()
        log.info(f"Ground truth abundances: shape={A_gt.shape}")

        # Alignment
        if cfg.get("force_align", False):
            aligner = AbundancesAligner(Aref=A_gt)
            A1 = aligner.fit_transform(A_hat)
        else:
            index = hsi.get_index()
            A1 = A_hat[index]
        log.info(f"Aligned abundances: shape={A1.shape}")

        # Get labels
        labels = hsi.get_labels()
        log.info(f"Labels: {labels}")

        # Compute metrics
        sre_result = compute_metric(
            SRE(),
            A_gt,
            A1,
            labels,
            detail=False,
            on_endmembers=False,
        )
        metrics_results["SRE"] = sre_result

        armse_result = compute_metric(
            aRMSE(),
            A_gt,
            A1,
            labels,
            detail=True,
            on_endmembers=False,
        )
        metrics_results["aRMSE"] = armse_result

        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_results, f, indent=4)
        log.info(f"Metrics saved to {metrics_path}")

        # Plot results
        hsi.plot_abundances(A0=A1, run="estimated")

    log.info("Semi-Supervised Unmixing (SUnSAL) - [END]")


if __name__ == "__main__":
    main()
"""
