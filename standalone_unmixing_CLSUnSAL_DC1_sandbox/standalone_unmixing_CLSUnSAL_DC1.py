# ==============================================================================
# STANDALONE UNMIXING SCRIPT - CLSUnSAL on DC1 Dataset
# ==============================================================================
#
# ORIGINAL SCRIPT PATH:
#   /fs-computility-new/UPDZ02_sunhe/shared/jiayx/HySupp/unmixing.py
#   Command: python unmixing.py data=DC1 model=CLSUnSAL SNR=30
#
# DEPENDENCIES:
#   Inputs:
#     - ./data_standalone/DC1.mat (HSI data with ground truth)
#     - ./data_standalone/standalone_unmixing_CLSUnSAL_DC1.json (configuration)
#   Outputs:
#     - ./exp_standalone_unmixing_CLSUnSAL_DC1/estimates.mat (estimated abundances)
#     - ./exp_standalone_unmixing_CLSUnSAL_DC1/metrics.json (SRE and aRMSE metrics)
#     - ./exp_standalone_unmixing_CLSUnSAL_DC1/DC1-abundances-estimated.png (abundance maps)
#     - ./exp_standalone_unmixing_CLSUnSAL_DC1/DC1-abundances-GT.png (ground truth abundance maps)
#
# ITERATION COUNT CHANGES:
#   None. Original AL_iters=1000 is preserved.
#
# INLINED PACKAGES:
#   All high-level package calls (mlxp, src.*) have been inlined.
#   The following modules were inlined:
#     - src.model.semisupervised.CLSUnSAL (CLSUnSAL algorithm)
#     - src.model.semisupervised.base (SemiSupervisedUnmixingModel)
#     - src.model.base (UnmixingModel)
#     - src.data.base (HSI, HSIWithGT, Estimate)
#     - src.data.noise (AdditiveWhiteGaussianNoise)
#     - src.data.utils (SVD_projection)
#     - src.utils.metrics (SRE, aRMSE, compute_metric)
#     - src.utils.aligners (AbundancesAligner, HungarianAligner)
#     - src.semisupervised (main entry point)
#
# ==============================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import logging
import logging.config
import time

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data_standalone")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "exp_standalone_unmixing_CLSUnSAL_DC1")
CONFIG_FILE = os.path.join(DATA_DIR, "standalone_unmixing_CLSUnSAL_DC1.json")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load configuration
with open(CONFIG_FILE, "r") as f:
    CONFIG = json.load(f)

# Global constant
EPS = CONFIG["EPS"]

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging_config = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            "datefmt": "%d-%b-%y %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"]
    }
}

logging.config.dictConfig(logging_config)
log = logging.getLogger(__name__)

# ==============================================================================
# SEED SETTING
# ==============================================================================

def set_seeds(seed):
    np.random.seed(seed)

set_seeds(CONFIG["seed"])

# ==============================================================================
# DATA CLASSES (inlined from src.data.base)
# ==============================================================================

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")


class HSI:
    def __init__(
        self,
        dataset: str,
        data_dir: str,
        figs_dir: str,
    ) -> None:

        # Populate with Null data
        # integers
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
        path = os.path.join(data_dir, filename)
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
        self.figs_dir = figs_dir
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
        """
        Display endmembers
        """
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
    def __init__(
        self,
        dataset,
        data_dir,
        figs_dir,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            figs_dir=figs_dir,
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
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint (ENC)
        assert np.all(self.E >= -EPS)

    def get_GT(self):
        return (
            self.E,
            self.A,
        )

    def has_GT(self):
        return True


# ==============================================================================
# NOISE CLASS (inlined from src.data.noise)
# ==============================================================================

class AdditiveWhiteGaussianNoise:
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y
        """
        log.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        L, N = Y.shape
        log.info(f"Desired SNR => {self.SNR}")

        #######
        # Fit #
        #######
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

        #############
        # Transform #
        #############
        noise = np.diag(sigmas) @ np.random.randn(L, N)

        # Return additive noise
        return Y + noise


# ==============================================================================
# DATA UTILITIES (inlined from src.data.utils)
# ==============================================================================

def SVD_projection(Y, p):
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)


# ==============================================================================
# METRICS (inlined from src.utils.metrics)
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
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.sqrt(normE.T**2 + normEref**2 - 2 * (E.T @ Eref))


class aRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        A, Aref = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())


class SRE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))


def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """
    Return individual and global metric
    """
    d = {}
    d["Overall"] = round(metric(X_hat, X_gt), 4)
    if detail:
        for ii, label in enumerate(labels):
            if on_endmembers:
                x_gt, x_hat = X_gt[:, ii][:, None], X_hat[:, ii][:, None]
                d[label] = round(metric(x_hat, x_gt), 4)
            else:
                d[label] = round(metric(X_hat[ii], X_gt[ii]), 4)

    log.info(f"{metric} => {d}")
    return d


# ==============================================================================
# ALIGNERS (inlined from src.utils.aligners)
# ==============================================================================

try:
    from munkres import Munkres
    HAS_MUNKRES = True
except ImportError:
    HAS_MUNKRES = False
    log.warning("munkres package not available. AbundancesAligner will not work.")


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
        if not HAS_MUNKRES:
            raise ImportError("munkres package required for HungarianAligner")

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
# MODEL BASE CLASSES (inlined from src.model.base and src.model.semisupervised.base)
# ==============================================================================

class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"


class SemiSupervisedUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_abundances(
        self,
        Y,
        D,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(f"Solver is not implemented for {self}")


# ==============================================================================
# CLSUnSAL MODEL (inlined from src.model.semisupervised.CLSUnSAL)
# ==============================================================================

class CLSUnSAL(SemiSupervisedUnmixingModel):
    """
    CLSUnSAL - Collaborative Sparse Unmixing via Split Augmented Lagrangian
    
    Source: https://github.com/etienne-monier/lib-unmixing/blob/master/unmixing.py
    """
    def __init__(
        self,
        AL_iters=1000,
        lambd=0.0,
        verbose=True,
        tol=1e-4,
        mu=0.1,
        x0=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.tol = tol
        self.x0 = x0
        self.mu = mu

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        norm_d = np.sqrt(np.mean(D**2))
        log.debug(f"Norm D => {norm_d:.3e}")
        # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        log.debug(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        mu = self.mu
        log.debug(f"Mu initial value => {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        AA = LA.inv(D.T @ D + 2 * np.eye(M))

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        u = x
        v1 = D @ x
        v2 = x
        v3 = x
        # # Scaled Lagrange Multipliers
        d1 = v1
        d2 = v2
        d3 = v3

        # # AL iterations
        tol = np.sqrt(N * M) * self.tol
        log.debug(f"Tolerance => {tol:.3e}")
        k = 1
        res_p = float("inf")
        res_d = float("inf")

        while (k <= self.AL_iters) and ((np.abs(res_p) > tol) or (np.abs(res_d) > tol)):
            # Save u to be used later
            if k % 10 == 1:
                u0 = u

            # Minimize w.r.t. u
            # NOTE Legacy (might be faster than solving linear system)
            u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)

            # Minimize w.r.t. v1
            v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

            # Minimize w.r.t. v2
            def current_fn(b):
                return self.vect_soft_thresh(b, lambd / mu)
            v2 = np.apply_along_axis(current_fn, axis=1, arr=u - d2)

            # Minimize w.r.t. v3
            v3 = np.maximum(u - d3, 0)

            # Lagrange multipliers update
            d1 = d1 - D @ u + v1
            d2 = d2 - u + v2
            d3 = d3 - u + v3

            # Update mu to keep primal and dual residuals within a factor of 10
            if k % 10 == 1:
                # primal residual
                res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                # dual residual
                res_d = mu * LA.norm(u - u0)
                if self.verbose:
                    log.info(
                        f"k = {k}, res_p = {res_p:.3e}, res_d = {res_d:.3e}, mu = {mu:.3e}"
                    )

                # Update mu
                if res_p > 10 * res_d:
                    mu = mu * 2

                if res_d > 10 * res_p:
                    mu = mu / 2

            k += 1

        Ahat = v3
        self.time = time.time() - tic
        log.info(self.print_time())
        return Ahat

    @staticmethod
    def vect_soft_thresh(b, t):
        max_b = np.maximum(LA.norm(b) - t, 0)
        ret = b * (max_b) / (max_b + t)
        return ret


# ==============================================================================
# MAIN FUNCTION (inlined and adapted from src.semisupervised)
# ==============================================================================

def main():
    log.info("Semi-Supervised Unmixing - [START]...")
    
    # Configuration
    cfg = CONFIG
    
    # Get noise
    noise = AdditiveWhiteGaussianNoise(SNR=cfg["SNR"])
    
    # Get HSI
    hsi = HSIWithGT(
        dataset=cfg["dataset"],
        data_dir=DATA_DIR,
        figs_dir=OUTPUT_DIR,
    )
    
    # Print HSI information
    log.info(hsi)
    
    # Get data
    Y, p, D = hsi.get_data()
    
    # Get image dimensions
    H, W = hsi.get_img_shape()
    
    # Apply noise
    Y = noise.apply(Y)
    
    # L2 normalization
    if cfg["l2_normalization"]:
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
    
    # Apply SVD projection
    if cfg["projection"]:
        Y = SVD_projection(Y, p)
    
    # Build model
    model_cfg = cfg["model"]
    model = CLSUnSAL(
        AL_iters=model_cfg["AL_iters"],
        lambd=model_cfg["lambd"],
        verbose=model_cfg["verbose"],
        tol=model_cfg["tol"],
        mu=model_cfg["mu"],
        x0=model_cfg["x0"],
    )
    
    # Solve unmixing
    A_hat = model.compute_abundances(Y, D, p=p, H=H, W=W)
    
    # Prepare estimates for saving
    E_hat = np.zeros((Y.shape[0], p))
    
    # Save estimates to .mat file
    estimates_data = {"E": E_hat, "A": A_hat.reshape(-1, H, W)}
    estimates_path = os.path.join(OUTPUT_DIR, "estimates.mat")
    sio.savemat(estimates_path, estimates_data)
    log.info(f"Saved estimates to {estimates_path}")
    
    # Metrics dictionary for saving
    all_metrics = {}
    
    if hsi.has_GT():
        # Get ground truth
        _, A_gt = hsi.get_GT()
        
        # Select only the first relevant components
        if cfg["force_align"]:
            aligner = AbundancesAligner(Aref=A_gt)
            A1 = aligner.fit_transform(A_hat)
        else:
            index = hsi.get_index()
            A1 = A_hat[index]
        
        # Get labels
        labels = hsi.get_labels()
        
        # Compute metrics
        sre_metrics = compute_metric(
            SRE(),
            A_gt,
            A1,
            labels,
            detail=False,
            on_endmembers=False,
        )
        all_metrics["SRE"] = sre_metrics
        
        armse_metrics = compute_metric(
            aRMSE(),
            A_gt,
            A1,
            labels,
            detail=True,
            on_endmembers=False,
        )
        all_metrics["aRMSE"] = armse_metrics
        
        # Save metrics to JSON
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f"Saved metrics to {metrics_path}")
        
        # Plot ground truth abundances
        hsi.plot_abundances(A0=None, run=0)
        # Rename the GT plot
        gt_plot_src = os.path.join(OUTPUT_DIR, f"{cfg['dataset']}-abundances-GT.png")
        gt_plot_dst = os.path.join(OUTPUT_DIR, f"{cfg['dataset']}-abundances-GT.png")
        if os.path.exists(gt_plot_src) and gt_plot_src != gt_plot_dst:
            os.rename(gt_plot_src, gt_plot_dst)
        
        # Plot estimated abundances
        hsi.plot_abundances(A0=A1, run="estimated")
    
    log.info("Semi-Supervised Unmixing - [END]")
    
    # Print final summary
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"Dataset: {cfg['dataset']}")
    log.info(f"Model: CLSUnSAL")
    log.info(f"SNR: {cfg['SNR']} dB")
    if "SRE" in all_metrics:
        log.info(f"SRE: {all_metrics['SRE']['Overall']:.4f} dB")
    if "aRMSE" in all_metrics:
        log.info(f"aRMSE: {all_metrics['aRMSE']['Overall']:.4f} %")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
