"""python
# ============================================================================
# STANDALONE UNMIXING SCRIPT - SUnCNN on DC1 Dataset
# ============================================================================
#
# ORIGINAL SCRIPT PATH:
#   /fs-computility-new/UPDZ02_sunhe/shared/jiayx/HySupp/unmixing.py
#   with command: python unmixing.py data=DC1 model=SUnCNN SNR=30
#
# DEPENDENCIES:
#   Input files:
#     - ./data_standalone/DC1.mat (hyperspectral data with ground truth)
#     - ./data_standalone/standalone_unmixing_SUnCNN_DC1.json (configuration)
#
#   Output files:
#     - ./exp_standalone_unmixing_SUnCNN_DC1/DC1-abundances-estimated.png (abundances plot)
#     - ./exp_standalone_unmixing_SUnCNN_DC1/estimates.mat (estimated E and A matrices)
#     - ./exp_standalone_unmixing_SUnCNN_DC1/metrics.json (evaluation metrics)
#
# REQUIRED PACKAGES (standard/pip-installable):
#   - numpy
#   - scipy
#   - matplotlib
#   - torch (PyTorch)
#   - tqdm
#   - munkres (Hungarian algorithm implementation)
#
# ITERATION COUNT CHANGES:
#   - Original niters: 20000
#   - Reduced niters: 2000 (10x speedup)
#
# INLINED CODE:
#   All high-level package calls from src.* and mlxp have been inlined:
#     - src.data.base.HSIWithGT -> inlined as HSIWithGT class
#     - src.data.noise.AdditiveWhiteGaussianNoise -> inlined
#     - src.model.semisupervised.SUnCNN -> inlined as SUnCNN class
#     - src.model.semisupervised.base.SemiSupervisedUnmixingModel -> inlined
#     - src.model.base.UnmixingModel -> inlined
#     - src.utils.aligners.AbundancesAligner -> inlined
#     - src.utils.metrics.* -> inlined (SRE, aRMSE, MSE)
#     - src.data.utils.SVD_projection -> inlined (not used with current config)
#     - mlxp.* -> removed (experiment tracking not needed for standalone)
#
# NOTES:
#   - No task-specific imports from src.* or mlxp are used
#   - All paths are relative to the script location
#   - Plots are saved as PNG files (not displayed)
#   - GPU usage forced to single GPU (CUDA_VISIBLE_DEVICES="0")
#
# ============================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import time
import logging
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)
log = logging.getLogger(__name__)

# ============================================================================
# GET SCRIPT DIRECTORY FOR RELATIVE PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(*args):
    """Convert relative path to absolute path based on script location."""
    return os.path.join(SCRIPT_DIR, *args)

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
CONFIG_PATH = rel_path("data_standalone", "standalone_unmixing_SUnCNN_DC1.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

EPS = CONFIG["EPS"]

# ============================================================================
# INLINED: src.data.base (HSI and HSIWithGT classes)
# ============================================================================

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")

class HSI:
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
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
        path = rel_path(data_dir, filename)
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
        self.figs_dir = rel_path(figs_dir)
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

    def plot_endmembers(self, E0=None, suffix="-GT"):
        """
        Display endmembers
        """
        title = f"{self.name} - endmembers" + (" (GT)" if E0 is None else " (Estimated)")
        ylabel = "Reflectance"
        xlabel = "# Bands"
        E = np.copy(self.E) if E0 is None else np.copy(E0)

        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            label = self.labels[pp] if pp < len(self.labels) else f"#{pp}"
            plt.plot(E[:, pp], label=label)
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        figname = f"{self.name}-endmembers{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()
        log.info(f"Saved endmembers plot to {os.path.join(self.figs_dir, figname)}")

    def plot_abundances(self, A0=None, suffix="-GT"):
        nrows, ncols = (1, self.p)
        title = f"{self.name} - abundances" + (" (GT)" if A0 is None else " (Estimated)")

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
                    curr_ax = ax[jj] if ncols > 1 else ax
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(A[kk], vmin=0.0, vmax=1.0)
                label = self.labels[kk] if kk < len(self.labels) else f"#{kk}"
                curr_ax.set_title(f"{label}")
                curr_ax.axis("off")
                fig.colorbar(mappable, ax=curr_ax, location="right", shrink=0.5)

                kk += 1
                if kk == self.p:
                    break

        plt.suptitle(title)
        figname = f"{self.name}-abundances{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()
        log.info(f"Saved abundances plot to {os.path.join(self.figs_dir, figname)}")


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


# ============================================================================
# INLINED: src.data.noise.AdditiveWhiteGaussianNoise
# ============================================================================

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


# ============================================================================
# INLINED: src.data.utils.SVD_projection
# ============================================================================

def SVD_projection(Y, p):
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)


# ============================================================================
# INLINED: src.model.base.UnmixingModel
# ============================================================================

class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"


# ============================================================================
# INLINED: src.model.semisupervised.base.SemiSupervisedUnmixingModel
# ============================================================================

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


# ============================================================================
# INLINED: src.model.semisupervised.SUnCNN
# ============================================================================

class SUnCNN(nn.Module, SemiSupervisedUnmixingModel):
    def __init__(
        self,
        niters=4000,
        lr=0.001,
        exp_weight=0.99,
        noisy_input=True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.kernel_sizes = [3, 3, 3, 1, 1, 1]
        self.strides = [2, 1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.noisy_input = noisy_input

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        # MiSiCNet-like architecture
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[0]),
            nn.Conv2d(self.L, 256, self.kernel_sizes[0], stride=self.strides[0]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[1]),
            nn.Conv2d(256, 256, self.kernel_sizes[1], stride=self.strides[1]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.L, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(260),
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, 256, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[4]),
            nn.Conv2d(256, self.M, self.kernel_sizes[4], stride=self.strides[4]),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.upsample(self.layer2(self.layer1(x)))
        xskip = self.layerskip(x)
        xcat = self.custom_cat(x1, xskip)
        out = self.softmax(self.layer5(self.layer4(self.layer3(xcat))))
        return out

    @staticmethod
    def custom_cat(x1, xskip):
        inputs = [x1, xskip]
        inputs_shape2 = [x.shape[2] for x in inputs]
        inputs_shape3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shape2) == min(inputs_shape2)) and np.all(
            np.array(inputs_shape3) == min(inputs_shape3)
        ):
            inputs_ = inputs
        else:

            inputs_ = []

            target_shape2 = min(inputs_shape2)
            target_shape3 = min(inputs_shape3)

            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=1)

    def compute_abundances(self, Y, D, H, W, seed=0, *args, **kwargs):
        tic = time.time()
        log.debug("Solving started...")

        # Hyperparameters
        self.L, self.N = Y.shape
        LD, self.M = D.shape
        assert self.L == LD, "Inconsistent number of channels for Y and D"
        self.H, self.W = H, W

        self.init_architecture(seed=seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        n_channels, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, n_channels, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)
        D = torch.Tensor(D).to(self.device)

        noisy_input = torch.rand_like(Y) if self.noisy_input else Y

        progress = tqdm(range(self.niters))
        for ii in progress:
            optimizer.zero_grad()

            abund = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * self.exp_weight + abund.detach() * (
                    1 - self.exp_weight
                )

            # Reshape data
            loss = F.mse_loss(Y.view(-1, h * w), D @ abund.view(-1, h * w))

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            loss.backward()
            optimizer.step()

        A = out_avg.cpu().numpy().reshape(-1, h * w)
        self.time = time.time() - tic
        log.info(self.print_time())

        return A


# ============================================================================
# INLINED: src.utils.metrics (MSE, SRE, aRMSE)
# ============================================================================

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


# ============================================================================
# INLINED: src.utils.aligners (BaseAligner, HungarianAligner, AbundancesAligner)
# ============================================================================

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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    log.info("=" * 60)
    log.info("Standalone Semi-Supervised Unmixing - SUnCNN on DC1")
    log.info("=" * 60)
    
    # Set seed
    seed = CONFIG["seed"]
    set_seeds(seed)
    log.info(f"Random seed set to: {seed}")
    
    # Initialize noise
    noise = AdditiveWhiteGaussianNoise(SNR=CONFIG["SNR"])
    
    # Load HSI data
    hsi = HSIWithGT(
        dataset=CONFIG["dataset"],
        data_dir=CONFIG["data_dir"],
        figs_dir=CONFIG["figs_dir"],
    )
    log.info(hsi)
    
    # Get data
    Y, p, D = hsi.get_data()
    # Get image dimensions
    H, W = hsi.get_img_shape()
    
    # Apply noise
    Y = noise.apply(Y)
    
    # L2 normalization (if configured)
    if CONFIG["l2_normalization"]:
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    
    # Apply SVD projection (if configured)
    if CONFIG["projection"]:
        Y = SVD_projection(Y, p)
    
    # Build model with reduced iterations for speedup (original: 20000, reduced: 2000)
    model_config = CONFIG["model"]
    reduced_niters = max(1, model_config["niters"] // 10)  # 10x speedup
    log.info(f"Using reduced niters: {reduced_niters} (original: {model_config['niters']})")
    
    model = SUnCNN(
        niters=reduced_niters,
        lr=model_config["lr"],
        exp_weight=model_config["exp_weight"],
        noisy_input=model_config["noisy_input"],
    )
    
    log.info("Semi-Supervised Unmixing - [START]")
    
    # Solve unmixing
    A_hat = model.compute_abundances(
        Y,
        D,
        p=p,
        H=H,
        W=W,
        seed=seed,
    )
    
    # For semi-supervised, E_hat is zeros (no endmember estimation)
    E_hat = np.zeros((Y.shape[0], p))
    
    # Save estimates
    estimates_path = rel_path(CONFIG["figs_dir"], "estimates.mat")
    estimates_data = {"E": E_hat, "A": A_hat.reshape(-1, H, W)}
    sio.savemat(estimates_path, estimates_data)
    log.info(f"Saved estimates to {estimates_path}")
    
    # Initialize metrics dictionary
    all_metrics = {}
    
    if hsi.has_GT():
        # Get ground truth
        _, A_gt = hsi.get_GT()
        
        # Alignment based on force_align config or use index
        if CONFIG["force_align"]:
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
        metrics_path = rel_path(CONFIG["figs_dir"], "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f"Saved metrics to {metrics_path}")
        
        # Plot estimated abundances
        hsi.plot_abundances(A0=A1, suffix="-estimated")
        
        # Also plot ground truth for comparison
        hsi.plot_abundances(A0=None, suffix="-GT")
    
    log.info("Semi-Supervised Unmixing - [END]")
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for metric_name, metric_values in all_metrics.items():
        log.info(f"{metric_name}: {metric_values}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
"""
