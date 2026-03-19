# ==============================================================================
# STANDALONE SCRIPT DOCUMENTATION
# ==============================================================================
# Original script path: /fs-computility-new/UPDZ02_sunhe/shared/jiayx/HySupp/unmixing.py
# Command: conda activate /fs-computility-new/UPDZ02_sunhe/shared/hysupp && python unmixing.py data=DC1 model=EDAA SNR=30
#
# DEPENDENCIES:
# - Inputs:
#   - ./data_standalone/DC1.mat (hyperspectral image data)
#   - ./data_standalone/standalone_unmixing_EDAA_DC1.json (configuration parameters)
# - Outputs:
#   - ./exp_standalone/estimates.mat (estimated endmembers and abundances)
#   - ./exp_standalone/metrics.json (SRE, aRMSE, SAD, eRMSE metrics)
#   - ./exp_standalone/DC1-endmembers.png (endmembers plot)
#   - ./exp_standalone/DC1-abundances.png (abundances plot)
#
# ITERATION COUNT CHANGES FOR ~10x SPEEDUP:
# - EDAA.T: 100 -> 10 (outer iterations)
# - EDAA.M: 50 -> 5 (number of random initializations)
# - Original values preserved in JSON under "original_EDAA"
#
# CONFIRMATION:
# - All high-level package calls (mlxp, src.*) have been inlined
# - No project-specific imports remain
# ==============================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import logging
import time
from pathlib import Path

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from munkres import Munkres

# ==============================================================================
# Setup logging
# ==============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)
log = logging.getLogger(__name__)

# ==============================================================================
# Resolve script directory for relative paths
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_standalone"
OUTPUT_DIR = SCRIPT_DIR / "exp_standalone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Load configuration from JSON
# ==============================================================================
CONFIG_PATH = DATA_DIR / "standalone_unmixing_EDAA_DC1.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

EPS = CONFIG.get("EPS", 1e-10)

# ==============================================================================
# INLINED: src.data.noise.AdditiveWhiteGaussianNoise
# ==============================================================================
class AdditiveWhiteGaussianNoise:
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        log.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        L, N = Y.shape
        log.info(f"Desired SNR => {self.SNR}")

        if self.SNR is None:
            sigmas = np.zeros(L)
        else:
            assert self.SNR > 0, "SNR must be strictly positive"
            sigmas = np.ones(L)
            sigmas /= np.linalg.norm(sigmas)
            log.debug(f"Sigmas after normalization: {sigmas[0]}")
            num = np.sum(Y**2) / N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            log.debug(f"Sigma mean based on SNR: {sigmas_mean}")
            sigmas *= sigmas_mean
            log.debug(f"Final sigmas value: {sigmas[0]}")

        noise = np.diag(sigmas) @ np.random.randn(L, N)
        return Y + noise

# ==============================================================================
# INLINED: src.data.utils.SVD_projection
# ==============================================================================
def SVD_projection(Y, p):
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)

# ==============================================================================
# INLINED: src.data.base.HSIWithGT
# ==============================================================================
INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")

class HSI:
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
    ) -> None:
        self.H = 0
        self.W = 0
        self.M = 0
        self.L = 0
        self.p = 0
        self.N = 0
        self.Y = np.zeros((self.L, self.N))
        self.E = np.zeros((self.L, self.p))
        self.A = np.zeros((self.p, self.N))
        self.D = np.zeros((self.L, self.M))
        self.labels = []
        self.index = []

        self.name = dataset
        filename = f"{self.name}.mat"
        path = os.path.join(data_dir, filename)
        log.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path), f"Data file not found: {path}"

        data = sio.loadmat(path)
        log.debug(f"Data keys: {data.keys()}")

        for key in filter(lambda k: not k.startswith("__"), data.keys()):
            self.__setattr__(
                key, data[key].item() if key in INTEGER_VALUES else data[key]
            )

        if "N" not in data.keys():
            self.N = self.H * self.W

        assert self.N == self.H * self.W
        assert self.Y.shape == (self.L, self.N)

        self.has_dict = False
        if "D" in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.L, self.M)

        if "index" in data.keys():
            self.index = list(self.index.squeeze())

        self.figs_dir = figs_dir
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_data(self):
        return (self.Y, self.p, self.D)

    def get_img_shape(self):
        return (self.H, self.W)

    def get_labels(self):
        return self.labels

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
        title = f"{self.name} - endmembers" + (" (GT)" if E0 is None else "")
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

        suffix = "-GT" if E0 is None else f"-{run}"
        figname = f"{self.name}-endmembers{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

    def plot_abundances(self, A0=None, run=0):
        nrows, ncols = (1, self.p)
        title = f"{self.name} - abundances" + (" (GT)" if A0 is None else "")

        A = np.copy(self.A) if A0 is None else np.copy(A0)
        A = A.reshape(self.p, self.H, self.W)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
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
        suffix = "-GT" if A0 is None else f"-{run}"
        figname = f"{self.name}-abundances{suffix}.png"
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()


class HSIWithGT(HSI):
    def __init__(self, dataset, data_dir, figs_dir):
        super().__init__(dataset=dataset, data_dir=data_dir, figs_dir=figs_dir)

        assert self.E.shape == (self.L, self.p)
        assert self.A.shape == (self.p, self.N)

        try:
            assert len(self.labels) == self.p
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]
        except Exception:
            self.labels = [f"#{ii}" for ii in range(self.p)]

        assert np.allclose(self.A.sum(0), np.ones(self.N), rtol=1e-3, atol=1e-3)
        assert np.all(self.A >= -EPS)
        assert np.all(self.E >= -EPS)

    def get_GT(self):
        return (self.E, self.A)

    def has_GT(self):
        return True

# ==============================================================================
# INLINED: src.model.base.UnmixingModel
# ==============================================================================
class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def print_time(self):
        return f"{self} took {self.time:.2f}s"

# ==============================================================================
# INLINED: src.model.blind.base.BlindUnmixingModel
# ==============================================================================
class BlindUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):
        raise NotImplementedError(f"Solver is not implemented for {self}")

# ==============================================================================
# INLINED: src.model.blind.AA.EDAA
# ==============================================================================
class EDAA(BlindUnmixingModel):
    def __init__(self, T=100, K1=5, K2=5, M=50, normalize=True, *args, **kwargs):
        super().__init__()
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M
        self.normalize = normalize
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        best_E = None
        best_A = None
        _, N = Y.shape

        def loss(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def residual_l1(a, b):
            return (Y - (Y @ b) @ a).abs().sum()

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b):
            return F.softmax(torch.log(a) + b, dim=0)

        def computeLA(b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        def max_correl(e):
            return np.max(np.corrcoef(e.T) - np.eye(p))

        results = {}
        tic = time.time()

        Y = torch.Tensor(Y)

        for m in tqdm(range(self.M)):
            torch.manual_seed(m + seed)
            generator = np.random.RandomState(m + seed)

            with torch.no_grad():
                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                A = (1 / p) * torch.ones((p, N))

                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)

                factA = 2 ** generator.randint(-3, 4)

                self.etaA = factA / computeLA(B)
                self.etaB = self.etaA * ((p / N) ** 0.5)

                for _ in range(self.T):
                    for _ in range(self.K1):
                        A = update(A, -self.etaA * grad_A(A, B))
                    for _ in range(self.K2):
                        B = update(B, -self.etaB * grad_B(A, B))

                fit_m = residual_l1(A, B).item()
                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                B = B.cpu().numpy()
                Rm = max_correl(E)
                results[m] = {
                    "Rm": Rm,
                    "Em": E,
                    "Am": A,
                    "Bm": B,
                    "fit_m": fit_m,
                    "factA": factA,
                }

        min_fit_l1 = np.min([v["fit_m"] for v in results.values()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]["fit_m"]
            return (abs(val - min_fit_l1) / abs(val)) < tol

        sorted_indices = sorted(
            filter(fit_l1_cutoff, results),
            key=lambda x: results[x]["Rm"],
        )

        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]

        best_E = best_result["Em"]
        best_A = best_result["Am"]
        self.B = best_result["Bm"]

        toc = time.time()
        self.time = toc - tic
        log.info(self.print_time())

        return best_E, best_A

# ==============================================================================
# INLINED: src.utils.metrics
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


class SpectralAngleDistance(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)
        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)
        tmp = (E / normE).T @ (Eref / normEref)
        ret = np.minimum(tmp, 1.0)
        return np.arccos(ret)


class SADDegrees(SpectralAngleDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        tmp = super().__call__(E, Eref)
        return (np.diag(tmp) * (180 / np.pi)).mean()


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


class eRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)
        return 100 * np.sqrt(((E - Eref) ** 2).mean())


class SRE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))


def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
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
# INLINED: src.utils.aligners
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


class HungarianAligner(BaseAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, A):
        self.dists = self.criterion(A.T, self.Aref.T)
        p = A.shape[0]
        P = np.zeros((p, p))
        m = Munkres()
        indices = m.compute(self.dists.tolist())
        for row, col in indices:
            P[row, col] = 1.0
        self.P = P.T


class AbundancesAligner(HungarianAligner):
    def __init__(self, **kwargs):
        super().__init__(criterion=MSE(), **kwargs)

# ==============================================================================
# MAIN FUNCTION (INLINED from src.blind.main)
# ==============================================================================
def main():
    log.info("Blind Unmixing - [START]")

    # Set seeds
    seed = CONFIG.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get noise
    SNR = CONFIG.get("SNR", None)
    noise = AdditiveWhiteGaussianNoise(SNR=SNR)

    # Get HSI
    hsi = HSIWithGT(
        dataset=CONFIG["data"]["dataset"],
        data_dir=str(DATA_DIR),
        figs_dir=str(OUTPUT_DIR),
    )
    log.info(hsi)

    # Get data
    Y, p, _ = hsi.get_data()
    H, W = hsi.get_img_shape()

    # Apply noise
    Y = noise.apply(Y)

    # L2 normalization
    if CONFIG.get("l2_normalization", False):
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY

    # Apply SVD projection
    if CONFIG.get("projection", False):
        Y = SVD_projection(Y, p)

    # Build model (EDAA with reduced iterations for speedup)
    edaa_cfg = CONFIG["EDAA"]
    model = EDAA(
        T=edaa_cfg["T"],
        K1=edaa_cfg["K1"],
        K2=edaa_cfg["K2"],
        M=edaa_cfg["M"],
        normalize=edaa_cfg.get("normalize", True),
    )

    # Solve unmixing
    E_hat, A_hat = model.compute_endmembers_and_abundances(Y, p, H=H, W=W)

    # Save estimates
    estimates_path = OUTPUT_DIR / "estimates.mat"
    sio.savemat(str(estimates_path), {"E": E_hat, "A": A_hat.reshape(-1, H, W)})
    log.info(f"Estimates saved to {estimates_path}")

    # Metrics dictionary to save
    all_metrics = {}

    if hsi.has_GT():
        # Get ground truth
        E_gt, A_gt = hsi.get_GT()

        # Align based on abundances
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)

        # Get labels
        labels = hsi.get_labels()

        # Compute metrics
        all_metrics["SRE"] = compute_metric(
            SRE(), A_gt, A1, labels, detail=False, on_endmembers=False
        )
        all_metrics["aRMSE"] = compute_metric(
            aRMSE(), A_gt, A1, labels, detail=True, on_endmembers=False
        )
        all_metrics["SAD"] = compute_metric(
            SADDegrees(), E_gt, E1, labels, detail=True, on_endmembers=True
        )
        all_metrics["eRMSE"] = compute_metric(
            eRMSE(), E_gt, E1, labels, detail=True, on_endmembers=True
        )

        # Save metrics
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f"Metrics saved to {metrics_path}")

        # Plot endmembers (estimated, aligned)
        hsi.E = E1  # Set for plotting
        hsi.plot_endmembers(E0=E1, run="estimated")

        # Plot abundances (estimated, aligned)
        hsi.plot_abundances(A0=A1, run="estimated")

    log.info("Blind Unmixing - [END]")


if __name__ == "__main__":
    main()
