"""python
# ============================================================================
# STANDALONE UNMIXING SCRIPT - PGMSU on DC1 Dataset
# ============================================================================
#
# Original script path:
#     /fs-computility-new/UPDZ02_sunhe/shared/jiayx/HySupp/unmixing.py
#     with command: python unmixing.py data=DC1 model=PGMSU SNR=30
#
# Dependencies - Inputs:
#     - ./data_standalone/DC1.mat (hyperspectral data)
#     - ./data_standalone/standalone_unmixing_PGMSU_DC1.json (config)
#
# Dependencies - Outputs:
#     - ./exp_standalone_unmixing_PGMSU_DC1/estimates.mat (estimated E and A)
#     - ./exp_standalone_unmixing_PGMSU_DC1/metrics.json (evaluation metrics)
#     - ./exp_standalone_unmixing_PGMSU_DC1/DC1-endmembers-estimated.png
#     - ./exp_standalone_unmixing_PGMSU_DC1/DC1-abundances-estimated.png
#
# Iteration count changes:
#     - None. Original epochs = 200 (unchanged).
#
# Confirmation:
#     - All high-level package calls (mlxp, src.*) have been inlined.
#     - No project-specific imports remain.
#     - This script is fully self-contained.
#
# ============================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import logging
import time

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from munkres import Munkres


# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# DATA LOADING CLASSES (inlined from src/data/base.py)
# ============================================================================

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")


class HSI:
    """Base HSI data class."""
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
        EPS: float = 1e-10,
    ) -> None:
        # Populate with Null data
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
        self.EPS = EPS

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
        for key in filter(lambda k: not k.startswith("__"), data.keys()):
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
        """Display endmembers."""
        title = f"{self.name} - endmembers" + (" (GT)" if E0 is None else "")
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
        title = f"{self.name} - abundances" + (" (GT)" if A0 is None else "")

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
    """HSI with Ground Truth."""
    def __init__(self, dataset, data_dir, figs_dir, EPS=1e-10):
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
        return (self.E, self.A)

    def has_GT(self):
        return True


# ============================================================================
# NOISE CLASS (inlined from src/data/noise.py)
# ============================================================================

class AdditiveWhiteGaussianNoise:
    """Additive White Gaussian Noise."""
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

        # Fit
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

        return Y + noise


# ============================================================================
# VCA EXTRACTOR (inlined from src/model/extractors.py)
# ============================================================================

class VCA:
    """Vertex Component Analysis."""
    def __init__(self):
        self.seed = None
        self.indices = None

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm
        under a non-specified Copyright (c)
        Translation of last version at 22-February-2018
        (Matlab version 2.1 (7-May-2004))
        """
        L, N = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)

        # SNR Estimates
        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :p]
            x_p = np.dot(Ud.T, Y_o)
            SNR = self.estimate_snr(Y, y_m, x_p)
            log.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            log.info(f"input SNR = {SNR}[dB]")

        SNR_th = 15 + 10 * np.log10(p)

        # Choosing Projective Projection or projection to p-1 subspace
        if SNR < SNR_th:
            log.info("... Select proj. to R-1")
            d = p - 1
            if snr_input == 0:
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m
                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :d]
                x_p = np.dot(Ud.T, Y_o)

            Yp = np.dot(Ud, x_p[:d, :]) + y_m
            x = x_p[:d, :]
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            log.info("... Select the projective proj.")
            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]
            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(Ud, x_p[:d, :])
            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)
            y = x / np.dot(u.T, x)

        # VCA algorithm
        indices = np.zeros((p), dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1

        for i in range(p):
            w = generator.random(size=(p, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)
            v = np.dot(f.T, y)
            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]

        E = Yp[:, indices]
        log.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape
        p, N = x.shape
        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))
        return snr_est


# ============================================================================
# METRICS (inlined from src/utils/metrics.py)
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
    """Return individual and global metric."""
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


# ============================================================================
# ALIGNERS (inlined from src/utils/aligners.py)
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
        super().__init__(criterion=MSE(), **kwargs)


# ============================================================================
# PGMSU MODEL (inlined from src/model/blind/PGMSU.py)
# ============================================================================

class PGMSU(nn.Module):
    """
    Probabilistic Generative Model for Hyperspectral Unmixing (PGMSU)
    simple PyTorch implementation
    based on https://github.com/shuaikaishi/PGMSU
    """
    def __init__(
        self,
        z_dim=4,
        lr=1e-3,
        epochs=200,
        lambda_kl=0.1,
        lambda_sad=0,
        lambda_vol=0.5,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.lr = lr
        self.epochs = epochs

        self.lambda_kl = lambda_kl
        self.lambda_sad = lambda_sad
        self.lambda_vol = lambda_vol

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
        self.time = 0

    def init_architecture(self, seed):
        # Set random seed
        torch.manual_seed(seed)

        # encoder z fc1 -> fc5
        self.fc1 = nn.Linear(self.L, 32 * self.p)
        self.bn1 = nn.BatchNorm1d(32 * self.p)

        self.fc2 = nn.Linear(32 * self.p, 16 * self.p)
        self.bn2 = nn.BatchNorm1d(16 * self.p)

        self.fc3 = nn.Linear(16 * self.p, 4 * self.p)
        self.bn3 = nn.BatchNorm1d(4 * self.p)

        self.fc4 = nn.Linear(4 * self.p, self.z_dim)
        self.fc5 = nn.Linear(4 * self.p, self.z_dim)

        # encoder a
        self.fc9 = nn.Linear(self.L, 32 * self.p)
        self.bn9 = nn.BatchNorm1d(32 * self.p)

        self.fc10 = nn.Linear(32 * self.p, 16 * self.p)
        self.bn10 = nn.BatchNorm1d(16 * self.p)

        self.fc11 = nn.Linear(16 * self.p, 4 * self.p)
        self.bn11 = nn.BatchNorm1d(4 * self.p)

        self.fc12 = nn.Linear(4 * self.p, 4 * self.p)
        self.bn12 = nn.BatchNorm1d(4 * self.p)

        self.fc13 = nn.Linear(4 * self.p, self.p)

        # decoder
        self.fc6 = nn.Linear(self.z_dim, 4 * self.p)
        self.bn6 = nn.BatchNorm1d(4 * self.p)

        self.fc7 = nn.Linear(4 * self.p, 64 * self.p)
        self.bn7 = nn.BatchNorm1d(64 * self.p)

        self.fc8 = nn.Linear(64 * self.p, self.L * self.p)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h1 = F.relu(h1)

        h1 = self.fc3(h1)
        h1 = self.bn3(h1)
        h1 = F.relu(h1)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_a(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.relu(h1)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.relu(h1)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.relu(h1)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)
        h1 = F.relu(h1)

        h1 = self.fc13(h1)
        a = F.softmax(h1, dim=1)
        return a

    def reparametrize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=self.device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.relu(h1)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.relu(h1)

        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        return em

    def forward(self, inputs):
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_a(inputs)

        # reparametrization trick
        z = self.reparametrize(mu, log_var)
        em = self.decoder(z)

        em_tensor = em.view(-1, self.p, self.L)
        a_tensor = a.view(-1, 1, self.p)
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        return y_hat, mu, log_var, a, em_tensor

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find("Linear") != -1:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def print_time(self):
        return f"PGMSU took {self.time:.2f}s"

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()
        log.debug("Solving started...")

        L, N = Y.shape
        # Hyperparameters
        self.L = L
        self.p = p
        self.N = N

        self.batchsz = self.N // 10

        self.init_architecture(seed=seed)

        # Process data
        train_db = torch.utils.data.TensorDataset(torch.Tensor(Y.T))
        dataloader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batchsz,
            shuffle=True,
        )

        # Endmembers initialization
        extractor = VCA()
        EM = extractor.extract_endmembers(Y, p, seed=seed)
        EM = EM.T
        EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype("float32")
        EM = torch.Tensor(EM).to(self.device)

        # Model initialization
        self = self.to(self.device)
        self.apply(self.weights_init)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.epochs))

        self.train()
        for ee in progress:
            for _, y in enumerate(dataloader):
                y = y[0].to(self.device)

                y_hat, mu, log_var, a, em_tensor = self(y)

                loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

                kl_div = -0.5 * (log_var + 1 - mu**2 - log_var.exp())
                kl_div = kl_div.sum() / y.shape[0]
                # KL balance of VAE
                kl_div = torch.max(kl_div, torch.Tensor([0.2]).to(self.device))

                if ee < self.epochs // 2:
                    # pre-train process
                    loss_vca = (em_tensor - EM).square().sum() / y.shape[0]
                    loss = loss_rec + self.lambda_kl * kl_div + 0.1 * loss_vca
                else:
                    # training process
                    # constraint 1 min_vol of EMs
                    em_bar = em_tensor.mean(dim=1, keepdim=True)
                    loss_minvol = ((em_tensor - em_bar) ** 2).sum() / (
                        y.shape[0] * self.p * self.L
                    )

                    # constraint 2 SAD for same materials
                    em_bar = em_tensor.mean(dim=0, keepdim=True)
                    aa = (em_tensor * em_bar).sum(dim=2)
                    em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                    em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

                    sad = torch.acos(
                        aa / ((em_bar_norm + 1e-6) * (em_tensor_norm + 1e-6))
                    )
                    loss_sad = sad.sum() / (y.shape[0] * self.p)
                    loss = (
                        loss_rec
                        + self.lambda_kl * kl_div
                        + self.lambda_vol * loss_minvol
                        + self.lambda_sad * loss_sad
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            y_hat, mu, log_var, A, E = self(torch.Tensor(Y.T).to(self.device))
            Ahat = A.cpu().numpy().T
            Ehat = E.cpu().numpy().mean(0).T

        self.time = time.time() - tic
        log.info(self.print_time())

        return Ehat, Ahat


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    log.info("Standalone PGMSU Unmixing - [START]")

    # Get script directory for relative paths
    script_dir = get_script_dir()
    os.chdir(script_dir)

    # Load configuration
    config_path = os.path.join(
        "data_standalone",
        "standalone_unmixing_PGMSU_DC1.json"
    )
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Set seeds
    seed = cfg["seed"]
    set_seeds(seed)

    # Create output directory
    output_dir = "exp_standalone_unmixing_PGMSU_DC1"
    os.makedirs(output_dir, exist_ok=True)

    # Get noise parameters
    noise = AdditiveWhiteGaussianNoise(SNR=cfg["noise"]["SNR"])

    # Get HSI data
    data_dir = cfg["data"]["data_dir"]
    dataset = cfg["data"]["dataset"]
    EPS = cfg["EPS"]

    hsi = HSIWithGT(
        dataset=dataset,
        data_dir=data_dir,
        figs_dir=output_dir,
        EPS=EPS,
    )

    # Print HSI information
    log.info(hsi)

    # Get data
    Y, p, _ = hsi.get_data()

    # Get image dimensions
    H, W = hsi.get_img_shape()

    # Apply noise
    Y = noise.apply(Y)

    # L2 normalization (disabled by config)
    if cfg.get("l2_normalization", False):
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY

    # Build model
    model_cfg = cfg["model"]
    model = PGMSU(
        z_dim=model_cfg["z_dim"],
        lr=model_cfg["lr"],
        epochs=model_cfg["epochs"],
        lambda_kl=model_cfg["lambda_kl"],
        lambda_sad=model_cfg["lambda_sad"],
        lambda_vol=model_cfg["lambda_vol"],
    )

    # Solve unmixing
    E_hat, A_hat = model.compute_endmembers_and_abundances(
        Y,
        p,
        H=H,
        W=W,
        seed=seed,
    )

    # Save estimates
    estimates_path = os.path.join(output_dir, "estimates.mat")
    sio.savemat(estimates_path, {"E": E_hat, "A": A_hat.reshape(-1, H, W)})
    log.info(f"Estimates saved to {estimates_path}")

    # Evaluate if ground truth is available
    metrics = {}
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
        metrics["SRE"] = compute_metric(
            SRE(), A_gt, A1, labels, detail=False, on_endmembers=False
        )
        metrics["aRMSE"] = compute_metric(
            aRMSE(), A_gt, A1, labels, detail=True, on_endmembers=False
        )
        metrics["SAD"] = compute_metric(
            SADDegrees(), E_gt, E1, labels, detail=True, on_endmembers=True
        )
        metrics["eRMSE"] = compute_metric(
            eRMSE(), E_gt, E1, labels, detail=True, on_endmembers=True
        )

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        log.info(f"Metrics saved to {metrics_path}")

        # Plot results using aligned estimates
        hsi.plot_endmembers(E0=E1, run="estimated")
        hsi.plot_abundances(A0=A1, run="estimated")
        log.info(f"Plots saved to {output_dir}")

    log.info("Standalone PGMSU Unmixing - [END]")


if __name__ == "__main__":
    main()
"""
