import os
import json
import time
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from munkres import Munkres


# --- Extracted Dependencies ---

logger = logging.getLogger(__name__)

def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def SVD_projection(Y, p):
    """SVD projection utility function."""
    logger.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    logger.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)

class AdditiveWhiteGaussianNoise:
    """Additive White Gaussian Noise generator."""
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y
        """
        logger.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        L, N = Y.shape
        logger.info(f"Desired SNR => {self.SNR}")

        if self.SNR is None:
            sigmas = np.zeros(L)
        else:
            assert self.SNR > 0, "SNR must be strictly positive"
            sigmas = np.ones(L)
            sigmas /= np.linalg.norm(sigmas)
            logger.debug(f"Sigmas after normalization: {sigmas[0]}")
            num = np.sum(Y**2) / N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            logger.debug(f"Sigma mean based on SNR: {sigmas_mean}")
            sigmas *= sigmas_mean
            logger.debug(f"Final sigmas value: {sigmas[0]}")

        noise = np.diag(sigmas) @ np.random.randn(L, N)
        return Y + noise

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
        self.EPS = EPS
        
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

        self.name = dataset
        filename = f"{self.name}.mat"
        
        # Resolve path relative to script directory
        script_dir = get_script_dir()
        path = os.path.join(script_dir, data_dir, filename)
        logger.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path), f"Data file not found: {path}"

        data = sio.loadmat(path)
        logger.debug(f"Data keys: {data.keys()}")

        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
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

        # Create output figures folder
        self.figs_dir = os.path.join(script_dir, figs_dir)
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_data(self):
        return (self.Y, self.p, self.D)

    def get_HSI_dimensions(self):
        return {
            "bands": self.L,
            "pixels": self.N,
            "lines": self.H,
            "samples": self.W,
            "atoms": self.M,
        }

    def get_img_shape(self):
        return (self.H, self.W)

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
    """HSI class with ground truth."""
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

        assert self.E.shape == (self.L, self.p)
        assert self.A.shape == (self.p, self.N)

        try:
            assert len(self.labels) == self.p
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]
        except Exception:
            self.labels = [f"#{ii}" for ii in range(self.p)]

        assert np.allclose(
            self.A.sum(0),
            np.ones(self.N),
            rtol=1e-3,
            atol=1e-3,
        )
        assert np.all(self.A >= -self.EPS)
        assert np.all(self.E >= -self.EPS)

    def get_GT(self):
        return (self.E, self.A)

    def has_GT(self):
        return True

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

    logger.info(f"{metric} => {d}")
    return d

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
        super().__init__(
            criterion=MSE(),
            **kwargs,
        )

class BaseExtractor:
    def __init__(self):
        self.seed = None

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        return NotImplementedError

    def __repr__(self):
        msg = f"{self.__class__.__name__}_seed{self.seed}"
        return msg

    def print_time(self, timer):
        msg = f"{self} took {timer:.2f} seconds..."
        return msg

class SiVM(BaseExtractor):
    """Simplex Volume Maximization extractor."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def Eucli_dist(x, y):
        a = np.subtract(x, y)
        return np.dot(a.T, a)

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):
        x, p = Y, p
        [D, N] = x.shape
        Z1 = np.zeros((1, 1))
        O1 = np.ones((1, 1))
        d = np.zeros((p, N))
        index = np.zeros((p, 1))
        V = np.zeros((1, N))
        ZD = np.zeros((D, 1))
        
        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), ZD)

        index = np.argmax(d[0, :])

        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), x[:, index].reshape(D, 1))

        for v in range(1, p):
            D1 = np.concatenate(
                (d[0:v, index].reshape((v, index.size)), np.ones((v, 1))), axis=1
            )
            D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
            D4 = np.concatenate((D1, D2), axis=0)
            D4 = np.linalg.inv(D4)

            for i in range(N):
                D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
                V[0, i] = np.dot(np.dot(D3.T, D4), D3)

            index = np.append(index, np.argmax(V))
            for i in range(N):
                d[v, i] = self.Eucli_dist(
                    x[:, i].reshape(D, 1), x[:, index[v]].reshape(D, 1)
                )

        per = np.argsort(index)
        index = np.sort(index)
        d = d[per, :]
        E = x[:, index]
        logger.debug(f"Indices chosen: {index}")
        return E

class UnmixingModel:
    """Base unmixing model class."""
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"

class BlindUnmixingModel(UnmixingModel):
    """Base class for blind unmixing models."""
    def __init__(self):
        super().__init__()

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):
        raise NotImplementedError(f"Solver is not implemented for {self}")

class MiSiCNet(nn.Module, BlindUnmixingModel):
    """MiSiCNet PyTorch implementation."""
    def __init__(
        self,
        niters=8000,
        lr=0.001,
        exp_weight=0.99,
        lambd=100.0,
        kernel_size=3,
        *args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        BlindUnmixingModel.__init__(self)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.kernel_sizes = [kernel_size] * 4 + [1]
        self.strides = [1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.lambd = lambd

    def init_architecture(self, seed):
        torch.manual_seed(seed)
        
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

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.L, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, self.p, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(self.p),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.layer1(x)
        xskip = self.layerskip(x)
        xcat = torch.cat([x1, xskip], dim=1)
        abund = self.softmax(self.layer4(self.layer3(xcat)))
        abund_reshape = torch.transpose(abund.squeeze().view(-1, self.H * self.W), 0, 1)
        img = self.decoder(abund_reshape)
        return abund_reshape, img

    def loss(self, target, output):
        N, L = output.shape
        target_reshape = target.squeeze().reshape(L, N)
        fit_term = 0.5 * torch.linalg.norm(target_reshape.t() - output, "fro") ** 2
        mean = target_reshape.mean(1, keepdims=True)
        reg_term = torch.linalg.norm(self.decoder.weight - mean, "fro") ** 2
        return fit_term + self.lambd * reg_term

    def compute_endmembers_and_abundances(self, Y, p, H, W, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        L, N = Y.shape

        self.L = L
        self.p = p
        self.H = H
        self.W = W

        self.init_architecture(seed=seed)

        # Initialize endmembers using SiVM extractor
        extractor = SiVM()
        Ehat = extractor.extract_endmembers(Y, p, seed=seed)
        self.decoder.weight.data = torch.Tensor(Ehat)

        num_channels, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, num_channels, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)

        noisy_input = torch.rand_like(Y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.niters))
        for ii in progress:
            optimizer.zero_grad()

            abund, output = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * self.exp_weight + abund.detach() * (
                    1 - self.exp_weight
                )

            loss = self.loss(Y, output)

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            loss.backward()
            optimizer.step()
            # Enforce physical constraints on endmembers
            self.decoder.weight.data[self.decoder.weight <= 0] = 0
            self.decoder.weight.data[self.decoder.weight >= 1] = 1

        Ahat = out_avg.cpu().T.numpy()
        Ehat = self.decoder.weight.detach().cpu().numpy()
        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat

def main():
    """Main function to run MiSiCNet unmixing."""
    logger.info("Blind Unmixing - MiSiCNet - [START]")
    
    # Get script directory for relative paths
    script_dir = get_script_dir()
    
    # Load configuration
    config_path = os.path.join(
        script_dir, 
        "data_standalone", 
        "standalone_unmixing_MiSiCNet_DC1.json"
    )
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Set seed
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = os.path.join(script_dir, cfg["figs_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize noise
    noise = AdditiveWhiteGaussianNoise(SNR=cfg["SNR"])
    
    # Initialize HSI data
    hsi = HSIWithGT(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        figs_dir=cfg["figs_dir"],
        EPS=cfg["EPS"],
    )
    
    # Print HSI information
    logger.info(hsi)
    
    # Get data
    Y, p, _ = hsi.get_data()
    H, W = hsi.get_img_shape()
    
    # Apply noise
    Y = noise.apply(Y)
    
    # L2 normalization (disabled by config)
    if cfg["l2_normalization"]:
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    
    # Apply SVD projection (disabled by config)
    if cfg["projection"]:
        Y = SVD_projection(Y, p)
    
    # Build model
    model = MiSiCNet(
        niters=cfg["model"]["niters"],
        lr=cfg["model"]["lr"],
        exp_weight=cfg["model"]["exp_weight"],
        lambd=cfg["model"]["lambd"],
        kernel_size=cfg["model"]["kernel_size"],
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
    logger.info(f"Estimates saved to {estimates_path}")
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    if hsi.has_GT():
        # Get ground truth
        E_gt, A_gt = hsi.get_GT()
        
        # Plot ground truth
        hsi.plot_endmembers(E0=None, run=0)
        hsi.plot_abundances(A0=None, run=0)
        
        # Align based on abundances
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)
        
        # Plot estimates
        hsi.plot_endmembers(E0=E1, run=0)
        hsi.plot_abundances(A0=A1, run=0)
        
        # Get labels
        labels = hsi.get_labels()
        
        # Compute metrics
        all_metrics["SRE"] = compute_metric(
            SRE(),
            A_gt,
            A1,
            labels,
            detail=False,
            on_endmembers=False,
        )
        
        all_metrics["aRMSE"] = compute_metric(
            aRMSE(),
            A_gt,
            A1,
            labels,
            detail=True,
            on_endmembers=False,
        )
        
        all_metrics["SAD"] = compute_metric(
            SADDegrees(),
            E_gt,
            E1,
            labels,
            detail=True,
            on_endmembers=True,
        )
        
        all_metrics["eRMSE"] = compute_metric(
            eRMSE(),
            E_gt,
            E1,
            labels,
            detail=True,
            on_endmembers=True,
        )
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("Blind Unmixing - MiSiCNet - [END]")
    
    return all_metrics
