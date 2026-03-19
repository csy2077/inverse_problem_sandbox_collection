import os
import json
import time
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt
import spams
from munkres import Munkres


# --- Extracted Dependencies ---

log = logging.getLogger(__name__)

SNR = CONFIG["SNR"]

PROJECTION = CONFIG["projection"]

L2_NORMALIZATION = CONFIG["l2_normalization"]

DATASET = CONFIG["dataset"]

DATA_DIR = CONFIG["data_dir"]

FIGS_DIR = CONFIG["figs_dir"]

OUTPUT_DIR = CONFIG["output_dir"]

EPS = CONFIG["EPS"]

INTEGER_VALUES = ("H", "W", "M", "L", "p", "N")

class HSI:
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
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
        self.figs_dir = os.path.join(os.getcwd(), figs_dir)
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
        """Display endmembers"""
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

def SVD_projection(Y, p):
    log.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f"projected Y shape => {denoised_image_reshape.shape}")
    return np.clip(denoised_image_reshape, 0, 1)

class VCA:
    def __init__(self):
        self.seed = None
        self.indices = None

    def __repr__(self):
        msg = f"{self.__class__.__name__}_seed{self.seed}"
        return msg

    def print_time(self, timer):
        msg = f"{self} took {timer:.2f} seconds..."
        return msg

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm
        under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 
        (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        L, N = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)

        #############################################
        # SNR Estimates
        #############################################

        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                :, :p
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            log.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            log.info(f"input SNR = {SNR}[dB]\n")

        SNR_th = 15 + 10 * np.log10(p)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            log.info("... Select proj. to R-1")

            d = p - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            log.info("... Select the projective proj.")

            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((p), dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1

        for i in range(p):
            w = generator.random(size=(p, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        log.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape  # L number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est

class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"

class SupervisedUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_abundances(self, Y, E, *args, **kwargs):
        raise NotImplementedError(f"Solver is not implemented for {self}")

class DecompSimplex(SupervisedUnmixingModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_abundances(self, Y, E, *args, **kwargs):
        tic = time.time()

        YY = np.asfortranarray(Y)
        EE = np.asfortranarray(E)
        A = np.array(spams.decompSimplex(YY, EE).todense())

        self.time = time.time() - tic
        log.info(self.print_time())

        return A

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
        ret = np.minimum(tmp, 1.0)  # Handle floating errors
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
    """Return individual and global metric"""
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
        indices = m.compute(self.dists.tolist())  # munkres needs list input
        for row, col in indices:
            P[row, col] = 1.0

        self.P = P.T

class AbundancesAligner(HungarianAligner):
    def __init__(self, **kwargs):
        super().__init__(
            criterion=MSE(),
            **kwargs,
        )

class Estimate:
    ext = ".mat"

    def __init__(self, Ehat, Ahat, H, W):
        self.data = {"E": Ehat, "A": Ahat.reshape(-1, H, W)}

    def save(self, fname="estimates"):
        sio.savemat(f"{fname}{self.ext}", self.data)

def main():
    log.info("Supervised Unmixing (DecompSimplex) - [START]...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize noise
    noise = AdditiveWhiteGaussianNoise(SNR=SNR)

    # Load HSI data
    hsi = HSIWithGT(
        dataset=DATASET,
        data_dir=DATA_DIR,
        figs_dir=FIGS_DIR,
    )
    log.info(hsi)

    # Get data
    Y, p, _ = hsi.get_data()

    # Get image dimensions
    H, W = hsi.get_img_shape()

    # Apply noise
    Y = noise.apply(Y)

    # L2 normalization (if enabled)
    if L2_NORMALIZATION:
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)

    # Apply SVD projection (if enabled)
    if PROJECTION:
        Y = SVD_projection(Y, p)

    # Initialize extractor and model
    extractor = VCA()
    model = DecompSimplex()

    # Endmember extraction
    E_hat = extractor.extract_endmembers(Y, p, H=H, W=W)

    # Abundance estimation
    A_hat = model.compute_abundances(Y, E_hat, p=p, H=H, W=W)

    # Save estimates
    estimate = Estimate(E_hat, A_hat, H, W)
    estimate.save(os.path.join(OUTPUT_DIR, "estimates"))

    # All metrics dictionary
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
        sre_result = compute_metric(
            SRE(),
            A_gt,
            A1,
            labels,
            detail=False,
            on_endmembers=False,
        )
        all_metrics["SRE"] = sre_result

        armse_result = compute_metric(
            aRMSE(),
            A_gt,
            A1,
            labels,
            detail=True,
            on_endmembers=False,
        )
        all_metrics["aRMSE"] = armse_result

        sad_result = compute_metric(
            SADDegrees(),
            E_gt,
            E1,
            labels,
            detail=True,
            on_endmembers=True,
        )
        all_metrics["SAD"] = sad_result

        ermse_result = compute_metric(
            eRMSE(),
            E_gt,
            E1,
            labels,
            detail=True,
            on_endmembers=True,
        )
        all_metrics["eRMSE"] = ermse_result

        # Plot ground truth
        hsi.plot_endmembers(E0=None, run=0)  # GT endmembers
        hsi.plot_abundances(A0=None, run=0)  # GT abundances

        # Plot estimated (aligned)
        hsi.plot_endmembers(E0=E1, run=0)
        hsi.plot_abundances(A0=A1, run=0)

    # Save metrics to JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    log.info(f"Metrics saved to {metrics_path}")

    log.info("Supervised Unmixing (DecompSimplex) - [END]...")
