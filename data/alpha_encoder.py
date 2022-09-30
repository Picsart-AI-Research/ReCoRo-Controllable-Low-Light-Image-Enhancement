import numpy as np
from hydra.utils import get_original_cwd

class AlphaEncoder():
    def __init__(self, cfg):
    # def __init__(self, name, dist, encoding="rand", dim=128, load=False):
        self.cfg = cfg
        if self.cfg.dist.type == "const":
            self.dist = np.array([self.cfg.dist.value])
        elif self.cfg.dist.type == "uniform":
            self.dist = np.linspace(self.cfg.dist.min, self.cfg.dist.max, num=self.cfg.dist.n)
        else:
            raise "Unknown alpha distribution"

        self.dist_idxs = np.arange(self.dist.size)

        save_path = f"encoding_mat_{self.cfg.name}.npy"
        if self.cfg.load:
            self.orig_cwd = get_original_cwd()
            self.encoding_mat = np.load(f"{self.orig_cwd}/{self.cfg.load_dir}/{save_path}")
        else:
            self.encoding_mat = self.gen_encoding_mat()
            np.save(save_path, self.encoding_mat)

    def gen_encoding_mat(self):
        # encoding matrix:
        if self.cfg.encoding == 'onehot':
            I_mat = np.eye(self.cfg.dim)
            encoding_mat = I_mat
        elif self.cfg.encoding == 'dct':
            from scipy.fftpack import dct
            dct_mat = dct(np.eye(self.cfg.dim), axis=0)
            encoding_mat = dct_mat
        elif self.cfg.encoding == 'random':
            rand_mat = np.random.randn(self.cfg.dim, self.cfg.dim)
            rand_otho_mat, _ = np.linalg.qr(rand_mat)
            encoding_mat = rand_otho_mat
        elif self.cfg.encoding is None:
            encoding_mat = None
        return encoding_mat



    def get_random_alpha(self):
        alpha_idx = np.random.choice(self.dist_idxs)
        alpha = self.dist[alpha_idx]
        if self.encoding_mat is not None:
            alpha_vec = self.encoding_mat[alpha_idx]
        else:
            alpha_vec = np.array([alpha])
        return alpha, alpha_vec

    def get_random_alpha_subset(self, subset_idxs):
        alpha_idx = np.random.choice(subset_idxs)
        alpha = self.dist[alpha_idx]
        if self.encoding_mat is not None:
            alpha_vec = self.encoding_mat[alpha_idx]
        else:
            alpha_vec = np.array([alpha])
        return alpha, alpha_vec

    def get_alpha(self, alpha):
        alpha_idx = np.where(np.isclose(self.dist, alpha))[0][0]
        if self.encoding_mat is not None:
            alpha_vec = self.encoding_mat[alpha_idx]
        else:
            alpha_vec = np.array([alpha])

        return alpha, alpha_vec

    def sample(self, cfg):
        if cfg.sampler == "const":
            return self.get_alpha(cfg.value)
        elif cfg.sampler == "random":
            return self.get_random_alpha()
        elif cfg.sampler == "subset_random":
            return self.get_random_alpha_subset(cfg.subset_idxs)
        else:
            raise "Unknown alpha sampler"