import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.spatial import distance



def compute_ks_test(real_data, generated_data):
    ks_stats = []
    for i in range(real_data.shape[1]):
        ks_stat, _ = stats.ks_2samp(real_data[:, i], generated_data[:, i])
        ks_stats.append(ks_stat)
    return np.mean(ks_stats)

def compute_jsd(real_data, generated_data, bins=100):
    jsd_stats = []
    for i in range(real_data.shape[1]):
        real_hist, _ = np.histogram(real_data[:, i], bins=bins, density=True)
        gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()
        jsd = distance.jensenshannon(real_hist, gen_hist)
        jsd_stats.append(jsd)
    return np.mean(jsd_stats)

def compute_kl_divergence(real_data, generated_data, bins=100):
    kl_divs = []
    for i in range(real_data.shape[1]):
        real_hist, bin_edges = np.histogram(real_data[:, i], bins=bins, density=True)
        gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
        real_hist = real_hist + 1e-10  # To avoid division by zero
        gen_hist = gen_hist + 1e-10  # To avoid log(0)
        kl_div = stats.entropy(real_hist, gen_hist)
        kl_divs.append(kl_div)
    return np.mean(kl_divs)