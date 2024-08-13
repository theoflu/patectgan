import numpy as np
from sklearn.metrics import mean_squared_error

from .metrics import compute_ks_test
from .metrics import compute_jsd
from .metrics import compute_kl_divergence



# def compute_ks_test(real_data, generated_data):
#     ks_stats = []
#     for i in range(real_data.shape[1]):
#         ks_stat, _ = stats.ks_2samp(real_data[:, i], generated_data[:, i])
#         ks_stats.append(ks_stat)
#     return np.mean(ks_stats)

# def compute_jsd(real_data, generated_data, bins=100):
#     jsd_stats = []
#     for i in range(real_data.shape[1]):
#         real_hist, _ = np.histogram(real_data[:, i], bins=bins, density=True)
#         gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
#         real_hist = real_hist / real_hist.sum()
#         gen_hist = gen_hist / gen_hist.sum()
#         jsd = distance.jensenshannon(real_hist, gen_hist)
#         jsd_stats.append(jsd)
#     return np.mean(jsd_stats)

# def compute_kl_divergence(real_data, generated_data, bins=100):
#     kl_divs = []
#     for i in range(real_data.shape[1]):
#         real_hist, bin_edges = np.histogram(real_data[:, i], bins=bins, density=True)
#         gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
#         real_hist = real_hist + 1e-10  # To avoid division by zero
#         gen_hist = gen_hist + 1e-10  # To avoid log(0)
#         kl_div = stats.entropy(real_hist, gen_hist)
#         kl_divs.append(kl_div)
#     return np.mean(kl_divs)

# def evaluate_model(model, test_data, scaler, result_file_path):
#     generated_data = model.generate(len(test_data))
#     generated_data = scaler.inverse_transform(generated_data)
#     test_data = scaler.inverse_transform(test_data)
    
#     mse = mean_squared_error(test_data, generated_data)
#     print(f"Mean Squared Error on Test Data: {mse}")

#     ks_test_result = compute_ks_test(test_data, generated_data)
#     jsd_result = compute_jsd(test_data, generated_data)
#     kl_divergence_result = compute_kl_divergence(test_data, generated_data)
    
#     print(f"KS Test Statistic: {ks_test_result}")
#     print(f"Jensen-Shannon Divergence: {jsd_result}")
#     print(f"KL Divergence: {kl_divergence_result}")

#     results = {
#         "Mean Squared Error": mse,
#         "KS Test Statistic": ks_test_result,
#         "Jensen-Shannon Divergence": jsd_result,
#         "KL Divergence": kl_divergence_result
#     }

#     hyperparameters = {
#         "embedding_dim": model.embedding_dim,
#         "gen_dim": model.gen_dim,
#         "dis_dim": model.dis_dim,
#         "l2scale": model.l2scale,
#         "batch_size": model.batch_size,
#         "epochs": model.epochs,
#         "pack": model.pack,
#         "loss": model.loss
#     }

#     # Sonuçları Dosyaya Yazma
#     with open(result_file_path, "w") as f:
#         f.write("Hyperparameters:\n")
#         for key, value in hyperparameters.items():
#             f.write(f"{key}: {value}\n")
#         f.write("\nResults:\n")
#         for key, value in results.items():
#             f.write(f"{key}: {value}\n")


# def compute_ks_test(real_data, generated_data):
#     ks_stats = []
#     for i in range(real_data.shape[1]):
#         ks_stat, _ = stats.ks_2samp(real_data[:, i], generated_data[:, i])
#         ks_stats.append(ks_stat)
#     return np.mean(ks_stats)

# def compute_jsd(real_data, generated_data, bins=100):
#     jsd_stats = []
#     for i in range(real_data.shape[1]):
#         real_hist, _ = np.histogram(real_data[:, i], bins=bins, density=True)
#         gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
#         real_hist = real_hist / real_hist.sum()
#         gen_hist = gen_hist / gen_hist.sum()
#         jsd = distance.jensenshannon(real_hist, gen_hist)
#         jsd_stats.append(jsd)
#     return np.mean(jsd_stats)

# def compute_kl_divergence(real_data, generated_data, bins=100):
#     kl_divs = []
#     for i in range(real_data.shape[1]):
#         real_hist, bin_edges = np.histogram(real_data[:, i], bins=bins, density=True)
#         gen_hist, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
#         real_hist = real_hist + 1e-10  # To avoid division by zero
#         gen_hist = gen_hist + 1e-10  # To avoid log(0)
#         kl_div = stats.entropy(real_hist, gen_hist)
#         kl_divs.append(kl_div)
#     return np.mean(kl_divs)

    
def evaluate_model(model, test_data, scaler, way):
    generated_data = model.generate(len(test_data))
    generated_data = scaler.inverse_transform(generated_data)
    test_data = scaler.inverse_transform(test_data)
    
    mse = mean_squared_error(test_data, generated_data)
    print(f"Mean Squared Error on Test Data: {mse}")

    ks_test_result = compute_ks_test(test_data, generated_data)
    jsd_result = compute_jsd(test_data, generated_data)
    kl_divergence_result = compute_kl_divergence(test_data, generated_data)
    
    results = {
        "Mean Squared Error": mse,
        "KS Test Statistic": ks_test_result,
        "Jensen-Shannon Divergence": jsd_result,
        "KL Divergence": kl_divergence_result
    }

    hyperparameters = {
        "embedding_dim": model.embedding_dim,
        "gen_dim": model.gen_dim,
        "dis_dim": model.dis_dim,
        "l2scale": model.l2scale,
        "batch_size": model.batch_size,
        "epochs": model.epochs,
        "pack": model.pack,
        "loss": model.loss
    }

    # Sonuçları Dosyaya Yazma
    with open(way, "w") as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\nResults:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
