import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model.patectgan import PATECTGAN
from utils.data_proccess.data_utils import load_and_preprocess_data
from utils.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    # Veri Yükleme
    train_data, test_data, scaler = load_and_preprocess_data("diabetes_data_train.csv", "diabetes_data_test.csv")

    # DataLoader oluşturma
    train_dataset = TensorDataset(torch.tensor(train_data))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Modeli Eğitme
    model = PATECTGAN(
        embedding_dim=train_data.shape[1],
        gen_dim=(256, 256),
        dis_dim=(256, 256),
        l2scale=1e-6,
        batch_size=64,
        epochs=300,
        pack=1,
        loss='cross_entropy'
    )
    model.train(train_loader)
    
    # Modeli Test Etme
    evaluate_model(model, test_data, scaler, f"data/{model.epochs}-similarity_results.txt")

    # Eğer modeliniz eğitim verisinde iyi performans gösterirken doğrulama verisinde performans düşüyorsa, 
    # aşırı uyuma (overfitting) olmuş olabilir ve epoch sayısını azaltmak veya modelinizi basitleştirmek gerekebilir.
    # Tam tersi durumda, modeliniz hem eğitim hem de doğrulama verisinde iyi performans gösteriyorsa, 
    # daha fazla epoch ekleyerek modelinizin daha iyi sonuçlar elde etmesini sağlayabilirsiniz.
