import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# def load_and_preprocess_data(train_path, test_path):
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
    
#     # Eksik değerleri doldurma (örnek: ortalama ile doldurma)
#     train_df.fillna(train_df.mean(), inplace=True)
#     test_df.fillna(test_df.mean(), inplace=True)
    
#     # Verileri numpy array ve float32 olarak dönüştürme
#     train_data = train_df.values.astype(np.float32)
#     test_data = test_df.values.astype(np.float32)
    
#     # Veriyi normalize etme.  Veriyi normalize etmek, modelin performansını artırmak 
#     ve daha güvenilir sonuçlar elde etmek için önemli bir adımdır.
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)
    
#     return train_data, test_data, scaler

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df.fillna(train_df.mean(), inplace=True)
    test_df.fillna(test_df.mean(), inplace=True)
    
    train_data = train_df.values.astype(np.float32)
    test_data = test_df.values.astype(np.float32)
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    return train_data, test_data, scaler
