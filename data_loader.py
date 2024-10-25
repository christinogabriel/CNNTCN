import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    def load_data(data_path, chunksize=100000):
        all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        data_chunks = []
        for file in all_files:
            for chunk in pd.read_csv(file, chunksize=chunksize):
                data_chunks.append(chunk)
        data = pd.concat(data_chunks, ignore_index=True)
        return data


def preprocess_data(train_vibration, train_temp, val_vibration, val_temp, test_vibration, test_temp):
    scaler = MinMaxScaler()
    train_vibration = scaler.fit_transform(train_vibration)
    train_temp = scaler.fit_transform(train_temp)
    val_vibration = scaler.transform(val_vibration)
    val_temp = scaler.transform(val_temp)
    test_vibration = scaler.transform(test_vibration)
    test_temp = scaler.transform(test_temp)
    return train_vibration, train_temp, val_vibration, val_temp, test_vibration, test_temp

def preprocess_data(train_vibration, train_temperature, val_vibration, val_temperature, test_vibration,
                        test_temperature):
        scaler = StandardScaler()

        # Remover ou substituir NaNs no conjunto de treinamento
        train_vibration = train_vibration.fillna(method='ffill').fillna(method='bfill')
        train_temperature = train_temperature.fillna(method='ffill').fillna(method='bfill')
        val_vibration = val_vibration.fillna(method='ffill').fillna(method='bfill')
        val_temperature = val_temperature.fillna(method='ffill').fillna(method='bfill')
        test_vibration = test_vibration.fillna(method='ffill').fillna(method='bfill')
        test_temperature = test_temperature.fillna(method='ffill').fillna(method='bfill')

        # Escalar os dados após tratar os NaNs
        train_vibration = scaler.fit_transform(train_vibration)
        train_temperature = scaler.fit_transform(train_temperature)
        val_vibration = scaler.transform(val_vibration)
        val_temperature = scaler.transform(val_temperature)
        test_vibration = scaler.transform(test_vibration)
        test_temperature = scaler.transform(test_temperature)

        return train_vibration, train_temperature, val_vibration, val_temperature, test_vibration, test_temperature

def create_sequences(data, seq_length=50):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)
