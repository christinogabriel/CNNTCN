import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from data_loader import load_data, preprocess_data, create_sequences
from model_builder import build_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Mensagem de início
print("Iniciando o carregamento de dados...")

# Especificar caminhos dos dados
train_vibration_path = 'datasets/Training_set/vibration'
train_temperature_path = 'datasets/Training_set/temperature'
val_vibration_path = 'datasets/Validation_set/vibration'
val_temperature_path = 'datasets/Validation_set/temperature'
test_vibration_path = 'datasets/Test_set/vibration'
test_temperature_path = 'datasets/Test_set/temperature'

# Carregar os dados
train_vibration_data = load_data(train_vibration_path)
print("Dados de vibração de treinamento carregados.")
train_temperature_data = load_data(train_temperature_path)
print("Dados de temperatura de treinamento carregados.")
val_vibration_data = load_data(val_vibration_path)
print("Dados de vibração de validação carregados.")
val_temperature_data = load_data(val_temperature_path)
print("Dados de temperatura de validação carregados.")
test_vibration_data = load_data(test_vibration_path)
print("Dados de vibração de teste carregados.")
test_temperature_data = load_data(test_temperature_path)
print("Dados de temperatura de teste carregados.")
# mensagem check
print("Verificando o carregamento dos dados:")
print("Dados de vibração de treinamento:", train_vibration.head() if train_vibration is not None else "None")
print("Dados de temperatura de treinamento:", train_temperature.head() if train_temperature is not None else "None")
print("Dados de vibração de validação:", val_vibration.head() if val_vibration is not None else "None")
print("Dados de temperatura de validação:", val_temperature.head() if val_temperature is not None else "None")
print("Dados de vibração de teste:", test_vibration.head() if test_vibration is not None else "None")
print("Dados de temperatura de teste:", test_temperature.head() if test_temperature is not None else "None")


# Pré-processar e normalizar
train_vib, train_temp, val_vib, val_temp, test_vib, test_temp = preprocess_data(
    train_vibration_data, train_temperature_data,
    val_vibration_data, val_temperature_data,
    test_vibration_data, test_temperature_data
)
print("Dados pré-processados e normalizados.")

# Criar sequências
seq_length = 50
X_train_vib = create_sequences(train_vib, seq_length)
X_train_temp = create_sequences(train_temp, seq_length)
X_val_vib = create_sequences(val_vib, seq_length)
X_val_temp = create_sequences(val_temp, seq_length)
X_test_vib = create_sequences(test_vib, seq_length)
X_test_temp = create_sequences(test_temp, seq_length)
print("Sequências criadas.")

# Concatenar vibração e temperatura
X_train = np.concatenate((X_train_vib, X_train_temp), axis=-1)
X_val = np.concatenate((X_val_vib, X_val_temp), axis=-1)
X_test = np.concatenate((X_test_vib, X_test_temp), axis=-1)
print("Dados de entrada concatenados.")

# Definir variáveis de saída (exemplo binário)
y_train = np.random.randint(0, 2, X_train.shape[0])
y_val = np.random.randint(0, 2, X_val.shape[0])
y_test = np.random.randint(0, 2, X_test.shape[0])
print("Variáveis de saída definidas.")

# Construir e treinar o modelo
input_shape = X_train.shape[1:]
model = build_model(input_shape)
print("Modelo compilado. Iniciando treinamento...")

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

print("Treinamento concluído. Gerando previsões...")

# Previsões para o conjunto de teste
predictions = model.predict(X_test)

# Cálculo do MAE entre as previsões e os valores reais
mae = mean_absolute_error(y_test, predictions)
print(f"MAE das previsões no conjunto de teste: {mae}")

# Ajustar o limiar de decisão
limiar = 0.5  # Altere conforme necessário
y_pred_classes = (predictions > limiar).astype(int).flatten()

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Falha", "Falha"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()
