import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar os dados diretamente do GitHub
url = "https://raw.githubusercontent.com/hermannvargens/doutorado/refs/heads/main/espectros_derivada.csv"
df = pd.read_csv(url, delimiter=";")
df = df[:-3]

# Substituir '.' por '' e ',' por '.' nos nomes das colunas
df.columns = [col.replace('.', '').replace(',', '.') for col in df.columns]

# Substituir o ponto por '' e a vírgula por ponto em todas as colunas do df
for col in df.columns:
    if df[col].dtype == 'object':  # Verificar se a coluna contém strings
        df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

# Converter todas as colunas para float
df = df.astype(float)

# Separar os valores de absorbância
df_espectros = df.iloc[:, 3:]
df_espectros = df_espectros.T  # Transpor os dados

# Carregar o modelo PLS treinado
pls_mistura = joblib.load('pls_mistura_model.joblib')

# Título do Streamlit
st.title("Visualização de Espectros NIR")

# Opção de selecionar os espectros
espectros_disponiveis = df_espectros.columns.tolist()

# Adicionar opção de visualizar todos os espectros
visualizar_todos = st.checkbox("Visualizar todos os espectros")

if visualizar_todos:
    espectros_selecionados = espectros_disponiveis
else:
    espectros_selecionados = st.multiselect("Escolha os espectros para visualizar:", espectros_disponiveis)

# Plotar os espectros selecionados
if espectros_selecionados:
    st.subheader("Espectros NIR Selecionados")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for espectro in espectros_selecionados:
        ax.plot(df_espectros.index, df_espectros[espectro], label=espectro)
    
    ax.set_xlabel("Comprimento de Onda")
    ax.set_ylabel("Absorbância")
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(df_espectros.index), 10))
    
    st.pyplot(fig)
else:
    st.write("Selecione pelo menos um espectro para visualizar.")

# Seleção de amostra para predição
df_teste = df.iloc[53:, :].reset_index(drop=True)

# Opção de escolher a amostra para a predição
linha = st.number_input("Escolha o número da amostra de 1 a 21:", min_value=1, max_value=21, value=1) - 1

# Obter os dados da amostra
X_new = df_teste.iloc[linha, 3:]
X_new = X_new.values.reshape(1, -1)

y_new = df_teste.iloc[linha, 0:3]
y_new = y_new.values.reshape(1, -1)

# Realizar a predição
y_pred_new = pls_mistura.predict(X_new)

# Calcular RMSE e MAE
rmse = np.sqrt(mean_squared_error(y_new, y_pred_new))
mae = mean_absolute_error(y_new, y_pred_new)

# Calcular os erros em porcentagem
rmse_percent = (rmse / np.mean(y_new)) * 100
mae_percent = (mae / np.mean(y_new)) * 100

# Exibir resultados
st.subheader("Resultados da Predição")
st.write(f"Predição: {y_pred_new[0]}")
st.write(f"Valores reais: {y_new[0]}")
st.write(f"RMSE (%): {rmse_percent:.6f}%")
st.write(f"MAE (%): {mae_percent:.6f}%")
