import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Função para carregar o modelo PLS salvo
@st.cache_resource
def load_model():
    pls = joblib.load('pls.pkl')  # Certifique-se de que o caminho do arquivo está correto
    return pls

# Função para processar o arquivo CSV
def process_csv(file):
    df = pd.read_csv(file)
    return df

# Início da interface do Streamlit
st.title('Previsões com PLS Regression')

# Carregar o modelo PLS
pls = load_model()

# Passo 1: Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregue o arquivo CSV com os dados de entrada", type=["csv"])

if uploaded_file is not None:
    # Passo 2: Processar o arquivo CSV
    df = process_csv(uploaded_file)
    st.write("Dados carregados:")
    st.write(df.head())  # Exibe as primeiras linhas dos dados carregados

    # Passo 3: Selecione as variáveis de entrada (X) e, se disponível, o alvo (y)
    # Supondo que o arquivo tenha colunas para predição e as variáveis X de forma geral:
    X = df  # Aqui, estamos assumindo que todas as colunas são variáveis independentes para a previsão.

    # Passo 4: Fazer previsões com o modelo
    y_pred = pls.predict(X)

    # Exibir as previsões
    st.write("Previsões feitas pelo modelo:")
    st.write(y)
