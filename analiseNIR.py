import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Função para carregar o modelo PLS salvo
@st.cache_resource
def load_model():
    pls = joblib.load('pls.pkl')  # Substitua com o caminho correto se necessário
    return pls

# Função para processar o arquivo CSV com o mesmo pré-processamento usado no Jupyter
def process_csv(file):
    df = pd.read_csv(file, delimiter=";")
    
    # Corrigir nomes das colunas (virgula para ponto e remover ponto extra)
    df.columns = [col.replace('.', '').replace(',', '.') for col in df.columns]
    
    # Corrigir valores decimais e converter para float
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Início da interface do Streamlit
st.title('Previsões com PLS Regression')

# Carregar o modelo PLS
pls = load_model()

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregue o arquivo CSV com os dados de entrada", type=["csv"])

if uploaded_file is not None:
    # Processar o arquivo CSV
    df = process_csv(uploaded_file)
    st.write("Dados carregados:")
    st.write(df)

    # Selecionar variáveis independentes (assumindo que as 3 primeiras colunas são alvos)
    X = df.iloc[:, 3:]

    # Verificar se o número de colunas bate com o modelo
    if X.shape[1] != pls.estimators_[0].x_weights_.shape[0]:
        st.error(f"O número de colunas ({X.shape[1]}) não é compatível com o modelo treinado ({pls.estimators_[0].x_weights_.shape[0]}).")
    else:
        # Fazer previsões com o modelo
        y_pred = pls.predict(X)

        # Exibir previsões
        st.write("Previsões feitas pelo modelo (xÁgua, xEtanol, xDEC):")
        st.dataframe(pd.DataFrame(y_pred, columns=['xAgua_pred', 'xEtanol_pred', 'xDEC_pred']))
