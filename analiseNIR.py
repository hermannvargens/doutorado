import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
st.title("Visualização de Espectros NIR")

uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=";")
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
    
    # Criar o gráfico
    st.subheader("Espectros NIR")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for column in df_espectros.columns:
        ax.plot(df_espectros.index, df_espectros[column], label=column)
    
    ax.set_xlabel("Comprimento de Onda")
    ax.set_ylabel("Absorbância")
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(df_espectros.index), 10))
    
    st.pyplot(fig)
