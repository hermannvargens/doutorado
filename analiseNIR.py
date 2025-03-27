import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
