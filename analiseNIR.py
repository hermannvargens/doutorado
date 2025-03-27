import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados (Substitua pelo carregamento adequado do seu dataset)
@st.cache_data
def load_data():
    # Simulação de um dataset (Substituir pelo real)
    df = pd.read_csv("espectros_derivada.csv")
    return df

df = load_data()

# Dividir o df, tomando da 4ª coluna em diante, onde estão os valores de Absorbance
df_espectros = df.iloc[:, 3:]

# Transpor os dados
df_espectros = df_espectros.T

# Criar interface
st.title("Visualização de Espectros NIR")
st.write("Selecione os espectros que deseja visualizar:")

# Selecionar amostras específicas
samples = st.multiselect("Amostras", df_espectros.columns, default=df_espectros.columns[:5])

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for column in samples:
    ax.plot(df_espectros.index, df_espectros[column], label=f"Amostra {column}")

ax.set_xlabel("Wavelength")
ax.set_ylabel("Absorbance")
ax.grid(True)
ax.set_xticks(np.arange(0, len(df_espectros.index), 10))
ax.legend()
st.pyplot(fig)
