from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

import base64
from io import BytesIO

app = Flask(__name__)

# Carregar modelo (substitua pelo nome correto do modelo)
model = joblib.load('rf.pkl')  # ou 'pls.pkl', 'knn.pkl'

def process_csv(file_stream):
    df = pd.read_csv(file_stream)
    df.columns = [col.replace('.', '').replace(',', '.') for col in df.columns]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def plot_spectrum(df):
    step = 5
    x_values = np.arange(0, len(df.columns), step)
    x_labels = [str(i) for i in range(0, len(df.columns), step)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.iloc[0, :])
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True)
    ax.set_title("Espectro")
    ax.set_xlabel("Absorvância")
    ax.set_ylabel("Wavelength (nm)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    table_html = ""
    spectrum_img = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = process_csv(file)
            X = df.values
            y_pred = model.predict(X)
            prediction = pd.DataFrame(y_pred, columns=['xAgua_pred', 'xEtanol_pred', 'xDEC_pred'])

            table_html = prediction.to_html(index=False)
            spectrum_img = plot_spectrum(df)

    return render_template("index.html", table=table_html, spectrum_img=spectrum_img)

if __name__ == "__main__":
    app.run()
