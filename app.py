import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------
# Simulación de datos
# ----------------------
np.random.seed(42)

estaciones = ["Portal Norte", "Av. Jimenez", "Calle 72", "Portal Sur", "Universidades"]
horas = ["6:00", "7:00", "8:00", "12:00", "16:00", "18:00", "20:00"]
dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
climas = ["Soleado", "Lluvia", "Nublado"]
eventos = ["Sí", "No"]

data = {
    "estacion": np.random.choice(estaciones, 1000),
    "hora": np.random.choice(horas, 1000),
    "dia": np.random.choice(dias, 1000),
    "clima": np.random.choice(climas, 1000),
    "evento_cercano": np.random.choice(eventos, 1000)
}

df = pd.DataFrame(data)

# ----------------------
# Asignar afluencia según reglas
# ----------------------
def asignar_afluencia(row):
    if row["hora"] in ["6:00", "7:00", "8:00", "16:00", "18:00"] and row["dia"] in dias[:5]:
        if row["clima"] == "Lluvia" or row["evento_cercano"] == "Sí":
            return "Alta"
        return np.random.choice(["Alta", "Media"], p=[0.7, 0.3])
    elif row["dia"] in ["Sábado", "Domingo"]:
        return np.random.choice(["Media", "Baja"], p=[0.3, 0.7])
    else:
        return "Media"

df["afluencia"] = df.apply(asignar_afluencia, axis=1)

# ----------------------
# Codificar variables categóricas
# ----------------------
encoders = {}
for col in df.columns[:-1]:  # todas menos 'afluencia'
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Codificar afluencia (target)
le_afluencia = LabelEncoder()
y = le_afluencia.fit_transform(df["afluencia"])

X = df.drop("afluencia", axis=1)

# Entrenar modelo
model = RandomForestClassifier()
model.fit(X, y)

# ----------------------
# Interfaz con Streamlit
# ----------------------
st.title("🔍 Predicción de Afluencia en TransMilenio")

estacion = st.selectbox("🚏 Estación", estaciones)
hora = st.selectbox("🕒 Hora del día", horas)
dia = st.selectbox("📅 Día de la semana", dias)
clima = st.selectbox("🌤️ Clima", climas)
evento = st.radio("🎤 ¿Hay evento cercano?", eventos)

if st.button("Predecir afluencia"):
    # Preprocesar inputs
    input_data = pd.DataFrame({
        "estacion": [encoders["estacion"].transform([estacion])[0]],
        "hora": [encoders["hora"].transform([hora])[0]],
        "dia": [encoders["dia"].transform([dia])[0]],
        "clima": [encoders["clima"].transform([clima])[0]],
        "evento_cercano": [encoders["evento_cercano"].transform([evento])[0]]
    })

    # Predecir
    pred = model.predict(input_data)
    resultado = le_afluencia.inverse_transform(pred)[0]

    # Mostrar resultado
    st.success(f"📈 La afluencia esperada es: **{resultado}**")
