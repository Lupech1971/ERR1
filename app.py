
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="Modelo de Regresión", layout="wide")

st.title("📊 Pronóstico mediante Modelo de Regresión")

# Cargar archivo de datos
uploaded_file = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    # Mostrar datos originales
    st.subheader("Datos cargados")
    st.dataframe(df)

    # Transponer si es necesario
    if st.checkbox("¿Transponer datos?"):
        df = df.transpose()
        df.columns = df.iloc[0]
        df = df[1:]
        df.reset_index(drop=True, inplace=True)

    # Convertir columnas a numéricas si es posible
    df = df.apply(pd.to_numeric, errors='coerce')

    # Selección de variable dependiente
    target_variable = st.selectbox("Selecciona la variable dependiente (Ventas)", df.columns)

    # Panel lateral para ingresar valores de pronóstico
    st.sidebar.header("🔢 Pronóstico manual")
    input_values = {}
    for col in df.columns:
        if col != target_variable:
            input_values[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # Cálculo de regresiones simples
    st.subheader("📈 Regresiones Simples")
    results = []
    for col in df.columns:
        if col != target_variable:
            x = df[[col]].dropna()
            y = df[target_variable].dropna()
            common_index = x.index.intersection(y.index)
            x = x.loc[common_index]
            y = y.loc[common_index]

            model = LinearRegression()
            model.fit(x, y)
            r2 = model.score(x, y)
            alpha = model.intercept_
            beta = model.coef_[0]
            forecast = alpha + beta * input_values[col]
            results.append({
                "Variable": col,
                "Pendiente (β)": beta,
                "Intersección (α)": alpha,
                "R²": r2,
                "Pronóstico": forecast
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Pronóstico múltiple ponderado por R²
    st.subheader("📊 Pronóstico Múltiple Ponderado por R²")
    if not results_df.empty:
        total_r2 = results_df["R²"].sum()
        if total_r2 > 0:
            weighted_forecast = sum(results_df["Pronóstico"] * results_df["R²"]) / total_r2
            st.metric("Pronóstico ponderado", round(weighted_forecast, 2))
        else:
            st.warning("R² total es cero. No se puede calcular pronóstico ponderado.")

    # Botón para descargar resultados
    st.subheader("📥 Descargar resultados")
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Resultados')
        return output.getvalue()

    excel_data = to_excel(results_df)
    st.download_button(
        label="Descargar Excel",
        data=excel_data,
        file_name="resultados_regresion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Por favor, carga un archivo para comenzar.")
