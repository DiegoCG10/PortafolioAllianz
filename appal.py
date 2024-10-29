# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


# Estilo CSS para ajustar el ancho de la columna que muestra el nombre completo del ETF
st.markdown(
    """
    <style>
    .dataframe th {
        white-space: nowrap;
        width: auto; /* El ancho automático permite que la columna ajuste su tamaño */
    }
    .dataframe td:nth-child(1) {
        width: 300px; /* Ajusta este valor para la columna de nombres de ETFs */
    }
    .dataframe th:nth-child(4) {
        width: 150px; /* Ajusta este valor para la columna de "Crecimiento Final Histórico" */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción
st.title("Simulador Comparativo de ETFs")
st.markdown("Simula y compara diferentes ETFs en función de una inversión inicial y rendimiento esperado.")

# Parámetros de inversión
st.sidebar.header("Parámetros de la inversión")
inversion_inicial = st.sidebar.number_input("Inversión Inicial (USD)", value=10000, step=500)
rendimiento_esperado = st.sidebar.slider("Rendimiento Esperado Anual (%)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)

# Slider para los primeros 5 años
plazo_inversion = st.sidebar.slider("Plazo de Inversión (años)", min_value=1, max_value=5, value=1)

# Selectbox para opciones de 5 o 10 años si se selecciona 5 en el slider
if plazo_inversion == 5:
    plazo_inversion = st.sidebar.selectbox("Extiende el Plazo a 10 años?", [5, 10], index=0)

etf_list = {
    "SPDR S&P 500 (SPY)": "SPY",
    "iShares MSCI Emerging (EEM)": "EEM",
    "Vanguard Total Stock (VTI)": "VTI",
    "Invesco QQQ (QQQ)": "QQQ",
    "iShares Russell (IWM)": "IWM",
    "SPDR DJIA Trust (DIA)": "DIA",
    "Vanguard Emerging Market (VWO)": "VWO",
    "Financial Select Sector SPDR (XLF)": "XLF",
    "Health Care Select Sector (XLV)": "XLV",
    "DJ US Home Construct (ITB)": "ITB",
    "Silver Trust (SLV)": "SLV",
    "MSCI Taiwan Index FD (EWT)": "EWT",
    "MSCI United Kingdom (EWU)": "EWU",
    "MSCI South Korea IND (EWY)": "EWY",
    "MSCI Japan Index FD (EWJ)": "EWJ"
}
selected_etfs = st.sidebar.multiselect("Seleccione los ETFs a comparar", list(etf_list.keys()), default=[])

# Función para obtener datos del ETF
def get_etf_data(ticker, period="5y"):
    return yf.download(ticker, period=period)["Close"]

# Simulación de crecimiento esperado
def calculate_expected_growth(inicial, rate, years):
    return inicial * ((1 + rate) ** years)

# Cálculo del Valor en Riesgo (VaR)
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

# Cálculo del Ratio de Sharpe
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(returns)

# Diccionario para almacenar resultados
etf_results = {}

# Realizar la simulación y mostrar los resultados
st.write(f"### Comparación de Inversión: {plazo_inversion} años con una Inversión Inicial de ${inversion_inicial:,.2f}")

for etf_name in selected_etfs:
    ticker_symbol = etf_list[etf_name]
    etf_data = get_etf_data(ticker_symbol, period=f"{plazo_inversion}y")
    
    if etf_data.empty:
        st.warning(f"No se encontraron datos para {etf_name}")
        continue
        # Calcular rendimientos anuales
    etf_data["Return"] = etf_data.pct_change()
    annual_return = etf_data["Return"].mean() * 252  # asumiendo 252 días de operación al año
    volatility = etf_data["Return"].std() * np.sqrt(252)  # Volatilidad anualizada
    
    # Calcular crecimiento final basado en rendimientos históricos
    crecimiento_historico = inversion_inicial * (1 + annual_return) ** plazo_inversion

        # Calcular el Valor en Riesgo (VaR)
    var = calculate_var(etf_data["Return"].dropna())  # Excluyendo NaN
    
    # Calcular el Ratio de Sharpe
    sharpe_ratio = calculate_sharpe_ratio(etf_data["Return"].dropna())
    
     # Almacenar resultados para la tabla comparativa
    etf_results[etf_name] = {
        "Símbolo": ticker_symbol,
        "Rendimiento Anual (%)": f"{annual_return * 100:.2f}%",
        "Volatilidad (%)": f"{volatility * 100:.2f}%",
        "Crecimiento Final Histórico": crecimiento_historico,
        "VaR (95%)": f"{var * 100:.2f}%",  # Convertir a porcentaje
        "Ratio de Sharpe": f"{sharpe_ratio:.2f}"
    }

# Mostrar tabla comparativa de resultados
resultados_df = pd.DataFrame(etf_results).T
st.write("### Resultados Comparativos de ETFs")
st.dataframe(resultados_df, use_container_width=True)

# Establecer estilo de Seaborn
sns.set(style="whitegrid")

# Gráfico de rendimiento de cada ETF durante el plazo seleccionado
plt.figure(figsize=(10, 6))

# Agregar cada ETF al gráfico de rendimiento como línea
for etf_name in selected_etfs:
    ticker_symbol = etf_list[etf_name]
    etf_data = get_etf_data(ticker_symbol, period=f"{plazo_inversion}y")
    
    # Calcular el rendimiento diario acumulado
    etf_data['Crecimiento Acumulado'] = (1 + etf_data.pct_change()).cumprod() - 1  # Rendimiento acumulado

    plt.plot(etf_data.index, etf_data['Crecimiento Acumulado'], label=etf_name)

# Configuración del gráfico de rendimiento
plt.title("Comportamiento del Rendimiento Acumulado de los ETFs")
plt.xlabel("Fecha")
plt.ylabel("Rendimiento Acumulado (%)")
plt.legend()
plt.grid(True)

# Mostrar el gráfico de rendimiento acumulado
st.pyplot(plt)

# Crecimiento esperado al final del periodo
rendimiento_esperado_decimal = rendimiento_esperado / 100
crecimiento_esperado = calculate_expected_growth(inversion_inicial, rendimiento_esperado_decimal, plazo_inversion)

# Gráfico de comparación de crecimiento
plt.figure(figsize=(10, 6))

# Agregar cada ETF al gráfico
for etf_name, resultados in etf_results.items():
    plt.bar(etf_name, resultados["Crecimiento Final Histórico"], label=f"{etf_name} - Crecimiento Final")

# Agregar línea de crecimiento esperado
plt.axhline(y=crecimiento_esperado, color='r', linestyle='--', label="Crecimiento Esperado")

plt.title("Comparación de Crecimiento Final vs. Crecimiento Esperado")
plt.xlabel("ETF")
plt.ylabel("Valor Final de la Inversión (USD)")
plt.legend()
plt.grid(axis='y')

# Mostrar gráfico de comparación de crecimiento
st.pyplot(plt)
# Resultados finales mejorados
st.markdown("## 📊 Resumen de Resultados Finales")

for etf_name, resultados in etf_results.items():
    # Encabezado del ETF con estilo y color
    st.markdown(f"<h3 style='color:#4CAF50;'>🚀 {etf_name}</h3>", unsafe_allow_html=True)
    
    # Tarjeta de rendimiento y volatilidad
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; padding: 16px; border-radius: 8px; background-color: #f9f9f9;">
            <p><strong>📈 Rendimiento y Volatilidad:</strong></p>
            <ul>
                <li><strong>Crecimiento Histórico:</strong> <span style="color:#4CAF50;">${resultados['Crecimiento Final Histórico']:,.2f}</span></li>
                <li><strong>Rendimiento Anualizado:</strong> {resultados['Rendimiento Anual (%)']}</li>
                <li><strong>Volatilidad Anualizada:</strong> {resultados['Volatilidad (%)']}</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Evaluación del rendimiento en comparación con el crecimiento esperado
    if resultados["Crecimiento Final Histórico"] > crecimiento_esperado:
        st.markdown(
            "<p style='color:green;'><strong>✅ Superó el crecimiento esperado</strong> 👍 ¡Este ETF tuvo un rendimiento excelente y superó las expectativas!</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='color:red;'><strong>❌ No alcanzó el crecimiento esperado</strong> 👎 El rendimiento fue inferior a lo esperado; considera analizar los factores de riesgo.</p>",
            unsafe_allow_html=True
        )
    
    # Tarjeta de análisis de riesgo
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; padding: 16px; border-radius: 8px; background-color: #f2f2f2;">
            <p><strong>🔹 Análisis de Riesgo:</strong></p>
            <ul>
                <li><strong>VaR (95%):</strong> {resultados['VaR (95%)']}</li>
                <li><strong>Ratio de Sharpe:</strong> {resultados['Ratio de Sharpe']} (indica rendimiento ajustado por riesgo)</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Tarjeta de conclusiones y recomendaciones
    st.markdown(
        """
        <div style="border: 1px solid #ddd; padding: 16px; border-radius: 8px; background-color: #e7f3fe;">
            <p><strong>📌 Conclusiones y Recomendaciones:</strong></p>
            <ul>
                <li><strong>Diversificación:</strong> Este ETF puede complementar un portafolio diversificado dependiendo de su rendimiento y nivel de riesgo.</li>
                <li><strong>Monitoreo de Volatilidad:</strong> Se recomienda revisar su volatilidad periódicamente para ajustar al perfil de riesgo personal.</li>
                <li><strong>Alineación de Objetivos:</strong> Evalúa si el rendimiento de este ETF está alineado con tus objetivos financieros a largo plazo.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Separador visual entre ETFs
    st.write("<hr style='border:1px solid #ddd; margin:20px 0;'>", unsafe_allow_html=True)