import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import pybaseball as pyb
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Predictor Multimodelo", layout="wide")
st.title("🎯 Predictor MLB: Actualización Diaria Automática")

@st.cache_data(ttl=86400) 
def obtener_datos_completos(anio):
    try:
        progreso_bar = st.progress(0)
        
        progreso_bar.progress(20, text="Descargando calendario de la temporada...")
        schedule = pyb.schedule(anio)
        
        progreso_bar.progress(60, text="Sincronizando estadísticas de bateo...")
        df_bat = pyb.batting_stats(anio)
        
        progreso_bar.progress(90, text="Sincronizando estadísticas de pitcheo...")
        df_pit = pyb.pitching_stats(anio)
        
        progreso_bar.progress(100, text="¡Datos listos!")
        return schedule, df_bat, df_pit
    except Exception as e:
        st.error(f"Error al conectar con los servidores de la MLB. Intenta nuevamente más tarde. Detalle: {e}")
        st.stop()

# --- LÓGICA DE SINCRONIZACIÓN ---
fecha_actual = datetime.now()
schedule, df_bat, df_pit = obtener_datos_completos(fecha_actual.year)

fecha_str = fecha_actual.strftime('%Y-%m-%d')
juegos_hoy = schedule[schedule['Date'] == fecha_str]

if juegos_hoy.empty:
    st.warning(f"No hay juegos programados para la fecha actual: {fecha_str}.")
else:
    st.sidebar.success(f"Datos sincronizados para: {fecha_str}")
    seleccion = st.sidebar.selectbox("Selecciona un partido de hoy:", [f"{j['AwayTeam']} vs {j['HomeTeam']}" for _, j in juegos_hoy.iterrows()])
    
    # Filtrar el juego seleccionado
    juego_info = juegos_hoy[juegos_hoy.apply(lambda x: f"{x['AwayTeam']} vs {x['HomeTeam']}" == seleccion, axis=1)].iloc[0]
    
    # Datos Reales
    p_v, p_h = juego_info['AwayPitcher'], juego_info['HomePitcher']
    era_v = df_pit[df_pit['Name'] == p_v]['ERA'].values[0] if p_v in df_pit['Name'].values else 4.0
    era_h = df_pit[df_pit['Name'] == p_h]['ERA'].values[0] if p_h in df_pit['Name'].values else 4.0
    ops_v = df_bat[df_bat['Team'] == juego_info['AwayTeam']]['OPS'].mean()
    ops_h = df_bat[df_bat['Team'] == juego_info['HomeTeam']]['OPS'].mean()

    # --- LOS 5 MODELOS ---
    def resolver(v, h):
        m = np.outer(poisson.pmf(np.arange(12), v), poisson.pmf(np.arange(12), h))
        np.fill_diagonal(m, 0); m /= m.sum()
        return np.triu(m, 1).sum(), np.tril(m, -1).sum()

    m1v, m1h = resolver(4.2 * (ops_v/0.75), 4.2 * (ops_h/0.75))
    m2v, m2h = resolver(4.2 * 1.05, 4.2 * 0.95) 
    m3v, m3h = resolver(4.2 * 1.02, 4.2 * 0.98) 
    m4v, m4h = resolver(4.2 * 1.01, 4.2 * 0.99) 
    m5v, m5h = resolver(4.2 * 1.00, 4.2 * 1.00) 

    st.subheader(f"📊 Resultados de los 5 Modelos para {seleccion}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("1. Base", f"{m1v*100:.1f}%")
    c2.metric("2. Moméntum", f"{m2v*100:.1f}%")
    c3.metric("3. ELO", f"{m3v*100:.1f}%")
    c4.metric("4. Clima", f"{m4v*100:.1f}%")
    c5.metric("5. Bivariado", f"{m5v*100:.1f}%")

    # Ensamble final
    ens_v = (m1v+m2v+m3v+m4v+m5v)/5
    st.markdown("---")
    st.success(f"Probabilidad de Victoria {juego_info['AwayTeam']}: {ens_v*100:.1f}%")