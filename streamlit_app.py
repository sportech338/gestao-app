import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# =========================
# Config
# =========================
st.set_page_config(page_title="Metas & Performance — Simples", layout="wide")
st.title("📊 Metas & Performance — Simples")
st.caption("Defina a META MENSAL e o app reparte automaticamente a META SEMANAL da semana selecionada.")

# =========================
# Sidebar — Parâmetros essenciais
# =========================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    target_roas = st.number_input("ROAS alvo", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.subheader("📅 Referências de Tempo")
    # Semana baseia-se nesta data (segunda a domingo)
    week_start = st.date_input(
        "Início da semana (segunda)",
        value=(datetime.today() - timedelta(days=datetime.today().weekday())).date(),
    )
    include_weekends = st.checkbox("Metas consideram finais de semana", value=True, help="Se desmarcar, metas diárias ignoram sábados e domingos.")

    # Mês de referência para a META MENSAL
    month_ref = st.date_input("Qualquer dia do mês da meta", value=datetime.today().date())

    st.subheader("🎯 META MENSAL (base de tudo)")
    goal_type_m = st.radio("Definir por", ["Faturamento", "Compras"], index=0, horizontal=True)
    monthly_goal_value = st.number_input("Valor da meta mensal (R$ se Faturamento; nº se Compras)", value=40000.0, min_value=0.0, step=500.0)

    st.subheader("📥 CSV do Gerenciador")
    uploaded = st.file_uploader("Envie o CSV (separador vírgula)", type=["csv"]) 

# =========================
# Helpers
# =========================

def daterange(start_date, end_date, include_weekends=True):
    days, d = [], start_date
    while d <= end_date:
        if include_weekends or d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

@st.cache_data(show_spinner=False)
def read_csv_flex(file):
    import re, unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = s.replace("\n"," ").replace("\r"," ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("(brl)", "").replace(" r$", "").strip()
        return s
    def _read(file):
        for enc in ["utf-8", "latin-1", "utf-16", "cp1252"]:
            try:
                return pd.read_csv(file, sep=None, engine="python", encoding=enc)
            except Exception:
                file.seek(0)
                continue
        file.seek(0)
        return pd.read_csv(file)
    raw = _read(file)

    # mapa simples de aliases
    ALIASES = {
        "campanha": ("nome da campanha", "campanha", "campaign name", "nome da campanha (id)"),
        "status": ("desativado/ativado", "ativado/desativado", "estado", "status da campanha"),
        "veiculacao": ("veiculacao", "veiculacao da campanha", "posicionamento"),
        "gasto": ("valor usado", "valor gasto", "amount spent", "spend", "valor usado brl"),
        "faturamento": ("valor de conversao da compra", "valor de conversão da compra", "purchase conversion val
