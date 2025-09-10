import streamlit as st
from datetime import date, timedelta, datetime
from ..configuracao import APP_TZ
from ..configuracao import PRODUTOS

def _range_from_preset(preset: str):
    local_today = datetime.now(APP_TZ).date()
    base_end = local_today - timedelta(days=1)
    if preset == "Hoje":               return local_today, local_today
    if preset == "Ontem":              return local_today - timedelta(days=1), local_today - timedelta(days=1)
    if preset == "Últimos 7 dias":     return base_end - timedelta(days=6), base_end
    if preset == "Últimos 14 dias":    return base_end - timedelta(days=13), base_end
    if preset == "Últimos 30 dias":    return base_end - timedelta(days=29), base_end
    if preset == "Últimos 90 dias":    return base_end - timedelta(days=89), base_end
    if preset == "Esta semana":
        start_week = local_today - timedelta(days=local_today.weekday())
        return start_week, local_today
    if preset == "Este mês":
        start_month = local_today.replace(day=1)
        return start_month, local_today
    if preset == "Máximo":             return date(2017, 1, 1), base_end
    return base_end - timedelta(days=6), base_end

def montar_sidebar():
    st.sidebar.header("Configuração")
    act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
    token = st.sidebar.text_input("Access Token", type="password")
    api_version = st.sidebar.text_input("API Version", value="v23.0")
    level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign"], index=0)

    preset = st.sidebar.radio(
        "Período rápido",
        ["Hoje","Ontem","Últimos 7 dias","Últimos 14 dias","Últimos 30 dias","Últimos 90 dias",
         "Esta semana","Este mês","Máximo","Personalizado"],
        index=2,
    )

    _since_auto, _until_auto = _range_from_preset(preset)

    if preset == "Personalizado":
        since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
        until = st.sidebar.date_input("Até",   value=_until_auto, key="until_custom")
    else:
        since, until = _since_auto, _until_auto
        st.sidebar.caption(f"**Desde:** {since}  \n**Até:** {until}")

    pronto = bool(act_id and token)

    return {
        "act_id": act_id,
        "token": token,
        "api_version": api_version,
        "level": level,
        "since": since,
        "until": until,
        "preset": preset,
        "pronto": pronto,
        "produtos": ["(Todos)"] + PRODUTOS
    }
