import streamlit as st
from datetime import date, timedelta, datetime
from config.constants import PRESETS, APP_TZ

def _range_from_preset(p):
    local_today = datetime.now(APP_TZ).date()
    base_end = local_today - timedelta(days=1)
    if p=="Hoje": return local_today, local_today
    if p=="Ontem": return local_today-timedelta(days=1), local_today-timedelta(days=1)
    if p=="Últimos 7 dias": return base_end-timedelta(days=6), base_end
    if p=="Últimos 14 dias": return base_end-timedelta(days=13), base_end
    if p=="Últimos 30 dias": return base_end-timedelta(days=29), base_end
    if p=="Últimos 90 dias": return base_end-timedelta(days=89), base_end
    if p=="Esta semana":
        start_week = local_today - timedelta(days=local_today.weekday())
        return start_week, local_today
    if p=="Este mês": return local_today.replace(day=1), local_today
    if p=="Máximo": return date(2017,1,1), base_end
    return base_end-timedelta(days=6), base_end

def sidebar_controls():
    st.sidebar.header("Configuração")
    act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
    token = st.sidebar.text_input("Access Token", type="password")
    api_version = st.sidebar.text_input("API Version", value="v23.0")
    level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign"], index=0)
    preset = st.sidebar.radio("Período rápido", PRESETS, index=2)

    _since_auto, _until_auto = _range_from_preset(preset)
    if preset=="Personalizado":
        since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
        until = st.sidebar.date_input("Até", value=_until_auto, key="until_custom")
    else:
        since, until = _since_auto, _until_auto
    st.sidebar.caption(f"**Desde:** {since} \n**Até:** {until}")
    ready = bool(act_id and token)
    return act_id, token, api_version, level, preset, since, until, ready
