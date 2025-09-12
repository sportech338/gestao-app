import streamlit as st

st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

from views.sidebar import sidebar_controls
from services.facebook_api import fetch_insights_daily
from views.daily import render_daily_tab
from views.daypart import render_daypart_tab
from views.detail import render_detail_tab

st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra.")

act_id, token, api_version, level, preset, since, until, ready = sidebar_controls()
if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

with st.spinner("Buscando dados da Meta…"):
    df_daily = fetch_insights_daily(
        act_id=act_id, token=token, api_version=api_version,
        since_str=str(since), until_str=str(until), level=level,
        product_name=st.session_state.get("daily_produto")
    )

if df_daily.empty:
    st.warning("Sem dados para o período.")
    st.stop()

tab_daily, tab_daypart, tab_detail = st.tabs(["📅 Visão diária", "⏱️ Horários", "📊 Detalhamento"])
with tab_daily:
    render_daily_tab(df_daily, act_id, token, api_version, level, since, until)
with tab_daypart:
    render_daypart_tab(act_id, token, api_version, since, until)
with tab_detail:
    render_detail_tab(act_id, token, api_version, level, since, until)
