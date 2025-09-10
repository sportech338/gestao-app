import streamlit as st

from src.ui.estilos import aplicar_config_e_css
from src.ui.barra_lateral import montar_sidebar
from src.servicos.meta_insights import fetch_insights_daily
from src.telas import visao_diaria, horarios, detalhamento

aplicar_config_e_css()
params = montar_sidebar()

if not params["pronto"]:
    st.title("📊 Meta Ads — Paridade + Funil")
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

@st.cache_data(ttl=600, show_spinner=True)
def _cache_daily(act_id, token, api_version, since, until, level, produto):
    return fetch_insights_daily(act_id=act_id, token=token, api_version=api_version,
                                since_str=str(since), until_str=str(until), level=level, product_name=produto)

df_daily = _cache_daily(params["act_id"], params["token"], params["api_version"],
                        params["since"], params["until"], params["level"],
                        st.session_state.get("daily_produto"))

if df_daily is None or df_daily.empty:
    st.warning("Sem dados para o período."); st.stop()

tab1, tab2, tab3 = st.tabs(["📅 Visão diária", "⏱️ Horários (principal)", "📊 Detalhamento"])
with tab1: visao_diaria.render(df_daily, params)
with tab2: horarios.render(params)
with tab3: detalhamento.render(params, df_daily)
""")
