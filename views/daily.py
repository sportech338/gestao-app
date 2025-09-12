from __future__ import annotations


import streamlit as st, numpy as np, pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta


from config.constants import PRODUTOS
from utils.formatting import (
fmt_money_br, fmt_ratio_br, fmt_int_br, fmt_pct_br, fmt_int_signed_br, safe_div,
)
from utils.metrics import funnel_fig, enforce_monotonic
from utils.helpers import filter_by_product


try:
# Opcional – usado para período anterior no bloco de taxas por dia
from services.facebook_api import fetch_insights_daily # type: ignore
except Exception: # pragma: no cover
fetch_insights_daily = None # noqa: F401




def render_daily_tab(df_daily: pd.DataFrame, act_id: str, token: str, api_version: str, level: str, since: date, until: date):
# moeda detectada
currency_detected = (df_daily["currency"].dropna().iloc[0]
if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty
else "BRL")
colA, colB = st.columns([1,2])
with colA:
use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True, key="daily_use_brl")
with colB:
if use_brl_display and currency_detected != "BRL":
st.caption("⚠️ Símbolo **R$** só para formatação visual. Valores permanecem na moeda da conta.")
st.caption(f"Moeda da conta: **{currency_detected}**")


# filtro por produto
produto_sel = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daily_produto")
df_view = filter_by_product(df_daily, produto_sel)
if df_view.empty:
st.info("Sem dados para o produto selecionado nesse período.")
return


# KPIs totais
tot_spend = float(df_view["spend"].sum())
tot_purch = float(df_view["purchases"].sum())
tot_rev = float(df_view["revenue"].sum())
roas_g = (tot_rev / tot_spend) if tot_spend > 0 else np.nan


c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="kpi-card"><div class="small-muted">Valor usado</div><div class="big-number">{fmt_money_br(tot_spend)}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="small-muted">Vendas</div><div class="big-number">{fmt_int_br(tot_purch)}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="small-muted">Valor de conversão</div><div class="big-number">{fmt_money_br(tot_rev)}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card"><div class="small-muted">ROAS</div><div class="big-number">{fmt_ratio_br(roas_g)}</div></div>', unsafe_allow_html=True)


st.divider()


# Série diária
st.subheader("Série diária — Investimento e Conversão")
daily = df_view.groupby("date", as_index=False)[["spend", "revenue"]].sum()
daily = daily.rename(columns={"spend":"Gasto", "revenue":"Faturamento"})
st.line_chart(daily.set_index("date")[ ["Faturamento", "Gasto"] ])
""")
