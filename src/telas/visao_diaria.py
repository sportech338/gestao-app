import streamlit as st
import pandas as pd
import numpy as np
from ..recursos.filtros import filtrar_por_produto
from ..regras.formatacao import dinheiro_br, ratio_br, int_br, pct_br
from ..regras.metricas import rate, safe_div
from ..recursos.graficos.funil import funnel_fig, enforce_monotonic

def render(df_daily: pd.DataFrame, params: dict):
    st.title("📊 Meta Ads — Paridade com Filtro + Funil")
    st.caption("KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra.")

    if df_daily is None or df_daily.empty:
        st.warning("Sem dados para o período."); st.stop()

    produto_sel = st.selectbox("Filtrar por produto (opcional)", params["produtos"], key="daily_produto")
    df = filtrar_por_produto(df_daily, produto_sel) if produto_sel != "(Todos)" else df_daily.copy()
    if df.empty:
        st.info("Sem dados para o produto selecionado."); st.stop()

    # KPIs
    tot_spend = float(df["spend"].sum())
    tot_purch = float(df["purchases"].sum())
    tot_rev   = float(df["revenue"].sum())
    roas_g    = (tot_rev / tot_spend) if tot_spend > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="small-muted">Valor usado</div><div class="big-number">{dinheiro_br(tot_spend)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="small-muted">Vendas</div><div class="big-number">{int_br(tot_purch)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="small-muted">Valor de conversão</div><div class="big-number">{dinheiro_br(tot_rev)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="small-muted">ROAS</div><div class="big-number">{ratio_br(roas_g) if pd.notnull(roas_g) else "—"}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Série diária
    daily = df.groupby("date", as_index=False)[["spend","revenue"]].sum().rename(columns={"spend":"Gasto","revenue":"Faturamento"})
    st.line_chart(daily.set_index("date")[["Faturamento","Gasto"]])

    # Funil total
    f_clicks = float(df["link_clicks"].sum())
    f_lpv    = float(df["lpv"].sum())
    f_ic     = float(df["init_checkout"].sum())
    f_api    = float(df["add_payment"].sum())
    f_pur    = float(df["purchases"].sum())

    labels_total = ["Cliques","LPV","Checkout","Add Pagamento","Compra"]
    values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)), int(round(f_api)), int(round(f_pur))]
    force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
    values_plot = enforce_monotonic(values_total) if force_shape else values_total
    st.plotly_chart(funnel_fig(labels_total, values_plot, title="Funil do período"), use_container_width=True)
""")
