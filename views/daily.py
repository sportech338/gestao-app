import streamlit as st, numpy as np, pandas as pd
from config.constants import PRODUTOS
from utils.formatting import fmt_money_br, fmt_ratio_br, fmt_int_br
from utils.metrics import funnel_fig, enforce_monotonic
from utils.helpers import filter_by_product

def render_daily_tab(df_daily, act_id, token, api_version, level, since, until):
    # moeda detectada
    currency_detected = (df_daily["currency"].dropna().iloc[0]
                         if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty
                         else "BRL")
    colA, colB = st.columns([1,2])
    with colA:
        use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True)
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
    st.line_chart(daily.set_index("date")[["Faturamento", "Gasto"]])
    st.caption("Linhas diárias de Receita e Gasto.")

    st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")
    f_clicks = float(df_view["link_clicks"].sum())
    f_lpv    = float(df_view["lpv"].sum())
    f_ic     = float(df_view["init_checkout"].sum())
    f_api    = float(df_view["add_payment"].sum())
    f_pur    = float(df_view["purchases"].sum())

    labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
    values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)), int(round(f_api)), int(round(f_pur))]
    force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
    values_plot = enforce_monotonic(values_total) if force_shape else values_total
    st.plotly_chart(funnel_fig(labels_total, values_plot, title="Funil do período"), use_container_width=True)

    # espaço para evolução futura (taxas/bandas/comparativos)
    with st.expander("Extras (em breve)"):
        st.write("Você pode colar aqui os gráficos de taxas/deltas/comparativos do seu código original.")
