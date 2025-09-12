# views/daypart.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

from services.facebook_api import fetch_insights_hourly
from config.constants import PRODUTOS
from utils.formatting import fmt_money_br, fmt_ratio_br
from utils.helpers import filter_by_product


def render_daypart_tab(act_id: str, token: str, api_version: str, since, until):
    """
    Aba de horários completa:
      1) Heatmap Hora x Dia (métrica escolhida)
      2) Apanhado geral por hora (tabela + barras)
      3) Taxas por hora (% por etapa) + cumulativas
      4) Comparação A vs B (hora a hora) com 2 gráficos e insights
    """
    st.caption("Explore desempenho por hora: Heatmap, taxas, e comparação entre dois períodos.")

    # -------- Lazy-load hourly (cache local em sessão) --------
    level_hourly = "campaign"
    cache = st.session_state.setdefault("hourly_cache", {})
    hourly_key = (act_id, api_version, level_hourly, str(since), str(until))

    if hourly_key not in cache:
        with st.spinner("Carregando breakdown por hora…"):
            cache[hourly_key] = fetch_insights_hourly(
                act_id=act_id, token=token, api_version=api_version,
                since_str=str(since), until_str=str(until), level=level_hourly
            )
    dfall = cache[hourly_key]
    if dfall is None or dfall.empty:
        st.info("A conta/período não retornou breakdown por hora. Use a visão diária.")
        return

    # --------- Filtro por produto (opcional) ---------
    produto_sel_hr = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daypart_produto")
    d = filter_by_product(dfall, produto_sel_hr)
    if d is None or d.empty:
        st.info("Sem dados para o filtro selecionado.")
        return

    min_spend = st.slider("Gasto mínimo para considerar o horário (R$)", 0.0, 1000.0, 0.0, 10.0)

    d = d.dropna(subset=["hour"]).copy()
    d["hour"] = d["hour"].astype(int).clip(0, 23)
    d["date_only"] = d["date"].dt.date

    # ============== 1) HEATMAP HORA × DIA ==============
    st.subheader("📆 Heatmap — Hora × Dia")
    cube_hm = d.groupby(["dow_label", "hour"], as_index=False)[
        ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
    ].sum()
    cube_hm["roas"] = np.where(cube_hm["spend"] > 0, cube_hm["revenue"] / cube_hm["spend"], np.nan)
    if min_spend > 0:
        cube_hm = cube_hm[cube_hm["spend"] >= min_spend]

    metric_hm = st.selectbox("Métrica do heatmap", ["Compras", "Faturamento", "Gasto", "ROAS"], index=0, key="hm_metric_top")
    mcol = {"Compras": "purchases", "Faturamento": "revenue", "Gasto": "spend", "ROAS": "roas"}[metric_hm]

    if mcol == "roas":
        pvt = cube_hm.groupby(["dow_label", "hour"], as_index=False)[mcol].mean()
    else:
        pvt = cube_hm.groupby(["dow_label", "hour"], as_index=False)[mcol].sum()

    order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    pvt["dow_label"] = pd.Categorical(pvt["dow_label"], categories=order, ordered=True)
    pvt = pvt.sort_values(["dow_label", "hour"])
    heat = pvt.pivot(index="dow_label", columns="hour", values=mcol).fillna(0)
    hours_full = list(range(24))
    heat = heat.reindex(columns=hours_full, fill_value=0.0)
    heat.columns = list(range(24))

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorbar=dict(title=metric_hm),
            hovertemplate="Dia: %{y}<br>Hora: %{x}h<br>" + metric_hm + ": %{z}<extra></extra>",
        )
    )
    fig_hm.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white", separators=".,")
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ============== 2) APANHADO GERAL POR HORA ==============
    st.subheader("📦 Apanhado geral por hora (período selecionado)")
    cube_hr = d.groupby("hour", as_index=False)[
        ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
    ].sum()
    cube_hr["ROAS"] = np.where(cube_hr["spend"] > 0, cube_hr["revenue"] / cube_hr["spend"], np.nan)
    if min_spend > 0:
        cube_hr = cube_hr[cube_hr["spend"] >= min_spend]

    # tabela
    top_hr = cube_hr.sort_values(["purchases", "ROAS"], ascending=[False, False]).copy()
    show_cols = ["hour", "purchases", "ROAS", "spend", "revenue", "link_clicks", "lpv", "init_checkout", "add_payment"]
    disp_top = top_hr[show_cols].rename(
        columns={"hour": "Hora", "purchases": "Compras", "spend": "Valor usado", "revenue": "Valor de conversão"}
    )
    disp_top["Valor usado"] = disp_top["Valor usado"].apply(fmt_money_br)
    disp_top["Valor de conversão"] = disp_top["Valor de conversão"].apply(fmt_money_br)
    disp_top["ROAS"] = disp_top["ROAS"].map(fmt_ratio_br)
    st.dataframe(disp_top, use_container_width=True, height=360)

    # barras
    fig_bar = go.Figure(go.Bar(x=cube_hr.sort_values("hour")["hour"], y=cube_hr.sort_values("hour")["purchases"]))
    fig_bar.update_layout(
        title="Compras por hora (total do período)",
        xaxis_title="Hora do dia",
        yaxis_title="Compras",
        height=380,
        template="plotly_white",
        margin=dict(l=10, r=10, t=48, b=10),
        separators=".,",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info("Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos.")

    st.markdown("---")

    # ============== 3) TAXAS POR HORA ==============
    st.subheader("🎯 Taxas por hora — médias diárias (com cap de funil)")
    cube_hr_all = d.groupby("hour", as_index=False)[
        ["link_clicks", "lpv", "init_checkout", "add_payment", "purchases"]
    ].sum()
    # garantir 0..23
    cube_hr_all = (
        cube_hr_all.set_index("hour").reindex(list(range(24)), fill_value=0.0).rename_axis("hour").reset_index()
    )

    # cap para evitar >100%
    cube_hr_all["LPV_cap"] = np.minimum(cube_hr_all["lpv"], cube_hr_all["link_clicks"])
    cube_hr_all["Checkout_cap"] = np.minimum(cube_hr_all["init_checkout"], cube_hr_all["LPV_cap"])

    # taxas instantâneas
    def _safe_div(n, d):
        n = float(n or 0.0)
        d = float(d or 0.0)
        return (n / d) if d > 0 else np.nan

    cube_hr_all["tx_lpv_clicks"] = cube_hr_all.apply(lambda r: _safe_div(r["LPV_cap"], r["link_clicks"]), axis=1)
    cube_hr_all["tx_checkout_lpv"] = cube_hr_all.apply(lambda r: _safe_div(r["Checkout_cap"], r["LPV_cap"]), axis=1)
    cube_hr_all["tx_compra_checkout"] = cube_hr_all.apply(lambda r: _safe_div(r["purchases"], r["Checkout_cap"]), axis=1)

    # cumulativas
    cum = cube_hr_all.sort_values("hour").copy()
    cum["cum_clicks"] = cum["link_clicks"].cumsum()
    cum["cum_lpv"] = cum["lpv"].cumsum()
    cum["cum_ic"] = cum["init_checkout"].cumsum()
    cum["cum_purch"] = cum["purchases"].cumsum()
    cum["LPV_cap_cum"] = np.minimum(cum["cum_lpv"], cum["cum_clicks"])
    cum["Checkout_cap_cum"] = np.minimum(cum["cum_ic"], cum["LPV_cap_cum"])

    tx_lpv_clicks_cum = np.divide(
        cum["LPV_cap_cum"], cum["cum_clicks"], out=np.full(len(cum), np.nan), where=cum["cum_clicks"] > 0
    )
    tx_checkout_lpv_cum = np.divide(
        cum["Checkout_cap_cum"], cum["LPV_cap_cum"], out=np.full(len(cum), np.nan), where=cum["LPV_cap_cum"] > 0
    )
    tx_compra_checkout_cum = np.divide(
        cum["cum_purch"], cum["Checkout_cap_cum"], out=np.full(len(cum), np.nan), where=cum["Checkout_cap_cum"] > 0
    )

    show_cum = st.checkbox("Mostrar linha cumulativa (até a hora)", value=True, key="hr_show_cum")
    show_band = st.checkbox("Mostrar banda saudável (faixa alvo)", value=True, key="hr_show_band")

    # herdar ou defaults
    def _get_band_from_state(key, default_pair):
        v = st.session_state.get(key)
        return v if (isinstance(v, tuple) and len(v) == 2) else default_pair

    _lpv_lo_def, _lpv_hi_def = _get_band_from_state("tx_lpv_cli_band", (70, 85))
    _co_lo_def, _co_hi_def = _get_band_from_state("tx_co_lpv_band", (10, 20))
    _buy_lo_def, _buy_hi_def = _get_band_from_state("tx_buy_co_band", (25, 40))

    b1, b2, b3 = st.columns(3)
    with b1:
        lpv_cli_low, lpv_cli_high = st.slider("LPV/Cliques alvo (%)", 0, 100, (_lpv_lo_def, _lpv_hi_def), 1)
    with b2:
        co_lpv_low, co_lpv_high = st.slider("Checkout/LPV alvo (%)", 0, 100, (_co_lo_def, _co_hi_def), 1)
    with b3:
        buy_co_low, buy_co_high = st.slider("Compra/Checkout alvo (%)", 0, 100, (_buy_lo_def, _buy_hi_def), 1)

    def _line_hour_pct(x, y, title, band_range=None, y_aux=None, aux_label="Cumulativa"):
        fig = go.Figure(
            go.Scatter(
                x=x, y=y, mode="lines+markers", name=title,
                hovertemplate=f"<b>{title}</b><br>Hora: %{{x}}h<br>Taxa: %{{y:.2f}}%<extra></extra>",
            )
        )
        if show_band and band_range and len(band_range) == 2:
            lo, hi = band_range
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=-0.5, x1=23.5, y0=lo, y1=hi,
                fillcolor="rgba(34,197,94,0.10)", line=dict(width=0), layer="below"
            )
        if show_cum and y_aux is not None:
            fig.add_trace(go.Scatter(x=x, y=y_aux, mode="lines", name=f"{aux_label}", line=dict(width=3)))
        fig.update_layout(
            title=title, xaxis_title="Hora do dia", yaxis_title="Taxa (%)", height=340,
            template="plotly_white", margin=dict(l=10, r=10, t=48, b=10), separators=".,",
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        return fig

    x_hours = cube_hr_all["hour"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            _line_hour_pct(
                x_hours,
                cube_hr_all["tx_lpv_clicks"] * 100,
                "LPV/Cliques (%)",
                band_range=(lpv_cli_low, lpv_cli_high),
                y_aux=tx_lpv_clicks_cum * 100,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _line_hour_pct(
                x_hours,
                cube_hr_all["tx_checkout_lpv"] * 100,
                "Checkout/LPV (%)",
                band_range=(co_lpv_low, co_lpv_high),
                y_aux=tx_checkout_lpv_cum * 100,
            ),
            use_container_width=True,
        )
    with col3:
        st.plotly_chart(
            _line_hour_pct(
                x_hours,
                cube_hr_all["tx_compra_checkout"] * 100,
                "Compra/Checkout (%)",
                band_range=(buy_co_low, buy_co_high),
                y_aux=tx_compra_checkout_cum * 100,
            ),
            use_container_width=True,
        )

    st.caption("A linha amarela (opcional) mostra as taxas cumulativas até a hora. As faixas verdes indicam a banda saudável.")

    st.markdown("---")

    # ============== 4) COMPARAR DOIS PERÍODOS (A vs B) — HORA A HORA ==============
    st.subheader("🆚 Comparar dois períodos (A vs B) — hora a hora")

    base_len = (until - since).days + 1
    default_sinceA = since - timedelta(days=base_len)
    default_untilA = since - timedelta(days=1)

    colA1, colA2, colB1, colB2 = st.columns(4)
    with colA1:
        period_sinceA = st.date_input("Desde (A)", value=default_sinceA, key="cmp_sinceA")
    with colA2:
        period_untilA = st.date_input("Até (A)", value=default_untilA, key="cmp_untilA")
    with colB1:
        period_sinceB = st.date_input("Desde (B)", value=since, key="cmp_sinceB")
    with colB2:
        period_untilB = st.date_input("Até (B)", value=until, key="cmp_untilB")

    if period_sinceA > period_untilA or period_sinceB > period_untilB:
        st.warning("Confira as datas: em cada período, 'Desde' não pode ser maior que 'Até'.")
        return

    union_since = min(period_sinceA, period_sinceB)
    union_until = max(period_untilA, period_untilB)
    union_key = (act_id, api_version, level_hourly, str(union_since), str(union_until))

    if union_key not in cache:
        with st.spinner("Carregando dados por hora dos períodos selecionados…"):
            cache[union_key] = fetch_insights_hourly(
                act_id=act_id, token=token, api_version=api_version,
                since_str=str(union_since), until_str=str(union_until), level=level_hourly
            )
    df_union = cache[union_key]
    if df_union is None or df_union.empty:
        st.info("Sem dados no intervalo combinado dos períodos selecionados.")
        return

    # filtro produto no union
    if produto_sel_hr != "(Todos)":
        df_union = filter_by_product(df_union, produto_sel_hr)

    d_cmp = df_union.dropna(subset=["hour"]).copy()
    d_cmp["hour"] = d_cmp["hour"].astype(int).clip(0, 23)
    d_cmp["date_only"] = d_cmp["date"].dt.date

    A_mask = (d_cmp["date_only"] >= period_sinceA) & (d_cmp["date_only"] <= period_untilA)
    B_mask = (d_cmp["date_only"] >= period_sinceB) & (d_cmp["date_only"] <= period_untilB)
    datA, datB = d_cmp[A_mask], d_cmp[B_mask]
    if datA.empty or datB.empty:
        st.info("Sem dados em um dos períodos selecionados.")
        return

    agg_cols = ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
    gA = datA.groupby("hour", as_index=False)[agg_cols].sum()
    gB = datB.groupby("hour", as_index=False)[agg_cols].sum()

    merged = pd.merge(gA, gB, on="hour", how="outer", suffixes=(" (A)", " (B)")).fillna(0.0)
    if min_spend > 0:
        keep = (merged["spend (A)"] >= min_spend) | (merged["spend (B)"] >= min_spend)
        merged = merged[keep]

    # garantir 0..23
    merged = merged.set_index("hour").reindex(list(range(24)), fill_value=0.0).rename_axis("hour").reset_index()

    x = merged["hour"].astype(int)

    barsA_max = (merged["spend (A)"] + merged["revenue (A)"]).max()
    barsB_max = (merged["spend (B)"] + merged["revenue (B)"]).max()
    bars_max = max(barsA_max, barsB_max)
    bars_max = (bars_max if np.isfinite(bars_max) and bars_max > 0 else 1.0) * 1.05

    lineA_max = merged["purchases (A)"].max()
    lineB_max = merged["purchases (B)"].max()
    line_max = max(lineA_max, lineB_max)
    line_max = (line_max if np.isfinite(line_max) and line_max > 0 else 1.0) * 1.05

    # Gráfico A
    fig_A = make_subplots(specs=[[{"secondary_y": True}]])
    fig_A.add_trace(go.Bar(name="Gasto (A)", x=x, y=merged["spend (A)"], legendgroup="A", offsetgroup="A"))
    fig_A.add_trace(go.Bar(name="Faturamento (A)", x=x, y=merged["revenue (A)"], legendgroup="A", offsetgroup="A"))
    fig_A.add_trace(
        go.Scatter(
            name=f"Compras (A) — {period_sinceA} a {period_untilA}",
            x=x,
            y=merged["purchases (A)"],
            mode="lines+markers",
            legendgroup="A",
        ),
        secondary_y=True,
    )
    fig_A.update_layout(
        title=f"Período A — {period_sinceA} a {period_untilA} (Gasto + Faturamento + Compras)",
        barmode="stack",
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=48, b=10),
        separators=".,",
        legend_title_text="",
    )
    fig_A.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
    fig_A.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
    fig_A.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])
    st.plotly_chart(fig_A, use_container_width=True)

    # Gráfico B
    fig_B = make_subplots(specs=[[{"secondary_y": True}]])
    fig_B.add_trace(go.Bar(name="Gasto (B)", x=x, y=merged["spend (B)"], legendgroup="B", offsetgroup="B"))
    fig_B.add_trace(go.Bar(name="Faturamento (B)", x=x, y=merged["revenue (B)"], legendgroup="B", offsetgroup="B"))
    fig_B.add_trace(
        go.Scatter(
            name=f"Compras (B) — {period_sinceB} a {period_untilB}",
            x=x,
            y=merged["purchases (B)"],
            mode="lines+markers",
            legendgroup="B",
        ),
        secondary_y=True,
    )
    fig_B.update_layout(
        title=f"Período B — {period_sinceB} a {period_untilB} (Gasto + Faturamento + Compras)",
        barmode="stack",
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=48, b=10),
        separators=".,",
        legend_title_text="",
    )
    fig_B.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
    fig_B.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
    fig_B.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])
    st.plotly_chart(fig_B, use_container_width=True)

    # Insights rápidos
    st.markdown("### 🔎 Insights")
    def _sum(s): return float(s.sum())
    a_spend, a_rev, a_purch = _sum(merged["spend (A)"]), _sum(merged["revenue (A)"]), int(_sum(merged["purchases (A)"]))
    b_spend, b_rev, b_purch = _sum(merged["spend (B)"]), _sum(merged["revenue (B)"]), int(_sum(merged["purchases (B)"]))
    roasA = (a_rev / a_spend) if a_spend > 0 else np.nan
    roasB = (b_rev / b_spend) if b_spend > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor usado (A)", fmt_money_br(a_spend))
    c2.metric("Faturamento (A)", fmt_money_br(a_rev))
    c3.metric("Vendas (A)", f"{a_purch:,}".replace(",", "."))
    c4.metric("ROAS (A)", fmt_ratio_br(roasA) if np.isfinite(roasA) else "—")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Valor usado (B)", fmt_money_br(b_spend))
    d2.metric("Faturamento (B)", fmt_money_br(b_rev))
    d3.metric("Vendas (B)", f"{b_purch:,}".replace(",", "."))
    d4.metric("ROAS (B)", fmt_ratio_br(roasB) if np.isfinite(roasB) else "—")

    # Horas com gasto e 0 compras
    wastedA = merged[(merged["spend (A)"] > 0) & (merged["purchases (A)"] == 0)]
    wastedB = merged[(merged["spend (B)"] > 0) & (merged["purchases (B)"] == 0)]
    wastedA_hours = ", ".join(f"{int(h)}h" for h in wastedA["hour"].tolist()) if not wastedA.empty else "—"
    wastedB_hours = ", ".join(f"{int(h)}h" for h in wastedB["hour"].tolist()) if not wastedB.empty else "—"
    st.markdown(f"**Horas com gasto e 0 compras — A:** {wastedA_hours}")
    st.markdown(f"**Horas com gasto e 0 compras — B:** {wastedB_hours}")
