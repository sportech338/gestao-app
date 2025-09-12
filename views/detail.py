# views/detail.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from services.facebook_api import fetch_insights_breakdown
from config.constants import PRODUTOS
from utils.helpers import filter_by_product
from utils.formatting import fmt_money_br, fmt_ratio_br


def _ensure_cols_exist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _agg_and_format(df: pd.DataFrame, group_cols: list[str], min_spend: float):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df2 = _ensure_cols_exist(df)
    g = df2.groupby(group_cols, dropna=False, as_index=False)[
        ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
    ].sum()
    g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)
    if min_spend and float(min_spend) > 0:
        g = g[g["spend"] >= float(min_spend)]
    if not g.empty:
        g = g.sort_values(["purchases", "ROAS"], ascending=[False, False])

    gf = g.copy()
    if not gf.empty:
        gf["Valor usado"] = gf["spend"].apply(fmt_money_br)
        gf["Valor de conversão"] = gf["revenue"].apply(fmt_money_br)
        gf["ROAS"] = gf["ROAS"].map(fmt_ratio_br)
        gf = gf.drop(columns=["spend", "revenue"])
    return g, gf


def _bar_chart(x_labels, y_values, title, x_title, y_title):
    fig = go.Figure(go.Bar(x=x_labels, y=y_values))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=48, b=10),
        separators=".,",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_detail_tab(act_id: str, token: str, api_version: str, level: str, since, until):
    """
    Aba de detalhamento:
      - 'Populares' (TOP 5 por Compras e por ROAS)
      - Demais dimensões: Idade, Gênero, Idade+Gênero, Região, País, Plataforma, Posicionamento
    """
    st.caption("Explore por dimensão: Idade, Gênero, País, Plataforma, Posicionamento — e a aba 'Populares'.")

    colf1, colf2 = st.columns([2, 1])
    with colf1:
        produto_sel_det = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="det_produto")
    with colf2:
        min_spend_det = st.slider("Gasto mínimo para considerar (R$)", 0.0, 2000.0, 0.0, 10.0, key="det_min_spend")

    dimensao = st.radio(
        "Dimensão",
        ["Populares", "Idade", "Gênero", "Idade + Gênero", "Região", "País", "Plataforma", "Posicionamento"],
        index=0,
        horizontal=True,
    )

    # ============ POPULARES ============
    if dimensao == "Populares":
        st.subheader("TOP 5 — Campanhas")
        # Para populares, o ideal é usar o df diário agregado já filtrado.
        # Como estamos nesta view, vamos buscar via breakdown por campanha_id (1 dia) — mas para performance,
        # o app principal já traz df_daily; se preferir, passe-o por parâmetro.
        df_bd = fetch_insights_breakdown(
            act_id, token, api_version, str(since), str(until), [], "campaign", product_name=produto_sel_det
        )
        if df_bd is None or df_bd.empty:
            st.info("Sem dados para o período/filtro.")
            return
        base = filter_by_product(df_bd, produto_sel_det)
        if base is None or base.empty:
            st.info("Sem dados após aplicar o filtro de produto.")
            return

        agg_cols = ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
        g = base.groupby(["campaign_id", "campaign_name"], as_index=False)[agg_cols].sum()
        g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)
        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]

        top_comp = g.sort_values(["purchases", "ROAS"], ascending=[False, False]).head(5).copy()
        top_roas = g[g["spend"] > 0].sort_values("ROAS", ascending=False).head(5).copy()

        def _fmt_disp(df_):
            out = df_.copy()
            out["Valor usado"] = out["spend"].apply(fmt_money_br)
            out["Valor de conversão"] = out["revenue"].apply(fmt_money_br)
            out["ROAS"] = out["ROAS"].map(fmt_ratio_br)
            out.rename(
                columns={
                    "campaign_name": "Campanha",
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                },
                inplace=True,
            )
            return out

        disp_comp = _fmt_disp(top_comp)[
            ["Campanha", "Compras", "Valor usado", "Valor de conversão", "ROAS", "Cliques", "LPV", "Checkout", "Add Pagto"]
        ]
        disp_roas = _fmt_disp(top_roas)[
            ["Campanha", "ROAS", "Compras", "Valor usado", "Valor de conversão", "Cliques", "LPV", "Checkout", "Add Pagto"]
        ]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Por Compras**")
            st.dataframe(disp_comp, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV — TOP 5 por Compras",
                data=disp_comp.to_csv(index=False).encode("utf-8-sig"),
                file_name="top5_campanhas_por_compras.csv",
                mime="text/csv",
            )
        with c2:
            st.markdown("**Por ROAS** (gasto > 0)")
            st.dataframe(disp_roas, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV — TOP 5 por ROAS",
                data=disp_roas.to_csv(index=False).encode("utf-8-sig"),
                file_name="top5_campanhas_por_roas.csv",
                mime="text/csv",
            )
        st.stop()  # não processar outras dimensões quando Populares está ativa

    # ============ DEMAIS DIMENSÕES ============
    dim_to_breakdowns = {
        "Idade": ["age"],
        "Gênero": ["gender"],
        "Idade + Gênero": ["age", "gender"],
        "Região": ["region"],
        "País": ["country"],
        "Plataforma": ["publisher_platform"],
        "Posicionamento": ["publisher_platform", "platform_position"],
    }

    # Proteção: 'Posicionamento' exige adset/ad; usamos adset se nível atual for campaign
    level_bd = level
    if dimensao == "Posicionamento" and level_bd not in ["adset", "ad"]:
        level_bd = "adset"

    if dimensao in dim_to_breakdowns:
        bks = dim_to_breakdowns[dimensao]

        df_bd = fetch_insights_breakdown(
            act_id, token, api_version, str(since), str(until), bks, level_bd, product_name=produto_sel_det
        )
        if df_bd is None or df_bd.empty:
            st.info(f"Sem dados para {dimensao} no período/filtro.")
            return

        rename_map = {
            "age": "Idade",
            "gender": "Gênero",
            "region": "Região",
            "country": "País",
            "publisher_platform": "Plataforma",
            "platform_position": "Posicionamento",
        }
        df_bd = df_bd.rename(columns=rename_map)

        group_cols = [rename_map.get(c, c) for c in bks]
        raw_base = df_bd.copy()
        g_raw, g_disp = _agg_and_format(raw_base, group_cols, min_spend_det)
        if g_disp.empty:
            st.info(f"Sem dados para {dimensao} após aplicar filtros.")
            return

        st.subheader(f"Desempenho por {dimensao}")
        base_cols = group_cols + [
            "Compras",
            "ROAS",
            "Valor usado",
            "Valor de conversão",
            "Cliques",
            "LPV",
            "Checkout",
            "Add Pagto",
        ]
        g_disp = g_disp.rename(
            columns={
                "purchases": "Compras",
                "link_clicks": "Cliques",
                "lpv": "LPV",
                "init_checkout": "Checkout",
                "add_payment": "Add Pagto",
            }
        )
        cols_presentes = [c for c in base_cols if c in g_disp.columns]
        st.dataframe(g_disp[cols_presentes], use_container_width=True, height=520)

        # gráfico
        if len(group_cols) == 1:
            xlab = group_cols[0]
            _bar_chart(g_raw[xlab].astype(str), g_raw["purchases"], f"Compras por {xlab}", xlab, "Compras")
        else:
            idx, col = group_cols
            pvt = g_raw.pivot_table(index=idx, columns=col, values="purchases", aggfunc="sum").fillna(0.0)
            fig = go.Figure(
                data=go.Heatmap(
                    z=pvt.values,
                    x=pvt.columns.astype(str),
                    y=pvt.index.astype(str),
                    colorbar=dict(title="Compras"),
                    hovertemplate=f"{idx}: " + "%{y}<br>" + f"{col}: " + "%{x}<br>Compras: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"Heatmap — Compras por {idx} × {col}",
                height=460,
                template="plotly_white",
                margin=dict(l=10, r=10, t=48, b=10),
                separators=".,",
            )
            st.plotly_chart(fig, use_container_width=True)
