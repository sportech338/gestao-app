import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ..servicos.meta_insights import fetch_insights_breakdown
from ..recursos.filtros import filtrar_por_produto
from ..configuracao import PRODUTOS

def render(params: dict, df_daily: pd.DataFrame):
    st.caption("Detalhe por dimensão.")

    colf1, colf2 = st.columns([2,1])
    with colf1:
        produto_sel = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="det_produto")
    with colf2:
        min_spend_det = st.slider("Gasto mínimo para considerar (R$)", 0.0, 2000.0, 0.0, 10.0, key="det_min_spend")

    dimensao = st.radio("Dimensão", ["Populares","Idade","Gênero","Idade + Gênero", "País","Plataforma"], index=0, horizontal=True)

    base = filtrar_por_produto(df_daily, produto_sel) if produto_sel != "(Todos)" else df_daily.copy()

    if dimensao == "Populares":
        g = (base.groupby(["campaign_id","campaign_name"], as_index=False)
                  [["spend","revenue","purchases"]].sum())
        g["ROAS"] = np.where(g["spend"]>0, g["revenue"]/g["spend"], np.nan)
        top = g.sort_values(["purchases","ROAS"], ascending=[False,False]).head(10)
        st.subheader("TOP Campanhas")
        st.dataframe(top.rename(columns={"campaign_name":"Campanha","purchases":"Compras","spend":"Gasto","revenue":"Faturamento"}),
                     use_container_width=True, height=360)
        st.stop()

    dim_to_breakdowns = {
        "Idade": ["age"],
        "Gênero": ["gender"],
        "Idade + Gênero": ["age","gender"],
        "País": ["country"],
        "Plataforma": ["publisher_platform"],
    }
    bks = dim_to_breakdowns[dimensao]
    df_bd = fetch_insights_breakdown(params["act_id"], params["token"], params["api_version"],
                                     str(params["since"]), str(params["until"]), bks, params["level"],
                                     product_name=produto_sel if produto_sel != "(Todos)" else None)
    if df_bd.empty:
        st.info("Sem dados para a dimensão no período."); st.stop()

    rename_map = {"age":"Idade","gender":"Gênero","country":"País","publisher_platform":"Plataforma"}
    df_bd = df_bd.rename(columns=rename_map)
    group_cols = [rename_map.get(c, c) for c in bks]

    agg_cols = ["spend","revenue","purchases"]
    g = df_bd.groupby(group_cols, as_index=False)[agg_cols].sum()
    if float(min_spend_det) > 0:
        g = g[g["spend"] >= float(min_spend_det)]
    g["ROAS"] = np.where(g["spend"]>0, g["revenue"]/g["spend"], np.nan)
    g = g.sort_values(["purchases","ROAS"], ascending=[False, False])

    st.subheader(f"Desempenho por {dimensao}")
    st.dataframe(g, use_container_width=True, height=520)

    if len(group_cols) == 1:
        xlab = group_cols[0]
        fig = go.Figure(go.Bar(x=g[xlab].astype(str), y=g["purchases"]))
        fig.update_layout(title=f"Compras por {xlab}", xaxis_title=xlab, yaxis_title="Compras", height=420,
                          template="plotly_white", margin=dict(l=10, r=10, t=48, b=10), separators=",.")
        st.plotly_chart(fig, use_container_width=True)
""")
