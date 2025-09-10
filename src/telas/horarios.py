import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ..servicos.meta_insights import fetch_insights_hourly
from ..recursos.filtros import filtrar_por_produto
from ..configuracao import PRODUTOS

@st.cache_data(ttl=600, show_spinner=True)
def _cache_hourly(act_id, token, api_version, since, until):
    return fetch_insights_hourly(act_id=act_id, token=token, api_version=api_version,
                                 since_str=str(since), until_str=str(until), level="campaign")

def render(params: dict):
    st.caption("Explore desempenho por hora.")

    df_hourly = _cache_hourly(params["act_id"], params["token"], params["api_version"], params["since"], params["until"])
    if df_hourly is None or df_hourly.empty:
        st.info("Sem dados horários no período."); st.stop()

    produto_sel = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daypart_produto")
    d = filtrar_por_produto(df_hourly, produto_sel) if produto_sel != "(Todos)" else df_hourly.copy()

    d = d.dropna(subset=["hour"]).copy()
    d["hour"] = d["hour"].astype(int).clip(0,23)

    # Heatmap simples: Compras
    cube = d.groupby(["dow_label","hour"], as_index=False)["purchases"].sum()
    order = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
    cube["dow_label"] = pd.Categorical(cube["dow_label"], categories=order, ordered=True)
    cube = cube.sort_values(["dow_label","hour"])
    heat = cube.pivot(index="dow_label", columns="hour", values="purchases").fillna(0)
    hours_full = list(range(24))
    heat = heat.reindex(columns=hours_full, fill_value=0)
    fig_hm = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorbar=dict(title="Compras"),
        hovertemplate="Dia: %{y}<br>Hora: %{x}h<br>Compras: %{z}<extra></extra>"))
    fig_hm.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white", separators=",.")
    st.subheader("📆 Heatmap — Hora × Dia (Compras)")
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("📦 Compras por hora (total do período)")
    ser = d.groupby("hour")["purchases"].sum().reindex(hours_full, fill_value=0)
    fig_bar = go.Figure(go.Bar(x=ser.index, y=ser.values))
    fig_bar.update_layout(title="Compras por hora", xaxis_title="Hora", yaxis_title="Compras", height=380,
                          template="plotly_white", margin=dict(l=10, r=10, t=48, b=10), separators=",.")
    st.plotly_chart(fig_bar, use_container_width=True)
""")
