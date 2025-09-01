import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Metas de Tráfego Pago — Semanal & Mensal", layout="wide")

st.title("📊 Dashboard de Metas — Tráfego Pago")
st.caption("Planeje orçamento, metas de pessoas/sessões, conversões e ROI estimado. Suba/baixe planilhas para acompanhar diariamente.")

# ---------- Sidebar: parâmetros globais ----------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    to_checkout = st.number_input("Taxa Sessão → Checkout (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, format="%.2f") / 100.0
    checkout_conv = st.number_input("Taxa Checkout → Compra (%)", value=40.0, min_value=0.0, max_value=100.0, step=1.0, format="%.2f") / 100.0
    target_roas = st.number_input("ROAS alvo (ex: 2.0 = 200%)", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.markdown("---")
    st.subheader("📆 Janela Semanal")
    week_start = st.date_input("Início da semana (segunda)", value=(datetime.today() - timedelta(days=datetime.today().weekday())).date())
    include_weekends = st.checkbox("Incluir finais de semana", value=True)

    st.subheader("📅 Janela Mensal")
    month_ref = st.date_input("Ref. do mês (qualquer dia do mês)", value=datetime.today().date())

    st.markdown("---")
    st.subheader("🎯 Tipo de Meta")
    goal_type = st.radio("Defina por:", options=["Faturamento", "Compras"], index=0, horizontal=True)
    weekly_goal_value = st.number_input("Meta SEMANAL (R$ se Faturamento; nº se Compras)", value=10000.0, min_value=0.0, step=100.0)
    monthly_goal_value = st.number_input("Meta MENSAL (R$ se Faturamento; nº se Compras)", value=40000.0, min_value=0.0, step=500.0)

# ---------- Helpers ----------
def daterange(start_date, end_date, include_weekends=True):
    days = []
    d = start_date
    while d <= end_date:
        if include_weekends or d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def build_targets(dates, period_goal_value, goal_type, aov, to_checkout, checkout_conv, target_roas):
    if goal_type == "Faturamento":
        rev_goal = float(period_goal_value)
        purchases_goal = rev_goal / aov if aov > 0 else 0.0
    else:
        purchases_goal = float(period_goal_value)
        rev_goal = purchases_goal * aov

    sessions_needed = purchases_goal / (to_checkout * checkout_conv) if (to_checkout > 0 and checkout_conv > 0) else 0.0
    spend_budget = rev_goal / target_roas if target_roas > 0 else 0.0
    roi = (rev_goal - spend_budget) / spend_budget if spend_budget > 0 else 0.0

    n_days = max(1, len(dates))
    daily_targets = pd.DataFrame({
        "data": dates,
        "meta_faturamento": rev_goal / n_days,
        "meta_compras": purchases_goal / n_days,
        "meta_sessoes": sessions_needed / n_days,
        "meta_investimento": spend_budget / n_days
    })
    totals = {
        "faturamento": rev_goal,
        "compras": purchases_goal,
        "sessoes": sessions_needed,
        "investimento": spend_budget,
        "roi_estimado": roi
    }
    return daily_targets, totals

# ---------- Computa janelas ----------
# SEMANA
week_start_dt = datetime.combine(week_start, datetime.min.time())
week_end_dt = week_start_dt + timedelta(days=6)
week_days = daterange(week_start_dt.date(), week_end_dt.date(), include_weekends)

weekly_df, weekly_totals = build_targets(
    week_days,
    weekly_goal_value,
    goal_type,
    aov,
    to_checkout,
    checkout_conv,
    target_roas,
)

# MÊS
month_first = month_ref.replace(day=1)
if month_first.month == 12:
    next_month_first = month_first.replace(year=month_first.year+1, month=1)
else:
    next_month_first = month_first.replace(month=month_first.month+1)
month_last = next_month_first - timedelta(days=1)

month_days = daterange(month_first, month_last, include_weekends=True)
monthly_df, monthly_totals = build_targets(
    month_days,
    monthly_goal_value,
    goal_type,
    aov,
    to_checkout,
    checkout_conv,
    target_roas,
)

# ---------- Área de acompanhamento diário ----------
st.subheader("✅ Acompanhamento Diário — Semana Atual")
st.caption("Edite os campos de realizado. Baixe a planilha, atualize no Excel/Sheets e reenvie quando quiser.")

week_live = weekly_df.copy()
week_live["real_investimento"] = 0.0
week_live["real_sessoes"] = 0.0
week_live["real_checkouts"] = 0.0
week_live["real_compras"] = 0.0
week_live["real_faturamento"] = 0.0

edited_week = st.data_editor(
    week_live,
    column_config={
        "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY", step=1),
        "meta_faturamento": st.column_config.NumberColumn("Meta Faturamento (R$)"),
        "meta_compras": st.column_config.NumberColumn("Meta Compras (nº)"),
        "meta_sessoes": st.column_config.NumberColumn("Meta Sessões (nº)"),
        "meta_investimento": st.column_config.NumberColumn("Meta Investimento (R$)"),
        "real_investimento": st.column_config.NumberColumn("Real Investimento (R$)"),
        "real_sessoes": st.column_config.NumberColumn("Real Sessões (nº)"),
        "real_checkouts": st.column_config.NumberColumn("Real Checkouts (nº)"),
        "real_compras": st.column_config.NumberColumn("Real Compras (nº)"),
        "real_faturamento": st.column_config.NumberColumn("Real Faturamento (R$)"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
)

# KPIs semanais
st.markdown("### 📌 KPIs Semanais")
kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("Meta Faturamento", f"R$ {weekly_totals['faturamento']:,.0f}".replace(",", "."))
with kpi_cols[1]:
    st.metric("Meta Compras", f"{weekly_totals['compras']:,.0f}".replace(",", "."))
with kpi_cols[2]:
    st.metric("Meta Sessões", f"{weekly_totals['sessoes']:,.0f}".replace(",", "."))
with kpi_cols[3]:
    st.metric("Orçamento (ROAS alvo)", f"R$ {weekly_totals['investimento']:,.0f}".replace(",", "."))
with kpi_cols[4]:
    st.metric("ROI Estimado", f"{weekly_totals['roi_estimado']*100:,.0f}%".replace(",", "."))

# KPIs realizados acumulados
realized = edited_week[["real_investimento","real_sessoes","real_checkouts","real_compras","real_faturamento"]].sum()
realized_cols = st.columns(5)
with realized_cols[0]:
    st.metric("Investimento (Real)", f"R$ {realized['real_investimento']:,.0f}".replace(",", "."))
with realized_cols[1]:
    st.metric("Sessões (Real)", f"{realized['real_sessoes']:,.0f}".replace(",", "."))
with realized_cols[2]:
    st.metric("Checkouts (Real)", f"{realized['real_checkouts']:,.0f}".replace(",", "."))
with realized_cols[3]:
    st.metric("Compras (Real)", f"{realized['real_compras']:,.0f}".replace(",", "."))
with realized_cols[4]:
    st.metric("Faturamento (Real)", f"R$ {realized['real_faturamento']:,.0f}".replace(",", "."))

st.markdown("---")

# ---------- Gráficos ----------
left, right = st.columns(2)
with left:
    df_plot = edited_week.copy()
    df_plot["data"] = pd.to_datetime(df_plot["data"])
    df_plot = df_plot.sort_values("data")
    df_plot["meta_fat_cum"] = df_plot["meta_faturamento"].cumsum()
    df_plot["real_fat_cum"] = df_plot["real_faturamento"].cumsum()
    fig = px.line(df_plot, x="data", y=["meta_fat_cum","real_fat_cum"], title="Faturamento Acumulado — Meta vs Real")
    st.plotly_chart(fig, use_container_width=True)

with right:
    df_inv = edited_week.copy()
    df_inv["data"] = pd.to_datetime(df_inv["data"])
    df_inv = df_inv.sort_values("data")
    df_inv["meta_inv_cum"] = df_inv["meta_investimento"].cumsum()
    df_inv["real_inv_cum"] = df_inv["real_investimento"].cumsum()
    fig2 = px.line(df_inv, x="data", y=["meta_inv_cum","real_inv_cum"], title="Investimento Acumulado — Meta vs Real")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Downloads ----------
st.markdown("### ⬇️ Exportar Planos")
wcsv = weekly_df.copy()
wcsv["data"] = pd.to_datetime(wcsv["data"]).dt.strftime("%Y-%m-%d")
mcsv = monthly_df.copy()
mcsv["data"] = pd.to_datetime(mcsv["data"]).dt.strftime("%Y-%m-%d")

st.download_button("Baixar Plano Semanal (CSV)", data=wcsv.to_csv(index=False).encode("utf-8"), file_name="plano_semanal.csv", mime="text/csv")
st.download_button("Baixar Plano Mensal (CSV)", data=mcsv.to_csv(index=False).encode("utf-8"), file_name="plano_mensal.csv", mime="text/csv")

st.info("💾 Para persistir dados no Cloud, utilize os botões de download e depois reenvie seu CSV atualizado.")
