import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests, json, time

# =========================
# Config
# =========================
st.set_page_config(page_title="Metas & Performance — Simples", page_icon="📊", layout="wide")
st.title("📊 Metas & Performance — Simples")
st.caption("Analise por intervalo de datas (API Meta). Para meta, escolha o mês e informe o valor (R$).")

# =========================
# Estilo / helpers
# =========================
COLORWAY = ["#7C3AED","#06B6D4","#22C55E","#F59E0B","#94A3B8","#0EA5E9","#EF4444","#10B981","#3B82F6"]

def style_fig(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial", size=13),
        title=dict(text=title or fig.layout.title.text, x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        colorway=COLORWAY,
    )
    return fig

def _retry_call(fn, max_retries=5, base_wait=1.5):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit","retry","temporarily unavailable","timeout"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

@st.cache_data(show_spinner=True, ttl=600)
def pull_meta_insights_http(act_id: str, token: str, api_version: str, level: str,
                            since: datetime, until: datetime) -> pd.DataFrame:
    """
    Busca insights no Graph API e retorna colunas:
    data, gasto, faturamento (valor de purchase), compras (qtd), campanha (conforme level).
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend",
        "actions",         # contagem de purchase
        "action_values",   # valor de purchase
        "date_start","date_stop",
        "campaign_name","campaign_id",
        "adset_name","adset_id",
        "ad_name","ad_id",
        "account_name","account_id",
    ]
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({
            "since": since.strftime("%Y-%m-%d"),
            "until": until.strftime("%Y-%m-%d"),
        }),
        "time_increment": 1,      # diário
        "limit": 500,
        "fields": ",".join(fields),
        "action_types": '["purchase"]',  # pedimos só purchase
    }

    def _sum_actions(rec_list) -> float:
        total = 0.0
        for it in rec_list or []:
            if str(it.get("action_type","")).lower().find("purchase") != -1:
                try:
                    total += float(it.get("value", 0) or 0)
                except:
                    pass
        return total

    rows = []
    next_url, next_params = base_url, params.copy()
    while next_url:
        def _do():
            return requests.get(next_url, params=next_params, timeout=60)
        resp = _retry_call(_do)

        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            st.error(
                f"Graph API error {resp.status_code} | code={err.get('code')} "
                f"subcode={err.get('error_subcode')}\nmessage: {err.get('message')}\n"
                f"fbtrace_id: {err.get('fbtrace_id')}"
            )
            return pd.DataFrame()

        for rec in payload.get("data", []):
            linha = {
                "data": pd.to_datetime(rec.get("date_start")),
                "gasto": float(rec.get("spend", 0) or 0),
                "compras": _sum_actions(rec.get("actions")),
                "faturamento": _sum_actions(rec.get("action_values")),
            }
            # Nome conforme level
            if level == "campaign":
                linha["campanha"] = rec.get("campaign_name", "")
            elif level == "adset":
                linha["campanha"] = rec.get("adset_name", "")
            elif level == "ad":
                linha["campanha"] = rec.get("ad_name", "")
            else:
                linha["campanha"] = rec.get("account_name", "")
            rows.append(linha)

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy()
            next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("data")
        for c in ["gasto","faturamento","compras"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# =========================
# Sidebar – só API e filtros
# =========================
with st.sidebar:
    st.subheader("🔌 Meta Marketing API")
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)", value="")
    access_token = st.text_input("Access Token", type="password", value="")
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("📅 Intervalo de Datas (Performance)")
    since_api = st.date_input("Desde", value=(datetime.today() - timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())

    st.subheader("🎯 Meta Mensal")
    month_ref = st.date_input("Mês da meta (escolha qualquer dia do mês)", value=datetime.today().date())
    monthly_goal_value = st.number_input("Meta mensal (R$)", value=40000.0, min_value=0.0, step=500.0)

# Carrega dados APENAS da API
df_api = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Conectando ao Meta Ads e coletando insights..."):
        df_api = pull_meta_insights_http(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            level="campaign",  # fixo (mais útil). Se quiser, torne um select.
            since=datetime.combine(since_api, datetime.min.time()),
            until=datetime.combine(until_api, datetime.min.time()),
        )

any_data = not df_api.empty
if act_id and access_token and not any_data:
    st.warning("Conectei, mas não vieram dados para o intervalo/level escolhido. Verifique permissões, datas e se há eventos atribuídos.")

# =========================
# Abas simples
# =========================
tab_perf, tab_goal = st.tabs(["Performance (intervalo)", "Meta Mensal"])

# ------------ Performance
with tab_perf:
    st.subheader("📥 Performance no intervalo")
    if not any_data:
        st.info("Informe conta, token e intervalo para carregar dados.")
    else:
        df = df_api.copy()

        # KPIs agregados do intervalo
        invest_total = float(df["gasto"].sum())
        fatur_total = float(df["faturamento"].sum())
        compras_total = float(df["compras"].sum())
        roas = (fatur_total/invest_total) if invest_total>0 else 0.0
        cpa = (invest_total/compras_total) if compras_total>0 else 0.0

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("💰 Investimento", f"R$ {invest_total:,.0f}".replace(",",".")) 
        c2.metric("🏪 Faturamento", f"R$ {fatur_total:,.0f}".replace(",",".")) 
        c3.metric("🛒 Compras", f"{compras_total:,.0f}".replace(",",".")) 
        c4.metric("📈 ROAS", f"{roas:,.2f}".replace(",",".")) 
        c5.metric("🎯 CPA", f"R$ {cpa:,.2f}".replace(",","."))

        # Série diária (spend vs revenue)
        daily = df.groupby("data", as_index=False)[["gasto","faturamento","compras"]].sum().sort_values("data")
        if not daily.empty:
            fig1 = px.bar(daily, x="data", y=["gasto","faturamento"], barmode="group", title="Investimento x Faturamento (diário)")
            st.plotly_chart(style_fig(fig1), use_container_width=True)

            daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
            fig2 = px.line(daily, x="data", y="ROAS", markers=True, title="ROAS diário")
            st.plotly_chart(style_fig(fig2), use_container_width=True)

        # Top campanhas
        if "campanha" in df.columns:
            grp = (df.groupby("campanha", as_index=False)
                     .agg(gasto=("gasto","sum"), faturamento=("faturamento","sum"), compras=("compras","sum")))
            grp["ROAS"] = grp["faturamento"] / grp["gasto"].replace(0, np.nan)
            grp["CPA"]  = grp["gasto"] / grp["compras"].replace(0, np.nan)
            grp = grp.sort_values(["ROAS","faturamento","gasto"], ascending=[False, False, True]).head(10)
            st.markdown("### 🏆 Campanhas (Top 10 por ROAS)")
            st.dataframe(grp.rename(columns={
                "campanha":"Campanha","gasto":"Investimento (R$)","faturamento":"Faturamento (R$)"}), use_container_width=True)

# ------------ Meta Mensal
with tab_goal:
    st.subheader("🎯 Progresso da Meta do Mês")
    if not act_id or not access_token:
        st.info("Conecte a API para calcular o progresso do mês.")
    else:
        # período do mês escolhido (primeiro ao último dia)
        month_first = month_ref.replace(day=1)
        next_month_first = (month_first.replace(year=month_first.year+1, month=1)
                            if month_first.month == 12 else month_first.replace(month=month_first.month+1))
        month_last = next_month_first - timedelta(days=1)

        # Se os dados carregados não cobrem o mês inteiro, buscamos só o mês para o progresso
        with st.spinner("Calculando progresso mensal..."):
            df_month = pull_meta_insights_http(
                act_id=act_id,
                token=access_token,
                api_version=api_version,
                level="campaign",
                since=datetime.combine(month_first, datetime.min.time()),
                until=datetime.combine(month_last, datetime.min.time()),
            )

        if df_month.empty:
            st.info("Sem dados para o mês selecionado.")
        else:
            fatur_m = float(df_month["faturamento"].sum())
            invest_m = float(df_month["gasto"].sum())
            compras_m = float(df_month["compras"].sum())
            roas_m = (fatur_m/invest_m) if invest_m>0 else 0.0
            cpa_m  = (invest_m/compras_m) if compras_m>0 else 0.0

            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Mês — Investimento", f"R$ {invest_m:,.0f}".replace(",",".")) 
            k2.metric("Mês — Faturamento", f"R$ {fatur_m:,.0f}".replace(",",".")) 
            k3.metric("Mês — ROAS", f"{roas_m:,.2f}".replace(",",".")) 
            k4.metric("Mês — CPA", f"R$ {cpa_m:,.2f}".replace(",",".")) 
            k5.metric("Mês — Compras", f"{compras_m:,.0f}".replace(",","."))

            meta = float(monthly_goal_value or 0.0)
            progresso = min(1.0, fatur_m / max(1.0, meta))
            st.progress(progresso, text=f"Progresso: R$ {fatur_m:,.0f} / R$ {meta:,.0f}".replace(",","."))

            # Curva diária do mês
            daily_m = (df_month.groupby("data", as_index=False)[["gasto","faturamento"]].sum().sort_values("data"))
            if not daily_m.empty:
                figm = px.line(daily_m, x="data", y=["faturamento","gasto"], markers=True, title=f"Evolução diária — {month_first.strftime('%m/%Y')}")
                st.plotly_chart(style_fig(figm), use_container_width=True)

st.info("Dica: se aparecer erro 400 da Graph API, verifique permissões do token, datas e se há eventos de purchase atribuídos.")
