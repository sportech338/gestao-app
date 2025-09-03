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
st.set_page_config(page_title="Performance Ads — Simples", page_icon="📊", layout="wide")
st.title("📊 Performance Ads — Simples")
st.caption("Analise os dados da API Meta pelo intervalo de datas escolhido.")

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
    Busca insights no Graph API e retorna:
    data, gasto, faturamento (valor de purchase), compras (qtd), campanha (conforme level).
    Usa 'conversions'/'conversion_values' (preferido) e faz fallback para 'actions'/'action_values'.
    Soma janelas (1d_click/7d_click/1d_view etc.) quando 'value' não existir.
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend",
        "conversions","conversion_values",  # preferível
        "actions","action_values",          # fallback
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
        "time_increment": 1,
        "fields": ",".join(fields),
        "action_report_time": "conversion",
        "use_unified_attribution_setting": "true",
        "limit": 500,
    }

    # preferências de tipos de compra (ordem importa para evitar dupla contagem)
    PREFER_ORDER = [
        "omni_purchase",
        "purchase",
        "offsite_conversion.fb_pixel_purchase",
        "onsite_conversion.purchase",
        "app_custom_event.fb_mobile_purchase",
        "website_purchase",
    ]

    def _sum_numeric_fields(item: dict) -> float:
        """Soma todos os campos numéricos do item (value, 1d_click, 7d_click, 1d_view, etc.)."""
        s = 0.0
        for k, v in (item or {}).items():
            if k == "action_type": 
                continue
            try:
                s += float(v or 0)
            except:
                pass
        return s

    def _best_purchase_from(list_of_dicts) -> dict:
        """
        Monta um mapa action_type -> soma(números).
        Retorna a melhor métrica seguindo a ordem de preferência.
        Se não achar nenhum preferido, retorna o total de tudo que contenha 'purchase'.
        """
        acc = {}
        for it in (list_of_dicts or []):
            at = str(it.get("action_type") or "").lower()
            if "purchase" not in at:
                continue
            acc[at] = acc.get(at, 0.0) + _sum_numeric_fields(it)

        # prioriza
        for key in PREFER_ORDER:
            if key in acc:
                return {"value": acc[key], "chosen": key}
        # fallback: soma de todos os 'purchase'
        if acc:
            return {"value": sum(acc.values()), "chosen": "any_purchase"}
        return {"value": 0.0, "chosen": None}

    rows, next_url, next_params = [], base_url, params.copy()
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
            spend = float(rec.get("spend", 0) or 0)

            # 1) Tenta conversions/conversion_values (preferido)
            conv_cnt   = _best_purchase_from(rec.get("conversions"))
            conv_value = _best_purchase_from(rec.get("conversion_values"))

            # 2) Se não veio nada ali, faz fallback p/ actions/action_values
            if conv_cnt["value"] == 0 and conv_value["value"] == 0:
                conv_cnt   = _best_purchase_from(rec.get("actions"))
                conv_value = _best_purchase_from(rec.get("action_values"))

            linha = {
                "data": pd.to_datetime(rec.get("date_start")),
                "gasto": spend,
                "compras": float(conv_cnt["value"]),
                "faturamento": float(conv_value["value"]),
            }

            # nome conforme o level
            if level == "campaign":
                linha["campanha"] = rec.get("campaign_name", "")
            elif level == "adset":
                linha["campanha"] = rec.get("adset_name", "")
            elif level == "ad":
                linha["campanha"] = rec.get("ad_name", "")
            else:
                linha["campanha"] = rec.get("account_name", "")

            rows.append(linha)

        # paginação
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
# Sidebar
# =========================
with st.sidebar:
    st.subheader("🔌 Meta Marketing API")
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)", value="")
    access_token = st.text_input("Access Token", type="password", value="")
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("📅 Intervalo de Datas")
    since_api = st.date_input("Desde", value=(datetime.today() - timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())

# =========================
# Carregar dados
# =========================
since_dt = datetime.combine(since_api, datetime.min.time())
until_dt = datetime.combine(until_api, datetime.min.time())
if since_dt > until_dt:
    since_dt, until_dt = until_dt, since_dt

df_api = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Conectando ao Meta Ads e coletando insights..."):
        df_api = pull_meta_insights_http(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            level="campaign",
            since=since_dt,
            until=until_dt,
        )

any_data = not df_api.empty
if act_id and access_token and not any_data:
    st.warning("Conectei, mas não vieram dados para o intervalo/level escolhido.")

# =========================
# Performance
# =========================
st.subheader("📥 Performance no intervalo")
if not any_data:
    st.info("Informe conta, token e intervalo para carregar dados.")
else:
    df = df_api.copy()

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

    daily = df.groupby("data", as_index=False)[["gasto","faturamento","compras"]].sum().sort_values("data")
    if not daily.empty:
        fig1 = px.bar(daily, x="data", y=["gasto","faturamento"], barmode="group", title="Investimento x Faturamento (diário)")
        st.plotly_chart(style_fig(fig1), use_container_width=True)

        daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
        fig2 = px.line(daily, x="data", y="ROAS", markers=True, title="ROAS diário")
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    if "campanha" in df.columns:
        grp = (df.groupby("campanha", as_index=False)
                 .agg(gasto=("gasto","sum"), faturamento=("faturamento","sum"), compras=("compras","sum")))
        grp["ROAS"] = grp["faturamento"] / grp["gasto"].replace(0, np.nan)
        grp["CPA"]  = grp["gasto"] / grp["compras"].replace(0, np.nan)
        grp = grp.sort_values(["ROAS","faturamento","gasto"], ascending=[False, False, True]).head(10)
        st.markdown("### 🏆 Campanhas (Top 10 por ROAS)")
        st.dataframe(grp.rename(columns={
            "campanha":"Campanha","gasto":"Investimento (R$)","faturamento":"Faturamento (R$)"}), use_container_width=True)
