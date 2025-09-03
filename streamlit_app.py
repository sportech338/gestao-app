import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import datetime, timedelta

st.set_page_config(page_title="Meta Ads — Base Correta", page_icon="📊", layout="wide")
st.title("📊 Meta Ads — Somatórios Corretos (Compras, Faturamento, Spend)")

# -------------- helpers --------------
def _retry_call(fn, max_retries=5, base_wait=1.5):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout"]):
                time.sleep(base_wait * (2 ** i)); continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def _sum_item(item: dict) -> float:
    # soma qualquer valor numérico dentro do dict (p/ janelas 1d_click, 7d_click etc.)
    if not isinstance(item, dict): 
        try: return float(item or 0)
        except: return 0.0
    s=0.0
    for k,v in item.items():
        if k=="action_type": continue
        try: s += float(v or 0)
        except: pass
    return s

def _sum_list_by_type(lst, wanted_substr="purchase"):
    """Soma valores de uma lista de dicts filtrando por action_type que contenha wanted_substr."""
    tot=0.0
    for it in (lst or []):
        at = str(it.get("action_type") or "").lower()
        if wanted_substr in at:
            # preferir 'value' se existir; senão somar as janelas (1d_click etc.)
            if "value" in it:
                try: tot += float(it["value"] or 0)
                except: pass
            else:
                tot += _sum_item(it)
    return tot

# -------------- coleta (APENAS actions/action_values) --------------
@st.cache_data(ttl=600, show_spinner=True)
def pull_meta_insights(act_id: str, token: str, api_version: str,
                       since: datetime, until: datetime,
                       report_time: str) -> pd.DataFrame:
    """
    Retorna por dia: data, gasto (spend), compras (actions[purchase]), faturamento (action_values[purchase]),
    campanha (nome). Usa atribuição unificada e report_time: 'impression' ou 'conversion'.
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = ["spend","actions","action_values","date_start","campaign_name","account_name"]
    params = {
        "access_token": token,
        "level": "campaign",
        "time_range": json.dumps({"since": since.strftime("%Y-%m-%d"), "until": until.strftime("%Y-%m-%d")}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
        "use_unified_attribution_setting": "true",
        "action_report_time": report_time,      # "impression" (padrão do Gerenciador) ou "conversion"
        "action_types": '["purchase"]'          # filtra só PURCHASE já na API
    }

    rows = []
    next_url, next_params = base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=60))
        try:
            payload = resp.json()
        except Exception:
            st.error("Não consegui decodificar a resposta da API."); return pd.DataFrame()
        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            st.error(f"Graph API error {resp.status_code} | code={err.get('code')} subcode={err.get('error_subcode')}\n"
                     f"message: {err.get('message')}\nfbtrace_id: {err.get('fbtrace_id')}")
            return pd.DataFrame()

        for rec in payload.get("data", []):
            spend = float(rec.get("spend", 0) or 0)
            # COMO CONTAMOS:
            compras = _sum_list_by_type(rec.get("actions"), "purchase")
            receita = _sum_list_by_type(rec.get("action_values"), "purchase")

            rows.append({
                "data":        pd.to_datetime(rec.get("date_start")),
                "gasto":       spend,
                "compras":     float(compras),
                "faturamento": float(receita),
                "campanha":    rec.get("campaign_name") or rec.get("account_name") or ""
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy(); next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows).sort_values("data") if rows else pd.DataFrame()
    if not df.empty:
        for c in ["gasto","faturamento","compras"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# -------------- UI --------------
with st.sidebar:
    st.header("🔌 Meta Marketing API")
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
    access_token = st.text_input("Access Token", type="password")
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("📅 Intervalo")
    since_api = st.date_input("Desde", value=(datetime.today()-timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())
    if since_api > until_api:
        since_api, until_api = until_api, since_api

    st.subheader("🗓️ Reporting time (como o Gerenciador conta datas)")
    report_mode = st.radio(
        "Modo de data",
        ["Impressão (igual ao Gerenciador)", "Conversão (quando ocorreu)"],
        index=0
    )
    report_time = "impression" if "Impressão" in report_mode else "conversion"

# -------------- run --------------
df = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Puxando dados…"):
        df = pull_meta_insights(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            since=datetime.combine(since_api, datetime.min.time()),
            until=datetime.combine(until_api, datetime.min.time()),
            report_time=report_time
        )

st.subheader("Resultado do intervalo")
if df.empty:
    st.info("Preencha conta/token/intervalo. Se vier vazio, cheque permissões, intervalo e se há compras atribuídas.")
else:
    # Somatórios corretos (intervalo)
    gasto_total = float(df["gasto"].sum())
    fat_total   = float(df["faturamento"].sum())
    comp_total  = float(df["compras"].sum())
    roas_global = (fat_total / gasto_total) if gasto_total > 0 else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💰 Valor usado (soma)", f"R$ {gasto_total:,.0f}".replace(",", "."))
    c2.metric("🏪 Faturamento (soma)", f"R$ {fat_total:,.0f}".replace(",", "."))
    c3.metric("🛒 Compras (soma)", f"{comp_total:,.0f}".replace(",", "."))
    c4.metric("📈 ROAS (global)", f"{roas_global:,.2f}".replace(",", "."), help="Faturamento total ÷ Gasto total no intervalo")

    # ROAS médio diário (opcional, para comparação)
    daily = df.groupby("data", as_index=False)[["gasto","faturamento"]].sum()
    daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
    roas_medio_dia = float(daily["ROAS"].replace([np.inf,-np.inf], np.nan).mean(skipna=True) or 0.0)
    st.caption(f"ROAS médio diário (média aritmética dos dias): {roas_medio_dia:.2f}")

    with st.expander("Dados por dia (auditoria)"):
        st.dataframe(df.sort_values(["data","campanha"]), use_container_width=True)
