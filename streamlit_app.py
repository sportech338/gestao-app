import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import requests, json, time

st.set_page_config(page_title="Performance Ads — Limpo", page_icon="📊", layout="wide")
st.title("📊 Performance Ads — Limpo")
st.caption("Investimento, Vendas (compras), Receita e ROAS corretos para o intervalo escolhido (API Meta).")

# ---------- Util ----------
def _num(x):
    try: return float(x or 0)
    except: return 0.0

def _item_value(item: dict) -> float:
    """Usa item['value'] se existir; senão soma janelas (1d_click, 7d_view etc.)."""
    if not isinstance(item, dict):
        return 0.0
    if "value" in item and item["value"] not in (None, "", "0"):
        return _num(item["value"])
    s = 0.0
    for k, v in item.items():
        if k == "action_type": 
            continue
        s += _num(v)
    return s

def _best_value(list_of_dicts, ordered_keys) -> float:
    """
    Soma por action_type e devolve na ordem de preferência (ex.: omni_purchase -> purchase).
    Se não achar nenhuma chave preferida, tenta somar tudo que 'contenha' o termo base (ex.: 'purchase').
    """
    if not list_of_dicts:
        return 0.0
    acc = {}
    for it in list_of_dicts:
        at = str(it.get("action_type") or "").lower()
        acc[at] = acc.get(at, 0.0) + _item_value(it)
    for k in ordered_keys:
        if k in acc:
            return acc[k]
    # fallback amplo pelo sufixo
    base = (ordered_keys[0] if ordered_keys else "").split("_")[-1]
    if base:
        tot = sum(v for k, v in acc.items() if base in k)
        if tot > 0:
            return tot
    return 0.0

# ordem de preferência de eventos (alinha com unified/omni)
PREF = {
    "purchase": [
        "omni_purchase",
        "purchase",
        "offsite_conversion.fb_pixel_purchase",
        "onsite_conversion.purchase",
        "app_custom_event.fb_mobile_purchase",
        "website_purchase",
    ],
}

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
def pull_insights(act_id: str, token: str, api_version: str,
                  since: datetime, until: datetime, level: str = "campaign") -> pd.DataFrame:
    """
    Retorna por dia e por campanha:
      data, gasto, compras, faturamento, roas, campanha
    Coleta conversions/conversion_values (unified). Fallback: actions/action_values.
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend",
        "conversions","conversion_values",   # preferido (unified)
        "actions","action_values",          # fallback (legado)
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
        "time_increment": 1,  # diário
        "fields": ",".join(fields),
        "action_report_time": "conversion",
        "use_unified_attribution_setting": "true",
        "limit": 500,
    }

    rows = []
    next_url, next_params = base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=60))
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
            spend = _num(rec.get("spend"))

            # Purchases/Revenue: try conversions first, fallback to actions
            compras = _best_value(rec.get("conversions"),       PREF["purchase"])
            receita = _best_value(rec.get("conversion_values"), PREF["purchase"])
            if compras == 0 and receita == 0:
                compras = _best_value(rec.get("actions"),       PREF["purchase"])
                receita = _best_value(rec.get("action_values"), PREF["purchase"])

            linha = {
                "data":        pd.to_datetime(rec.get("date_start")),
                "gasto":       spend,
                "compras":     _num(compras),
                "faturamento": _num(receita),
                "roas":        (_num(receita)/spend) if spend > 0 else 0.0,
            }

            # rótulo por level
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
    if df.empty:
        return df

    df = df.sort_values("data")
    for c in ["gasto","compras","faturamento","roas"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("🔌 Meta Marketing API")
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
    access_token = st.text_input("Access Token", type="password")
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("📅 Intervalo")
    since_api = st.date_input("Desde", value=(datetime.today() - timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())

# ---------- Carregar ----------
since_dt = datetime.combine(since_api, datetime.min.time())
until_dt = datetime.combine(until_api, datetime.min.time())
if since_dt > until_dt:
    since_dt, until_dt = until_dt, since_dt

df = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Buscando dados..."):
        df = pull_insights(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            since=since_dt,
            until=until_dt,
            level="campaign",  # pode trocar p/ adset/ad se quiser
        )

# ---------- UI ----------
st.subheader("📥 Performance no intervalo")
if df.empty:
    st.info("Informe conta, token e intervalo para carregar dados.")
else:
    # KPIs do período
    inv = float(df["gasto"].sum())
    fat = float(df["faturamento"].sum())
    comp = float(df["compras"].sum())
    roas = (fat/inv) if inv>0 else 0.0
    cpa  = (inv/comp) if comp>0 else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Investimento", f"R$ {inv:,.0f}".replace(",",".")) 
    c2.metric("🏪 Faturamento", f"R$ {fat:,.0f}".replace(",",".")) 
    c3.metric("🛒 Compras", f"{comp:,.0f}".replace(",",".")) 
    c4.metric("📈 ROAS", f"{roas:,.2f}".replace(",",".")) 
    c5.metric("🎯 CPA", f"R$ {cpa:,.2f}".replace(",","."))

    # Diário
    daily = df.groupby("data", as_index=False)[["gasto","faturamento","compras"]].sum().sort_values("data")
    if not daily.empty:
        st.plotly_chart(
            px.bar(daily, x="data", y=["gasto","faturamento"], barmode="group", title="Investimento × Faturamento (diário)"),
            use_container_width=True
        )
        daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
        st.plotly_chart(
            px.line(daily, x="data", y="ROAS", markers=True, title="ROAS diário"),
            use_container_width=True
        )

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

st.caption("Obs.: usando action_report_time=conversion + unified attribution (igual ao Gerenciador). Se ainda divergir, pode ser a janela de atribuição do conjunto de anúncios.")
