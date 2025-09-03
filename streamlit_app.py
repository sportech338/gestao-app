import streamlit as st
import pandas as pd
import requests, json, time
from datetime import datetime, timedelta

# -------------------------
# Helper para re-tentativa
# -------------------------
def _retry_call(fn, max_retries=5, base_wait=1.5):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

# -------------------------
# Função de coleta da API
# -------------------------
@st.cache_data(ttl=600, show_spinner=True)
def pull_insights(act_id: str, token: str, api_version: str,
                  since: datetime, until: datetime, level: str = "campaign") -> pd.DataFrame:
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend",
        "conversions","conversion_values",  # unified attribution
        "actions","action_values",          # fallback
        "date_start","campaign_name","account_name"
    ]
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({"since": since.strftime("%Y-%m-%d"),
                                  "until": until.strftime("%Y-%m-%d")}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "action_report_time": "conversion",
        "use_unified_attribution_setting": "true",
        "limit": 500,
    }

    def _item_value(item):
        if "value" in item: return float(item["value"] or 0)
        return sum(float(v or 0) for k,v in item.items() if k!="action_type")

    def _best(list_of_dicts, keys):
        if not list_of_dicts: return 0.0
        acc = {}
        for it in list_of_dicts:
            at = str(it.get("action_type") or "").lower()
            val = _item_value(it)
            acc[at] = acc.get(at, 0) + val
        for k in keys:
            if k in acc: return acc[k]
        return 0.0

    PREF = {"purchase": ["omni_purchase","purchase","offsite_conversion.fb_pixel_purchase"]}

    rows = []
    next_url, next_params = base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=60))
        payload = resp.json()
        if resp.status_code != 200:
            st.error(payload.get("error",{}).get("message"))
            return pd.DataFrame()

        for rec in payload.get("data", []):
            spend = float(rec.get("spend",0) or 0)
            compras = _best(rec.get("conversions"), PREF["purchase"])
            receita = _best(rec.get("conversion_values"), PREF["purchase"])
            if compras == 0 and receita == 0:
                compras = _best(rec.get("actions"), PREF["purchase"])
                receita = _best(rec.get("action_values"), PREF["purchase"])

            rows.append({
                "data": pd.to_datetime(rec.get("date_start")),
                "gasto": spend,
                "compras": compras,
                "faturamento": receita,
                "roas": (receita/spend) if spend>0 else 0.0,
                "campanha": rec.get("campaign_name","")
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy(); next_params["after"] = after
        else:
            break

    return pd.DataFrame(rows)

# -------------------------
# Sidebar para parâmetros
# -------------------------
with st.sidebar:
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
    access_token = st.text_input("Access Token", type="password")
    api_version = st.text_input("API version", value="v23.0")
    since_api = st.date_input("Desde", value=(datetime.today()-timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())

# -------------------------
# Executa coleta
# -------------------------
df = pd.DataFrame()
if act_id and access_token:
    with st.spinner("🔌 Conectando ao Meta Ads..."):
        df = pull_insights(act_id, access_token, api_version,
                           datetime.combine(since_api, datetime.min.time()),
                           datetime.combine(until_api, datetime.min.time()))

if df.empty:
    st.info("Informe conta, token e intervalo para carregar dados.")
else:
    st.success("✅ Dados carregados da API!")
    st.dataframe(df)
