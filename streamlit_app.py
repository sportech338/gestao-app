import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import datetime, timedelta

st.set_page_config(page_title="Performance Ads — Base", page_icon="📊", layout="wide")
st.title("📊 Performance Ads — Base (somatórios + ROAS médio)")
st.caption("Puxa dados da Meta (Graph API). Sempre soma Compras, Gasto e Faturamento no intervalo. ROAS pode ser média diária, média por campanha ou global (ponderado).")

# -------------------------
# Re-tentativa simples
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
# Coleta da API (mínima e correta p/ compras e faturamento)
# -------------------------
@st.cache_data(ttl=600, show_spinner=True)
def pull_insights(act_id: str, token: str, api_version: str,
                  since: datetime, until: datetime, level: str = "campaign") -> pd.DataFrame:
    """
    Retorna linhas diárias com: data, gasto, compras, faturamento, campanha.
    Compra e faturamento vêm de conversions/conversion_values (unified);
    se não vier, caímos no fallback actions/action_values.
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend",
        "conversions","conversion_values",   # preferido (unified attribution)
        "actions","action_values",           # fallback
        "date_start",
        "campaign_name","account_name",
    ]
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({
            "since": since.strftime("%Y-%m-%d"),
            "until": until.strftime("%Y-%m-%d"),
        }),
        "time_increment": 1,                 # diário
        "fields": ",".join(fields),
        "action_report_time": "conversion",  # atribui na data da conversão
        "use_unified_attribution_setting": "true",
        "limit": 500,
    }

    # preferências p/ tipos de evento de compra (ordem importa)
    PREF = {"purchase": ["omni_purchase","purchase","offsite_conversion.fb_pixel_purchase"]}

    def _item_value(item: dict) -> float:
        # 'conversion_values' costuma ter 'value'; 'conversions' pode ter várias chaves
        if "value" in (item or {}):
            try: return float(item["value"] or 0)
            except: return 0.0
        s = 0.0
        for k, v in (item or {}).items():
            if k == "action_type": 
                continue
            try: s += float(v or 0)
            except: pass
        return s

    def _best(list_of_dicts, keys) -> float:
        if not list_of_dicts:
            return 0.0
        acc = {}
        for it in list_of_dicts:
            at = str(it.get("action_type") or "").lower()
            acc[at] = acc.get(at, 0.0) + _item_value(it)
        # tenta na ordem preferida
        for k in keys:
            if k in acc:
                return acc[k]
        # fallback amplo: se nada das chaves preferidas existir, soma tudo que contenha 'purchase'
        tot = sum(v for k, v in acc.items() if "purchase" in k)
        return tot

    rows = []
    next_url, next_params = base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=60))
        try:
            payload = resp.json()
        except Exception:
            st.error("Falha ao decodificar resposta da API.")
            return pd.DataFrame()

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

            # preferência unified (conversions/conversion_values)
            compras = _best(rec.get("conversions"),       PREF["purchase"])
            receita = _best(rec.get("conversion_values"), PREF["purchase"])

            # fallback clássico (actions/action_values)
            if compras == 0 and receita == 0:
                compras = _best(rec.get("actions"),       PREF["purchase"])
                receita = _best(rec.get("action_values"), PREF["purchase"])

            rows.append({
                "data":        pd.to_datetime(rec.get("date_start")),
                "gasto":       spend,
                "compras":     float(compras),
                "faturamento": float(receita),
                "campanha":    rec.get("campaign_name") or rec.get("account_name") or "",
            })

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

# -------------------------
# Sidebar (parâmetros)
# -------------------------
with st.sidebar:
    st.header("🔌 Meta Marketing API")
    act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
    access_token = st.text_input("Access Token", type="password")
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("📅 Intervalo")
    since_api = st.date_input("Desde", value=(datetime.today()-timedelta(days=7)).date())
    until_api = st.date_input("Até", value=datetime.today().date())

    st.subheader("📈 ROAS — como calcular")
    roas_mode = st.radio(
        "Escolha o cálculo do ROAS mostrado no card",
        ["Média diária (simples)", "Média por campanha (simples)", "Global (ponderado)"],
        index=0
    )

# -------------------------
# Carregar dados
# -------------------------
df = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Conectando e puxando dados..."):
        df = pull_insights(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            since=datetime.combine(since_api, datetime.min.time()),
            until=datetime.combine(until_api, datetime.min.time()),
            level="campaign"  # simples e útil para começar
        )

if df.empty:
    st.info("Informe conta, token e intervalo para carregar dados.")
else:
    # --- SOMATÓRIOS do período (sempre corretos) ---
    gasto_total = float(df["gasto"].sum())
    fat_total   = float(df["faturamento"].sum())
    comp_total  = float(df["compras"].sum())

    # --- ROAS médio conforme seleção ---
    if roas_mode == "Média diária (simples)":
        daily = df.groupby("data", as_index=False)[["gasto","faturamento"]].sum()
        daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
        roas_card = float(daily["ROAS"].replace([np.inf, -np.inf], np.nan).mean(skipna=True) or 0.0)
        roas_hint = "Média aritmética do ROAS diário"
    elif roas_mode == "Média por campanha (simples)":
        per_camp = df.groupby("campanha", as_index=False)[["gasto","faturamento"]].sum()
        per_camp = per_camp[per_camp["gasto"]>0]
        per_camp["ROAS"] = per_camp["faturamento"]/per_camp["gasto"]
        roas_card = float(per_camp["ROAS"].replace([np.inf, -np.inf], np.nan).mean(skipna=True) or 0.0)
        roas_hint = "Média aritmética do ROAS por campanha"
    else:  # Global (ponderado)
        roas_card = (fat_total/gasto_total) if gasto_total>0 else 0.0
        roas_hint = "Faturamento total ÷ Gasto total"

    # --- Cards principais ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Valor usado (soma)", f"R$ {gasto_total:,.0f}".replace(",", "."))
    c2.metric("🏪 Faturamento (soma)", f"R$ {fat_total:,.0f}".replace(",", "."))
    c3.metric("🛒 Compras (soma)", f"{comp_total:,.0f}".replace(",", "."))
    c4.metric("📈 ROAS", f"{roas_card:,.2f}".replace(",", "."), help=roas_hint)

    st.markdown("---")

    # Série diária (opcional para visual)
    daily_view = df.groupby("data", as_index=False)[["gasto","faturamento","compras"]].sum()
    if not daily_view.empty:
        st.bar_chart(daily_view.set_index("data")[["gasto","faturamento"]])
        roas_daily = np.where(daily_view["gasto"]>0, daily_view["faturamento"]/daily_view["gasto"], np.nan)
        st.line_chart(pd.DataFrame({"ROAS": roas_daily}, index=daily_view["data"]))

    # Tabela crua (para auditoria rápida)
    with st.expander("Ver dados brutos (por dia x campanha)"):
        st.dataframe(df, use_container_width=True)

    # Resumo por campanha (útil pra gestão)
    per_camp = df.groupby("campanha", as_index=False).agg(
        gasto=("gasto","sum"), faturamento=("faturamento","sum"), compras=("compras","sum")
    )
    per_camp["ROAS"] = np.where(per_camp["gasto"]>0, per_camp["faturamento"]/per_camp["gasto"], np.nan)
    per_camp["CPA"]  = np.where(per_camp["compras"]>0, per_camp["gasto"]/per_camp["compras"], np.nan)
    st.markdown("### 📊 Resumo por campanha")
    st.dataframe(per_camp.sort_values(["ROAS","faturamento","gasto"], ascending=[False, False, True]), use_container_width=True)

    st.caption("Obs.: ROAS *Global (ponderado)* = Faturamento total ÷ Gasto total. ROAS *Médio* pode ser calculado por dia ou por campanha (média simples).")
