import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import datetime, timedelta

# -------------------- helpers --------------------
def _retry_call(fn, max_retries=5, base_wait=1.5):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout"]):
                time.sleep(base_wait * (2 ** i)); continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

# --- NOVO: some as janelas da atribuição da conta (o Manager faz isso) ---
ATTR_WINDOWS = ["7d_click", "1d_view"]   # ajuste se sua conta usar outra regra

def _to_float(x):
    try:
        return float(x or 0)
    except:
        return 0.0

def _one_item_total(item: dict) -> float:
    """Se o item tem 'value', usa esse valor; senão, soma as janelas definidas em ATTR_WINDOWS."""
    if not isinstance(item, dict):
        return _to_float(item)
    if "value" in item:
        return _to_float(item.get("value"))
    return sum(_to_float(item.get(k)) for k in ATTR_WINDOWS if k in item)

def _sum_by_action_type(rows: list, wanted_types: list) -> float:
    """Soma por action_type usando _one_item_total; retorna a 1ª ação disponível conforme prioridade."""
    if not rows:
        return 0.0
    acc = {}
    for it in rows:
        at = str(it.get("action_type") or "").lower()
        acc[at] = acc.get(at, 0.0) + _one_item_total(it)
    for t in wanted_types:
        if t in acc:
            return float(acc[t])
    # fallback amplo (qualquer 'purchase')
    return float(sum(v for k, v in acc.items() if "purchase" in k) or 0.0)


# -------------------- coleta da API --------------------
@st.cache_data(ttl=600, show_spinner=True)
def pull_meta_insights_correct(act_id: str, token: str, api_version: str,
                               since: datetime, until: datetime,
                               report_time: str = "impression") -> pd.DataFrame:
    """
    Retorna por dia:
      data, gasto, compras (qtd), faturamento (R$), campanha.
    Usa a atribuição unificada da conta e report_time: 'impression' ou 'conversion'.
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = ["spend", "actions", "action_values", "date_start", "campaign_name", "account_name"]
    params = {
        "access_token": token,
        "level": "campaign",
        "time_range": json.dumps({"since": since.strftime("%Y-%m-%d"), "until": until.strftime("%Y-%m-%d")}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
        "use_unified_attribution_setting": "true",
        "action_report_time": report_time,  # "impression" ou "conversion"
        # peça todas as variações de purchase para garantir compatibilidade
        "action_types": json.dumps(PURCHASE_PREF),
    }

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=60))
        try:
            payload = resp.json()
        except Exception:
            st.error("Resposta inválida da Graph API."); return pd.DataFrame()

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

            # quantidade de compras (actions) e valor de compra (action_values)
            purchases_cnt = _sum_by_action_type(rec.get("actions"), PURCHASE_PREF)
            revenue_val   = _sum_by_action_type(rec.get("action_values"), PURCHASE_PREF)

            rows.append({
                "data":        pd.to_datetime(rec.get("date_start")),
                "gasto":       spend,
                "compras":     float(purchases_cnt),
                "faturamento": float(revenue_val),
                "campanha":    rec.get("campaign_name") or rec.get("account_name") or "",
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy(); next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("data")
        for c in ["gasto","compras","faturamento"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# -------------------- exemplo de uso / somatórios --------------------
if __name__ == "__main__":
    st.header("Teste rápido (somatórios do intervalo)")

    with st.sidebar:
        act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
        access_token = st.text_input("Access Token", type="password")
        api_version = st.text_input("API version", value="v23.0")
        since_api = st.date_input("Desde", value=(datetime.today()-timedelta(days=7)).date())
        until_api = st.date_input("Até", value=datetime.today().date())
        mode = st.radio("Modo de data", ["Impressão (padrão)", "Conversão"], index=0)
        report_time = "impression" if mode.startswith("Impressão") else "conversion"

    df = pd.DataFrame()
    if act_id and access_token:
        with st.spinner("Buscando…"):
            df = pull_meta_insights_correct(
                act_id=act_id,
                token=access_token,
                api_version=api_version,
                since=datetime.combine(since_api, datetime.min.time()),
                until=datetime.combine(until_api, datetime.min.time()),
                report_time=report_time
            )

    if df.empty:
        st.info("Informe conta/token/intervalo. Se vier vazio, verifique permissões, intervalo e se há compras atribuídas.")
    else:
        # somatórios do intervalo
        gasto_total = float(df["gasto"].sum())
        comp_total  = float(df["compras"].sum())
        fat_total   = float(df["faturamento"].sum())
        roas_global = (fat_total / gasto_total) if gasto_total > 0 else 0.0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("💰 Valor usado (R$)", f"R$ {gasto_total:,.0f}".replace(",", "."))
        c2.metric("🛒 Compras (qtd)", f"{comp_total:,.0f}".replace(",", "."))
        c3.metric("🏪 Faturamento (R$)", f"R$ {fat_total:,.0f}".replace(",", "."))
        c4.metric("📈 ROAS (global)", f"{roas_global:,.2f}".replace(",", "."))

        with st.expander("Amostra (por dia e campanha)"):
            st.dataframe(df, use_container_width=True)

        # (opcional) ROAS médio diário
        daily = df.groupby("data", as_index=False)[["gasto","faturamento"]].sum()
        daily["ROAS"] = np.where(daily["gasto"]>0, daily["faturamento"]/daily["gasto"], np.nan)
        st.caption(f"ROAS médio diário: {float(daily['ROAS'].mean(skipna=True) or 0):.2f}")
