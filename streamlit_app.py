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
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "retry", "temporarily unavailable", "timeout"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

# Preferências de tipos de compra (vamos somar "omni" e "específicos" e usar o MAIOR)
PURCHASE_OMNI = ["omni_purchase"]
PURCHASE_SPECIFIC = [
    "purchase",
    "onsite_conversion.purchase",
    "offsite_conversion.fb_pixel_purchase",
]

def _to_float(x):
    try:
        return float(x or 0)
    except:
        return 0.0

def _sum_windows_or_value(item: dict) -> float:
    """
    Alguns itens trazem 'value' (total agregado).
    Outros trazem por janelas (ex.: '7d_click', '1d_view', etc.).
    Vamos priorizar 'value' se existir, senão somar todas as chaves numéricas.
    """
    if not isinstance(item, dict):
        return _to_float(item)
    if "value" in item:
        return _to_float(item.get("value"))

    s = 0.0
    for k, v in item.items():
        # soma apenas números (ex.: '7d_click', '1d_view', '28d_click'...).
        s += _to_float(v)
    return s

def _sum_by_types(rows: list, types: list) -> float:
    """Soma total para uma lista de action_types."""
    if not rows:
        return 0.0
    acc = 0.0
    for it in rows:
        at = str(it.get("action_type") or "").lower()
        if any(at == t for t in types):
            acc += _sum_windows_or_value(it)
    # fallback: se lista vazia, tenta qualquer 'purchase'
    if acc == 0.0:
        for it in rows:
            if "purchase" in str(it.get("action_type", "")).lower():
                acc += _sum_windows_or_value(it)
    return float(acc)

def _ensure_act_prefix(ad_account_id: str) -> str:
    s = ad_account_id.strip()
    return s if s.startswith("act_") else f"act_{s}"

# -------------------- coleta da API --------------------
@st.cache_data(ttl=600, show_spinner=True)
def pull_meta_insights_correct(act_id: str, token: str, api_version: str,
                               since: datetime, until: datetime,
                               report_time: str = "conversion",
                               audit: bool = False) -> pd.DataFrame:
    """
    Retorna por dia: data, gasto, compras (qtd), faturamento (R$), campanha.
    Usa a Atribuição Unificada do conjunto.
    'report_time': 'conversion' (recomendado) ou 'impression'.
    """
    if not act_id or not token:
        return pd.DataFrame()

    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    fields = [
        "spend",
        "actions",
        "action_values",
        "date_start",
        "campaign_name",
        "account_name",
    ]

    params = {
        "access_token": token,
        "level": "campaign",
        "time_range": json.dumps({
            "since": since.strftime("%Y-%m-%d"),
            "until": until.strftime("%Y-%m-%d"),
        }),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
        "use_unified_attribution_setting": "true",
        "action_report_time": report_time,  # 'conversion' bate com o dia da compra
        # IMPORTANTE: NÃO usar 'action_types' aqui; filtraremos no cliente.
        # IMPORTANTE: NÃO enviar 'action_attribution_windows' quando unified=true.
    }

    # Modo auditoria: queremos ver por action_type também
    if audit:
        params["action_breakdowns"] = "action_type"

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=120))
        try:
            payload = resp.json()
        except Exception:
            st.error("Resposta inválida da Graph API.")
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
            spend = _to_float(rec.get("spend"))
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            # QUANTIDADE de compras
            omni_cnt = _sum_by_types(actions, PURCHASE_OMNI)
            spec_cnt = _sum_by_types(actions, PURCHASE_SPECIFIC)
            purchases_cnt = max(omni_cnt, spec_cnt)

            # VALOR de compras (faturamento)
            omni_rev = _sum_by_types(action_values, PURCHASE_OMNI)
            spec_rev = _sum_by_types(action_values, PURCHASE_SPECIFIC)
            revenue_val = max(omni_rev, spec_rev)

            rows.append({
                "data":        pd.to_datetime(rec.get("date_start")),
                "gasto":       spend,
                "compras":     float(purchases_cnt),
                "faturamento": float(revenue_val),
                "campanha":    rec.get("campaign_name") or rec.get("account_name") or "",
                "_raw_actions": actions if audit else None,
                "_raw_action_values": action_values if audit else None,
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
        for c in ["gasto", "compras", "faturamento"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# -------------------- exemplo de uso / somatórios --------------------
if __name__ == "__main__":
    st.header("Teste rápido (somatórios do intervalo)")

    with st.sidebar:
        act_id = st.text_input("Ad Account ID (ex.: act_1234567890)")
        access_token = st.text_input("Access Token", type="password")
        api_version = st.text_input("API version", value="v23.0")
        since_api = st.date_input("Desde", value=(datetime.today() - timedelta(days=7)).date())
        until_api = st.date_input("Até", value=datetime.today().date())
        mode = st.radio("Modo de data", ["Conversão (recomendado)", "Impressão"], index=0)
        report_time = "conversion" if mode.startswith("Conversão") else "impression"
        audit_mode = st.checkbox("Auditar por action_type (debug)", value=False)

    df = pd.DataFrame()
    if act_id and access_token:
        with st.spinner("Buscando…"):
            df = pull_meta_insights_correct(
                act_id=act_id,
                token=access_token,
                api_version=api_version,
                since=datetime.combine(since_api, datetime.min.time()),
                until=datetime.combine(until_api, datetime.min.time()),
                report_time=report_time,
                audit=audit_mode
            )

    if df.empty:
        st.info("Informe conta/token/intervalo. Se vier vazio, verifique permissões, intervalo e se há compras atribuídas.")
    else:
        # Somatórios do intervalo
        gasto_total = float(df["gasto"].sum())
        comp_total  = float(df["compras"].sum())
        fat_total   = float(df["faturamento"].sum())
        roas_global = (fat_total / gasto_total) if gasto_total > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Valor usado (R$)", f"R$ {gasto_total:,.0f}".replace(",", "."))
        c2.metric("🛒 Compras (qtd)", f"{comp_total:,.0f}".replace(",", "."))
        c3.metric("🏪 Faturamento (R$)", f"R$ {fat_total:,.0f}".replace(",", "."))
        c4.metric("📈 ROAS (global)", f"{roas_global:,.2f}".replace(",", "."))

        with st.expander("Amostra (por dia e campanha)"):
            st.dataframe(df.drop(columns=[c for c in df.columns if c.startswith("_raw")]), use_container_width=True)

        if audit_mode:
            st.warning("Auditoria ligada: use para comparar tipos de compra (omni vs específicos).")
            st.caption("Se houver divergência, é comum ver 1 compra num tipo específico e não no 'omni' ou vice-versa.")
