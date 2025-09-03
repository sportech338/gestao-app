# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Meta Ads — Dashboard Limpo",
    page_icon="📈",
    layout="wide",
)

# =========================
# Estilos leves
# =========================
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.big-number { font-size:28px; font-weight:700; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def _retry_call(fn, max_retries=5, base_wait=1.3):
    """Retry exponencial para chamadas HTTP instáveis."""
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

def _ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def _to_float(x):
    try:
        return float(x or 0)
    except:
        return 0.0

def _sum_actions_generic(rows, contains: str) -> float:
    """
    Soma qualquer item de 'actions'/'action_values' cujo action_type contenha a substring (ex.: 'purchase').
    Se houver 'value', usa; caso contrário soma chaves numéricas (ex.: '7d_click', '1d_view', etc.).
    """
    total = 0.0
    if not rows:
        return 0.0
    for it in rows:
        at = str(it.get("action_type", "")).lower()
        if contains in at:
            if "value" in it:
                total += _to_float(it.get("value"))
            else:
                # soma apenas valores numéricos possíveis (janelas)
                for k, v in it.items():
                    if k not in ("action_type", "action_target", "action_carousel_card_id",
                                 "action_destination", "action_device", "action_channel", "value"):
                        total += _to_float(v)
    return float(total)

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str,
                         token: str,
                         api_version: str,
                         since_str: str,
                         until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True) -> pd.DataFrame:
    """
    Busca insights diários no padrão dos dashboards prontos:
      - use_unified_attribution_setting=true
      - action_report_time=conversion
      - sem 'action_types' no request (não cortamos vendas)
      - level fixo (default: 'campaign')
      - time_increment=1
    Fallback automático: se erro #100, refaz sem campos extras.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    base_fields = [
        "spend",
        "impressions",
        "clicks",
        "actions",
        "action_values",
        "account_currency",
        "date_start",
        "campaign_id",
        "campaign_name",
    ]
    extra_fields = ["link_clicks", "landing_page_views"]  # podem falhar em combinações específicas (#100)

    fields = base_fields + (extra_fields if try_extra_fields else [])
    params = {
        "access_token": token,
        "level": level,  # "campaign" recomendado
        "time_range": json.dumps({"since": since_str, "until": until_str}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
        "use_unified_attribution_setting": "true",
        "action_report_time": "conversion",
        # IMPORTANTE: não enviar action_attribution_windows quando unified=true.
        # IMPORTANTE: não enviar action_types — filtramos no cliente.
    }

    rows = []
    next_url = base_url
    next_params = params.copy()

    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=90))
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("Resposta inválida da Graph API.")

        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code = err.get("code")
            sub = err.get("error_subcode")
            msg = err.get("message")
            # Fallback: se fields extras quebraram (#100), refaça sem extras uma única vez
            if code == 100 and try_extra_fields:
                return fetch_insights_daily(
                    act_id=act_id, token=token, api_version=api_version,
                    since_str=since_str, until_str=until_str,
                    level=level, try_extra_fields=False
                )
            raise RuntimeError(f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}")

        for rec in payload.get("data", []):
            spend = _to_float(rec.get("spend"))
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            purchases_cnt = _sum_actions_generic(actions, "purchase")
            revenue_val   = _sum_actions_generic(action_values, "purchase")

            rows.append({
                "date":           pd.to_datetime(rec.get("date_start")),
                "spend":          spend,
                "purchases":      purchases_cnt,
                "revenue":        revenue_val,
                "impressions":    _to_float(rec.get("impressions")),
                "clicks":         _to_float(rec.get("clicks")),
                "link_clicks":    _to_float(rec.get("link_clicks")) if "link_clicks" in rec else np.nan,
                "lpv":            _to_float(rec.get("landing_page_views")) if "landing_page_views" in rec else np.nan,
                "currency":       rec.get("account_currency", "BRL"),
                "campaign_id":    rec.get("campaign_id", ""),
                "campaign_name":  rec.get("campaign_name", ""),
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url = base_url
            next_params = params.copy()
            next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Tipagem e métricas derivadas
    num_cols = ["spend", "purchases", "revenue", "impressions", "clicks", "link_clicks", "lpv"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df["cpa"]  = np.where(df["purchases"] > 0, df["spend"] / df["purchases"], np.nan)
    df = df.sort_values("date")
    return df

# =========================
# Sidebar — parâmetros
# =========================
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", options=["campaign", "account"], index=0)

today = date.today()
since = st.sidebar.date_input("Desde", value=today - timedelta(days=7))
until = st.sidebar.date_input("Até", value=today)

do_fetch = st.sidebar.button("Atualizar dados", type="primary")

# =========================
# Busca e UI
# =========================
st.title("📈 Meta Ads — Dashboard (Paridade com Ads Manager)")
st.caption("Padrão: use_unified_attribution_setting=true • action_report_time=conversion • sem action_types no request.")

if do_fetch:
    if not act_id or not token:
        st.error("Informe **Ad Account ID** e **Access Token**.")
        st.stop()

    with st.spinner("Buscando dados da Meta…"):
        try:
            df = fetch_insights_daily(
                act_id=act_id,
                token=token,
                api_version=api_version,
                since_str=str(since),
                until_str=str(until),
                level=level,
                try_extra_fields=True,  # se der #100, o código refaz sem extras
            )
        except Exception as e:
            st.error(f"Erro na coleta: {e}")
            st.stop()

    if df.empty:
        st.info("Nenhum dado retornado para o período. Verifique permissões, conta, intervalo e eventos de Purchase (valor/moeda).")
        st.stop()

    # =========================
    # KPIs do período
    # =========================
    tot_spend = float(df["spend"].sum())
    tot_purch = float(df["purchases"].sum())
    tot_rev   = float(df["revenue"].sum())
    roas_g    = (tot_rev / tot_spend) if tot_spend > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><div class="small-muted">Investimento</div>'
                    f'<div class="big-number">{_fmt_money_br(tot_spend)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="small-muted">Compras</div>'
                    f'<div class="big-number">{int(round(tot_purch)):,}</div></div>'.replace(",", "."),
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="small-muted">Faturamento</div>'
                    f'<div class="big-number">{_fmt_money_br(tot_rev)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-card"><div class="small-muted">ROAS</div>'
                    f'<div class="big-number">{roas_g:,.2f}</div></div>'.replace(",", "X").replace(".", ",").replace("X", "."),
                    unsafe_allow_html=True)

    st.divider()

    # =========================
    # Curvas diárias
    # =========================
    daily = df.groupby("date", as_index=False)[["spend", "revenue", "purchases"]].sum()
    st.subheader("Série diária")
    st.line_chart(daily.set_index("date")[["spend", "revenue"]])
    st.caption("Curva diária de investimento e faturamento. Compras exibidas na tabela abaixo.")

    # =========================
    # Agregação por campanha
    # =========================
    st.subheader("Campanhas (somatório no período)")
    agg_cols = ["spend", "purchases", "revenue", "impressions", "clicks", "link_clicks", "lpv"]
    by_campaign = df.groupby(["campaign_id", "campaign_name"], as_index=False)[agg_cols].sum()
    by_campaign["roas"] = np.where(by_campaign["spend"] > 0, by_campaign["revenue"] / by_campaign["spend"], np.nan)
    by_campaign["cpa"]  = np.where(by_campaign["purchases"] > 0, by_campaign["spend"] / by_campaign["purchases"], np.nan)

    # Ordena por gasto (principal driver em auditoria)
    by_campaign = by_campaign.sort_values(["spend"], ascending=False)

    # Formatação amigável
    show = by_campaign.copy()
    show["spend"] = show["spend"].apply(_fmt_money_br)
    show["revenue"] = show["revenue"].apply(_fmt_money_br)
    show["roas"] = show["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
    show["cpa"] = show["cpa"].apply(lambda x: _fmt_money_br(x) if pd.notnull(x) else "")

    st.dataframe(
        show.rename(columns={
            "campaign_id": "ID da campanha",
            "campaign_name": "Campanha",
            "spend": "Investimento",
            "purchases": "Compras",
            "revenue": "Faturamento",
            "roas": "ROAS",
            "cpa": "CPA",
            "impressions": "Impr.",
            "clicks": "Cliques",
            "link_clicks": "Link Clicks",
            "lpv": "LPV",
        }),
        use_container_width=True,
        height=520
    )

    with st.expander("Ver dados diários (detalhe)"):
        dd = df.copy()
        dd["date"] = dd["date"].dt.date
        dd_fmt = dd[["date","campaign_name","spend","purchases","revenue","impressions","clicks","link_clicks","lpv","roas","cpa"]].copy()
        dd_fmt["spend"] = dd_fmt["spend"].apply(_fmt_money_br)
        dd_fmt["revenue"] = dd_fmt["revenue"].apply(_fmt_money_br)
        dd_fmt["roas"] = dd_fmt["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
        dd_fmt["cpa"] = dd_fmt["cpa"].apply(lambda x: _fmt_money_br(x) if pd.notnull(x) else "")
        st.dataframe(dd_fmt, use_container_width=True)

else:
    st.info("Preencha a conta/token, selecione o período e clique **Atualizar dados**.")
    st.markdown(
        "<span class='small-muted'>Padrões para bater com o Ads Manager: "
        "<b>use_unified_attribution_setting=true</b>, <b>action_report_time=conversion</b>, "
        "sem <b>action_types</b> no request e agregação no cliente.</span>", unsafe_allow_html=True
    )
