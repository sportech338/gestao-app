
# app.py — Meta Ads com Funil completo
import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
APP_TZ = ZoneInfo("America/Sao_Paulo")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============== Config & Estilos ===============
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)
ATTR_KEYS = ["7d_click", "1d_view"]

# --- ADD: constantes e parser para breakdown por hora
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"

def _parse_hour_bucket(h):
    """Normaliza bucket de hora ('0'..'23' ou '00:00'..'23:00') para int [0..23]."""
    if h is None:
        return None
    try:
        s = str(h).strip()
        val = int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except Exception:
        return None

# =============== Helpers de rede/parse ===============
def _retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout"]):
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

def _sum_item(item, allowed_keys=None):
    """Usa 'value' quando existir; senão soma SOMENTE as chaves permitidas (ex.: 7d_click, 1d_view)."""
    if not isinstance(item, dict):
        return _to_float(item)
    if "value" in item:
        return _to_float(item.get("value"))
    keys = allowed_keys or ATTR_KEYS
    s = 0.0
    for k in keys:
        s += _to_float(item.get(k))
    return s

def _sum_actions_exact(rows, exact_names, allowed_keys=None) -> float:
    """Soma totals de actions pelos nomes exatos (case-insensitive), respeitando a janela (allowed_keys)."""
    if not rows:
        return 0.0
    names = {str(n).lower() for n in exact_names}
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type", "")).lower()
        if at in names:
            tot += _sum_item(r, allowed_keys)
    return float(tot)

def _sum_actions_contains(rows, substrs, allowed_keys=None) -> float:
    """Soma totals de actions que CONTENHAM qualquer substring, respeitando a janela (allowed_keys)."""
    if not rows:
        return 0.0
    ss = [str(s).lower() for s in substrs]
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type", "")).lower()
        if any(s in at for s in ss):
            tot += _sum_item(r, allowed_keys)
    return float(tot)

def _pick_purchase_totals(rows, allowed_keys=None) -> float:
    """Prioriza omni_purchase; senão pega o MAIOR entre tipos específicos (sem duplicar janelas)."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type", "")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))
    candidates = {
        "purchase": 0.0,
        "onsite_conversion.purchase": 0.0,
        "offsite_conversion.fb_pixel_purchase": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)
    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))
    # fallback amplo (respeitando allowed_keys)
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def funnel_fig(labels, values, title=None):
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value",
            textposition="inside",
            texttemplate="<b>%{value}</b>",  # deixa o número em negrito
            textfont=dict(size=35),
            opacity=0.95,
            connector={"line": {"dash": "dot", "width": 1}},
        )
    )
    fig.update_layout(
        title=title or "",
        margin=dict(l=10, r=10, t=48, b=10),
        height=540,                         # <<< AUMENTE AQUI (ex.: 600–720)
        template="plotly_white",
        separators=",.",                    # pt-BR
        uniformtext=dict(minsize=12, mode="show")
    )
    return fig



def enforce_monotonic(values):
    """Garante formato de funil: cada etapa <= etapa anterior (só para o desenho)."""
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

# ==== Helpers de comparação/formatos ====
def _rate(a, b):
    return (a / b) if b and b > 0 else np.nan

def _fmt_pct_br(x):
    return (
        f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_ratio_br(x):  # ROAS "1,23x"
    return (
        f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_int_br(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except:
        return ""

def _fmt_int_signed_br(x):
    try:
        v = int(round(float(x)))
        s = f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v > 0 else (f"-{s}" if v < 0 else "0")
    except:
        return ""

def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return (n / d) if d > 0 else np.nan

# =============== Coleta (com fallback de campos extras) ===============
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único ('campaign' recomendado)
    - Usa action_report_time=conversion e action_attribution_windows fixos (paridade com Ads Manager)
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    base_fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name"
    ]
    extra_fields = ["link_clicks", "landing_page_views"]

    fields = base_fields + (extra_fields if try_extra_fields else [])
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({"since": since_str, "until": until_str}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
    # >>> Fixos para paridade com o Ads Manager
        "action_report_time": "conversion",
        "action_attribution_windows": ",".join(ATTR_KEYS),  # "7d_click,1d_view"
    }

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=90))
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("Resposta inválida da Graph API.")

        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
            if code == 100 and try_extra_fields:
                # refaz sem extras
                return fetch_insights_daily(act_id, token, api_version, since_str, until_str, level, try_extra_fields=False)
            raise RuntimeError(f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}")

        for rec in payload.get("data", []):
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            # Cliques em link (preferir field; fallback action com janela)
            link_clicks = rec.get("link_clicks", None)
            if link_clicks is None:
                link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

            # LPV (preferir field; fallback landing_page_view → view_content → contains "landing_page")
            lpv = rec.get("landing_page_views", None)
            if lpv is None:
                lpv = _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                if lpv == 0:
                    lpv = (_sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                           or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))

            # Iniciar checkout / add payment info com janela definida
            ic  = _sum_actions_exact(actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS)
            api = _sum_actions_exact(actions, ["add_payment_info"], allowed_keys=ATTR_KEYS)

            # Purchase (qtd) e Revenue (valor) respeitando janela
            purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
            revenue_val   = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

            rows.append({
                "date":           pd.to_datetime(rec.get("date_start")),
                "currency":       rec.get("account_currency", "BRL"),
                "campaign_id":    rec.get("campaign_id", ""),
                "campaign_name":  rec.get("campaign_name", ""),

                # métricas básicas
                "spend":          _to_float(rec.get("spend")),
                "impressions":    _to_float(rec.get("impressions")),
                "clicks":         _to_float(rec.get("clicks")),

                # funil
                "link_clicks":    _to_float(link_clicks),
                "lpv":            _to_float(lpv),
                "init_checkout":  _to_float(ic),
                "add_payment":    _to_float(api),
                "purchases":      _to_float(purchases_cnt),
                "revenue":        _to_float(revenue_val),
            })


        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy()
            next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # >>> mantenha estes 4 passos DENTRO da função diária
    num_cols = ["spend", "impressions", "clicks", "link_clicks",
                "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Métricas derivadas (do período/dia; taxas serão calculadas em agregações)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df = df.sort_values("date")
    return df

# --- ADD: coleta por hora (dayparting)
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(act_id: str, token: str, api_version: str,
                          since_str: str, until_str: str,
                          level: str = "campaign") -> pd.DataFrame:
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]

    def _run_hourly(action_rt: str) -> pd.DataFrame:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": since_str, "until": until_str}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": action_rt,            # conversion (primeira tentativa) ou impression (fallback)
            "action_attribution_windows": ",".join(ATTR_KEYS),
            "breakdowns": HOUR_BREAKDOWN,
        }
        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=90))
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API (hourly).")

            if resp.status_code != 200:
                # Propaga o erro para o chamador decidir fallback
                err = (payload or {}).get("error", {})
                code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
                raise RuntimeError(f"Graph API error hourly | code={code} subcode={sub} | {msg}")

            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []
                hour_bucket = _parse_hour_bucket(rec.get(HOUR_BREAKDOWN))

                link_clicks = rec.get("link_clicks")
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

                lpv = rec.get("landing_page_views")
                if lpv is None:
                    lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                           or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                           or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))

                ic   = _sum_actions_exact(actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS)
                api_ = _sum_actions_exact(actions, ["add_payment_info"], allowed_keys=ATTR_KEYS)
                pur  = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev  = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows.append({
                    "date":          pd.to_datetime(rec.get("date_start")),
                    "hour":          hour_bucket,
                    "currency":      rec.get("account_currency","BRL"),
                    "campaign_id":   rec.get("campaign_id",""),
                    "campaign_name": rec.get("campaign_name",""),
                    "spend":         _to_float(rec.get("spend")),
                    "impressions":   _to_float(rec.get("impressions")),
                    "clicks":        _to_float(rec.get("clicks")),
                    "link_clicks":   _to_float(link_clicks),
                    "lpv":           _to_float(lpv),
                    "init_checkout": _to_float(ic),
                    "add_payment":   _to_float(api_),
                    "purchases":     _to_float(pur),
                    "revenue":       _to_float(rev),
                })

            # paginação: preferir paging.next; senão usar cursor.after
            paging = payload.get("paging") or {}
            next_link = paging.get("next")
            if next_link:
                next_url, next_params = next_link, None
            else:
                after = ((paging.get("cursors") or {}).get("after"))
                if after:
                    next_url, next_params = base_url, params.copy()
                    next_params["after"] = after
                else:
                    break

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["dow"] = df["date"].dt.dayofweek
        order = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
        df["dow_label"] = df["dow"].map(order)
        df["roas"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
        return df.sort_values(["date","hour"])

    # 1ª tentativa: conversion
    try:
        df = _run_hourly("conversion")
        if not df.empty:
            return df
    except Exception as e1:
        # mostra aviso com o erro real
        st.warning(f"Hour breakdown (conversion) falhou: {e1}")

    # fallback: impression
    try:
        st.info("Tentando breakdown por hora com `action_report_time=impression`…")
        return _run_hourly("impression")
    except Exception as e2:
        st.error(f"Hour breakdown (impression) também falhou: {e2}")
        return pd.DataFrame()

# =============== Sidebar (filtros) ===============
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign", "account"], index=0)
today = datetime.now(APP_TZ).date()

preset = st.sidebar.radio(
    "Período rápido",
    ["Hoje", "Ontem", "Últimos 7 dias", "Últimos 14 dias", "Últimos 30 dias", "Personalizado"],
    index=2,  # "Últimos 7 dias"
)

def _range_from_preset(p):
    local_today = datetime.now(APP_TZ).date()
    base_end = local_today - timedelta(days=1)  # períodos “terminando ontem”, igual ao Ads Manager

    if p == "Hoje":
        return local_today, local_today
    if p == "Ontem":
        return local_today - timedelta(days=1), local_today - timedelta(days=1)
    if p == "Últimos 7 dias":
        return base_end - timedelta(days=6), base_end
    if p == "Últimos 14 dias":
        return base_end - timedelta(days=13), base_end
    if p == "Últimos 30 dias":
        return base_end - timedelta(days=29), base_end
    return base_end - timedelta(days=6), base_end

_since_auto, _until_auto = _range_from_preset(preset)

if preset == "Personalizado":
    since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
    until = st.sidebar.date_input("Até",   value=_until_auto, key="until_custom")
else:
    # Mostra as datas calculadas, bloqueadas (só leitura)
    since = st.sidebar.date_input("Desde", value=_since_auto, disabled=True, key="since_auto")
    until = st.sidebar.date_input("Até",   value=_until_auto, disabled=True, key="until_auto")

ready = bool(act_id and token)

# =============== Tela ===============
st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Finalização → Add Pagamento → Compra. Tudo alinhado ao período selecionado.")

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

# ===================== NOVO LAYOUT COM ABAS =====================
with st.spinner("Buscando dados da Meta…"):
    df_daily = fetch_insights_daily(
        act_id=act_id, token=token, api_version=api_version,
        since_str=str(since), until_str=str(until), level=level
    )
    df_hourly = fetch_insights_hourly(
        act_id=act_id, token=token, api_version=api_version,
        since_str=str(since), until_str=str(until), level=level
    )

if df_daily.empty and (df_hourly is None or df_hourly.empty):
    st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
    st.stop()

tab_daily, tab_daypart = st.tabs(["📅 Visão diária", "⏱️ Horários (principal)"])

# -------------------- ABA 1: VISÃO DIÁRIA (seu conteúdo atual) --------------------
with tab_daily:
    # === Moeda detectada e override opcional ===
    currency_detected = (df_daily["currency"].dropna().iloc[0]
                         if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty else "BRL")
    col_curA, col_curB = st.columns([1, 2])
    with col_curA:
        use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True)

    currency_label = "BRL" if use_brl_display else currency_detected

    with col_curB:
        if use_brl_display and currency_detected != "BRL":
            st.caption("⚠️ Exibindo com símbolo **R$** apenas para **formatação visual**. "
                       "Os valores permanecem na moeda da conta.")

    st.caption(f"Moeda da conta detectada: **{currency_detected}** — Exibindo como: **{currency_label}**")


    # ========= KPIs do período =========
    tot_spend = float(df_daily["spend"].sum())
    tot_purch = float(df_daily["purchases"].sum())
    tot_rev   = float(df_daily["revenue"].sum())
    roas_g    = (tot_rev / tot_spend) if tot_spend > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><div class="small-muted">Valor usado</div>'
                    f'<div class="big-number">{_fmt_money_br(tot_spend)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="small-muted">Vendas</div>'
                    f'<div class="big-number">{int(round(tot_purch)):,}</div></div>'.replace(",", "."),
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="small-muted">Valor de conversão</div>'
                    f'<div class="big-number">{_fmt_money_br(tot_rev)}</div></div>', unsafe_allow_html=True)
    with c4:
        roas_txt = (f"{roas_g:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    if pd.notnull(roas_g) else "—")
        st.markdown('<div class="kpi-card"><div class="small-muted">ROAS</div>'
                    f'<div class="big-number">{roas_txt}</div></div>',
                    unsafe_allow_html=True)
                    
    st.divider()

    # ========= Série diária =========
    st.subheader("Série diária — Investimento e Conversão")

    daily = df_daily.groupby("date", as_index=False)[["spend", "revenue", "purchases"]].sum()
    daily_pt = daily.rename(columns={"spend": "Gasto", "revenue": "Receita"})
    st.line_chart(daily_pt.set_index("date")[["Receita", "Gasto"]])
    st.caption("Linhas diárias de Receita e Gasto. Vendas na tabela abaixo.")

    # ========= FUNIL (Período) — FUNIL VISUAL =========
    st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")

    f_clicks = float(df_daily["link_clicks"].sum())
    f_lpv    = float(df_daily["lpv"].sum())
    f_ic     = float(df_daily["init_checkout"].sum())
    f_api    = float(df_daily["add_payment"].sum())
    f_pur    = float(df_daily["purchases"].sum())

    labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
    values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)), int(round(f_api)), int(round(f_pur))]

    force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
    values_plot = enforce_monotonic(values_total) if force_shape else values_total

    st.plotly_chart(
        funnel_fig(labels_total, values_plot, title="Funil do período"),
        use_container_width=True
    )

    core_rows = [
        ("LPV / Cliques",     _rate(values_total[1], values_total[0])),
        ("Checkout / LPV",    _rate(values_total[2], values_total[1])),
        ("Compra / Checkout", _rate(values_total[4], values_total[2])),
    ]
    extras_def = {
        "Add Pagto / Checkout": _rate(values_total[3], values_total[2]),
        "Compra / Add Pagto":   _rate(values_total[4], values_total[3]),
        "Compra / LPV":         _rate(values_total[4], values_total[1]),
        "Compra / Cliques":     _rate(values_total[4], values_total[0]),
        "Checkout / Cliques":   _rate(values_total[2], values_total[0]),
        "Add Pagto / LPV":      _rate(values_total[3], values_total[1]),
    }

    with st.expander("Comparar outras taxas (opcional)"):
        extras_selected = st.multiselect(
            "Escolha métricas adicionais para visualizar:",
            options=list(extras_def.keys()),
            default=[],
        )
    rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
    sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
    sr["Valor"] = sr["Valor"].map(lambda x: f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                                  if pd.notnull(x) else "")
    base_h, row_h = 160, 36
    height = base_h + row_h * len(extras_selected)
    st.dataframe(sr, use_container_width=True, height=height)

    # ========= COMPARATIVOS (Período A vs Período B) =========
    with st.expander("Comparativos — Período A vs Período B (opcional)", expanded=False):
        st.subheader("Comparativos — descubra o que mudou e por quê")

        period_len = (until - since).days + 1
        default_sinceA = since - timedelta(days=period_len)
        default_untilA = since - timedelta(days=1)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Período A**")
            sinceA = st.date_input("Desde (A)", value=default_sinceA, key="sinceA")
            untilA = st.date_input("Até (A)",   value=default_untilA, key="untilA")
        with colB:
            st.markdown("**Período B**")
            sinceB = st.date_input("Desde (B)", value=since, key="sinceB")
            untilB = st.date_input("Até (B)",   value=until, key="untilB")

        if sinceA > untilA or sinceB > untilB:
            st.warning("Confira as datas: 'Desde' não pode ser maior que 'Até'.")
        else:
            with st.spinner("Comparando períodos…"):
                dfA = fetch_insights_daily(act_id, token, api_version, str(sinceA), str(untilA), level)
                dfB = fetch_insights_daily(act_id, token, api_version, str(sinceB), str(untilB), level)

            if dfA.empty or dfB.empty:
                st.info("Sem dados em um dos períodos selecionados.")
            else:
                def _agg(d):
                    return {
                        "spend": d["spend"].sum(),
                        "revenue": d["revenue"].sum(),
                        "purchases": d["purchases"].sum(),
                        "clicks": d["link_clicks"].sum(),
                        "lpv": d["lpv"].sum(),
                        "checkout": d["init_checkout"].sum(),
                        "add_payment": d["add_payment"].sum(),
                    }

                A = _agg(dfA); B = _agg(dfB)

                roasA = _safe_div(A["revenue"], A["spend"])
                roasB = _safe_div(B["revenue"], B["spend"])
                cpaA  = _safe_div(A["spend"], A["purchases"])
                cpaB  = _safe_div(B["spend"], B["purchases"])
                cpcA  = _safe_div(A["spend"], A["clicks"])
                cpcB  = _safe_div(B["spend"], B["clicks"])

                dir_map = {
                    "Valor usado": "neutral",
                    "Faturamento": "higher",
                    "Vendas":      "higher",
                    "ROAS":        "higher",
                    "CPC":         "lower",
                    "CPA":         "lower",
                }
                delta_map = {
                    "Valor usado":  B["spend"] - A["spend"],
                    "Faturamento":  B["revenue"] - A["revenue"],
                    "Vendas":       B["purchases"] - A["purchases"],
                    "ROAS":         (roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else np.nan,
                    "CPC":          (cpcB - cpcA)   if pd.notnull(cpcA) and pd.notnull(cpcB) else np.nan,
                    "CPA":          (cpaB - cpaA)   if pd.notnull(cpaA) and pd.notnull(cpaB) else np.nan,
                }

                kpi_rows = [
                    ("Valor usado", _fmt_money_br(A["spend"]),   _fmt_money_br(B["spend"]),   _fmt_money_br(B["spend"] - A["spend"])),
                    ("Faturamento", _fmt_money_br(A["revenue"]), _fmt_money_br(B["revenue"]), _fmt_money_br(B["revenue"] - A["revenue"])),
                    ("Vendas",      _fmt_int_br(A["purchases"]), _fmt_int_br(B["purchases"]), _fmt_int_br(B["purchases"] - A["purchases"])),
                    ("ROAS",        _fmt_ratio_br(roasA),        _fmt_ratio_br(roasB),
                                     (_fmt_ratio_br(roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else "")),
                    ("CPC",         _fmt_money_br(cpcA) if pd.notnull(cpcA) else "",
                                     _fmt_money_br(cpcB) if pd.notnull(cpcB) else "",
                                     _fmt_money_br(cpcB - cpcA) if pd.notnull(cpcA) and pd.notnull(cpcB) else ""),
                    ("CPA",         _fmt_money_br(cpaA) if pd.notnull(cpaA) else "",
                                     _fmt_money_br(cpaB) if pd.notnull(cpaB) else "",
                                     _fmt_money_br(cpaB - cpaA) if pd.notnull(cpaA) and pd.notnull(cpaB) else ""),
                ]
                kpi_df_disp = pd.DataFrame(kpi_rows, columns=["Métrica", "Período A", "Período B", "Δ (B - A)"])

                def _style_kpi(row):
                    metric = row["Métrica"]
                    d      = delta_map.get(metric, np.nan)
                    rule   = dir_map.get(metric, "neutral")
                    styles = [""] * len(row)
                    try:
                        idxB = list(row.index).index("Período B")
                        idxD = list(row.index).index("Δ (B - A)")
                    except Exception:
                        return styles
                    if pd.isna(d) or rule == "neutral" or d == 0:
                        return styles
                    better = (d > 0) if rule == "higher" else (d < 0)
                    color  = "#16a34a" if better else "#dc2626"
                    weight = "700"
                    styles[idxB] = f"color:{color}; font-weight:{weight};"
                    styles[idxD] = f"color:{color}; font-weight:{weight};"
                    return styles

                st.markdown("**KPIs do período (A vs B)**")
                st.dataframe(kpi_df_disp.style.apply(_style_kpi, axis=1), use_container_width=True, height=260)

                st.markdown("---")

                # Taxas do funil
                rates_num = pd.DataFrame({
                    "Taxa": ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout"],
                    "Período A": [
                        _safe_div(A["lpv"], A["clicks"]),
                        _safe_div(A["checkout"], A["lpv"]),
                        _safe_div(A["purchases"], A["checkout"]),
                    ],
                    "Período B": [
                        _safe_div(B["lpv"], B["clicks"]),
                        _safe_div(B["checkout"], B["lpv"]),
                        _safe_div(B["purchases"], B["checkout"]),
                    ],
                })
                rates_num["Δ"] = rates_num["Período B"] - rates_num["Período A"]

                rates_disp = rates_num.copy()
                for col in ["Período A", "Período B", "Δ"]:
                    rates_disp[col] = rates_disp[col].map(_fmt_pct_br)

                delta_by_taxa = dict(zip(rates_num["Taxa"], rates_num["Δ"]))

                def _style_rate(row):
                    taxa = row["Taxa"]
                    d    = delta_by_taxa.get(taxa, np.nan)
                    styles = [""] * len(row)
                    try:
                        idxB = list(row.index).index("Período B")
                        idxD = list(row.index).index("Δ")
                    except Exception:
                        return styles
                    if pd.isna(d) or d == 0:
                        return styles
                    better = d > 0
                    color  = "#16a34a" if better else "#dc2626"
                    weight = "700"
                    styles[idxB] = f"color:{color}; font-weight:{weight};"
                    styles[idxD] = f"color:{color}; font-weight:{weight};"
                    return styles

                st.markdown("**Taxas do funil (A vs B)**")
                st.dataframe(rates_disp.style.apply(_style_rate, axis=1), use_container_width=True, height=180)

                # Funis lado a lado
                labels_funnel = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
                valsA = [int(round(A["clicks"])), int(round(A["lpv"])), int(round(A["checkout"])),
                         int(round(A["add_payment"])), int(round(A["purchases"]))]
                valsB = [int(round(B["clicks"])), int(round(B["lpv"])), int(round(B["checkout"])),
                         int(round(B["add_payment"])), int(round(B["purchases"]))]
                valsA_plot = enforce_monotonic(valsA)
                valsB_plot = enforce_monotonic(valsB)

                cA, cB = st.columns(2)
                with cA:
                    st.plotly_chart(
                        funnel_fig(labels_funnel, valsA_plot, title=f"Funil — Período A ({sinceA} a {untilA})"),
                        use_container_width=True
                    )
                with cB:
                    st.plotly_chart(
                        funnel_fig(labels_funnel, valsB_plot, title=f"Funil — Período B ({sinceB} a {untilB})"),
                        use_container_width=True
                    )

                # Δ por etapa
                delta_counts = [b - a for a, b in zip(valsA, valsB)]
                delta_df = pd.DataFrame({
                    "Etapa": labels_funnel,
                    "Período A": valsA,
                    "Período B": valsB,
                    "Δ (B - A)": delta_counts,
                })
                delta_disp = delta_df.copy()
                delta_disp["Período A"]  = delta_disp["Período A"].map(_fmt_int_br)
                delta_disp["Período B"]  = delta_disp["Período B"].map(_fmt_int_br)
                delta_disp["Δ (B - A)"]  = delta_disp["Δ (B - A)"].map(_fmt_int_signed_br)

                delta_by_stage = dict(zip(delta_df["Etapa"], delta_df["Δ (B - A)"]))

                def _style_delta_counts(row):
                    d = delta_by_stage.get(row["Etapa"], np.nan)
                    styles = [""] * len(row)
                    try:
                        idxB = list(row.index).index("Período B")
                        idxD = list(row.index).index("Δ (B - A)")
                    except Exception:
                        return styles
                    if pd.isna(d) or d == 0:
                        return styles
                    color  = "#16a34a" if d > 0 else "#dc2626"
                    weight = "700"
                    styles[idxB] = f"color:{color}; font-weight:{weight};"
                    styles[idxD] = f"color:{color}; font-weight:{weight};"
                    return styles

                st.markdown("**Pessoas a mais/menos em cada etapa (B − A)**")
                st.dataframe(delta_disp.style.apply(_style_delta_counts, axis=1), use_container_width=True, height=240)

                st.markdown("---")

    # ========= FUNIL por CAMPANHA =========
    if level == "campaign":
        st.subheader("Campanhas — Funil e Taxas (somatório no período)")

        agg_cols = ["spend", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
        camp = df_daily.groupby(["campaign_id", "campaign_name"], as_index=False)[agg_cols].sum()

        # Taxas principais
        camp["LPV/Cliques"]     = np.where(camp["link_clicks"] > 0, camp["lpv"] / camp["link_clicks"], np.nan)
        camp["Checkout/LPV"]    = np.where(camp["lpv"] > 0, camp["init_checkout"] / camp["lpv"], np.nan)
        camp["Compra/Checkout"] = np.where(camp["init_checkout"] > 0, camp["purchases"] / camp["init_checkout"], np.nan)
        camp["ROAS"]            = np.where(camp["spend"] > 0, camp["revenue"] / camp["spend"], np.nan)

        extras_cols = {
            "Add Pagto / Checkout":  np.where(camp["init_checkout"] > 0, camp["add_payment"] / camp["init_checkout"], np.nan),
            "Compra / Add Pagto":    np.where(camp["add_payment"] > 0, camp["purchases"] / camp["add_payment"], np.nan),
            "Compra / LPV":          np.where(camp["lpv"] > 0, camp["purchases"] / camp["lpv"], np.nan),
            "Compra / Cliques":      np.where(camp["link_clicks"] > 0, camp["purchases"] / camp["link_clicks"], np.nan),
            "Checkout / Cliques":    np.where(camp["link_clicks"] > 0, camp["init_checkout"] / camp["link_clicks"], np.nan),
            "Add Pagto / LPV":       np.where(camp["lpv"] > 0, camp["add_payment"] / camp["lpv"], np.nan),
        }

        with st.expander("Comparar outras taxas (opcional)"):
            extras_selected = st.multiselect(
                "Escolha métricas adicionais:",
                options=list(extras_cols.keys()),
                default=[],
            )

        cols_base = [
            "campaign_id", "campaign_name",
            "spend", "revenue", "ROAS",
            "link_clicks", "lpv", "init_checkout", "purchases",
            "LPV/Cliques", "Checkout/LPV", "Compra/Checkout"
        ]
        camp_view = camp[cols_base].copy()

        for name in extras_selected:
            camp_view[name] = extras_cols[name]

        # Ordena ANTES de formatar moeda
        camp_view = camp_view.sort_values("spend", ascending=False)

        # Formatação
        for c in ["spend", "revenue"]:
            camp_view[c] = camp_view[c].apply(_fmt_money_br)

        pct_cols = ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout"] + list(extras_selected)
        for c in pct_cols:
            camp_view[c] = camp_view[c].map(lambda x: (f"{x*100:,.2f}%" if pd.notnull(x) else "")
                                            .replace(",", "X").replace(".", ",").replace("X", "."))

        camp_view["ROAS"] = camp_view["ROAS"].map(
            lambda x: (f"{x:,.2f}x" if pd.notnull(x) else "").replace(",", "X").replace(".", ",").replace("X", ".")
        )

        camp_view = camp_view.rename(columns={
            "campaign_id": "ID campanha",
            "campaign_name": "Campanha",
            "spend": "Valor usado",
            "revenue": "Valor de conversão",
            "link_clicks": "Cliques",
            "lpv": "LPV",
            "init_checkout": "Checkout",
            "purchases": "Compras",
        })

        st.dataframe(camp_view, use_container_width=True, height=520)

        with st.expander("Dados diários (detalhe por campanha)"):
            dd = df_daily.copy()
            dd["date"] = dd["date"].dt.date
            cols = ["date", "campaign_name", "spend", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases", "revenue", "roas"]
            dd["roas"] = np.where(dd["spend"]>0, dd["revenue"]/dd["spend"], np.nan)
            dd_fmt = dd[cols].copy()
            dd_fmt["spend"]   = dd_fmt["spend"].apply(_fmt_money_br)
            dd_fmt["revenue"] = dd_fmt["revenue"].apply(_fmt_money_br)
            dd_fmt["roas"]    = dd_fmt["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
            st.dataframe(dd_fmt.rename(columns={
                "date": "Data", "campaign_name": "Campanha", "spend": "Valor usado",
                "link_clicks": "Cliques", "lpv": "LPV", "init_checkout": "Finalização",
                "add_payment": "Add Pagamento", "purchases": "Compras",
                "revenue": "Valor de conversão", "roas": "ROAS"
            }), use_container_width=True)
    else:
        st.info("Troque o nível para 'campaign' para ver o detalhamento por campanha.")

# -------------------- ABA DE HORÁRIOS (Heatmap no topo) --------------------
with tab_daypart:
    st.caption("Explore desempenho por hora: Heatmap no topo, depois comparação de dias e apanhado geral.")
    if df_hourly is None or df_hourly.empty:
        st.info("A conta/período não retornou breakdown por hora. Use a visão diária.")
    else:
        # ---------------- Filtros + base ----------------
        min_spend = st.slider("Gasto mínimo para considerar o horário (R$)", 0.0, 1000.0, 0.0, 10.0)
        d = df_hourly.copy()
        d = d.dropna(subset=["hour"])
        d["hour"] = d["hour"].astype(int).clip(0, 23)
        d["date_only"] = d["date"].dt.date

        # ============== 1) HEATMAP HORA × DIA (TOPO) ==============
        st.subheader("📆 Heatmap — Hora × Dia")
        cube_hm = d.groupby(["dow_label","hour"], as_index=False)[
            ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
        ].sum()
        cube_hm["roas"] = np.where(cube_hm["spend"]>0, cube_hm["revenue"]/cube_hm["spend"], np.nan)

        if min_spend > 0:
            cube_hm = cube_hm[cube_hm["spend"] >= min_spend]

        metric_hm = st.selectbox("Métrica para o heatmap", ["Compras","Receita","Gasto","ROAS"], index=0, key="hm_metric_top")
        mcol_hm = {"Compras":"purchases","Receita":"revenue","Gasto":"spend","ROAS":"roas"}[metric_hm]

        if mcol_hm == "roas":
            pvt = cube_hm.groupby(["dow_label","hour"], as_index=False)[mcol_hm].mean()
        else:
            pvt = cube_hm.groupby(["dow_label","hour"], as_index=False)[mcol_hm].sum()

        order = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        pvt["dow_label"] = pd.Categorical(pvt["dow_label"], categories=order, ordered=True)
        pvt = pvt.sort_values(["dow_label","hour"])
        heat = pvt.pivot(index="dow_label", columns="hour", values=mcol_hm).fillna(0)

        hours_full = list(range(24))
        heat = heat.reindex(columns=hours_full, fill_value=0)
        heat.columns = list(range(24))

        fig_hm = go.Figure(data=go.Heatmap(
            z=heat.values, x=heat.columns, y=heat.index,
            colorbar=dict(title=metric_hm),
            hovertemplate="Dia: %{y}<br>Hora: %{x}h<br>"+metric_hm+": %{z}<extra></extra>"
        ))
        fig_hm.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10), template="plotly_white")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")

        # ============== 3) APANHADO GERAL POR HORA (no período) ==============
        st.subheader("📦 Apanhado geral por hora (período selecionado)")
        cube_hr = d.groupby("hour", as_index=False)[
            ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
        ].sum()
        cube_hr["ROAS"] = np.where(cube_hr["spend"]>0, cube_hr["revenue"]/cube_hr["spend"], np.nan)
        if min_spend > 0:
            cube_hr = cube_hr[cube_hr["spend"] >= min_spend]

        top_hr = cube_hr.sort_values(["purchases","ROAS"], ascending=[False,False]).copy()
        show_cols = ["hour","purchases","ROAS","spend","revenue","link_clicks","lpv","init_checkout","add_payment"]
        disp_top = top_hr[show_cols].rename(columns={
            "hour":"Hora","purchases":"Compras","spend":"Valor usado","revenue":"Valor de conversão"
        })
        disp_top["Valor usado"] = disp_top["Valor usado"].apply(_fmt_money_br)
        disp_top["Valor de conversão"] = disp_top["Valor de conversão"].apply(_fmt_money_br)
        disp_top["ROAS"] = disp_top["ROAS"].map(_fmt_ratio_br)
        st.dataframe(disp_top, use_container_width=True, height=360)

        fig_bar = go.Figure(go.Bar(x=cube_hr.sort_values("hour")["hour"], y=cube_hr.sort_values("hour")["purchases"]))
        fig_bar.update_layout(
            title="Compras por hora (total do período)",
            xaxis_title="Hora do dia", yaxis_title="Compras",
            height=380, template="plotly_white", margin=dict(l=10,r=10,t=48,b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info("Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos.")

        st.subheader("🆚 Comparar dois períodos (A vs B) — hora a hora")

        # Defaults: B = período atual (since/until), A = período anterior com mesma duração
        base_len = (until - since).days + 1
        default_sinceA = (since - timedelta(days=base_len))
        default_untilA = (since - timedelta(days=1))

        colA1, colA2, colB1, colB2 = st.columns(4)
        with colA1:
            period_sinceA = st.date_input("Desde (A)", value=default_sinceA, key="cmp_sinceA")
        with colA2:
            period_untilA = st.date_input("Até (A)",   value=default_untilA, key="cmp_untilA")
        with colB1:
            period_sinceB = st.date_input("Desde (B)", value=since, key="cmp_sinceB")
        with colB2:
            period_untilB = st.date_input("Até (B)",   value=until, key="cmp_untilB")

        # Validação rápida
        if period_sinceA > period_untilA or period_sinceB > period_untilB:
            st.warning("Confira as datas: em cada período, 'Desde' não pode ser maior que 'Até'.")
        else:
            # Buscar dados por hora cobrindo A ∪ B
            union_since = min(period_sinceA, period_sinceB)
            union_until = max(period_untilA, period_untilB)

            with st.spinner("Carregando dados por hora dos períodos selecionados…"):
                df_hourly_union = fetch_insights_hourly(
                    act_id=act_id, token=token, api_version=api_version,
                    since_str=str(union_since), until_str=str(union_until), level=level
                )

            if df_hourly_union is None or df_hourly_union.empty:
                st.info("Sem dados no intervalo combinado dos períodos selecionados.")
            else:
                # Base preparada
                d_cmp = df_hourly_union.dropna(subset=["hour"]).copy()
                d_cmp["hour"] = d_cmp["hour"].astype(int).clip(0, 23)
                d_cmp["date_only"] = d_cmp["date"].dt.date

                # Filtra pelos períodos A e B
                A_mask = (d_cmp["date_only"] >= period_sinceA) & (d_cmp["date_only"] <= period_untilA)
                B_mask = (d_cmp["date_only"] >= period_sinceB) & (d_cmp["date_only"] <= period_untilB)
                datA, datB = d_cmp[A_mask], d_cmp[B_mask]

                if datA.empty or datB.empty:
                    st.info("Sem dados em um dos períodos selecionados.")
                else:
                    agg_cols = ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]

                    # Soma por hora
                    gA = datA.groupby("hour", as_index=False)[agg_cols].sum()
                    gB = datB.groupby("hour", as_index=False)[agg_cols].sum()

                    # Merge A vs B
                    merged = pd.merge(gA, gB, on="hour", how="outer", suffixes=(" (A)", " (B)")).fillna(0.0)

                    # Filtro de gasto mínimo (descarta só se AMBOS forem baixos)
                    if min_spend > 0:
                        keep = (merged["spend (A)"] >= min_spend) | (merged["spend (B)"] >= min_spend)
                        merged = merged[keep]

                    if merged.empty:
                        st.info("Após o filtro de gasto mínimo, não sobraram horas para comparar.")
                    else:
                        # ---------- GRÁFICO COMBINADO ----------
                        x = merged["hour"].astype(int)
                        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

                        # Barras empilhadas — Período A
                        fig_combo.add_trace(
                            go.Bar(name="Gasto (A)", x=x, y=merged["spend (A)"],
                                   offsetgroup="A", legendgroup="A"),
                            secondary_y=False
                        )
                        fig_combo.add_trace(
                            go.Bar(name="Receita (A)", x=x, y=merged["revenue (A)"],
                                   offsetgroup="A", legendgroup="A"),
                            secondary_y=False
                        )

                        # Barras empilhadas — Período B (lado a lado de A)
                        fig_combo.add_trace(
                            go.Bar(name="Gasto (B)", x=x, y=merged["spend (B)"],
                                   offsetgroup="B", legendgroup="B"),
                            secondary_y=False
                        )
                        fig_combo.add_trace(
                            go.Bar(name="Receita (B)", x=x, y=merged["revenue (B)"],
                                   offsetgroup="B", legendgroup="B"),
                            secondary_y=False
                        )

                        # Linhas — Compras (eixo secundário)
                        fig_combo.add_trace(
                            go.Scatter(name=f"Compras (A) — {period_sinceA} a {period_untilA}",
                                       x=x, y=merged["purchases (A)"], mode="lines+markers", legendgroup="A"),
                            secondary_y=True
                        )
                        fig_combo.add_trace(
                            go.Scatter(name=f"Compras (B) — {period_sinceB} a {period_untilB}",
                                       x=x, y=merged["purchases (B)"], mode="lines+markers", legendgroup="B"),
                            secondary_y=True
                        )

                        fig_combo.update_layout(
                            title="Comparativo por hora — Barras empilhadas (Gasto+Receita) + linha de Compras",
                            barmode="relative",              # empilha Gasto e Receita dentro de cada período
                            bargap=0.15, bargroupgap=0.12,
                            template="plotly_white",
                            height=520, margin=dict(l=10, r=10, t=48, b=10),
                        )
                        fig_combo.update_xaxes(title_text="Hora do dia")
                        fig_combo.update_yaxes(title_text="Valores (R$)", secondary_y=False)
                        fig_combo.update_yaxes(title_text="Compras (unid.)", secondary_y=True)

                        st.plotly_chart(fig_combo, use_container_width=True)
