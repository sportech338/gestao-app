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
st.set_page_config(
    page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide"
)
st.markdown(
    """
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Janelas de atribuição (paridade com Ads Manager)
ATTR_KEYS = ["7d_click", "1d_view"]
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]

# --- Constantes e parser para breakdown por hora
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"


def _parse_hour_bucket(h):
    """Normaliza bucket de hora (\'0\'..\'23\' ou \'00:00\'..\'23:00\') para int [0..23]."""
    if h is None:
        return None
    try:
        s = str(h).strip()
        val = int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except Exception:
        return None


# =============== Helpers genéricos ===============
def _retry_call(fn, max_retries=5, base_wait=1.2):
    """Executa uma função com backoff exponencial para erros/transientes comuns."""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(
                k in msg
                for k in [
                    "rate limit",
                    "retry",
                    "temporarily unavailable",
                    "timeout",
                    "timed out",
                ]
            ):
                time.sleep(base_wait * (2**i))
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
    """Usa \'value\' quando existir; senão soma SOMENTE as chaves permitidas (ex.: 7d_click, 1d_view)."""
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
            texttemplate="<b>%{value}</b>",
            textfont=dict(size=35),
            opacity=0.95,
            connector={"line": {"dash": "dot", "width": 1}},
        )
    )
    fig.update_layout(
        title=title or "",
        margin=dict(l=10, r=10, t=48, b=10),
        height=540,
        template="plotly_white",
        separators=",.",  # pt-BR
        uniformtext=dict(minsize=12, mode="show"),
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
        if pd.notnull(x)
        else ""
    )


def _fmt_ratio_br(x):  # ROAS \"1,23x\"
    return (
        f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x)
        else ""
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
def fetch_insights_daily(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
    try_extra_fields: bool = True,
) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único (\'campaign\' recomendado)
    - Usa action_report_time=conversion e action_attribution_windows fixos (paridade com Ads Manager)
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
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
        "action_attribution_windows": ",".join(ATTR_KEYS),
    }

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(
            lambda: requests.get(next_url, params=next_params, timeout=90)
        )
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("Resposta inválida da Graph API.")

        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code, sub, msg = (
                err.get("code"),
                err.get("error_subcode"),
                err.get("message"),
            )
            if code == 100 and try_extra_fields:
                # refaz sem extras
                return fetch_insights_daily(
                    act_id,
                    token,
                    api_version,
                    since_str,
                    until_str,
                    level,
                    try_extra_fields=False,
                )
            raise RuntimeError(
                f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}"
            )

        for rec in payload.get("data", []):
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            # Cliques em link (preferir field; fallback action com janela)
            link_clicks = rec.get("link_clicks", None)
            if link_clicks is None:
                link_clicks = _sum_actions_exact(
                    actions, ["link_click"], allowed_keys=ATTR_KEYS
                )

            # LPV (preferir field; fallback landing_page_view → view_content → contains \"landing_page\")
            lpv = rec.get("landing_page_views", None)
            if lpv is None:
                lpv = _sum_actions_exact(
                    actions, ["landing_page_view"], allowed_keys=ATTR_KEYS
                )
                if lpv == 0:
                    lpv = _sum_actions_exact(
                        actions, ["view_content"], allowed_keys=ATTR_KEYS
                    ) or _sum_actions_contains(
                        actions, ["landing_page"], allowed_keys=ATTR_KEYS
                    )

            # Iniciar checkout / add payment info com janela definida
            ic = _sum_actions_exact(
                actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS
            )
            api = _sum_actions_exact(
                actions, ["add_payment_info"], allowed_keys=ATTR_KEYS
            )

            # Purchase (qtd) e Revenue (valor) respeitando janela
            purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
            revenue_val = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

            rows.append(
                {
                    "date": pd.to_datetime(rec.get("date_start")),
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    # métricas básicas
                    "spend": _to_float(rec.get("spend")),
                    "impressions": _to_float(rec.get("impressions")),
                    "clicks": _to_float(rec.get("clicks")),
                    # funil
                    "link_clicks": _to_float(link_clicks),
                    "lpv": _to_float(lpv),
                    "init_checkout": _to_float(ic),
                    "add_payment": _to_float(api),
                    "purchases": _to_float(purchases_cnt),
                    "revenue": _to_float(revenue_val),
                }
            )

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
    num_cols = [
        "spend",
        "impressions",
        "clicks",
        "link_clicks",
        "lpv",
        "init_checkout",
        "add_payment",
        "purchases",
        "revenue",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Métricas derivadas (do período/dia; taxas serão calculadas em agregações)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df = df.sort_values("date")
    return df


# --- Coleta por hora (dayparting)
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
) -> pd.DataFrame:
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
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

    def _run_hourly(action_rt: str) -> pd.DataFrame:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": since_str, "until": until_str}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": action_rt,
            "breakdowns": HOUR_BREAKDOWN,
        }

        if action_rt == "conversion":
            params["action_attribution_windows"] = ",".join(ATTR_KEYS)

        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            resp = _retry_call(
                lambda: requests.get(next_url, params=next_params, timeout=90)
            )
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API (hourly).")

            if resp.status_code != 200:
                # Propaga o erro para o chamador decidir fallback
                err = (payload or {}).get("error", {})
                code, sub, msg = (
                    err.get("code"),
                    err.get("error_subcode"),
                    err.get("message"),
                )
                raise RuntimeError(
                    f"Graph API error hourly | code={code} subcode={sub} | {msg}"
                )

            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []
                hour_bucket = _parse_hour_bucket(rec.get(HOUR_BREAKDOWN))

                link_clicks = rec.get("link_clicks")
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(
                        actions, ["link_click"], allowed_keys=ATTR_KEYS
                    )

                lpv = rec.get("landing_page_views")
                if lpv is None:
                    lpv = (
                        _sum_actions_exact(
                            actions, ["landing_page_view"], allowed_keys=ATTR_KEYS
                        )
                        or _sum_actions_exact(
                            actions, ["view_content"], allowed_keys=ATTR_KEYS
                        )
                        or _sum_actions_contains(
                            actions, ["landing_page"], allowed_keys=ATTR_KEYS
                        )
                    )

                ic = _sum_actions_exact(
                    actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS
                )
                api_ = _sum_actions_exact(
                    actions, ["add_payment_info"], allowed_keys=ATTR_KEYS
                )
                pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows.append(
                    {
                        "date": pd.to_datetime(rec.get("date_start")),
                        "hour": hour_bucket,
                        "currency": rec.get("account_currency", "BRL"),
                        "campaign_id": rec.get("campaign_id", ""),
                        "campaign_name": rec.get("campaign_name", ""),
                        "spend": _to_float(rec.get("spend")),
                        "impressions": _to_float(rec.get("impressions")),
                        "clicks": _to_float(rec.get("clicks")),
                        "link_clicks": _to_float(link_clicks),
                        "lpv": _to_float(lpv),
                        "init_checkout": _to_float(ic),
                        "add_payment": _to_float(api_),
                        "purchases": _to_float(pur),
                        "revenue": _to_float(rev),
                    }
                )

            # paginação: preferir paging.next; senão usar cursor.after
            paging = payload.get("paging") or {}
            next_link = paging.get("next")
            if next_link:
                next_url, next_params = next_link, None
            else:
                after = (paging.get("cursors") or {}).get("after")
                if after:
                    next_url, next_params = base_url, params.copy()
                    next_params["after"] = after
                else:
                    break

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["dow_label"] = pd.Categorical(
            df["date"].dt.day_name(locale="pt_BR"),
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )
        df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
        return df

    # Tenta com action_report_time=conversion primeiro
    try:
        return _run_hourly("conversion")
    except RuntimeError as e:
        # Se falhar com erro 100 (permissão/campo), tenta com impression
        if "code=100" in str(e):
            st.warning(
                "Não foi possível obter dados por hora com janela de conversão. Tentando com janela de impressão."
            )
            return _run_hourly("impression")
        raise  # Propaga outros erros


@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_breakdown(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    breakdowns: list[str],
    level: str = "campaign",
) -> pd.DataFrame:
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    fields = [
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

    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({"since": since_str, "until": until_str}),
        "time_increment": "all_days",  # para breakdown, queremos o total do período
        "fields": ",".join(fields),
        "limit": 500,
        "action_report_time": "conversion",
        "action_attribution_windows": ",".join(ATTR_KEYS),
        "breakdowns": ",".join(breakdowns),
    }

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(
            lambda: requests.get(next_url, params=next_params, timeout=90)
        )
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("Resposta inválida da Graph API (breakdown).")

        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code, sub, msg = (
                err.get("code"),
                err.get("error_subcode"),
                err.get("message"),
            )
            raise RuntimeError(
                f"Graph API error breakdown | code={code} subcode={sub} | {msg}"
            )

        for rec in payload.get("data", []):
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            link_clicks = rec.get("link_clicks")
            if link_clicks is None:
                link_clicks = _sum_actions_exact(
                    actions, ["link_click"], allowed_keys=ATTR_KEYS
                )

            lpv = rec.get("landing_page_views")
            if lpv is None:
                lpv = (
                    _sum_actions_exact(
                        actions, ["landing_page_view"], allowed_keys=ATTR_KEYS
                    )
                    or _sum_actions_exact(
                        actions, ["view_content"], allowed_keys=ATTR_KEYS
                    )
                    or _sum_actions_contains(
                        actions, ["landing_page"], allowed_keys=ATTR_KEYS
                    )
                )

            ic = _sum_actions_exact(
                actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS
            )
            api_ = _sum_actions_exact(
                actions, ["add_payment_info"], allowed_keys=ATTR_KEYS
            )
            pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
            rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

            base = {
                "currency": rec.get("account_currency", "BRL"),
                "campaign_id": rec.get("campaign_id", ""),
                "campaign_name": rec.get("campaign_name", ""),
                "spend": _to_float(rec.get("spend")),
                "impressions": _to_float(rec.get("impressions")),
                "clicks": _to_float(rec.get("clicks")),
                "link_clicks": _to_float(link_clicks),
                "lpv": _to_float(lpv),
                "init_checkout": _to_float(ic),
                "add_payment": _to_float(api_),
                "purchases": _to_float(pur),
                "revenue": _to_float(rev),
            }
            for b in breakdowns:
                base[b] = rec.get(b)
            rows.append(base)

        paging = (payload or {}).get("paging", {})
        if paging.get("next"):
            next_url, next_params = paging.get("next"), None
        else:
            after = (paging.get("cursors") or {}).get("after")
            if after:
                next_url, next_params = base_url, params.copy()
                next_params["after"] = after
            else:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df


# =============== Sidebar (filtros) ===============
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox(
    "Nível (recomendado: campaign)", ["campaign", "account"], index=0
)

preset = st.sidebar.radio(
    "Período rápido",
    [
        "Hoje",
        "Ontem",
        "Últimos 7 dias",
        "Últimos 14 dias",
        "Últimos 30 dias",
        "Últimos 90 dias",
        "Esta semana",
        "Este mês",
        "Máximo",
        "Personalizado",
    ],
    index=2,
)


def _range_from_preset(p):
    local_today = datetime.now(APP_TZ).date()
    base_end = local_today - timedelta(days=1)
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
    if p == "Últimos 90 dias":
        return base_end - timedelta(days=89), base_end
    if p == "Esta semana":
        # semana corrente (seg a hoje)
        start_week = local_today - timedelta(days=local_today.weekday())
        return start_week, local_today
    if p == "Este mês":
        start_month = local_today.replace(day=1)
        return start_month, local_today
    if p == "Máximo":
        # arbitrário: desde 2017-01-01 até ontem
        return date(2017, 1, 1), base_end

    # fallback
    return base_end - timedelta(days=6), base_end


_since_auto, _until_auto = _range_from_preset(preset)

if preset == "Personalizado":
    since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
    until = st.sidebar.date_input("Até", value=_until_auto, key="until_custom")
else:
    # Mostra as datas calculadas, bloqueadas (só leitura)
    since = st.sidebar.date_input(
        "Desde", value=_since_auto, disabled=True, key="since_auto"
    )
    until = st.sidebar.date_input(
        "Até", value=_until_auto, disabled=True, key="until_auto"
    )

ready = bool(act_id and token)

# =============== Tela ===============
st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption(
    "KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra. Tudo alinhado ao período selecionado."
)

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

# ===================== Coleta =====================
with st.spinner("Buscando dados da Meta…"):
    df_daily = fetch_insights_daily(
        act_id=act_id,
        token=token,
        api_version=api_version,
        since_str=str(since),
        until_str=str(until),
        level=level,
    )
    df_hourly = fetch_insights_hourly(
        act_id=act_id,
        token=token,
        api_version=api_version,
        since_str=str(since),
        until_str=str(until),
        level=level,
    )

if df_daily.empty and (df_hourly is None or df_hourly.empty):
    st.warning(
        "Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency)."
    )
    st.stop()

tab_daily, tab_daypart, tab_detail = st.tabs(
    ["📅 Visão diária", "⏱️ Horários (principal)", "📊 Detalhamento"]
)

# -------------------- ABA 1: VISÃO DIÁRIA --------------------
with tab_daily:
    # === Moeda detectada e override opcional ===
    currency_detected = (
        df_daily["currency"].dropna().iloc[0]
        if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty
        else "BRL"
    )
    col_curA, col_curB = st.columns([1, 2])
    with col_curA:
        use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True)

    currency_label = "BRL" if use_brl_display else currency_detected

    with col_curB:
        if use_brl_display and currency_detected != "BRL":
            st.caption(
                "⚠️ Exibindo com símbolo **R$** apenas para **formatação visual**. "
                "Os valores permanecem na moeda da conta."
            )

    st.caption(
        f"Moeda da conta detectada: **{currency_detected}** — Exibindo como: **{currency_label}**"
    )

    # ========= KPIs do período =========
    tot_spend = float(df_daily["spend"].sum())
    tot_purch = float(df_daily["purchases"].sum())
    tot_rev = float(df_daily["revenue"].sum())
    roas_g = (tot_rev / tot_spend) if tot_spend > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Valor usado</div>'
            f'<div class="big-number">{_fmt_money_br(tot_spend)}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Vendas</div>'
            f'<div class="big-number">{int(round(tot_purch)):,}</div></div>'.replace(
                ",", "."
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Valor de conversão</div>'
            f'<div class="big-number">{_fmt_money_br(tot_rev)}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        roas_txt = _fmt_ratio_br(roas_g) if pd.notnull(roas_g) else "—"
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">ROAS</div>'
            f'<div class="big-number">{roas_txt}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ========= Série diária =========
    st.subheader("Série diária — Investimento e Conversão")
    daily = df_daily.groupby("date", as_index=False)[
        ["spend", "revenue", "purchases"]
    ].sum()
    daily_pt = daily.rename(columns={"spend": "Gasto", "revenue": "Faturamento"})
    st.line_chart(daily_pt.set_index("date")[["Faturamento", "Gasto"]])
    st.caption("Linhas diárias de Receita e Gasto. Vendas na tabela abaixo.")

    # ========= FUNIL (Período) — FUNIL VISUAL =========
    st.subheader(
        "Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra"
    )

    f_clicks = float(df_daily["link_clicks"].sum())
    f_lpv = float(df_daily["lpv"].sum())
    f_ic = float(df_daily["init_checkout"].sum())
    f_api = float(df_daily["add_payment"].sum())
    f_pur = float(df_daily["purchases"].sum())

    labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
    values_total = [
        int(round(f_clicks)),
        int(round(f_lpv)),
        int(round(f_ic)),
        int(round(f_api)),
        int(round(f_pur)),
    ]

    force_shape = st.checkbox(
        "Forçar formato de funil (sempre decrescente)", value=True
    )
    values_plot = enforce_monotonic(values_total) if force_shape else values_total

    st.plotly_chart(
        funnel_fig(labels_total, values_plot, title="Funil do período"),
        use_container_width=True,
    )

    core_rows = [
        ("LPV / Cliques", _rate(values_total[1], values_total[0])),
        ("Checkout / LPV", _rate(values_total[2], values_total[1])),
        ("Compra / Checkout", _rate(values_total[4], values_total[2])),
    ]
    extras_def = {
        "Add Pagto / Checkout": _rate(values_total[3], values_total[2]),
        "Compra / Add Pagto": _rate(values_total[4], values_total[3]),
        "Compra / LPV": _rate(values_total[4], values_total[1]),
        "Compra / Cliques": _rate(values_total[4], values_total[0]),
        "Checkout / Cliques": _rate(values_total[2], values_total[0]),
        "Add Pagto / LPV": _rate(values_total[3], values_total[1]),
    }

    with st.expander("Comparar outras taxas (opcional)"):
        extras_selected = st.multiselect(
            "Escolha métricas adicionais para visualizar:",
            options=list(extras_def.keys()),
            default=[],
        )
    rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
    sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
    sr["Valor"] = sr["Valor"].map(
        lambda x: (
            f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x)
            else ""
        )
    )
    base_h, row_h = 160, 36
    height = base_h + row_h * len(extras_selected)
    st.dataframe(sr, use_container_width=True, height=height)

    # ========= COMPARATIVOS (Período A vs Período B) =========
    with st.expander(
        "Comparativos — Período A vs Período B (opcional)", expanded=False
    ):
        st.subheader("Comparativos — descubra o que mudou e por quê")

        period_len = (until - since).days + 1
        default_sinceA = since - timedelta(days=period_len)
        default_untilA = since - timedelta(days=1)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Período A**")
            sinceA = st.date_input("Desde (A)", value=default_sinceA, key="sinceA")
            untilA = st.date_input("Até (A)", value=default_untilA, key="untilA")
        with colB:
            st.markdown("**Período B**")
            sinceB = st.date_input("Desde (B)", value=since, key="sinceB")
            untilB = st.date_input("Até (B)", value=until, key="untilB")

        if sinceA > untilA or sinceB > untilB:
            st.warning("Confira as datas: 'Desde' não pode ser maior que 'Até'.")
        else:
            with st.spinner("Comparando períodos…"):
                dfA = fetch_insights_daily(
                    act_id, token, api_version, str(sinceA), str(untilA), level
                )
                dfB = fetch_insights_daily(
                    act_id, token, api_version, str(sinceB), str(untilB), level
                )

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

                A = _agg(dfA)
                B = _agg(dfB)

                roasA = _safe_div(A["revenue"], A["spend"])
                roasB = _safe_div(B["revenue"], B["spend"])
                cpaA = _safe_div(A["spend"], A["purchases"])
                cpaB = _safe_div(B["spend"], B["purchases"])
                cpcA = _safe_div(A["spend"], A["clicks"])
                cpcB = _safe_div(B["spend"], B["clicks"])

                dir_map = {
                    "Valor usado": "neutral",
                    "Faturamento": "higher",
                    "Vendas": "higher",
                    "ROAS": "higher",
                    "CPC": "lower",
                    "CPA": "lower",
                }
                delta_map = {
                    "Valor usado": B["spend"] - A["spend"],
                    "Faturamento": B["revenue"] - A["revenue"],
                    "Vendas": B["purchases"] - A["purchases"],
                    "ROAS": (
                        (roasB - roasA)
                        if pd.notnull(roasA) and pd.notnull(roasB)
                        else np.nan
                    ),
                    "CPC": (
                        (cpcB - cpcA)
                        if pd.notnull(cpcA) and pd.notnull(cpcB)
                        else np.nan
                    ),
                    "CPA": (
                        (cpaB - cpaA)
                        if pd.notnull(cpaA) and pd.notnull(cpaB)
                        else np.nan
                    ),
                }

                kpi_rows = [
                    (
                        "Valor usado",
                        _fmt_money_br(A["spend"]),
                        _fmt_money_br(B["spend"]),
                        _fmt_money_br(B["spend"] - A["spend"]),
                    ),
                    (
                        "Faturamento",
                        _fmt_money_br(A["revenue"]),
                        _fmt_money_br(B["revenue"]),
                        _fmt_money_br(B["revenue"] - A["revenue"]),
                    ),
                    (
                        "Vendas",
                        _fmt_int_br(A["purchases"]),
                        _fmt_int_br(B["purchases"]),
                        _fmt_int_br(B["purchases"] - A["purchases"]),
                    ),
                    (
                        "ROAS",
                        _fmt_ratio_br(roasA),
                        _fmt_ratio_br(roasB),
                        (
                            _fmt_ratio_br(roasB - roasA)
                            if pd.notnull(roasA) and pd.notnull(roasB)
                            else ""
                        ),
                    ),
                    (
                        "CPC",
                        _fmt_money_br(cpcA) if pd.notnull(cpcA) else "",
                        _fmt_money_br(cpcB) if pd.notnull(cpcB) else "",
                        (
                            _fmt_money_br(cpcB - cpcA)
                            if pd.notnull(cpcA) and pd.notnull(cpcB)
                            else ""
                        ),
                    ),
                    (
                        "CPA",
                        _fmt_money_br(cpaA) if pd.notnull(cpaA) else "",
                        _fmt_money_br(cpaB) if pd.notnull(cpaB) else "",
                        (
                            _fmt_money_br(cpaB - cpaA)
                            if pd.notnull(cpaA) and pd.notnull(cpaB)
                            else ""
                        ),
                    ),
                ]
                kpi_df_disp = pd.DataFrame(
                    kpi_rows, columns=["Métrica", "Período A", "Período B", "Δ (B - A)"]
                )

                def _style_kpi(row):
                    metric = row["Métrica"]
                    d = delta_map.get(metric, np.nan)
                    rule = dir_map.get(metric, "neutral")
                    styles = [""] * len(row)
                    try:
                        idxB = list(row.index).index("Período B")
                        idxD = list(row.index).index("Δ (B - A)")
                    except Exception:
                        return styles
                    if pd.isna(d) or rule == "neutral" or d == 0:
                        return styles
                    better = (d > 0) if rule == "higher" else (d < 0)
                    color = "#16a34a" if better else "#dc2626"
                    weight = "700"
                    styles[idxB] = f"color:{color}; font-weight:{weight};"
                    styles[idxD] = f"color:{color}; font-weight:{weight};"
                    return styles

                st.markdown("**KPIs do período (A vs B)**")
                st.dataframe(
                    kpi_df_disp.style.apply(_style_kpi, axis=1),
                    use_container_width=True,
                    height=260,
                )

                st.markdown("---")

                # Taxas do funil
                rates_num = pd.DataFrame(
                    {
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
                    }
                )
                rates_num["Δ"] = rates_num["Período B"] - rates_num["Período A"]

                rates_disp = rates_num.copy()
                for col in ["Período A", "Período B", "Δ"]:
                    rates_disp[col] = rates_disp[col].map(_fmt_pct_br)

                delta_by_taxa = dict(zip(rates_num["Taxa"], rates_num["Δ"]))

                def _style_rate(row):
                    taxa = row["Taxa"]
                    d = delta_by_taxa.get(taxa, np.nan)
                    styles = [""] * len(row)
                    try:
                        idxB = list(row.index).index("Período B")
                        idxD = list(row.index).index("Δ")
                    except Exception:
                        return styles
                    if pd.isna(d) or d == 0:
                        return styles
                    better = d > 0
                    color = "#16a34a" if better else "#dc2626"
                    weight = "700"
                    styles[idxB] = f"color:{color}; font-weight:{weight};"
                    styles[idxD] = f"color:{color}; font-weight:{weight};"
                    return styles

                st.markdown("**Taxas do funil (A vs B)**")
                st.dataframe(
                    rates_disp.style.apply(_style_rate, axis=1),
                    use_container_width=True,
                    height=180,
                )

    # ========= TABELA DIÁRIA =========
    st.subheader("Tabela diária")
    # Filtro por produto
    produto_sel = st.selectbox(
        "Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daily_produto"
    )

    def _apply_prod_filter(df):
        if produto_sel != "(Todos)":
            mask = df["campaign_name"].str.contains(produto_sel, case=False, na=False)
            return df[mask]
        return df

    dd = _apply_prod_filter(df_daily.copy())

    if dd.empty:
        st.info("Sem dados para o produto selecionado no período.")
    else:
        dd_fmt = dd.copy()
        dd_fmt["spend"] = dd_fmt["spend"].apply(_fmt_money_br)
        dd_fmt["revenue"] = dd_fmt["revenue"].apply(_fmt_money_br)
        dd_fmt["roas"] = dd_fmt["roas"].map(
            lambda x: (
                f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                if pd.notnull(x)
                else ""
            )
        )
        st.dataframe(
            dd_fmt.rename(
                columns={
                    "date": "Data",
                    "campaign_name": "Campanha",
                    "spend": "Valor usado",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagamento",
                    "purchases": "Compras",
                    "revenue": "Valor de conversão",
                    "roas": "ROAS",
                }
            ),
            use_container_width=True,
        )

# -------------------- ABA DE HORÁRIOS (Heatmap no topo) --------------------
with tab_daypart:
    st.caption(
        "Explore desempenho por hora: Heatmap no topo, depois comparação de dias e apanhado geral."
    )
    if df_hourly is None or df_hourly.empty:
        st.info("A conta/período não retornou breakdown por hora. Use a visão diária.")
    else:
        # ========= FILTRO POR PRODUTO =========
        produto_sel_hr = st.selectbox(
            "Filtrar por produto (opcional)",
            ["(Todos)"] + PRODUTOS,
            key="daypart_produto",
        )

        d = df_hourly.copy()
        if produto_sel_hr != "(Todos)":
            mask_hr = d["campaign_name"].str.contains(
                produto_sel_hr, case=False, na=False
            )
            d = d[mask_hr].copy()

        # ---------------- Filtros + base ----------------
        min_spend = st.slider(
            "Gasto mínimo para considerar o horário (R$)", 0.0, 1000.0, 0.0, 10.0
        )
        d = d.dropna(subset=["hour"])
        d["hour"] = d["hour"].astype(int).clip(0, 23)
        d["date_only"] = d["date"].dt.date

        # ============== 1) HEATMAP HORA × DIA (TOPO) ==============
        st.subheader("📆 Heatmap — Hora × Dia")
        cube_hm = d.groupby(["dow_label", "hour"], as_index=False)[
            [
                "spend",
                "revenue",
                "purchases",
                "link_clicks",
                "lpv",
                "init_checkout",
                "add_payment",
            ]
        ].sum()
        cube_hm["roas"] = np.where(
            cube_hm["spend"] > 0, cube_hm["revenue"] / cube_hm["spend"], np.nan
        )

        if min_spend > 0:
            cube_hm = cube_hm[cube_hm["spend"] >= min_spend]

        metric_hm = st.selectbox(
            "Métrica para o heatmap",
            ["Compras", "Faturamento", "Gasto", "ROAS"],
            index=0,
            key="hm_metric_top",
        )
        mcol_hm = {
            "Compras": "purchases",
            "Faturamento": "revenue",
            "Gasto": "spend",
            "ROAS": "roas",
        }[metric_hm]

        if mcol_hm == "roas":
            pvt = cube_hm.groupby(["dow_label", "hour"], as_index=False)[mcol_hm].mean()
        else:
            pvt = cube_hm.groupby(["dow_label", "hour"], as_index=False)[mcol_hm].sum()

        order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        pvt["dow_label"] = pd.Categorical(
            pvt["dow_label"], categories=order, ordered=True
        )
        pvt = pvt.sort_values(["dow_label", "hour"])
        heat = pvt.pivot(index="dow_label", columns="hour", values=mcol_hm).fillna(0)

        hours_full = list(range(24))
        heat = heat.reindex(columns=hours_full, fill_value=0)
        heat.columns = list(range(24))

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorbar=dict(title=metric_hm),
                hovertemplate="Dia: %{y}<br>Hora: %{x}h<br>"
                + metric_hm
                + ": %{z}<extra></extra>",
            )
        )
        fig_hm.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            separators=",.",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")

        # ============== 3) APANHADO GERAL POR HORA (no período) ==============
        st.subheader("📦 Apanhado geral por hora (período selecionado)")
        cube_hr = d.groupby("hour", as_index=False)[
            [
                "spend",
                "revenue",
                "purchases",
                "link_clicks",
                "lpv",
                "init_checkout",
                "add_payment",
            ]
        ].sum()
        cube_hr["ROAS"] = np.where(
            cube_hr["spend"] > 0, cube_hr["revenue"] / cube_hr["spend"], np.nan
        )
        if min_spend > 0:
            cube_hr = cube_hr[cube_hr["spend"] >= min_spend]

        top_hr = cube_hr.sort_values(
            ["purchases", "ROAS"], ascending=[False, False]
        ).copy()
        show_cols = [
            "hour",
            "purchases",
            "ROAS",
            "spend",
            "revenue",
            "link_clicks",
            "lpv",
            "init_checkout",
            "add_payment",
        ]
        disp_top = top_hr[show_cols].rename(
            columns={
                "hour": "Hora",
                "purchases": "Compras",
                "spend": "Valor usado",
                "revenue": "Valor de conversão",
            }
        )
        disp_top["Valor usado"] = disp_top["Valor usado"].apply(_fmt_money_br)
        disp_top["Valor de conversão"] = disp_top["Valor de conversão"].apply(
            _fmt_money_br
        )
        disp_top["ROAS"] = disp_top["ROAS"].map(_fmt_ratio_br)
        st.dataframe(disp_top, use_container_width=True, height=360)

        fig_bar = go.Figure(
            go.Bar(
                x=cube_hr.sort_values("hour")["hour"],
                y=cube_hr.sort_values("hour")["purchases"],
            )
        )
        fig_bar.update_layout(
            title="Compras por hora (total do período)",
            xaxis_title="Hora do dia",
            yaxis_title="Compras",
            height=380,
            template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10),
            separators=",.",
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos."
        )

        st.subheader("🆚 Comparar dois períodos (A vs B) — hora a hora")

        # Defaults: B = período atual (since/until), A = período anterior com mesma duração
        base_len = (until - since).days + 1
        default_sinceA = since - timedelta(days=base_len)
        default_untilA = since - timedelta(days=1)

        colA1, colA2, colB1, colB2 = st.columns(4)
        with colA1:
            period_sinceA = st.date_input(
                "Desde (A)", value=default_sinceA, key="cmp_sinceA"
            )
        with colA2:
            period_untilA = st.date_input(
                "Até (A)", value=default_untilA, key="cmp_untilA"
            )
        with colB1:
            period_sinceB = st.date_input("Desde (B)", value=since, key="cmp_sinceB")
        with colB2:
            period_untilB = st.date_input("Até (B)", value=until, key="cmp_untilB")

        # Validação rápida
        if period_sinceA > period_untilA or period_sinceB > period_untilB:
            st.warning(
                "Confira as datas: em cada período, 'Desde' não pode ser maior que 'Até'."
            )
        else:
            # Buscar dados por hora cobrindo A ∪ B
            union_since = min(period_sinceA, period_sinceB)
            union_until = max(period_untilA, period_untilB)

            with st.spinner("Carregando dados por hora dos períodos selecionados…"):
                df_hourly_union = fetch_insights_hourly(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(union_since),
                    until_str=str(union_until),
                    level=level,
                )

            if df_hourly_union is None or df_hourly_union.empty:
                st.info("Sem dados no intervalo combinado dos períodos selecionados.")
            else:
                # Base preparada
                d_cmp = df_hourly_union.dropna(subset=["hour"]).copy()
                d_cmp["hour"] = d_cmp["hour"].astype(int).clip(0, 23)
                d_cmp["date_only"] = d_cmp["date"].dt.date

                # Filtra pelos períodos A e B
                A_mask = (d_cmp["date_only"] >= period_sinceA) & (
                    d_cmp["date_only"] <= period_untilA
                )
                B_mask = (d_cmp["date_only"] >= period_sinceB) & (
                    d_cmp["date_only"] <= period_untilB
                )
                datA, datB = d_cmp[A_mask], d_cmp[B_mask]

                if datA.empty or datB.empty:
                    st.info("Sem dados em um dos períodos selecionados.")
                else:
                    agg_cols = [
                        "spend",
                        "revenue",
                        "purchases",
                        "link_clicks",
                        "lpv",
                        "init_checkout",
                        "add_payment",
                    ]

                    # Soma por hora
                    gA = datA.groupby("hour", as_index=False)[agg_cols].sum()
                    gB = datB.groupby("hour", as_index=False)[agg_cols].sum()

                    # Merge A vs B
                    merged = pd.merge(
                        gA, gB, on="hour", how="outer", suffixes=(" (A)", " (B)")
                    ).fillna(0.0)

                    # Filtro de gasto mínimo (descarta só se AMBOS forem baixos)
                    if min_spend > 0:
                        keep = (merged["spend (A)"] >= min_spend) | (
                            merged["spend (B)"] >= min_spend
                        )
                        merged = merged[keep]

                    if merged.empty:
                        st.info(
                            "Após o filtro de gasto mínimo, não sobraram horas para comparar."
                        )
                    else:
                        # ---------- GRÁFICOS SEPARADOS: Período A e Período B ----------
                        # 0..23 sempre presentes (preenche horas faltantes com 0)
                        hours_full = list(range(24))
                        merged = (
                            merged.set_index("hour")
                            .reindex(hours_full, fill_value=0)
                            .rename_axis("hour")
                            .reset_index()
                        )

                        # Eixo X (numérico 0..23)
                        x = merged["hour"].astype(int)

                        # Teto comum para as BARRAS (Gasto + Receita)
                        barsA_max = (merged["spend (A)"] + merged["revenue (A)"]).max()
                        barsB_max = (merged["spend (B)"] + merged["revenue (B)"]).max()
                        bars_max = max(barsA_max, barsB_max)
                        if not np.isfinite(bars_max) or bars_max <= 0:
                            bars_max = 1.0
                        bars_max *= 1.05  # folga de 5%

                        # Teto comum para a LINHA (Compras)
                        lineA_max = merged["purchases (A)"].max()
                        lineB_max = merged["purchases (B)"].max()
                        line_max = max(lineA_max, lineB_max)
                        if not np.isfinite(line_max) or line_max <= 0:
                            line_max = 1.0
                        line_max *= 1.05  # folga de 5%

                        # ===== Gráfico do Período A =====
                        fig_A = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_A.add_trace(
                            go.Bar(
                                name="Gasto (A)",
                                x=x,
                                y=merged["spend (A)"],
                                legendgroup="A",
                                offsetgroup="A",
                                hovertemplate="Hora: %{x}h<br>Gasto: R$ %{y:,.2f}<extra></extra>",
                                marker_color="#e51414",
                            ),
                            secondary_y=False,
                        )
                        fig_A.add_trace(
                            go.Bar(
                                name="Faturamento (A)",
                                x=x,
                                y=merged["revenue (A)"],
                                legendgroup="A",
                                offsetgroup="A",
                                hovertemplate="Hora: %{x}h<br>Faturamento: R$ %{y:,.2f}<extra></extra>",
                                marker_color="#636EFA",
                            ),
                            secondary_y=False,
                        )
                        fig_A.add_trace(
                            go.Scatter(
                                name=f"Compras (A) — {period_sinceA} a {period_untilA}",
                                x=x,
                                y=merged["purchases (A)"],
                                mode="lines+markers",
                                legendgroup="A",
                                hovertemplate="Hora: %{x}h<br>Compras: %{y}<extra></extra>",
                            ),
                            secondary_y=True,
                        )
                        fig_A.update_layout(
                            title=f"Período A — {period_sinceA} a {period_untilA} (Gasto + Faturamento + Compras)",
                            barmode="stack",
                            bargap=0.15,
                            bargroupgap=0.12,
                            template="plotly_white",
                            height=460,
                            margin=dict(l=10, r=10, t=48, b=10),
                            legend_title_text="",
                            separators=",.",
                        )
                        fig_A.update_xaxes(
                            title_text="Hora do dia",
                            tickmode="linear",
                            tick0=0,
                            dtick=1,
                            range=[-0.5, 23.5],
                        )
                        fig_A.update_yaxes(
                            title_text="Valores (R$)",
                            secondary_y=False,
                            range=[0, bars_max],
                        )
                        fig_A.update_yaxes(
                            title_text="Compras (unid.)",
                            secondary_y=True,
                            range=[0, line_max],
                        )

                        st.plotly_chart(fig_A, use_container_width=True)

                        # ===== Gráfico do Período B =====
                        fig_B = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_B.add_trace(
                            go.Bar(
                                name="Gasto (B)",
                                x=x,
                                y=merged["spend (B)"],
                                legendgroup="B",
                                offsetgroup="B",
                                hovertemplate="Hora: %{x}h<br>Gasto: R$ %{y:,.2f}<extra></extra>",
                                marker_color="#e51414",
                            ),
                            secondary_y=False,
                        )
                        fig_B.add_trace(
                            go.Bar(
                                name="Faturamento (B)",
                                x=x,
                                y=merged["revenue (B)"],
                                legendgroup="B",
                                offsetgroup="B",
                                hovertemplate="Hora: %{x}h<br>Faturamento: R$ %{y:,.2f}<extra></extra>",
                                marker_color="#636EFA",
                            ),
                            secondary_y=False,
                        )
                        fig_B.add_trace(
                            go.Scatter(
                                name=f"Compras (B) — {period_sinceB} a {period_untilB}",
                                x=x,
                                y=merged["purchases (B)"],
                                mode="lines+markers",
                                legendgroup="B",
                                hovertemplate="Hora: %{x}h<br>Compras: %{y}<extra></extra>",
                            ),
                            secondary_y=True,
                        )
                        fig_B.update_layout(
                            title=f"Período B — {period_sinceB} a {period_untilB} (Gasto + Faturamento + Compras)",
                            barmode="stack",
                            bargap=0.15,
                            bargroupgap=0.12,
                            template="plotly_white",
                            height=460,
                            margin=dict(l=10, r=10, t=48, b=10),
                            legend_title_text="",
                            separators=",.",
                        )
                        fig_B.update_xaxes(
                            title_text="Hora do dia",
                            tickmode="linear",
                            tick0=0,
                            dtick=1,
                            range=[-0.5, 23.5],
                        )
                        fig_B.update_yaxes(
                            title_text="Valores (R$)",
                            secondary_y=False,
                            range=[0, bars_max],
                        )
                        fig_B.update_yaxes(
                            title_text="Compras (unid.)",
                            secondary_y=True,
                            range=[0, line_max],
                        )

                        st.plotly_chart(fig_B, use_container_width=True)

                        # ===== INSIGHTS — Período A =====
                        st.markdown("### 🔎 Insights — Período A")

                        a = merged.sort_values("hour").copy()
                        a_spend = a["spend (A)"]
                        a_rev = a["revenue (A)"]
                        a_purch = a["purchases (A)"]
                        a_roas_ser = np.where(a_spend > 0, a_rev / a_spend, np.nan)

                        a_tot_spend = float(a_spend.sum())
                        a_tot_rev = float(a_rev.sum())
                        a_tot_purch = int(round(float(a_purch.sum())))
                        a_roas = (
                            (a_tot_rev / a_tot_spend) if a_tot_spend > 0 else np.nan
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Gasto", _fmt_money_br(a_tot_spend))
                        with c2:
                            st.metric("Faturamento", _fmt_money_br(a_tot_rev))
                        with c3:
                            st.metric("Compras", _fmt_int_br(a_tot_purch))
                        with c4:
                            st.metric("ROAS", _fmt_ratio_br(a_roas))

                        st.markdown("### 🔎 Insights — Período B")

                        b = merged.sort_values("hour").copy()
                        b_spend = b["spend (B)"]
                        b_rev = b["revenue (B)"]
                        b_purch = b["purchases (B)"]
                        b_roas_ser = np.where(b_spend > 0, b_rev / b_spend, np.nan)

                        b_tot_spend = float(b_spend.sum())
                        b_tot_rev = float(b_rev.sum())
                        b_tot_purch = int(round(float(b_purch.sum())))
                        b_roas = (
                            (b_tot_rev / b_tot_spend) if b_tot_spend > 0 else np.nan
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric(
                                "Gasto",
                                _fmt_money_br(b_tot_spend),
                                delta=_fmt_money_br(b_tot_spend - a_tot_spend),
                            )
                        with c2:
                            st.metric(
                                "Faturamento",
                                _fmt_money_br(b_tot_rev),
                                delta=_fmt_money_br(b_tot_rev - a_tot_rev),
                            )
                        with c3:
                            st.metric(
                                "Compras",
                                _fmt_int_br(b_tot_purch),
                                delta=_fmt_int_signed_br(b_tot_purch - a_tot_purch),
                            )
                        with c4:
                            st.metric(
                                "ROAS",
                                _fmt_ratio_br(b_roas),
                                delta=(
                                    _fmt_ratio_br(b_roas - a_roas)
                                    if pd.notnull(b_roas) and pd.notnull(a_roas)
                                    else None
                                ),
                            )

                        st.markdown("### 📈 Comparativo Hora a Hora")

                        # Tabela comparativa
                        comp_cols = [
                            "hour",
                            "spend (A)",
                            "spend (B)",
                            "revenue (A)",
                            "revenue (B)",
                            "purchases (A)",
                            "purchases (B)",
                        ]
                        comp_df = merged[comp_cols].copy()
                        comp_df["Δ Gasto"] = comp_df["spend (B)"] - comp_df["spend (A)"]
                        comp_df["Δ Faturamento"] = (
                            comp_df["revenue (B)"] - comp_df["revenue (A)"]
                        )
                        comp_df["Δ Compras"] = (
                            comp_df["purchases (B)"] - comp_df["purchases (A)"]
                        )

                        comp_df["ROAS (A)"] = np.where(
                            comp_df["spend (A)"] > 0,
                            comp_df["revenue (A)"] / comp_df["spend (A)"],
                            np.nan,
                        )
                        comp_df["ROAS (B)"] = np.where(
                            comp_df["spend (B)"] > 0,
                            comp_df["revenue (B)"] / comp_df["spend (B)"],
                            np.nan,
                        )
                        comp_df["Δ ROAS"] = comp_df["ROAS (B)"] - comp_df["ROAS (A)"]

                        # Formatação para exibição
                        disp_comp = comp_df.copy()
                        for col in [
                            "spend (A)",
                            "spend (B)",
                            "revenue (A)",
                            "revenue (B)",
                            "Δ Gasto",
                            "Δ Faturamento",
                        ]:
                            disp_comp[col] = disp_comp[col].apply(_fmt_money_br)
                        for col in ["purchases (A)", "purchases (B)"]:
                            disp_comp[col] = disp_comp[col].apply(_fmt_int_br)
                        disp_comp["Δ Compras"] = disp_comp["Δ Compras"].apply(
                            _fmt_int_signed_br
                        )
                        for col in ["ROAS (A)", "ROAS (B)", "Δ ROAS"]:
                            disp_comp[col] = disp_comp[col].map(_fmt_ratio_br)

                        st.dataframe(
                            disp_comp.rename(
                                columns={
                                    "hour": "Hora",
                                    "spend (A)": "Gasto A",
                                    "spend (B)": "Gasto B",
                                    "revenue (A)": "Faturamento A",
                                    "revenue (B)": "Faturamento B",
                                    "purchases (A)": "Compras A",
                                    "purchases (B)": "Compras B",
                                }
                            ),
                            use_container_width=True,
                            height=520,
                        )

                        st.info(
                            "A tabela acima mostra a comparação hora a hora entre os períodos A e B."
                        )

# -------------------- ABA 3: DETALHAMENTO --------------------
with tab_detail:
    st.caption(
        "Detalhe por dimensão: Campanha, Idade, Gênero, Região, País, Plataforma, Posicionamento, Dia, Hora."
    )

    dimensao = st.selectbox(
        "Detalhar por:",
        [
            "Campanha",
            "Populares",
            "Idade",
            "Gênero",
            "Idade + Gênero",
            "Região",
            "País",
            "Plataforma",
            "Posicionamento",
            "Dia",
            "Hora",
        ],
        index=0,
    )

    min_spend_det = st.slider(
        "Gasto mínimo para detalhamento (R$)",
        0.0,
        1000.0,
        0.0,
        10.0,
        key="min_spend_det",
    )

    # Funções auxiliares para detalhamento
    def _agg_and_format(df, group_cols):
        g = df.groupby(group_cols, as_index=False)[
            [
                "spend",
                "revenue",
                "purchases",
                "link_clicks",
                "lpv",
                "init_checkout",
                "add_payment",
            ]
        ].sum()
        g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)

        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]

        gf = g.copy()
        gf["Valor usado"] = gf["spend"].apply(_fmt_money_br)
        gf["Valor de conversão"] = gf["revenue"].apply(_fmt_money_br)
        gf["ROAS"] = gf["ROAS"].map(_fmt_ratio_br)
        gf = gf.drop(columns=["spend", "revenue"])
        return g, gf

    def _bar_chart(x_labels, y_values, title, x_title, y_title):
        fig = go.Figure(go.Bar(x=x_labels, y=y_values))
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=420,
            template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10),
            separators=",.",
        )
        st.plotly_chart(fig, use_container_width=True)

    # =================== Campanha ===================
    if dimensao == "Campanha":
        st.subheader("Desempenho por Campanha")
        if level != "campaign":
            st.info(
                "Troque o nível para 'campaign' na barra lateral para ver o detalhamento por campanha."
            )
            st.stop()

        d = _apply_prod_filter(df_daily.copy())
        if d.empty:
            st.info("Sem dados após filtro de produto.")
            st.stop()

        g, gf = _agg_and_format(d, ["campaign_id", "campaign_name"])
        if gf.empty:
            st.info("Sem dados por Campanha após aplicar filtros.")
            st.stop()

        st.dataframe(
            gf.rename(
                columns={
                    "campaign_id": "ID Campanha",
                    "campaign_name": "Campanha",
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                }
            ),
            use_container_width=True,
            height=520,
        )

        # Gráfico de compras por campanha (TOP 10)
        top_campaigns = g.sort_values("purchases", ascending=False).head(10)
        _bar_chart(
            top_campaigns["campaign_name"],
            top_campaigns["purchases"],
            "Compras por Campanha (TOP 10)",
            "Campanha",
            "Compras",
        )

    # =================== Populares (TOP 5 em cada dimensão principal) ===================
    if dimensao == "Populares":
        st.subheader("TOP 5 — Quem mais compra em cada dimensão")

        # Idade
        with st.container():
            st.markdown("**Idade**")
            df_age = fetch_insights_breakdown(
                act_id, token, api_version, str(since), str(until), ["age"], level
            )
            g, gf = (
                _agg_and_format(df_age, ["age"])
                if not df_age.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por Idade para o período/filtro.")
            else:
                top = gf.head(5).rename(
                    columns={
                        "age": "Idade",
                        "purchases": "Compras",
                        "link_clicks": "Cliques",
                        "lpv": "LPV",
                        "init_checkout": "Checkout",
                        "add_payment": "Add Pagto",
                    }
                )
                st.dataframe(
                    top[
                        [
                            "Idade",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                            "Cliques",
                            "LPV",
                            "Checkout",
                            "Add Pagto",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )
                _bar_chart(
                    top["Idade"],
                    g.head(5)["purchases"],
                    "Compras por Idade (TOP 5)",
                    "Idade",
                    "Compras",
                )

        st.markdown("---")

        # Gênero
        with st.container():
            st.markdown("**Gênero**")
            df_gen = fetch_insights_breakdown(
                act_id, token, api_version, str(since), str(until), ["gender"], level
            )
            g, gf = (
                _agg_and_format(df_gen, ["gender"])
                if not df_gen.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por Gênero para o período/filtro.")
            else:
                top = gf.head(5).rename(
                    columns={
                        "gender": "Gênero",
                        "purchases": "Compras",
                        "link_clicks": "Cliques",
                        "lpv": "LPV",
                        "init_checkout": "Checkout",
                        "add_payment": "Add Pagto",
                    }
                )
                st.dataframe(
                    top[
                        [
                            "Gênero",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                            "Cliques",
                            "LPV",
                            "Checkout",
                            "Add Pagto",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )
                _bar_chart(
                    top["Gênero"],
                    g.head(5)["purchases"],
                    "Compras por Gênero (TOP 5)",
                    "Gênero",
                    "Compras",
                )

        st.markdown("---")

        # Região
        with st.container():
            st.markdown("**Região**")
            df_reg = fetch_insights_breakdown(
                act_id, token, api_version, str(since), str(until), ["region"], level
            )
            g, gf = (
                _agg_and_format(df_reg, ["region"])
                if not df_reg.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por Região para o período/filtro.")
            else:
                top = gf.head(5).rename(
                    columns={
                        "region": "Região",
                        "purchases": "Compras",
                        "link_clicks": "Cliques",
                        "lpv": "LPV",
                        "init_checkout": "Checkout",
                        "add_payment": "Add Pagto",
                    }
                )
                st.dataframe(
                    top[
                        [
                            "Região",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                            "Cliques",
                            "LPV",
                            "Checkout",
                            "Add Pagto",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )
                _bar_chart(
                    top["Região"],
                    g.head(5)["purchases"],
                    "Compras por Região (TOP 5)",
                    "Região",
                    "Compras",
                )

        st.markdown("---")

        # País
        with st.container():
            st.markdown("**País**")
            df_cty = fetch_insights_breakdown(
                act_id, token, api_version, str(since), str(until), ["country"], level
            )
            g, gf = (
                _agg_and_format(df_cty, ["country"])
                if not df_cty.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por País para o período/filtro.")
            else:
                top = gf.head(5).rename(
                    columns={
                        "country": "País",
                        "purchases": "Compras",
                        "link_clicks": "Cliques",
                        "lpv": "LPV",
                        "init_checkout": "Checkout",
                        "add_payment": "Add Pagto",
                    }
                )
                st.dataframe(
                    top[
                        [
                            "País",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                            "Cliques",
                            "LPV",
                            "Checkout",
                            "Add Pagto",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )
                _bar_chart(
                    top["País"],
                    g.head(5)["purchases"],
                    "Compras por País (TOP 5)",
                    "País",
                    "Compras",
                )

        st.markdown("---")

        # Plataforma (publisher_platform) e Posicionamento (platform_position)
        cA, cB = st.columns(2)

        with cA:
            st.markdown("**Plataforma**")
            df_plat = fetch_insights_breakdown(
                act_id,
                token,
                api_version,
                str(since),
                str(until),
                ["publisher_platform"],
                level,
            )
            g, gf = (
                _agg_and_format(df_plat, ["publisher_platform"])
                if not df_plat.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por Plataforma.")
            else:
                top = gf.head(5).rename(
                    columns={"publisher_platform": "Plataforma", "purchases": "Compras"}
                )
                st.dataframe(
                    top[
                        [
                            "Plataforma",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )

        with cB:
            st.markdown("**Posicionamento**")
            df_pos = fetch_insights_breakdown(
                act_id,
                token,
                api_version,
                str(since),
                str(until),
                ["publisher_platform", "platform_position"],
                level,
            )
            g, gf = (
                _agg_and_format(df_pos, ["publisher_platform", "platform_position"])
                if not df_pos.empty
                else (pd.DataFrame(), pd.DataFrame())
            )
            if gf.empty:
                st.info("Sem dados por Posicionamento.")
            else:
                top = gf.head(5).rename(
                    columns={
                        "publisher_platform": "Plataforma",
                        "platform_position": "Posicionamento",
                        "purchases": "Compras",
                    }
                )
                st.dataframe(
                    top[
                        [
                            "Plataforma",
                            "Posicionamento",
                            "Compras",
                            "ROAS",
                            "Valor usado",
                            "Valor de conversão",
                        ]
                    ],
                    use_container_width=True,
                    height=230,
                )

        st.stop()  # encerra a aba no modo \"Populares\"

        # =================== Dimensões 1D e 2D (Idade, Gênero, Idade+Gênero, País, Plataforma, Posicionamento) ===================
        dim_to_breakdowns = {
            "Idade": ["age"],
            "Gênero": ["gender"],
            "Idade + Gênero": ["age", "gender"],
            "Região": ["region"],
            "País": ["country"],
            "Plataforma": ["publisher_platform"],
            "Posicionamento": ["publisher_platform", "platform_position"],
        }

        if dimensao in dim_to_breakdowns:
            bks = dim_to_breakdowns[dimensao]
            df_bd = fetch_insights_breakdown(
                act_id, token, api_version, str(since), str(until), bks, level
            )

            if df_bd.empty:
                st.info(f"Sem dados para {dimensao} no período/filtro.")
                st.stop()

            # Nomes amigáveis
            rename_map = {
                "age": "Idade",
                "gender": "Gênero",
                "region": "Região",
                "country": "País",
                "publisher_platform": "Plataforma",
                "platform_position": "Posicionamento",
            }
            group_cols = [rename_map.get(c, c) for c in bks]

            # agrega + formata
            raw, disp = _agg_and_format(df_bd.rename(columns=rename_map), group_cols)
            if disp.empty:
                st.info(f"Sem dados para {dimensao} após aplicar filtros.")
                st.stop()

            st.subheader(f"Desempenho por {dimensao}")
            # tabela
            base_cols = group_cols + [
                "Compras",
                "ROAS",
                "Valor usado",
                "Valor de conversão",
                "Cliques",
                "LPV",
                "Checkout",
                "Add Pagto",
            ]
            disp = disp.rename(
                columns={
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                }
            )
            # garante colunas na ordem
            for c in base_cols:
                if c not in disp.columns:
                    pass
            st.dataframe(disp[base_cols], use_container_width=True, height=520)

            # gráfico: compras por grupo (se 1D) ou heatmap (se 2D)
            if len(group_cols) == 1:
                xlab = group_cols[0]
                _bar_chart(
                    raw[xlab], raw["purchases"], f"Compras por {xlab}", xlab, "Compras"
                )
            else:
                idx, col = group_cols
                pvt = raw.pivot_table(
                    index=idx, columns=col, values="purchases", aggfunc="sum"
                ).fillna(0)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pvt.values,
                        x=pvt.columns.astype(str),
                        y=pvt.index.astype(str),
                        colorbar=dict(title="Compras"),
                        hovertemplate=f"{idx}: "
                        + "%{y}<br>"
                        + f"{col}: "
                        + "%{x}<br>Compras: %{z}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title=f"Heatmap — Compras por {idx} × {col}",
                    height=460,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=48, b=10),
                    separators=",.",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.stop()

        # =================== Dia ===================
        if dimensao == "Dia":
            st.subheader("Desempenho por Dia")
            if df_daily.empty:
                st.info("Sem dados diários.")
                st.stop()

            d = _apply_prod_filter(df_daily.copy())
            if d.empty:
                st.info("Sem dados após filtro de produto.")
                st.stop()

            g = d.groupby(d["date"].dt.date, as_index=False)[
                [
                    "spend",
                    "revenue",
                    "purchases",
                    "link_clicks",
                    "lpv",
                    "init_checkout",
                    "add_payment",
                ]
            ].sum()
            g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)

            if min_spend_det and float(min_spend_det) > 0:
                g = g[g["spend"] >= float(min_spend_det)]

            disp = g.rename(
                columns={
                    "date": "Data",
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                    "spend": "Valor usado",
                    "revenue": "Valor de conversão",
                }
            )
            disp["Valor usado"] = disp["Valor usado"].apply(_fmt_money_br)
            disp["Valor de conversão"] = disp["Valor de conversão"].apply(_fmt_money_br)
            disp["ROAS"] = disp["ROAS"].map(_fmt_ratio_br)

            st.dataframe(
                disp[
                    [
                        "Data",
                        "Compras",
                        "ROAS",
                        "Valor usado",
                        "Valor de conversão",
                        "Cliques",
                        "LPV",
                        "Checkout",
                        "Add Pagto",
                    ]
                ],
                use_container_width=True,
                height=520,
            )

            # gráfico: série de compras
            _bar_chart(
                g["date"].dt.strftime("%Y-%m-%d"),
                g["purchases"],
                "Compras por Dia",
                "Dia",
                "Compras",
            )
            st.stop()

        # =================== Hora ===================
        if dimensao == "Hora":
            st.subheader("Desempenho por Hora")
            if df_hourly is None or df_hourly.empty:
                st.info("Sem breakdown por hora para o período.")
                st.stop()

            d = _apply_prod_filter(df_hourly.copy())
            d = d.dropna(subset=["hour"])
            if d.empty:
                st.info("Sem dados após filtro de produto.")
                st.stop()

            g = d.groupby("hour", as_index=False)[
                [
                    "spend",
                    "revenue",
                    "purchases",
                    "link_clicks",
                    "lpv",
                    "init_checkout",
                    "add_payment",
                ]
            ].sum()
            g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)

            if min_spend_det and float(min_spend_det) > 0:
                g = g[g["spend"] >= float(min_spend_det)]

            disp = g.rename(
                columns={
                    "hour": "Hora",
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                    "spend": "Valor usado",
                    "revenue": "Valor de conversão",
                }
            )
            disp["Valor usado"] = disp["Valor usado"].apply(_fmt_money_br)
            disp["Valor de conversão"] = disp["Valor de conversão"].apply(_fmt_money_br)
            disp["ROAS"] = disp["ROAS"].map(_fmt_ratio_br)

            st.dataframe(
                disp[
                    [
                        "Hora",
                        "Compras",
                        "ROAS",
                        "Valor usado",
                        "Valor de conversão",
                        "Cliques",
                        "LPV",
                        "Checkout",
                        "Add Pagto",
                    ]
                ],
                use_container_width=True,
                height=520,
            )

            _bar_chart(
                g["hour"].astype(int),
                g["purchases"],
                "Compras por Hora",
                "Hora do dia",
                "Compras",
            )
            st.stop()
