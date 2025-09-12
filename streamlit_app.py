import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# =============================================================================
# --- Configuração e Estilos ---
# =============================================================================
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# Janelas de atribuição (paridade com Ads Manager)
ATTR_KEYS = ["7d_click", "1d_view"]
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus", "(Todos)"]

# Fuso horário da aplicação
APP_TZ = ZoneInfo("America/Sao_Paulo")

# Sessão de Requests para otimizar conexões HTTP
_session = None
def _get_session():
    """Retorna uma sessão de requests, criando-a se necessário."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session

# Constantes e parser para breakdown por hora
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"

# =============================================================================
# --- Funções Auxiliares de Tratamento de Dados e Formatação ---
# =============================================================================

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

def _retry_call(fn, max_retries=5, base_wait=1.2):
    """Executa uma função com backoff exponencial para erros/transientes."""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit","retry","temporarily unavailable","timeout","timed out"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def _ensure_act_prefix(ad_account_id: str) -> str:
    """Garante o prefixo 'act_' no ID da conta de anúncios."""
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def _to_float(x):
    """Converte um valor para float, tratando erros."""
    try:
        return float(x or 0)
    except:
        return 0.0

def _sum_item(item, allowed_keys=None):
    """Soma valores de uma ação, priorizando 'value' ou as chaves de atribuição."""
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
    """Soma ações por nomes exatos (case-insensitive)."""
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
    """Soma ações que contenham qualquer substring."""
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
    """Soma as compras, priorizando 'omni_purchase' ou a maior variante."""
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
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

def _pick_checkout_totals(rows, allowed_keys=None) -> float:
    """Soma os 'Initiate Checkout', priorizando 'omni'."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] in ("omni_initiated_checkout","omni_initiate_checkout")]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))
    candidates = {
        "initiate_checkout": 0.0,
        "initiated_checkout": 0.0,
        "onsite_conversion.initiated_checkout": 0.0,
        "offsite_conversion.fb_pixel_initiate_checkout": 0.0,
        "offsite_conversion.fb_pixel_initiated_checkout": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)
    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))
    grp = {}
    for r in rows:
        if "initiate" in r["action_type"] and "checkout" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

def _pick_add_payment_totals(rows, allowed_keys=None) -> float:
    """Soma os 'Add Payment Info', com suporte a variantes."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] in ("omni_add_payment_info","add_payment_info.omni")]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))
    candidates = {
        "add_payment_info": 0.0,
        "onsite_conversion.add_payment_info": 0.0,
        "offsite_conversion.fb_pixel_add_payment_info": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)
    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))
    grp = {}
    for r in rows:
        if "add_payment" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

def enforce_monotonic(values):
    """Garante que cada etapa do funil não tenha mais que a etapa anterior."""
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

# --- Funções de formatação de métricas ---
def _rate(a, b):
    """Calcula uma taxa (a/b), tratando divisão por zero."""
    return (a / b) if b and b > 0 else np.nan

def _fmt_money_br(v: float) -> str:
    """Formata um valor como moeda (R$)."""
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct_br(x):
    """Formata um valor como porcentagem em português."""
    return (
        f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_ratio_br(x):
    """Formata uma taxa (ex: ROAS '1,23x')."""
    return (
        f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_int_br(x):
    """Formata um número inteiro em português."""
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except:
        return ""

def _fmt_int_signed_br(x):
    """Formata um número inteiro com sinal (+/-)."""
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

def _pct(a, b):
    a = float(a or 0); b = float(b or 0)
    return (a / b) if b > 0 else np.nan

def _intensity_label(share):
    """Gera um rótulo de intensidade (Baixa, Média, Alta)."""
    if not np.isfinite(share):
        return "Baixa"
    if share > 0.60:
        return "Alta"
    if share >= 0.30:
        return "Média"
    return "Baixa"

def _decide_focus(r1, r2, r3, clicks, lpv, co, addpay, purch,
                  bm_r1, bm_r2, bm_r3, min_clicks, min_lpv, min_co, min_purch,
                  split_rmk=True):
    """
    Analisa as taxas e volumes do funil para sugerir o foco de otimização
    (Criativo, Interesse, Remarketing, ou Escala).
    """
    healthy = (
        (pd.notnull(r1) and r1 >= bm_r1/100.0) and
        (pd.notnull(r2) and r2 >= bm_r2/100.0) and
        (pd.notnull(r3) and r3 >= bm_r3/100.0) and
        (float(purch or 0) >= float(min_purch or 0))
    )
    low_volume_guard = (float(clicks or 0) < float(min_clicks or 0)) or \
                       (float(lpv or 0) < float(min_lpv or 0)) or \
                       (float(co or 0) < float(min_co or 0))
    drop1 = max(0.0, float(clicks or 0) - float(lpv or 0))
    drop2 = max(0.0, float(lpv or 0) - float(co or 0))
    drop3a = max(0.0, float(co or 0) - float(addpay or 0))
    drop3b = max(0.0, float(addpay or 0) - float(purch or 0))

    if healthy and not low_volume_guard:
        return "Escala", "Taxas ≥ benchmarks e volume OK — elegível a escalar.", "Média", False, drop1, drop2, drop3a, drop3b

    if split_rmk:
        gaps = {
            "Teste de criativo": drop1,
            "Teste de interesse": drop2,
            "Remarketing (checkout→pagto)": drop3a,
            "Remarketing (pagto→compra)": drop3b,
        }
    else:
        gaps = {
            "Teste de criativo": drop1,
            "Teste de interesse": drop2,
            "Remarketing (fundo do funil)": drop3a + drop3b,
        }

    major = max(gaps, key=gaps.get)
    max_drop = gaps[major]
    total_drop = sum(v for v in gaps.values() if v > 0)
    share = (max_drop / total_drop) if total_drop > 0 else np.nan
    intensity = _intensity_label(share)
    if max_drop <= 0 and not healthy:
        return "Diagnóstico", "Sem queda dominante; revisar tracking/UX/oferta.", "Baixa", low_volume_guard, drop1, drop2, drop3a, drop3b
    reason = f"Maior perda em **{major}** (Δ={int(round(max_drop))} pessoas)."
    if low_volume_guard:
        return f"{major} (c/ cautela)", reason, intensity, True, drop1, drop2, drop3a, drop3b
    return major, reason, intensity, False, drop1, drop2, drop3a, drop3b

def _chunks_by_days(since_str: str, until_str: str, max_days: int = 30):
    """Divide um intervalo de datas em pedaços de até `max_days`."""
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days - 1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)

def _filter_by_product(df: pd.DataFrame, produto: str) -> pd.DataFrame:
    """Filtra o DataFrame pelo nome do produto na campanha."""
    if not isinstance(df, pd.DataFrame) or df.empty or not produto or produto == "(Todos)":
        return df
    mask = df["campaign_name"].str.contains(produto, case=False, na=False)
    return df[mask].copy()

def funnel_fig(labels, values, title=None):
    """Cria um gráfico de funil com Plotly."""
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
        separators=",.",
        uniformtext=dict(minsize=12, mode="show")
    )
    return fig

def _bar_chart(x_data, y_data, title, x_label, y_label):
    """Cria um gráfico de barras simples com Plotly."""
    fig = go.Figure(go.Bar(x=x_data, y=y_data))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def _line_chart(df, x_col, y_col, title, x_label=None, y_label=None, color=None):
    """Cria um gráfico de linha com Plotly."""
    fig = go.Figure()
    if color:
        for val in df[color].unique():
            subset = df[df[color] == val]
            fig.add_trace(go.Scatter(x=subset[x_col], y=subset[y_col], mode='lines', name=str(val)))
    else:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# --- Funções de Coleta de Dados da Meta (API) ---
# =============================================================================

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True,
                         product_name: str | None = None) -> pd.DataFrame:
    """
    Busca insights diários da Meta Ads API, com chunking e paralelismo.
    Inclui fallback para campos extras e paridade com o Ads Manager.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    base_fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name"
    ]
    extra_fields = ["link_clicks", "landing_page_views"]

    def _fetch_range(_since: str, _until: str, _try_extra: bool) -> list[dict]:
        fields = base_fields + (extra_fields if _try_extra else [])
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": "conversion",
            "action_attribution_windows": ",".join(ATTR_KEYS),
        }
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps([{
                "field": "campaign.name", "operator": "CONTAIN", "value": product_name
            }])
        rows_local, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API.")
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
                if code == 100 and _try_extra:
                    return _fetch_range(_since, _until, _try_extra=False)
                raise RuntimeError(f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}")
            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []
                link_clicks = rec.get("link_clicks", None)
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)
                lpv = rec.get("landing_page_views", None)
                if lpv is None:
                    lpv = _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                    if lpv == 0:
                        lpv = (_sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                                or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))
                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                revenue_val = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)
                rows_local.append({
                    "date": pd.to_datetime(rec.get("date_start")),
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend": _to_float(rec.get("spend")),
                    "impressions": _to_float(rec.get("impressions")),
                    "clicks": _to_float(rec.get("clicks")),
                    "link_clicks": _to_float(link_clicks),
                    "lpv": _to_float(lpv),
                    "init_checkout": _to_float(ic),
                    "add_payment": _to_float(api),
                    "purchases": _to_float(purchases_cnt),
                    "revenue": _to_float(revenue_val),
                })
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
        return rows_local

    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u, try_extra_fields) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    num_cols = ["spend", "impressions", "clicks", "link_clicks",
                "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df = df.sort_values("date")
    return df

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(act_id: str, token: str, api_version: str,
                          since_str: str, until_str: str,
                          level: str = "campaign") -> pd.DataFrame:
    """Busca insights por hora, com chunking e fallback para 'impression'."""
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]
    def _fetch_range(_since: str, _until: str, action_rt: str) -> pd.DataFrame:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
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
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API (hourly).")
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
                raise RuntimeError(f"Graph API error hourly | code={code} subcode={sub} | {msg}")
            for rec in (payload.get("data") or []):
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
                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)
                rows.append({
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
                })
            paging = (payload.get("paging") or {})
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
        return df

    dfs = []
    for s_chunk, u_chunk in _chunks_by_days(since_str, until_str, max_days=30):
        try:
            df_chunk = _fetch_range(s_chunk, u_chunk, "conversion")
        except Exception:
            try:
                df_chunk = _fetch_range(s_chunk, u_chunk, "impression")
            except Exception:
                df_chunk = pd.DataFrame()
        if df_chunk is not None and not df_chunk.empty:
            dfs.append(df_chunk)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int).clip(0, 23)
    df["dow"] = df["date"].dt.dayofweek
    order = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
    df["dow_label"] = df["dow"].map(order)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df.sort_values(["date", "hour"])

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_breakdown(act_id: str, token: str, api_version: str,
                             since_str: str, until_str: str,
                             breakdowns: list[str],
                             level: str = "campaign",
                             product_name: str | None = None) -> pd.DataFrame:
    """
    Busca insights com breakdowns (ex: idade, gênero), com chunking e paralelismo.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]
    def _fetch_range(_since: str, _until: str) -> list[dict]:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": "conversion",
            "action_attribution_windows": ",".join(ATTR_KEYS),
            "breakdowns": ",".join(breakdowns[:2])
        }
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps([{
                "field": "campaign.name", "operator": "CONTAIN", "value": product_name
            }])
        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            payload = resp.json()
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                raise RuntimeError(
                    f"Graph API error breakdown | code={err.get('code')} "
                    f"sub={err.get('error_subcode')} | {err.get('message')}"
                )
            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []
                link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)
                lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                        or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                        or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))
                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)
                base = {
                    "currency": rec.get("account_currency","BRL"),
                    "campaign_id": rec.get("campaign_id",""),
                    "campaign_name": rec.get("campaign_name",""),
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
                for b in breakdowns[:2]:
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
        return rows
    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df

# =============================================================================
# --- Interface do Streamlit (Sidebar) ---
# =============================================================================

st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign"], index=0)

preset = st.sidebar.radio(
    "Período rápido",
    [
        "Hoje", "Ontem",
        "Últimos 7 dias", "Últimos 14 dias", "Últimos 30 dias", "Últimos 90 dias",
        "Esta semana", "Este mês", "Máximo",
        "Personalizado"
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
        start_week = local_today - timedelta(days=local_today.weekday())
        return start_week, local_today
    if p == "Este mês":
        start_month = local_today.replace(day=1)
        return start_month, local_today
    if p == "Máximo":
        return date(2017, 1, 1), base_end
    return base_end - timedelta(days=6), base_end

_since_auto, _until_auto = _range_from_preset(preset)

if preset == "Personalizado":
    since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
    until = st.sidebar.date_input("Até", value=_until_auto, key="until_custom")
else:
    since, until = _since_auto, _until_auto
    st.sidebar.caption(f"**Desde:** {since} \n**Até:** {until}")

st.sidebar.markdown("---")
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = "(Todos)"
selected_product = st.sidebar.selectbox(
    "Filtrar por Produto",
    PRODUTOS,
    index=PRODUTOS.index(st.session_state.selected_product)
)
st.session_state.selected_product = selected_product

ready = bool(act_id and token)

# =============================================================================
# --- Lógica Principal da Aplicação ---
# =============================================================================

st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra. Tudo alinhado ao período selecionado.")

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

# --- Coleta dos dados ---
try:
    with st.spinner("Buscando dados da Meta…"):
        df_daily = fetch_insights_daily(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(since),
            until_str=str(until),
            level=level,
            product_name=st.session_state.selected_product
        )
except Exception as e:
    st.error(f"Erro ao buscar dados: {e}")
    st.stop()

if df_daily.empty:
    st.warning("Não há dados para o período e filtros selecionados.")
    st.stop()

# --- Análise do Funil (totais agregados) ---
df_total = df_daily.groupby(["campaign_name"]).sum(numeric_only=True).reset_index()
tot = df_daily.sum(numeric_only=True)
clicks = float(tot["link_clicks"])
lpv = float(tot["lpv"])
init_checkout = float(tot["init_checkout"])
add_payment = float(tot["add_payment"])
purchases = float(tot["purchases"])

labels_funil = ["Cliques", "LPV", "Checkout", "Adicionar Pagamento", "Compra"]
values_funil = enforce_monotonic([clicks, lpv, init_checkout, add_payment, purchases])

# --- Análise de Funil e KPIs (Layout) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Funil de Conversão")
    fig_funnel = funnel_fig(labels_funil, values_funil)
    st.plotly_chart(fig_funnel, use_container_width=True)

with col2:
    st.subheader("Análise e KPIs")
    
    col2a, col2b, col2c = st.columns(3)
    with col2a:
        st.markdown(f'<div class="kpi-card">Taxa LPV/Clique<br><span class="big-number">{_fmt_pct_br(_pct(lpv, clicks))}</span></div>', unsafe_allow_html=True)
    with col2b:
        st.markdown(f'<div class="kpi-card">Taxa Checkout/LPV<br><span class="big-number">{_fmt_pct_br(_pct(init_checkout, lpv))}</span></div>', unsafe_allow_html=True)
    with col2c:
        st.markdown(f'<div class="kpi-card">Taxa Compra/Checkout<br><span class="big-number">{_fmt_pct_br(_pct(purchases, init_checkout))}</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(f"<div class='kpi-card'>Cliques<br><span class='big-number'>{_fmt_int_br(clicks)}</span></div>", unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"<div class='kpi-card'>Compras<br><span class='big-number'>{_fmt_int_br(purchases)}</span></div>", unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"<div class='kpi-card'>ROAS<br><span class='big-number'>{_fmt_ratio_br(_safe_div(tot['revenue'], tot['spend']))}</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    focus_area, reason, intensity, low_volume, d1, d2, d3a, d3b = _decide_focus(
        _pct(lpv, clicks), _pct(init_checkout, lpv), _pct(purchases, init_checkout),
        clicks, lpv, init_checkout, add_payment, purchases,
        bm_r1=25, bm_r2=10, bm_r3=5, min_clicks=100, min_lpv=50, min_co=20, min_purch=1
    )
    
    st.markdown(f"""
    **Foco de Otimização Sugerido**: **{focus_area}**
    <span class="small-muted">{reason}</span>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Intensidade do Problema**: {intensity}")

# --- Gráficos diários ---
st.markdown("---")
st.header("Análise Diária")
tab_funnel, tab_kpis = st.tabs(["Funil", "KPIs Financeiros"])

with tab_funnel:
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["link_clicks"], name="Cliques", mode='lines+markers'))
    fig_daily.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["lpv"], name="LPV", mode='lines+markers'))
    fig_daily.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["init_checkout"], name="Checkout", mode='lines+markers'))
    fig_daily.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["purchases"], name="Compras", mode='lines+markers'))
    fig_daily.update_layout(title="Funil Diário (números absolutos)", template="plotly_white")
    st.plotly_chart(fig_daily, use_container_width=True)

    fig_rates = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rates.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["purchases"], name="Compras", mode='lines+markers'))
    fig_rates.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["lpv"]/df_daily["link_clicks"], name="Taxa LPV/Clique", mode='lines+markers', yaxis='y2'))
    fig_rates.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["init_checkout"]/df_daily["lpv"], name="Taxa Checkout/LPV", mode='lines+markers', yaxis='y2'))
    fig_rates.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["purchases"]/df_daily["init_checkout"], name="Taxa Compra/Checkout", mode='lines+markers', yaxis='y2'))
    fig_rates.update_layout(title="Taxas de Funil Diárias", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_rates.update_yaxes(title_text="Compras", secondary_y=False)
    fig_rates.update_yaxes(title_text="Taxa de Conversão", secondary_y=True, tickformat=".1%")
    st.plotly_chart(fig_rates, use_container_width=True)

with tab_kpis:
    fig_kpis = go.Figure()
    fig_kpis.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["spend"], name="Custo", mode='lines+markers'))
    fig_kpis.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["revenue"], name="Receita", mode='lines+markers'))
    fig_kpis.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["roas"], name="ROAS", mode='lines+markers', yaxis='y2'))
    fig_kpis.update_layout(title="KPIs Financeiros Diários", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_kpis.update_yaxes(title_text="Custo / Receita (R$)", secondary_y=False)
    fig_kpis.update_yaxes(title_text="ROAS", secondary_y=True, tickformat=".2f")
    st.plotly_chart(fig_kpis, use_container_width=True)

# --- Análise de breakdowns ---
st.markdown("---")
st.header("Análise por Dimensão")
breakdown_opts = ["age", "gender", "publisher_platform", "country", "device_platform"]
dimensao = st.selectbox("Selecione uma dimensão para análise", breakdown_opts, key="breakdown_select")

try:
    with st.spinner(f"Buscando dados por {dimensao}..."):
        breakdown_df = fetch_insights_breakdown(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(since),
            until_str=str(until),
            breakdowns=[dimensao],
            level=level,
            product_name=st.session_state.selected_product
        )
except Exception as e:
    st.error(f"Erro ao buscar dados por dimensão: {e}")
    breakdown_df = pd.DataFrame()

if not breakdown_df.empty:
    agg_df = breakdown_df.groupby(dimensao).sum(numeric_only=True).reset_index()
    agg_df["CPA"] = _safe_div(agg_df["spend"], agg_df["purchases"])
    agg_df["ROAS"] = _safe_div(agg_df["revenue"], agg_df["spend"])

    tab_kpis_b, tab_funnel_b = st.tabs(["KPIs", "Funil"])

    with tab_kpis_b:
        st.subheader(f"Desempenho por {dimensao}")
        cols_to_display = [
            dimensao,
            "purchases",
            "ROAS",
            "spend",
            "revenue",
            "CPA"
        ]
        disp_kpis = agg_df[cols_to_display].rename(columns={
            "purchases": "Compras",
            "ROAS": "ROAS",
            "spend": "Custo",
            "revenue": "Receita",
            "CPA": "CPA"
        }).sort_values(by="Compras", ascending=False).set_index(dimensao)
        st.dataframe(disp_kpis, use_container_width=True)

    with tab_funnel_b:
        st.subheader(f"Funil por {dimensao}")
        cols_to_display = [
            dimensao,
            "link_clicks",
            "lpv",
            "init_checkout",
            "add_payment",
            "purchases"
        ]
        disp_funnel = agg_df[cols_to_display].rename(columns={
            "link_clicks": "Cliques",
            "lpv": "LPV",
            "init_checkout": "Checkout",
            "add_payment": "Add Pagto",
            "purchases": "Compras"
        }).sort_values(by="Cliques", ascending=False).set_index(dimensao)
        st.dataframe(disp_funnel, use_container_width=True)

        st.subheader(f"Comparativo de KPIs por {dimensao}")
        metric_to_plot = st.selectbox(
            "Selecione uma métrica para o gráfico de barras",
            ["Compras", "ROAS", "Custo", "Receita", "Cliques", "LPV"]
        )
        data = disp_kpis if metric_to_plot in disp_kpis.columns else disp_funnel
        _bar_chart(data.index, data[metric_to_plot], f"{metric_to_plot} por {dimensao}", dimensao, metric_to_plot)

# --- Análise de dayparting (hora do dia) ---
st.markdown("---")
st.header("Análise por Hora do Dia (Dayparting)")
try:
    with st.spinner("Buscando dados por hora..."):
        df_hourly = fetch_insights_hourly(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(since),
            until_str=str(until),
            level=level
        )
except Exception as e:
    st.error(f"Erro ao buscar dados por hora: {e}")
    df_hourly = pd.DataFrame()

if not df_hourly.empty:
    agg_hourly = df_hourly.groupby("hour").sum(numeric_only=True).reset_index()
    agg_hourly["ROAS"] = _safe_div(agg_hourly["revenue"], agg_hourly["spend"])
    
    st.subheader("Dayparting Geral (Médias por Hora)")
    
    fig_hourly_kpis = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hourly_kpis.add_trace(go.Bar(x=agg_hourly["hour"], y=agg_hourly["spend"], name="Custo"), secondary_y=False)
    fig_hourly_kpis.add_trace(go.Bar(x=agg_hourly["hour"], y=agg_hourly["purchases"], name="Compras"), secondary_y=False)
    fig_hourly_kpis.add_trace(go.Scatter(x=agg_hourly["hour"], y=agg_hourly["ROAS"], name="ROAS", mode="lines+markers"), secondary_y=True)
    fig_hourly_kpis.update_layout(
        title="Desempenho por Hora do Dia",
        template="plotly_white",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_hourly_kpis.update_yaxes(title_text="Custo / Compras", secondary_y=False)
    fig_hourly_kpis.update_yaxes(title_text="ROAS", secondary_y=True)
    st.plotly_chart(fig_hourly_kpis, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Matriz de Calor: Compras por Dia da Semana e Hora")
    
    pivot_table_purchases = df_hourly.pivot_table(
        index="dow_label", columns="hour", values="purchases", aggfunc="sum"
    ).fillna(0)
    
    days_order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    pivot_table_purchases = pivot_table_purchases.reindex(days_order)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table_purchases.values,
        x=pivot_table_purchases.columns,
        y=pivot_table_purchases.index,
        colorscale='Viridis',
        colorbar=dict(title="Compras"),
        hovertemplate="Dia da semana: %{y}<br>Hora: %{x}:00<br>Compras: %{z}<extra></extra>"
    ))
    fig_heatmap.update_layout(
        title="Compras por Hora e Dia da Semana",
        xaxis_title="Hora do Dia",
        yaxis_title="Dia da Semana",
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
