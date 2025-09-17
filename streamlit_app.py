import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
APP_TZ = ZoneInfo("America/Sao_Paulo")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

_session = None
def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session


# =============== Config & Estilos ===============
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
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]

# --- Constantes e parser para breakdown por hora
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

# =============== Helpers genéricos ===============
def _retry_call(fn, max_retries=5, base_wait=1.2):
    """Executa uma função com backoff exponencial para erros/transientes comuns."""
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
        uniformtext=dict(minsize=12, mode="show")
    )
    return fig

def _pick_checkout_totals(rows, allowed_keys=None) -> float:
    """
    Soma Initiate Checkout priorizando omni; senão pega o MAIOR entre variantes
    (sem duplicar janelas). Aceita initiated/initiate e offsite/onsite.
    """
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]

    # 1) preferir omni
    omni = [r for r in rows if r["action_type"] in ("omni_initiated_checkout","omni_initiate_checkout")]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))

    # 2) variantes conhecidas
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

    # 3) fallback amplo por substring
    grp = {}
    for r in rows:
        if "initiate" in r["action_type"] and "checkout" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def _pick_add_payment_totals(rows, allowed_keys=None) -> float:
    """
    Soma Add Payment Info com suporte a omni/onsite/offsite.
    """
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

def _pct(a, b):
    a = float(a or 0); b = float(b or 0)
    return (a / b) if b > 0 else np.nan

def _intensity_label(share):
    # share = fração do maior drop sobre a soma de todos os drops (>0)
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
    # saúde para Escala
    healthy = (
        (pd.notnull(r1) and r1 >= bm_r1/100.0) and
        (pd.notnull(r2) and r2 >= bm_r2/100.0) and
        (pd.notnull(r3) and r3 >= bm_r3/100.0) and
        (float(purch or 0) >= float(min_purch or 0))
    )
    low_volume_guard = (float(clicks or 0) < float(min_clicks or 0)) or \
                       (float(lpv or 0)    < float(min_lpv or 0))    or \
                       (float(co or 0)     < float(min_co or 0))

    # drops por volume
    drop1 = max(0.0, float(clicks or 0) - float(lpv or 0))        # Criativo
    drop2 = max(0.0, float(lpv or 0)    - float(co or 0))         # Interesse
    drop3a= max(0.0, float(co or 0)     - float(addpay or 0))     # RMK (checkout->pagto)
    drop3b= max(0.0, float(addpay or 0) - float(purch or 0))      # RMK (pagto->compra)

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
    """Divide [since, until] em janelas de até max_days (inclusive)."""
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days - 1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)

# --- Filtro simples por produto no nome da campanha ---
def _filter_by_product(df: pd.DataFrame, produto: str) -> pd.DataFrame:
    """
    Filtra pelo nome do produto dentro de campaign_name.
    Se produto == "(Todos)", retorna o df original.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not produto or produto == "(Todos)":
        return df
    mask = df["campaign_name"].str.contains(produto, case=False, na=False)
    return df[mask].copy()


# =============== Coleta (com fallback de campos extras) ===============
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True,
                         product_name: str | None = None) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único ('campaign' recomendado)
    - Usa action_report_time=conversion e action_attribution_windows fixos (paridade com Ads Manager)
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
    - Paraleliza chunks de 30d e usa requests.Session para keep-alive.
    - Opcional: filtering por nome da campanha (product_name) direto na API.
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
        # filtro por campanha (produto) direto na API, quando aplicável
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
                    # refaz sem extras só para ESTA janela
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

                ic  = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                revenue_val   = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows_local.append({
                    "date":           pd.to_datetime(rec.get("date_start")),
                    "currency":       rec.get("account_currency", "BRL"),
                    "campaign_id":    rec.get("campaign_id", ""),
                    "campaign_name":  rec.get("campaign_name", ""),
                    "spend":          _to_float(rec.get("spend")),
                    "impressions":    _to_float(rec.get("impressions")),
                    "clicks":         _to_float(rec.get("clicks")),
                    "link_clicks":    _to_float(link_clicks),
                    "lpv":            _to_float(lpv),
                    "init_checkout":  _to_float(ic),
                    "add_payment":    _to_float(api),
                    "purchases":      _to_float(purchases_cnt),
                    "revenue":        _to_float(revenue_val),
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

    # Busca em paralelo (3-5 workers é seguro)
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

# --- Coleta por hora (dayparting)
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(act_id: str, token: str, api_version: str,
                          since_str: str, until_str: str,
                          level: str = "campaign") -> pd.DataFrame:
    """
    Coleta por hora em janelas menores (default 30 dias) para evitar code=1
    e concatena o resultado. Tenta 'conversion' por chunk; se falhar, tenta 'impression'.
    """
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

                ic   = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur  = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev  = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows.append({
                    "date":          pd.to_datetime(rec.get("date_start")),
                    "hour":          hour_bucket,
                    "currency":      rec.get("account_currency", "BRL"),
                    "campaign_id":   rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
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

    # Agrega chunks menores (use 30 dias por segurança no hourly)
    dfs = []
    for s_chunk, u_chunk in _chunks_by_days(since_str, until_str, max_days=30):
        try:
            df_chunk = _fetch_range(s_chunk, u_chunk, "conversion")
        except Exception:
            # fallback silencioso por chunk
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
    Coleta insights com 1 ou 2 breakdowns.
    Mantém paridade (conversion + ATTR_KEYS), chunking 30d, requests.Session e paralelismo.
    Opcional: filtering por campanha (product_name) direto na API.
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
                ic   = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur  = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev  = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                base = {
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

    # 🔑 Junta em blocos de até 30 dias — em paralelo
    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
    return df

# =============== Sidebar (filtros) ===============
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign"],  index=0)

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
    since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom", format="DD/MM/YYYY")
    until = st.sidebar.date_input("Até",   value=_until_auto, key="until_custom", format="DD/MM/YYYY")
else:
    # ✅ NÃO usar date_input aqui (evita estado preso)
    since, until = _since_auto, _until_auto
    st.sidebar.caption(f"**Desde:** {since}  \n**Até:** {until}")

ready = bool(act_id and token)

# =============== Tela ===============
st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra. Tudo alinhado ao período selecionado.")

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
        product_name=st.session_state.get("daily_produto")  # pode ser None na primeira carga
    )

df_hourly = None  # será carregado apenas quando o usuário abrir a aba de horário

if df_daily.empty and (df_hourly is None or df_hourly.empty):
    st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
    st.stop()

tab_daily, tab_daypart, tab_detail = st.tabs(["📅 Visão diária", "⏱️ Horários (principal)", "📊 Detalhamento"])

# -------------------- ABA 1: VISÃO DIÁRIA --------------------
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

    # ---- Filtro por produto na Visão diária (AGORA DENTRO DA ABA) ----
    produto_sel_daily = st.selectbox(
        "Filtrar por produto (opcional)",
        ["(Todos)"] + PRODUTOS,
        key="daily_produto"
    )

    df_daily_view = _filter_by_product(df_daily, produto_sel_daily)

    if df_daily_view.empty:
        st.info("Sem dados para o produto selecionado nesse período.")
        st.stop()

    if produto_sel_daily != "(Todos)":
        st.caption(f"🔎 Filtrando por produto: **{produto_sel_daily}**")

    # ========= KPIs do período =========
    tot_spend = float(df_daily_view["spend"].sum())
    tot_purch = float(df_daily_view["purchases"].sum())
    tot_rev   = float(df_daily_view["revenue"].sum())
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
        roas_txt = _fmt_ratio_br(roas_g) if pd.notnull(roas_g) else "—"
        st.markdown('<div class="kpi-card"><div class="small-muted">ROAS</div>'
                    f'<div class="big-number">{roas_txt}</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # ========= Série diária =========
    st.subheader("Série diária — Investimento e Conversão")
    daily = df_daily_view.groupby("date", as_index=False)[["spend", "revenue", "purchases"]].sum()
    daily_pt = daily.rename(columns={"spend": "Gasto", "revenue": "Faturamento"})
    st.line_chart(daily_pt.set_index("date")[["Faturamento", "Gasto"]])
    st.caption("Linhas diárias de Receita e Gasto. Vendas na tabela abaixo.")

    # ========= FUNIL (Período) — FUNIL VISUAL =========
    st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")

    f_clicks = float(df_daily_view["link_clicks"].sum())
    f_lpv    = float(df_daily_view["lpv"].sum())
    f_ic     = float(df_daily_view["init_checkout"].sum())
    f_api    = float(df_daily_view["add_payment"].sum())
    f_pur    = float(df_daily_view["purchases"].sum())

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

    # ========= TAXAS POR DIA (variação diária — com banda saudável e resumo) =========
    st.markdown("### Taxas por dia — evolução e leitura guiada")

    with st.expander("Ajustes de exibição", expanded=True):
        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            min_clicks_day = st.slider("Ignorar dias com menos de X cliques", 0, 500, 30, 10)
            mark_weekends = st.checkbox("Marcar fins de semana no fundo", value=True)
            show_band = st.checkbox("Mostrar banda saudável (faixa alvo)", value=True)
        with col_cfg2:
            st.caption("Faixas saudáveis (%)")
            lpv_cli_low, lpv_cli_high = st.slider("LPV / Cliques", 0, 100, (70, 85), 1, key="tx_lpv_cli_band")
            co_lpv_low,  co_lpv_high  = st.slider("Checkout / LPV", 0, 100, (10, 20), 1, key="tx_co_lpv_band")
            buy_co_low,  buy_co_high  = st.slider("Compra / Checkout", 0, 100, (25, 40), 1, key="tx_buy_co_band")

    # agrega por dia
    daily_conv = (
        df_daily_view.groupby("date", as_index=False)[
            ["link_clicks","lpv","init_checkout","add_payment","purchases"]
        ].sum()
        .rename(columns={"link_clicks":"clicks","init_checkout":"checkout","add_payment":"addpay"})
    )

    # evita ruído (dias com pouquíssimos eventos)
    daily_conv = daily_conv[daily_conv["clicks"] >= min_clicks_day].copy()
    if daily_conv.empty:
        st.info("Sem dias suficientes após o filtro de cliques mínimos.")
    else:
        # taxas (frações 0–1)
        daily_conv["LPV/Cliques"]     = daily_conv.apply(lambda r: _safe_div(r["lpv"],       r["clicks"]),   axis=1)
        daily_conv["Checkout/LPV"]    = daily_conv.apply(lambda r: _safe_div(r["checkout"],  r["lpv"]),      axis=1)
        daily_conv["Compra/Checkout"] = daily_conv.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)

        ### ADD: calcula período anterior com mesma duração
        period_len = (until - since).days + 1
        prev_since = since - timedelta(days=period_len)
        prev_until = since - timedelta(days=1)

        df_prev = fetch_insights_daily(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(prev_since),
            until_str=str(prev_until),
            level=level,
            product_name=st.session_state.get("daily_produto")
        )

        daily_prev = pd.DataFrame()
        if df_prev is not None and not df_prev.empty:
            daily_prev = (
                df_prev.groupby("date", as_index=False)[
                    ["link_clicks","lpv","init_checkout","purchases"]
                ].sum()
                .rename(columns={"link_clicks":"clicks","init_checkout":"checkout"})
            )
            daily_prev["LPV/Cliques"]     = daily_prev.apply(lambda r: _safe_div(r["lpv"],       r["clicks"]),   axis=1)
            daily_prev["Checkout/LPV"]    = daily_prev.apply(lambda r: _safe_div(r["checkout"],  r["lpv"]),      axis=1)
            daily_prev["Compra/Checkout"] = daily_prev.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)


        def _fmt_pct_series(s):  # 0–1 -> 0–100
            return (s*100).round(2)

        # helper geral do gráfico
        # helper geral do gráfico
        def _line_pct_banded(df, col, lo_pct, hi_pct, title):
            import plotly.graph_objects as go

            x = df["date"]
            y = (df[col] * 100).round(2)

            def _status(v):
                if not pd.notnull(v):
                    return "sem"
                v_pct = float(v) * 100.0  # fração -> %
                if v_pct < lo_pct:
                    return "abaixo"
                if v_pct > hi_pct:
                    return "acima"
                return "dentro"

            status = df[col].map(_status).tolist()
            colors = [{"abaixo": "#dc2626", "dentro": "#16a34a", "acima": "#0ea5e9", "sem": "#9ca3af"}[s] for s in status]
            hover = [f"{title}<br>{d:%Y-%m-%d}<br>Taxa: {val:.2f}%" for d, val in zip(x, y.fillna(0))]

            fig = go.Figure()

            # banda saudável
            if show_band:
                fig.add_shape(
                    type="rect",
                    xref="x", yref="y",
                    x0=x.min(), x1=x.max(),
                    y0=lo_pct, y1=hi_pct,
                    fillcolor="rgba(34,197,94,0.08)", line=dict(width=0),
                    layer="below"
                )

            # fins de semana
            if mark_weekends:
                for d in x:
                    if d.weekday() >= 5:
                        fig.add_shape(
                            type="rect", xref="x", yref="paper",
                            x0=d, x1=d + pd.Timedelta(days=1),
                            y0=0, y1=1,
                            line=dict(width=0),
                            fillcolor="rgba(2,132,199,0.06)"
                        )

            # série
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                name="Diário",
                marker=dict(size=7, color=colors),
                line=dict(width=1.5, color="#1f77b4"),
                hovertext=hover, hoverinfo="text"
            ))

            # período anterior (se houver)
            if not daily_prev.empty and col in daily_prev.columns:
                x_aligned = df["date"].values[:len(daily_prev)]
                y_prev = (daily_prev[col] * 100).round(2)
                fig.add_trace(go.Scatter(
                    x=x_aligned, y=y_prev,
                    mode="lines",
                    name="Período anterior (sobreposto)",
                    line=dict(width=2.2, color="#ef4444", dash="dot")
                ))

            fig.update_layout(
                title=title,
                yaxis_title="%",
                xaxis_title="Data",
                height=340,
                template="plotly_white",
                margin=dict(l=10, r=10, t=48, b=10),
                separators=",.",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )

            # limites Y
            y_min = max(0, min(y.min(), lo_pct) - 5)
            y_max = min(100, max(y.max(), hi_pct) + 5)
            fig.update_yaxes(range=[y_min, y_max])

            # 🔧 hover e formato — DENTRO DA FUNÇÃO
            fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Taxa: %{y:.2f}%<extra></extra>")
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(hoverformat="%Y-%m-%d", showspikes=False)
            fig.update_yaxes(hoverformat=".2f%", ticksuffix="%", showspikes=False)

            return fig


        # ======== RESUMO DAS TAXAS (dinâmico por período) ========
        def _trend_vs_previous_period(series_vals: pd.Series,
                                      since_dt: date,
                                      until_dt: date,
                                      rate_name: str,
                                      produto_nome: str | None) -> tuple[float | None, float | None]:
            cur_mean = float(series_vals.mean() * 100.0) if series_vals.size else None
            period_len = (until_dt - since_dt).days + 1
            prev_since = since_dt - timedelta(days=period_len)
            prev_until = since_dt - timedelta(days=1)

            try:
                df_prev = fetch_insights_daily(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(prev_since),
                    until_str=str(prev_until),
                    level=level,
                    product_name=produto_nome
                )
                if df_prev is not None and not df_prev.empty:
                    prev = (
                        df_prev.groupby("date", as_index=False)[
                            ["link_clicks", "lpv", "init_checkout", "purchases"]
                        ]
                        .sum()
                        .rename(columns={"link_clicks": "clicks", "init_checkout": "checkout"})
                    )
                    if rate_name == "LPV/Cliques":
                        prev["rate"] = prev.apply(lambda r: _safe_div(r["lpv"], r["clicks"]), axis=1)
                    elif rate_name == "Checkout/LPV":
                        prev["rate"] = prev.apply(lambda r: _safe_div(r["checkout"], r["lpv"]), axis=1)
                    else:  # "Compra/Checkout"
                        prev["rate"] = prev.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)

                    prev_mean = float(prev["rate"].mean() * 100.0) if not prev.empty else None
                else:
                    prev_mean = None
            except Exception:
                prev_mean = None

            if prev_mean is not None and cur_mean is not None:
                return cur_mean, (cur_mean - prev_mean)

            # fallback: metade inicial vs. final do período atual
            n = series_vals.size
            if n == 0:
                return cur_mean, None
            mid = max(1, n // 2)
            first_mean = float(series_vals.iloc[:mid].mean() * 100.0)
            last_mean = float(series_vals.iloc[mid:].mean() * 100.0) if n - mid > 0 else first_mean
            return cur_mean, (last_mean - first_mean)

        def _resume_box(df_rates: pd.DataFrame,
                        col: str,
                        lo_pct: int,
                        hi_pct: int,
                        label: str) -> None:
            """Mostra: média do período, % de dias dentro da banda e tendência vs período anterior."""
            vals = df_rates[col].dropna()
            if vals.empty:
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric(label, "—")
                mcol2.metric("% dias dentro", "—")
                mcol3.metric("Tendência (período)", "—")
                return

            mean_pct = float(vals.mean() * 100.0)
            inside = float(((vals * 100.0 >= lo_pct) & (vals * 100.0 <= hi_pct)).mean() * 100.0)

            cur_mean, delta_pp = _trend_vs_previous_period(
                series_vals=vals,
                since_dt=since,
                until_dt=until,
                rate_name=label.split(" (")[0],  # tira o “(média)” do fim
                produto_nome=st.session_state.get("daily_produto")
            )

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric(label, f"{mean_pct:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
            mcol2.metric("% dias dentro", f"{inside:,.0f}%".replace(",", "X").replace(".", ",").replace("X", "."))
            if delta_pp is None:
                mcol3.metric("Tendência (período)", "—")
            else:
                delta_txt = f"{delta_pp:,.2f} pp".replace(",", "X").replace(".", ",").replace("X", ".")
                mcol3.metric("Tendência (período)", ("+" if delta_pp >= 0 else "") + delta_txt)

        # === Chame as três caixinhas de resumo ===
        st.markdown("**Resumo das taxas (período filtrado)**")
        _resume_box(daily_conv, "LPV/Cliques",     lpv_cli_low, lpv_cli_high, "LPV/Cliques (média)")
        _resume_box(daily_conv, "Checkout/LPV",    co_lpv_low,  co_lpv_high,  "Checkout/LPV (média)")
        _resume_box(daily_conv, "Compra/Checkout", buy_co_low,  buy_co_high,  "Compra/Checkout (média)")

        st.markdown("---")

        # === Gráficos lado a lado (mantém como estava) ===
        left, mid, right = st.columns(3)
        with left:
            st.plotly_chart(
                _line_pct_banded(daily_conv, "LPV/Cliques", lpv_cli_low, lpv_cli_high, "LPV/Cliques"),
                use_container_width=True
            )
        with mid:
            st.plotly_chart(
                _line_pct_banded(daily_conv, "Checkout/LPV", co_lpv_low, co_lpv_high, "Checkout/LPV"),
                use_container_width=True
            )
        with right:
            st.plotly_chart(
                _line_pct_banded(daily_conv, "Compra/Checkout", buy_co_low, buy_co_high, "Compra/Checkout"),
                use_container_width=True
            )

        st.caption(
            "Leitura: pontos **verdes** estão dentro da banda saudável, **vermelhos** abaixo e **azuis** acima. "
            "A área verde mostra o alvo; a linha vermelha tracejada mostra o **período anterior** para comparação. "
            "Fins de semana ganham fundo azul claro (opcional)."
        )

    # === NOTIFICAÇÃO DIDÁTICA DE ALOCAÇÃO DE VERBA =================================
    st.subheader("🔔 Para onde vai a verba? (recomendação automática)")

    # usa as MESMAS faixas saudáveis definidas no bloco "Taxas por dia"
    # lpv_cli_low/high, co_lpv_low/high, buy_co_low/high
    # adiciona apenas o volume mínimo para liberar "Escala"
    min_purchases_to_scale = st.number_input(
        "Compras mínimas para sugerir Escala (volume)",
        min_value=0, value=50, step=1
    )

    # taxas do período (a partir do funil total já calculado)
    r1 = _safe_div(values_total[1], values_total[0])   # LPV/Cliques
    r2 = _safe_div(values_total[2], values_total[1])   # Checkout/LPV
    r3 = _safe_div(values_total[4], values_total[2])   # Compra/Checkout

    # quedas absolutas por etapa (onde as pessoas “somem”)
    drop1 = max(0, values_total[0] - values_total[1])  # Cliques -> LPV (Criativo/LP)
    drop2 = max(0, values_total[1] - values_total[2])  # LPV -> Checkout (Interesse/Oferta)
    drop3 = max(0, values_total[2] - values_total[4])  # Checkout -> Compra (RMK/Pagamento)

    # helpers de status e forma de exibir “chips”
    def _band_status(val, lo, hi):
        if not pd.notnull(val):
            return "sem_dado"
        v = val * 100
        if v < lo:
            return "abaixo"
        if v > hi:
            return "acima"
        return "dentro"

    def _chip(label, val, lo, hi):
        status = _band_status(val, lo, hi)
        if status == "abaixo":
            return f"❌ **{label}** — {_fmt_pct_br(val)} (alvo {lo}–{hi}%)"
        if status == "dentro":
            return f"✅ **{label}** — {_fmt_pct_br(val)} (dentro de {lo}–{hi}%)"
        if status == "acima":
            return f"🟢 **{label}** — {_fmt_pct_br(val)} (acima de {hi}%)"
        return f"⛔ **{label}** — sem dados suficientes"

    # mapa didático das etapas (reaproveitando faixas)
    stages = {
        "Teste de criativo": {
            "rate": r1, "lo": lpv_cli_low, "hi": lpv_cli_high, "drop": drop1,
            "explain": "Perda entre Cliques → LPV (qualidade do clique, criativo, velocidade e UX da landing).",
            "todo": [
                "Testar variações de criativo (ângulo, thumb, 3s iniciais, CTA).",
                "Melhorar tempo de carregamento e primeira dobra da LP.",
                "Revisar promessa/título para alinhar com o anúncio."
            ]
        },
        "Teste de interesse": {
            "rate": r2, "lo": co_lpv_low, "hi": co_lpv_high, "drop": drop2,
            "explain": "Perda entre LPV → Checkout (público/segmentação e proposta de valor).",
            "todo": [
                "Refinar públicos/lookalikes e excluir desinteressados.",
                "Evidenciar prova social e benefícios acima do CTA.",
                "Harmonizar oferta (preço/parcelas/bundle) com o público certo."
            ]
        },
        "Remarketing": {
            "rate": r3, "lo": buy_co_low, "hi": buy_co_high, "drop": drop3,
            "explain": "Perda entre Checkout → Compra (confiança, meios de pagamento, follow-up).",
            "todo": [
                "RMK dinâmico com objeções, frete e garantia claros.",
                "Oferecer alternativas de pagamento (pix/boleto/parcelas).",
                "Recuperar carrinhos (e-mail/SMS/Whats) em até 24h."
            ]
        }
    }

    # decide foco principal
    abaixos = {k: v for k, v in stages.items() if _band_status(v["rate"], v["lo"], v["hi"]) == "abaixo"}

    if abaixos:
        # se >1 abaixo, escolhe onde há maior queda absoluta de pessoas
        foco, foco_dat = max(abaixos.items(), key=lambda kv: kv[1]["drop"])
    else:
        total_purch = values_total[4]
        todas_ok = all(_band_status(v["rate"], v["lo"], v["hi"]) in ["dentro", "acima"] for v in stages.values())
        if todas_ok and total_purch >= min_purchases_to_scale:
            foco, foco_dat = "Escala", {
                "rate": None, "lo": None, "hi": None, "drop": 0,
                "explain": "Taxas saudáveis e volume suficiente. Hora de aumentar alcance nas melhores campanhas."
            }
        else:
            # sem crítica clara: sugerir ganho de volume onde a queda é maior
            foco, foco_dat = max(stages.items(), key=lambda kv: kv[1]["drop"])

    # intensidade (ajuda a sugerir % de verba)
    total_drop = max(1, drop1 + drop2 + drop3)  # evita divisão por zero
    share = foco_dat["drop"] / total_drop
    if share > 0.60:
        intensidade = "Alta"; faixa_verba = "↑ realocar **20–30%** do budget"
    elif share >= 0.30:
        intensidade = "Média"; faixa_verba = "↑ realocar **10–15%** do budget"
    else:
        intensidade = "Baixa"; faixa_verba = "↑ realocar **5–10%** do budget"

    # cartão-resumo
    st.markdown("---")
    colA, colB = st.columns([1, 2])

    with colA:
        st.markdown("**Taxas do período**")
        st.markdown(_chip("LPV/Cliques", r1, lpv_cli_low, lpv_cli_high))
        st.markdown(_chip("Checkout/LPV", r2, co_lpv_low,  co_lpv_high))
        st.markdown(_chip("Compra/Checkout", r3, buy_co_low,  buy_co_high))

    with colB:
        if foco == "Escala":
            st.success(
                f"**✅ Recomendação: Escala**\n\n"
                f"- Motivo: {foco_dat['explain']}\n"
                f"- Compras no período: **{_fmt_int_br(values_total[4])}** "
                f"(mín. para escalar: **{_fmt_int_br(min_purchases_to_scale)}**)\n"
                f"- Ação: aumentar orçamento nas campanhas com melhor ROAS; manter horários e públicos vencedores."
            )
        else:
            st.warning(
                f"**⚠️ Recomendação: {foco}**\n\n"
                f"- Motivo: {foco_dat['explain']}\n"
                f"- Queda concentrada nessa etapa: **{_fmt_int_br(foco_dat['drop'])}** pessoas "
                f"(intensidade **{intensidade}** → {faixa_verba})."
            )
            st.markdown("**O que fazer agora**")
            for tip in foco_dat["todo"]:
                st.markdown(f"- {tip}")

    # ajuda didática
    with st.expander("ℹ️ Como interpretar"):
        st.markdown(
            """
- **LPV/Cliques** baixo → **Criativo/LP** (as pessoas clicam mas não engajam na página).
- **Checkout/LPV** baixo → **Interesse/Oferta** (as pessoas veem, mas não avançam).
- **Compra/Checkout** baixo → **Remarketing/Pagamento** (travou na finalização).
- Se tudo está saudável **e** há volume de compras → **Escala**.
            """
        )

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
                dfA = fetch_insights_daily(
                    act_id, token, api_version, str(sinceA), str(untilA), level,
                    product_name=produto_sel_daily
                )
                dfB = fetch_insights_daily(
                    act_id, token, api_version, str(sinceB), str(untilB), level,
                    product_name=produto_sel_daily
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
        camp = df_daily_view.groupby(["campaign_id", "campaign_name"], as_index=False)[agg_cols].sum()

        # (resto do bloco de campanhas igual ao seu, já usando df_daily_view)
        # ...
    else:
        st.info("Troque o nível para 'campaign' para ver o detalhamento por campanha.")
        
with tab_daypart:
    st.caption("Explore desempenho por hora: Heatmap no topo, depois comparação de dias e apanhado geral.")

    level_hourly = "campaign"

    # 2) Cache por chave (granularidade + período)
    cache = st.session_state.setdefault("hourly_cache", {})
    hourly_key = (act_id, api_version, level_hourly, str(since), str(until))

    # 3) Lazy-load: só busca quando precisa e guarda no cache
    if df_hourly is None or hourly_key not in cache:
        with st.spinner("Carregando breakdown por hora…"):
            cache[hourly_key] = fetch_insights_hourly(
                act_id=act_id, token=token, api_version=api_version,
                since_str=str(since), until_str=str(until), level=level_hourly
            )
    df_hourly = cache[hourly_key]

    # --------- Filtro por produto (opcional) ---------
    produto_sel_hr = st.selectbox(
        "Filtrar por produto (opcional)",
        ["(Todos)"] + PRODUTOS,
        key="daypart_produto"
    )

    # Guard: checa vazio antes de usar
    if df_hourly is None or df_hourly.empty:
        st.info("A conta/período não retornou breakdown por hora. Use a visão diária.")
        st.stop()

    # Agora aplicamos o filtro por produto no campaign_name
    d = df_hourly.copy()
    if produto_sel_hr != "(Todos)":
        mask_hr = d["campaign_name"].str.contains(produto_sel_hr, case=False, na=False)
        d = d[mask_hr].copy()

    # Slider de gasto mínimo
    min_spend = st.slider(
        "Gasto mínimo para considerar o horário (R$)",
        0.0, 1000.0, 0.0, 10.0
    )

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

    metric_hm = st.selectbox("Métrica para o heatmap", ["Compras","Faturamento","Gasto","ROAS"], index=0, key="hm_metric_top")
    mcol_hm = {"Compras":"purchases","Faturamento":"revenue","Gasto":"spend","ROAS":"roas"}[metric_hm]

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
    fig_hm.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        separators=",."
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ============== 3) APANHADO GERAL POR HORA (período) ==============
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
        xaxis_title="Hora do dia",
        yaxis_title="Compras",
        height=380,
        template="plotly_white",
        margin=dict(l=10, r=10, t=48, b=10),
        separators=",."
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos.")

    # ============== 2) TAXAS POR HORA (gráficos em cima + tabela de quantidades abaixo) ==============
    st.subheader("🎯 Taxas por hora — médias diárias (sinais puros, com cap de funil)")

    # --- Base SEM filtro de gasto: somas por hora no período selecionado
    cube_hr_all = d.groupby("hour", as_index=False)[
        ["link_clicks", "lpv", "init_checkout", "add_payment", "purchases"]
    ].sum()

    # Garante presença de 0..23 mesmo que falte alguma hora
    _hours_full = list(range(24))
    cube_hr_all = (
        cube_hr_all.set_index("hour")
                   .reindex(_hours_full, fill_value=0.0)
                   .rename_axis("hour")
                   .reset_index()
    )

    # ---- Cap de funil para evitar taxas >100%
    cube_hr_all["LPV_cap"] = np.minimum(cube_hr_all["lpv"], cube_hr_all["link_clicks"])
    cube_hr_all["Checkout_cap"] = np.minimum(cube_hr_all["init_checkout"], cube_hr_all["LPV_cap"])

    # ---- Taxas (frações 0–1) — taxa da HORA (instantânea)
    cube_hr_all["tx_lpv_clicks"]      = cube_hr_all.apply(lambda r: _safe_div(r["LPV_cap"],      r["link_clicks"]),  axis=1)
    cube_hr_all["tx_checkout_lpv"]    = cube_hr_all.apply(lambda r: _safe_div(r["Checkout_cap"], r["LPV_cap"]),      axis=1)
    cube_hr_all["tx_compra_checkout"] = cube_hr_all.apply(lambda r: _safe_div(r["purchases"],    r["Checkout_cap"]), axis=1)

    # ---- Linha cumulativa (até a hora) — soma horas para trás e calcula a taxa no acumulado
    show_cum = st.checkbox("Mostrar linha cumulativa (até a hora)", value=True, key="hr_show_cum")
    cum = cube_hr_all.sort_values("hour").copy()
    cum["cum_clicks"] = cum["link_clicks"].cumsum()
    cum["cum_lpv"]    = cum["lpv"].cumsum()
    cum["cum_ic"]     = cum["init_checkout"].cumsum()
    cum["cum_purch"]  = cum["purchases"].cumsum()

    # cap no ACUMULADO
    cum["LPV_cap_cum"]      = np.minimum(cum["cum_lpv"], cum["cum_clicks"])
    cum["Checkout_cap_cum"] = np.minimum(cum["cum_ic"],  cum["LPV_cap_cum"])

    # taxas cumulativas (frações 0–1)
    tx_lpv_clicks_cum      = np.divide(cum["LPV_cap_cum"],      cum["cum_clicks"],      out=np.full(len(cum), np.nan), where=cum["cum_clicks"]>0)
    tx_checkout_lpv_cum    = np.divide(cum["Checkout_cap_cum"], cum["LPV_cap_cum"],     out=np.full(len(cum), np.nan), where=cum["LPV_cap_cum"]>0)
    tx_compra_checkout_cum = np.divide(cum["cum_purch"],        cum["Checkout_cap_cum"],out=np.full(len(cum), np.nan), where=cum["Checkout_cap_cum"]>0)

    # =================== CONTROLES — banda saudável ===================
    def _get_band_from_state(key, default_pair):
        v = st.session_state.get(key)
        return v if (isinstance(v, tuple) and len(v) == 2) else default_pair

    # tenta herdar as faixas que você já definiu na aba diária, senão usa defaults
    _lpv_lo_def, _lpv_hi_def = _get_band_from_state("tx_lpv_cli_band", (70, 85))
    _co_lo_def,  _co_hi_def  = _get_band_from_state("tx_co_lpv_band",  (10, 20))
    _buy_lo_def, _buy_hi_def = _get_band_from_state("tx_buy_co_band",  (25, 40))

    with st.expander("Ajustes de exibição das bandas (opcional)", expanded=True):
        show_band_hour = st.checkbox("Mostrar banda saudável (faixa alvo)", value=True, key="hr_show_band")
        b1, b2, b3 = st.columns(3)
        with b1:
            lpv_cli_low, lpv_cli_high = st.slider("LPV/Cliques alvo (%)", 0, 100, (_lpv_lo_def, _lpv_hi_def), 1, key="hr_band_lpv_clicks")
        with b2:
            co_lpv_low,  co_lpv_high  = st.slider("Checkout/LPV alvo (%)", 0, 100, (_co_lo_def,  _co_hi_def),  1, key="hr_band_checkout_lpv")
        with b3:
            buy_co_low,  buy_co_high  = st.slider("Compra/Checkout alvo (%)", 0, 100, (_buy_lo_def, _buy_hi_def), 1, key="hr_band_buy_checkout")

    # =================== INDICADORES — médias do período (com cap) ===================
    st.markdown("### Resumo das taxas (período filtrado)")

    # pega o último ponto do acumulado do dia (representa o período inteiro)
    _last = cum.iloc[-1]
    _clicks_tot   = float(_last["cum_clicks"])
    _lpv_cap_tot  = float(_last["LPV_cap_cum"])
    _chk_cap_tot  = float(_last["Checkout_cap_cum"])
    _purch_tot    = float(_last["cum_purch"])

    # médias do período (frações 0–1) com cap no acumulado
    avg_lpv_clicks   = _safe_div(_lpv_cap_tot, _clicks_tot)
    avg_chk_lpv      = _safe_div(_chk_cap_tot, _lpv_cap_tot)
    avg_buy_checkout = _safe_div(_purch_tot,   _chk_cap_tot)

    # formato %
    def _pct(x): 
        return "–" if (x is None or np.isnan(x)) else f"{x*100:,.2f}%"

    # cards bonitos tipo o exemplo (escuros e com número grande)
    card_css = """
    <style>
    .kpi-wrap {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; margin: 6px 0 12px;}
    .kpi {
      background: #0f172a; /* slate-900 */
      border: 1px solid #1f2937; /* gray-800 */
      border-radius: 12px; padding: 14px 16px;
    }
    .kpi h4 {margin: 0 0 6px; font-size: 0.92rem; color: #cbd5e1; font-weight: 600;}
    .kpi .val {font-size: 2rem; font-weight: 700; color: #ffffff; letter-spacing: .2px;}
    @media (max-width: 900px){ .kpi-wrap{grid-template-columns:1fr;}}
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    cards_html = f"""
    <div class="kpi-wrap">
      <div class="kpi">
        <h4>LPV/Cliques (média)</h4>
        <div class="val">{_pct(avg_lpv_clicks)}</div>
      </div>
      <div class="kpi">
        <h4>Checkout/LPV (média)</h4>
        <div class="val">{_pct(avg_chk_lpv)}</div>
      </div>
      <div class="kpi">
        <h4>Compra/Checkout (média)</h4>
        <div class="val">{_pct(avg_buy_checkout)}</div>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # =================== GRÁFICOS — 3 linhas de TAXAS (%) (EM CIMA) ===================
    def _line_hour_pct(x, y, title, band_range=None, show_band=False, y_aux=None, aux_label="Cumulativa"):
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="lines+markers", name=title,
            hovertemplate=f"<b>{title}</b><br>Hora: %{{x}}h<br>Taxa: %{{y:.2f}}%<extra></extra>"
        ))
        # banda saudável (faixa alvo)
        if show_band and band_range and len(band_range) == 2:
            lo, hi = band_range
            # retângulo de -0.5 a 23.5 para cobrir o eixo inteiro
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=-0.5, x1=23.5, y0=lo, y1=hi,
                fillcolor="rgba(34,197,94,0.10)", line=dict(width=0), layer="below"
            )
        # linha amarela cumulativa (até a hora)
        if show_cum and y_aux is not None:
            fig.add_trace(go.Scatter(
                x=x, y=y_aux, mode="lines", name=f"{aux_label}",
                line=dict(width=3, color="#f59e0b")
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Hora do dia",
            yaxis_title="Taxa (%)",
            height=340,
            template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10),
            separators=",.",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        return fig

    x_hours = cube_hr_all["hour"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            _line_hour_pct(
                x_hours, cube_hr_all["tx_lpv_clicks"]*100,
                "LPV/Cliques (%)",
                band_range=(lpv_cli_low, lpv_cli_high),
                show_band=show_band_hour,
                y_aux=tx_lpv_clicks_cum*100, aux_label="Cumulativa (até a hora)"
            ),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            _line_hour_pct(
                x_hours, cube_hr_all["tx_checkout_lpv"]*100,
                "Checkout/LPV (%)",
                band_range=(co_lpv_low, co_lpv_high),
                show_band=show_band_hour,
                y_aux=tx_checkout_lpv_cum*100, aux_label="Cumulativa (até a hora)"
            ),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            _line_hour_pct(
                x_hours, cube_hr_all["tx_compra_checkout"]*100,
                "Compra/Checkout (%)",
                band_range=(buy_co_low, buy_co_high),
                show_band=show_band_hour,
                y_aux=tx_compra_checkout_cum*100, aux_label="Cumulativa (até a hora)"
            ),
            use_container_width=True
        )

    st.markdown("---")

    # =================== TABELA — mostrar APENAS QUANTIDADES (EMBAIXO) ===================
    taxas_qtd = cube_hr_all[[
        "hour", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases"
    ]].copy().rename(columns={
        "hour": "Hora",
        "link_clicks": "Cliques",
        "lpv": "LPV",
        "init_checkout": "Checkout",
        "add_payment": "Add Pagto",
        "purchases": "Compras",
    })
    st.caption("Contagens por hora no período selecionado")
    st.dataframe(taxas_qtd, use_container_width=True, height=360)


    # ============== 4) COMPARAR DOIS PERÍODOS (A vs B) — HORA A HORA ==============
    st.subheader("🆚 Comparar dois períodos (A vs B) — hora a hora")

    # Defaults: B = período atual (since/until), A = período anterior com mesma duração
    base_len = (until - since).days + 1
    default_sinceA = (since - timedelta(days=base_len))
    default_untilA = (since - timedelta(days=1))

    colA1, colA2, colB1, colB2 = st.columns(4)
    with colA1:
        period_sinceA = st.date_input("Desde (A)", value=default_sinceA, key="cmp_sinceA")
    with colA2:
        period_untilA = st.date_input("Até (A)", value=default_untilA, key="cmp_untilA")
    with colB1:
        period_sinceB = st.date_input("Desde (B)", value=since, key="cmp_sinceB")
    with colB2:
        period_untilB = st.date_input("Até (B)", value=until, key="cmp_untilB")

    # Validação rápida
    if period_sinceA > period_untilA or period_sinceB > period_untilB:
        st.warning("Confira as datas: em cada período, 'Desde' não pode ser maior que 'Até'.")
    else:
        # Buscar dados por hora cobrindo A ∪ B
        union_since = min(period_sinceA, period_sinceB)
        union_until = max(period_untilA, period_untilB)

        level_union = "campaign"

        with st.spinner("Carregando dados por hora dos períodos selecionados…"):
            df_hourly_union = fetch_insights_hourly(
                act_id=act_id, token=token, api_version=api_version,
                since_str=str(union_since), until_str=str(union_until), level=level_union
            )

        # aplica o filtro de produto no union (se houver)
        if df_hourly_union is not None and not df_hourly_union.empty and produto_sel_hr != "(Todos)":
            mask_union = df_hourly_union["campaign_name"].str.contains(produto_sel_hr, case=False, na=False)
            df_hourly_union = df_hourly_union[mask_union].copy()

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

                    # ==================== CORES ====================
                    COLOR_SPEND   = "#E74C3C"  # vermelho (Gasto)
                    COLOR_REVENUE = "#3498DB"  # azul (Faturamento)
                    COLOR_LINE    = "#2ECC71"  # verde (Compras)

                    # ==================== GRÁFICO — Período A ====================
                    fig_A = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_A.add_trace(
                        go.Bar(
                            name="Gasto (A)",
                            x=x, y=merged["spend (A)"],
                            legendgroup="A",
                            offsetgroup="A",
                            marker_color=COLOR_SPEND,
                        )
                    )
                    fig_A.add_trace(
                        go.Bar(
                            name="Faturamento (A)",
                            x=x, y=merged["revenue (A)"],
                            legendgroup="A",
                            offsetgroup="A",
                            marker_color=COLOR_REVENUE,
                        )
                    )
                    fig_A.add_trace(
                        go.Scatter(
                            name=f"Compras (A) — {period_sinceA} a {period_untilA}",
                            x=x, y=merged["purchases (A)"],
                            mode="lines+markers",
                            legendgroup="A",
                            line=dict(color=COLOR_LINE, width=2),
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
                    fig_A.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
                    fig_A.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
                    fig_A.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])

                    st.plotly_chart(fig_A, use_container_width=True)

                    # ==================== GRÁFICO — Período B ====================
                    fig_B = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_B.add_trace(
                        go.Bar(
                            name="Gasto (B)",
                            x=x, y=merged["spend (B)"],
                            legendgroup="B",
                            offsetgroup="B",
                            marker_color=COLOR_SPEND,
                        )
                    )
                    fig_B.add_trace(
                        go.Bar(
                            name="Faturamento (B)",
                            x=x, y=merged["revenue (B)"],
                            legendgroup="B",
                            offsetgroup="B",
                            marker_color=COLOR_REVENUE,
                        )
                    )
                    fig_B.add_trace(
                        go.Scatter(
                            name=f"Compras (B) — {period_sinceB} a {period_untilB}",
                            x=x, y=merged["purchases (B)"],
                            mode="lines+markers",
                            legendgroup="B",
                            line=dict(color=COLOR_LINE, width=2),
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
                    fig_B.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
                    fig_B.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
                    fig_B.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])

                    st.plotly_chart(fig_B, use_container_width=True)

                    # ===== INSIGHTS — Período A =====
                    st.markdown("### 🔎 Insights — Período A")
                    a = merged.sort_values("hour").copy()
                    a_spend     = a["spend (A)"]
                    a_rev       = a["revenue (A)"]
                    a_purch     = a["purchases (A)"]
                    a_roas_ser  = np.where(a_spend > 0, a_rev / a_spend, np.nan)

                    a_tot_spend = float(a_spend.sum())
                    a_tot_rev   = float(a_rev.sum())
                    a_tot_purch = int(round(float(a_purch.sum())))
                    a_roas      = (a_tot_rev / a_tot_spend) if a_tot_spend > 0 else np.nan

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Valor usado (A)", _fmt_money_br(a_tot_spend))
                    c2.metric("Faturamento (A)", _fmt_money_br(a_tot_rev))
                    c3.metric("Vendas (A)", f"{a_tot_purch:,}".replace(",", "."))
                    c4.metric("ROAS (A)", _fmt_ratio_br(a_roas) if pd.notnull(a_roas) else "—")

                    h_best_purch = int(a.loc[a["purchases (A)"].idxmax(), "hour"]) if len(a_purch) and a_purch.max() > 0 else None
                    best_purch_val = int(a_purch.max()) if len(a_purch) else 0

                    mask_roasA = (a_spend >= float(min_spend)) & (a_spend > 0)
                    if mask_roasA.any():
                        roasA_vals = a_roas_ser.copy()
                        roasA_vals[~mask_roasA] = np.nan
                        h_best_roasA = int(a.loc[np.nanargmax(roasA_vals), "hour"])
                        best_roasA_val = float(np.nanmax(roasA_vals))
                    else:
                        h_best_roasA, best_roasA_val = None, np.nan

                    rollA = a_purch.rolling(3, min_periods=1).sum()
                    iA = int(rollA.idxmax()) if len(rollA) else 0
                    def _bA(ix): return int(a.loc[min(max(ix, 0), len(a)-1), "hour"])
                    winA_start, winA_mid, winA_end = _bA(iA-1), _bA(iA), _bA(iA+1)
                    winA_sum = int(rollA.max()) if len(rollA) else 0

                    wastedA = a[(a_spend > 0) & (a_purch == 0)]
                    wastedA_hours = ", ".join(f"{int(h)}h" for h in wastedA["hour"].tolist()) if not wastedA.empty else "—"

                    st.markdown(
                        f"""
**Pontos-chave (A)**  
- 🕐 **Pico de compras:** **{str(h_best_purch)+'h' if h_best_purch is not None else '—'}** ({best_purch_val} compras).  
- 💹 **Melhor ROAS** (gasto ≥ R$ {min_spend:,.0f}): **{(str(h_best_roasA)+'h') if h_best_roasA is not None else '—'}** ({_fmt_ratio_br(best_roasA_val) if pd.notnull(best_roasA_val) else '—'}).  
- ⏱️ **Janela forte (3h):** **{winA_start}–{winA_end}h** (centro {winA_mid}h) somando **{winA_sum}** compras.  
- 🧯 **Horas com gasto e 0 compras:** {wastedA_hours}.
""".replace(",", "X").replace(".", ",").replace("X", ".")
                    )

                    st.markdown("**Top 5 horas (A)**")
                    colTA, colTB = st.columns(2)
                    with colTA:
                        topA_p = a[["hour","purchases (A)","spend (A)","revenue (A)"]].sort_values("purchases (A)", ascending=False).head(5).copy()
                        topA_p.rename(columns={"hour":"Hora","purchases (A)":"Compras","spend (A)":"Valor usado","revenue (A)":"Valor de conversão"}, inplace=True)
                        topA_p["Valor usado"] = topA_p["Valor usado"].apply(_fmt_money_br)
                        topA_p["Valor de conversão"] = topA_p["Valor de conversão"].apply(_fmt_money_br)
                        st.dataframe(topA_p, use_container_width=True, height=220)
                    with colTB:
                        if mask_roasA.any():
                            topA_r = a[mask_roasA][["hour","spend (A)","revenue (A)"]].copy()
                            topA_r["ROAS"] = a_roas_ser[mask_roasA]
                            topA_r = topA_r.sort_values("ROAS", ascending=False).head(5)
                            topA_r.rename(columns={"hour":"Hora","spend (A)":"Valor usado","revenue (A)":"Valor de conversão"}, inplace=True)
                            topA_r["Valor usado"] = topA_r["Valor usado"].apply(_fmt_money_br)
                            topA_r["Valor de conversão"] = topA_r["Valor de conversão"].apply(_fmt_money_br)
                            topA_r["ROAS"] = topA_r["ROAS"].map(_fmt_ratio_br)
                        else:
                            topA_r = pd.DataFrame(columns=["Hora","Valor usado","Valor de conversão","ROAS"])
                        st.dataframe(topA_r, use_container_width=True, height=220)

                    st.info("Sugestões (A): priorize a janela forte, aumente orçamento nas horas de melhor ROAS (com gasto mínimo atendido) e reavalie criativo/lance nas horas com gasto e 0 compras.")
                    st.markdown("---")

                    # ===== INSIGHTS — Período B =====
                    st.markdown("### 🔎 Insights — Período B")
                    b = merged.sort_values("hour").copy()
                    b_spend     = b["spend (B)"]
                    b_rev       = b["revenue (B)"]
                    b_purch     = b["purchases (B)"]
                    b_roas_ser  = np.where(b_spend > 0, b_rev / b_spend, np.nan)

                    b_tot_spend = float(b_spend.sum())
                    b_tot_rev   = float(b_rev.sum())
                    b_tot_purch = int(round(float(b_purch.sum())))
                    b_roas      = (b_tot_rev / b_tot_spend) if b_tot_spend > 0 else np.nan

                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Valor usado (B)", _fmt_money_br(b_tot_spend))
                    d2.metric("Faturamento (B)", _fmt_money_br(b_tot_rev))
                    d3.metric("Vendas (B)", f"{b_tot_purch:,}".replace(",", "."))
                    d4.metric("ROAS (B)", _fmt_ratio_br(b_roas) if pd.notnull(b_roas) else "—")

                    h_best_purchB = int(b.loc[b["purchases (B)"].idxmax(), "hour"]) if len(b_purch) and b_purch.max() > 0 else None
                    best_purch_valB = int(b_purch.max()) if len(b_purch) else 0

                    mask_roasB = (b_spend >= float(min_spend)) & (b_spend > 0)
                    if mask_roasB.any():
                        roasB_vals = b_roas_ser.copy()
                        roasB_vals[~mask_roasB] = np.nan
                        h_best_roasB = int(b.loc[np.nanargmax(roasB_vals), "hour"])
                        best_roasB_val = float(np.nanmax(roasB_vals))
                    else:
                        h_best_roasB, best_roasB_val = None, np.nan

                    rollB = b_purch.rolling(3, min_periods=1).sum()
                    iB = int(rollB.idxmax()) if len(rollB) else 0
                    def _bB(ix): return int(b.loc[min(max(ix, 0), len(b)-1), "hour"])
                    winB_start, winB_mid, winB_end = _bB(iB-1), _bB(iB), _bB(iB+1)
                    winB_sum = int(rollB.max()) if len(rollB) else 0

                    wastedB = b[(b_spend > 0) & (b_purch == 0)]
                    wastedB_hours = ", ".join(f"{int(h)}h" for h in wastedB["hour"].tolist()) if not wastedB.empty else "—"

                    st.markdown(
                        f"""
**Pontos-chave (B)**  
- 🕐 **Pico de compras:** **{(str(h_best_purchB)+'h') if h_best_purchB is not None else '—'}** ({best_purch_valB} compras).  
- 💹 **Melhor ROAS** (gasto ≥ R$ {min_spend:,.0f}): **{(str(h_best_roasB)+'h') if h_best_roasB is not None else '—'}** ({_fmt_ratio_br(best_roasB_val) if pd.notnull(best_roasB_val) else '—'}).  
- ⏱️ **Janela forte (3h):** **{winB_start}–{winB_end}h** (centro {winB_mid}h) somando **{winB_sum}** compras.  
- 🧯 **Horas com gasto e 0 compras:** {wastedB_hours}.
""".replace(",", "X").replace(".", ",").replace("X", ".")
                    )

                    colTB1, colTB2 = st.columns(2)
                    with colTB1:
                        topB_p = b[["hour","purchases (B)","spend (B)","revenue (B)"]].sort_values("purchases (B)", ascending=False).head(5).copy()
                        topB_p.rename(columns={"hour":"Hora","purchases (B)":"Compras","spend (B)":"Valor usado","revenue (B)":"Valor de conversão"}, inplace=True)
                        topB_p["Valor usado"] = topB_p["Valor usado"].apply(_fmt_money_br)
                        topB_p["Valor de conversão"] = topB_p["Valor de conversão"].apply(_fmt_money_br)
                        st.dataframe(topB_p, use_container_width=True, height=220)
                    with colTB2:
                        if mask_roasB.any():
                            topB_r = b[mask_roasB][["hour","spend (B)","revenue (B)"]].copy()
                            topB_r["ROAS"] = b_roas_ser[mask_roasB]
                            topB_r = topB_r.sort_values("ROAS", ascending=False).head(5)
                            topB_r.rename(columns={"hour":"Hora","spend (B)":"Valor usado","revenue (B)":"Valor de conversão"}, inplace=True)
                            topB_r["Valor usado"] = topB_r["Valor usado"].apply(_fmt_money_br)
                            topB_r["Valor de conversão"] = topB_r["Valor de conversão"].apply(_fmt_money_br)
                            topB_r["ROAS"] = topB_r["ROAS"].map(_fmt_ratio_br)
                        else:
                            topB_r = pd.DataFrame(columns=["Hora","Valor usado","Valor de conversão","ROAS"])
                        st.dataframe(topB_r, use_container_width=True, height=220)

                    st.info("Sugestões (B): direcione orçamento para as horas com melhor ROAS e pause/teste criativos nas horas com gasto e 0 compras.")

# -------------------- ABA 3: 📊 DETALHAMENTO --------------------
with tab_detail:
    st.caption(
        "Explore por dimensão: Idade, Gênero, Idade+Gênero, País, Plataforma, "
        "Posicionamento, Dia e Hora. Há um modo 'Populares' com os TOP 5."
    )

    # ===== Filtros =====
    colf1, colf2 = st.columns([2, 1])
    with colf1:
        produto_sel_det = st.selectbox(
            "Filtrar por produto (opcional)",
            ["(Todos)"] + PRODUTOS,
            key="det_produto",
        )
    with colf2:
        min_spend_det = st.slider(
            "Gasto mínimo para considerar (R$)",
            0.0, 2000.0, 0.0, 10.0,
            key="det_min_spend",
        )

    dimensao = st.radio(
        "Dimensão",
        [
            "Populares", "Idade", "Gênero", "Idade + Gênero",
            "Região", "País", "Plataforma", "Posicionamento",
        ],
        index=0,
        horizontal=True,
    )

    # ========= Helpers =========
    def _apply_prod_filter(df_base: pd.DataFrame) -> pd.DataFrame:
        if produto_sel_det and produto_sel_det != "(Todos)":
            mask = df_base["campaign_name"].str.contains(produto_sel_det, case=False, na=False)
            return df_base[mask].copy()
        return df_base

    def _ensure_cols_exist(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _agg_and_format(df: pd.DataFrame, group_cols: list[str]):
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df2 = _apply_prod_filter(df)
        if df2.empty:
            return pd.DataFrame(), pd.DataFrame()
        df2 = _ensure_cols_exist(df2)

        agg_cols = ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
        g = df2.groupby(group_cols, dropna=False, as_index=False)[agg_cols].sum()
        g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)

        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]
        if not g.empty:
            g = g.sort_values(["purchases", "ROAS"], ascending=[False, False])
        return g, g.copy()

    def _bar_chart(x_labels, y_values, title, x_title, y_title):
        fig = go.Figure(go.Bar(x=x_labels, y=y_values, text=y_values, textposition="outside"))
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=420,
            template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10),
            separators=".,",
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Cores e formatação de período ===
    COLOR_A = "#636EFA"   # azul
    COLOR_B = "#EF553B"   # laranja

    def _fmt_range_br(d1, d2) -> str:
        d1 = pd.to_datetime(str(d1)).date()
        d2 = pd.to_datetime(str(d2)).date()
        return f"{d1.strftime('%d/%m/%Y')} → {d2.strftime('%d/%m/%Y')}"

    # ========= Helpers p/ taxas =========
    def _rates_from_raw(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        df["LPV/Cliques"]        = df.apply(lambda r: _safe_div(r["lpv"], r["link_clicks"]), axis=1)
        df["Checkout/LPV"]       = df.apply(lambda r: _safe_div(r["init_checkout"], r["lpv"]), axis=1)
        df["Compra/Checkout"]    = df.apply(lambda r: _safe_div(r["purchases"], r["init_checkout"]), axis=1)
        df["Add Pagto/Checkout"] = df.apply(lambda r: _safe_div(r["add_payment"], r["init_checkout"]), axis=1)
        df["Compra/Add Pagto"]   = df.apply(lambda r: _safe_div(r["purchases"], r["add_payment"]), axis=1)
        return df

    # ========= POPULARES =========
    if dimensao == "Populares":
        base = df_daily.copy()
        base = _apply_prod_filter(base)
        base = _ensure_cols_exist(base)

        g = (
            base.groupby(["campaign_id", "campaign_name"], as_index=False)[
                ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
            ].sum()
        )
        g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)

        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]

        top_comp = g.sort_values(["purchases", "ROAS"], ascending=[False, False]).head(5).copy()
        top_roas = g[g["spend"] > 0].sort_values("ROAS", ascending=False).head(5).copy()

        def _fmt_disp(df_):
            out = df_.copy()
            out["Valor usado"]        = out["spend"].apply(_fmt_money_br)
            out["Valor de conversão"] = out["revenue"].apply(_fmt_money_br)
            out["ROAS"]               = out["ROAS"].map(_fmt_ratio_br)
            out.rename(
                columns={
                    "campaign_name": "Campanha",
                    "purchases": "Compras",
                    "link_clicks": "Cliques",
                    "lpv": "LPV",
                    "init_checkout": "Checkout",
                    "add_payment": "Add Pagto",
                },
                inplace=True,
            )
            return out

        disp_comp = _fmt_disp(top_comp)[
            ["Campanha", "Compras", "Valor usado", "Valor de conversão", "ROAS", "Cliques", "LPV", "Checkout", "Add Pagto"]
        ]
        disp_roas = _fmt_disp(top_roas)[
            ["Campanha", "ROAS", "Compras", "Valor usado", "Valor de conversão", "Cliques", "LPV", "Checkout", "Add Pagto"]
        ]

        st.subheader("TOP 5 — Campanhas")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(disp_comp, use_container_width=True, height=260)
        with c2:
            st.dataframe(disp_roas, use_container_width=True, height=260)
        st.stop()

    # ========= DEMAIS DIMENSÕES =========
    dim_to_breakdowns = {
        "Idade": ["age"],
        "Gênero": ["gender"],
        "Idade + Gênero": ["age", "gender"],
        "Região": ["region"],
        "País": ["country"],
        "Plataforma": ["publisher_platform"],
        "Posicionamento": ["publisher_platform", "platform_position"],
    }

    level_bd = level
    if dimensao == "Posicionamento" and level_bd not in ["adset", "ad"]:
        level_bd = "adset"

    if dimensao in dim_to_breakdowns:
        bks = dim_to_breakdowns[dimensao]
        df_bd = fetch_insights_breakdown(
            act_id, token, api_version, str(since), str(until),
            bks, level_bd, product_name=st.session_state.get("det_produto"),
        )

        if df_bd.empty:
            st.info(f"Sem dados para {dimensao} no período/filtro.")
            st.stop()

        rename_map = {
            "age": "Idade",
            "gender": "Gênero",
            "region": "Região",
            "country": "País",
            "publisher_platform": "Plataforma",
            "platform_position": "Posicionamento",
        }
        group_cols = [rename_map.get(c, c) for c in bks]

        raw, disp = _agg_and_format(df_bd.rename(columns=rename_map), group_cols)
        if disp.empty:
            st.info(f"Sem dados para {dimensao} após aplicar filtros.")
            st.stop()

        st.subheader(f"Desempenho por {dimensao}")

        # ----- gráfico -----
        if len(group_cols) == 1:
            xlab = group_cols[0]
            _bar_chart(raw[xlab], raw["purchases"], f"Compras por {xlab}", xlab, "Compras")
        else:
            idx, col = group_cols
            pvt = raw.pivot_table(index=idx, columns=col, values="purchases", aggfunc="sum").fillna(0)
            fig = go.Figure(
                data=go.Heatmap(
                    z=pvt.values,
                    x=pvt.columns.astype(str),
                    y=pvt.index.astype(str),
                    colorbar=dict(title="Compras"),
                    hovertemplate=(f"{idx}: " + "%{y}<br>" + f"{col}: " + "%{x}<br>" + "Compras: %{z}<extra></extra>"),
                )
            )
            fig.update_layout(
                title=f"Heatmap — Compras por {idx} × {col}",
                height=460,
                template="plotly_white",
                margin=dict(l=10, r=10, t=48, b=10),
                separators=".,",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ----- tabela integrada com taxas -----
        raw = _rates_from_raw(raw, group_cols)
        disp = raw.rename(
            columns={
                "link_clicks": "Cliques",
                "lpv": "LPV",
                "init_checkout": "Checkout",
                "add_payment": "Add Pagto",
                "purchases": "Compras",
                "revenue": "Valor de conversão",
                "spend": "Valor usado",
            }
        )

        # formatação para exibição
        disp["Valor usado"] = disp["Valor usado"].apply(_fmt_money_br)
        disp["Valor de conversão"] = disp["Valor de conversão"].apply(_fmt_money_br)
        disp["ROAS"] = disp["ROAS"].map(_fmt_ratio_br)
        for col_taxa in ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout", "Add Pagto/Checkout", "Compra/Add Pagto"]:
            if col_taxa in disp.columns:
                disp[col_taxa] = disp[col_taxa].map(_fmt_pct_br)
        for col_abs in ["Cliques", "LPV", "Checkout", "Add Pagto", "Compras"]:
            if col_abs in disp.columns:
                disp[col_abs] = disp[col_abs].astype(int)

        final_cols = group_cols + [
            "ROAS", "Valor usado", "Valor de conversão",
            "Cliques", "LPV", "LPV/Cliques",
            "Checkout", "Checkout/LPV",
            "Add Pagto", "Add Pagto/Checkout",
            "Compras", "Compra/Checkout", "Compra/Add Pagto",
        ]

        # cabeçalhos das taxas (destaque)
        taxa_cols = ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout"]

        def highlight_headers(x):
            return [
                "background-color: rgba(59, 130, 246, 0.15); font-weight: bold;"
                if col in taxa_cols else ""
                for col in x
            ]

        styled_disp = disp[final_cols].style.apply(
            lambda _: highlight_headers(disp[final_cols].columns),
            axis=1,
        )
        st.dataframe(styled_disp, use_container_width=True, height=520)

        # ====== Comparação por período ======
        st.markdown("### Comparar períodos")

        from datetime import timedelta

        since_dt = pd.to_datetime(str(since)).date()
        until_dt = pd.to_datetime(str(until)).date()
        delta = until_dt - since_dt

        colp1, colp2 = st.columns(2)
        with colp1:
            perA = st.date_input(
                "Período A",
                (since_dt, until_dt),
                key="perA_det_tbl",
                format="DD/MM/YYYY",
            )
        with colp2:
            default_b_end = since_dt - timedelta(days=1)
            default_b_start = default_b_end - delta
            perB = st.date_input(
                "Período B",
                (default_b_start, default_b_end),
                key="perB_det_tbl",
                format="DD/MM/YYYY",
            )

        since_A, until_A = perA
        since_B, until_B = perB

        def _load_rates_for_range(d1, d2):
            dfR = fetch_insights_breakdown(
                act_id, token, api_version, str(d1), str(d2),
                bks, level_bd, product_name=st.session_state.get("det_produto"),
            )
            rawR, _ = _agg_and_format(dfR.rename(columns=rename_map), group_cols)
            return _rates_from_raw(rawR, group_cols)

        # Carrega e calcula taxas para A e B
        raw_A = _load_rates_for_range(since_A, until_A)
        raw_B = _load_rates_for_range(since_B, until_B)

        def _alias_cols(df, suffix):
            m = {
                "spend": f"Valor usado {suffix}",
                "revenue": f"Valor de conversão {suffix}",
                "link_clicks": f"Cliques {suffix}",
                "lpv": f"LPV {suffix}",
                "init_checkout": f"Checkout {suffix}",
                "add_payment": f"Add Pagto {suffix}",
                "purchases": f"Compras {suffix}",
                "ROAS": f"ROAS {suffix}",
                "LPV/Cliques": f"LPV/Cliques {suffix}",
                "Checkout/LPV": f"Checkout/LPV {suffix}",
                "Compra/Checkout": f"Compra/Checkout {suffix}",
                "Add Pagto/Checkout": f"Add Pagto/Checkout {suffix}",
                "Compra/Add Pagto": f"Compra/Add Pagto {suffix}",
            }
            cols = [c for c in (group_cols + list(m.keys())) if c in df.columns]
            return df[cols].rename(columns=m)

        # Tabelas NUMÉRICAS (base para os cálculos)
        A_num = _alias_cols(raw_A, "A")
        B_num = _alias_cols(raw_B, "B")

        # ------- formatação para exibição (cada período em sua tabela) -------
        def _format_period_table(df_num: pd.DataFrame, suffix: str) -> pd.DataFrame:
            df = df_num.copy()

            money_cols = [f"Valor usado {suffix}", f"Valor de conversão {suffix}"]
            int_cols = [
                f"Cliques {suffix}", f"LPV {suffix}", f"Checkout {suffix}",
                f"Add Pagto {suffix}", f"Compras {suffix}",
            ]
            roas_cols = [f"ROAS {suffix}"]
            rate_cols = [
                f"LPV/Cliques {suffix}", f"Checkout/LPV {suffix}",
                f"Compra/Checkout {suffix}", f"Add Pagto/Checkout {suffix}",
                f"Compra/Add Pagto {suffix}",
            ]

            # Inteiros
            for c in int_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0).astype(float).round(0).astype(int)

            # Dinheiro
            for c in money_cols:
                if c in df.columns:
                    df[c] = df[c].apply(_fmt_money_br)

            # ROAS e taxas
            for c in roas_cols:
                if c in df.columns:
                    df[c] = df[c].map(_fmt_ratio_br)
            for c in rate_cols:
                if c in df.columns:
                    df[c] = df[c].map(_fmt_pct_br)

            ordered = group_cols + [
                f"Valor usado {suffix}", f"Valor de conversão {suffix}", f"ROAS {suffix}",
                f"Cliques {suffix}", f"LPV {suffix}", f"Checkout {suffix}",
                f"Add Pagto {suffix}", f"Compras {suffix}",
                f"LPV/Cliques {suffix}", f"Checkout/LPV {suffix}",
                f"Compra/Checkout {suffix}", f"Add Pagto/Checkout {suffix}",
                f"Compra/Add Pagto {suffix}",
            ]
            return df[[c for c in ordered if c in df.columns]]

        A_fmt = _format_period_table(A_num, "A")
        B_fmt = _format_period_table(B_num, "B")

        # Destaque visual das colunas de taxas em A e B
        RATE_COLS_A = ["LPV/Cliques A", "Checkout/LPV A", "Compra/Checkout A", "Add Pagto/Checkout A", "Compra/Add Pagto A"]
        RATE_COLS_B = ["LPV/Cliques B", "Checkout/LPV B", "Compra/Checkout B", "Add Pagto/Checkout B", "Compra/Add Pagto B"]

        RED_TRANSP    = "rgba(239, 68, 68, 0.15)"  # A
        YELLOW_TRANSP = "rgba(255, 255, 0, 0.30)"  # B

        def _style_rate_columns(df: pd.DataFrame, rate_cols: list[str], rgba_bg: str):
            sty = df.style.apply(
                lambda _row: [f"background-color: {rgba_bg};" if col in rate_cols else "" for col in df.columns],
                axis=1,
            )
            header_styles = [
                {
                    "selector": f"th.col_heading.level0.col{df.columns.get_loc(col)}",
                    "props": [("background-color", rgba_bg), ("font-weight", "bold")],
                }
                for col in rate_cols
                if col in df.columns
            ]
            return sty.set_table_styles(header_styles)

        A_styled = _style_rate_columns(A_fmt, RATE_COLS_A, RED_TRANSP)
        B_styled = _style_rate_columns(B_fmt, RATE_COLS_B, YELLOW_TRANSP)

        # ------- EXIBIÇÃO: duas tabelas separadas -------
        st.markdown("#### Período A")
        st.dataframe(A_styled, use_container_width=True, height=360)

        st.markdown("#### Período B")
        st.dataframe(B_styled, use_container_width=True, height=360)

        # ------- (Opcional) Tabela de variação A vs B (APENAS TAXAS) -------
        show_deltas = st.checkbox(
            "Mostrar variação entre A e B (apenas taxas em p.p.)",
            value=False,
            key="det_show_deltas_tbl",
        )

        if show_deltas:
            # ---- Merge numérico para cálculo ----
            comp_num = pd.merge(A_num, B_num, on=group_cols, how="outer")

            # ---- Variação em pontos percentuais (taxas) ----
            rate_specs = [
                ("LPV/Cliques",        "LPV/Cliques A",        "LPV/Cliques B"),
                ("Checkout/LPV",       "Checkout/LPV A",       "Checkout/LPV B"),
                ("Compra/Checkout",    "Compra/Checkout A",    "Compra/Checkout B"),
                ("Add Pagto/Checkout", "Add Pagto/Checkout A", "Add Pagto/Checkout B"),
                ("Compra/Add Pagto",   "Compra/Add Pagto A",   "Compra/Add Pagto B"),
            ]

            # DataFrame NUMÉRICO (sem strings) para aplicar cor por sinal
            deltas_num = comp_num[group_cols].copy()
            for label, colA, colB in rate_specs:
                if colA in comp_num.columns and colB in comp_num.columns:
                    deltas_num[f"Δ {label} (p.p.)"] = (
                        comp_num[colA].fillna(0).astype(float)
                        - comp_num[colB].fillna(0).astype(float)
                    ) * 100.0

            # ordem das colunas: dimensões + deltas
            ordered_cols = group_cols + [f"Δ {lbl} (p.p.)" for (lbl, _, _) in rate_specs]
            deltas_num = deltas_num[[c for c in ordered_cols if c in deltas_num.columns]]

            pp_cols = [c for c in deltas_num.columns if c.endswith("(p.p.)")]

            # ---- Estilo de fundo (+ verde, - vermelho) ----
            POS_BG = "rgba(22, 163, 74, 0.45)"   # positivo
            NEG_BG = "rgba(239, 68, 68, 0.45)"   # negativo

            def _bg_sign(val):
                try:
                    v = float(val)
                except Exception:
                    return ""
                if v > 0:
                    return f"background-color: {POS_BG}; font-weight: 700;"
                if v < 0:
                    return f"background-color: {NEG_BG}; font-weight: 700;"
                return ""

            # ---- Formatação pt-BR para exibição (+x,x p.p.) ----
            def _fmt_pp(v):
                if pd.isna(v) or np.isinf(v):
                    return "—"
                sign = "+" if v >= 0 else ""
                s = f"{sign}{v:.1f}".replace(".", ",")
                return f"{s} p.p."

            styled = (
                deltas_num.style
                .hide(axis="index")  # oculta índice para alinhar o cabeçalho à esquerda
                .applymap(_bg_sign, subset=pp_cols)
                .format({c: _fmt_pp for c in pp_cols})
                .set_properties(subset=pp_cols, **{"padding": "6px 8px", "text-align": "center"})
                .set_table_styles([
                    {"selector": "th.blank",       "props": [("display", "none")]},
                    {"selector": "th.col_heading", "props": [("text-align", "center"), ("white-space", "nowrap")]},
                    {"selector": "td",             "props": [("vertical-align", "middle")]},
                ])
            )

            st.markdown("#### Variação — Taxas (p.p.)")
            st.table(styled)

        st.caption(
            f"Período A: **{_fmt_range_br(since_A, until_A)}**  |  "
            f"Período B: **{_fmt_range_br(since_B, until_B)}**"
        )

        st.stop()
