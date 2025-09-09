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

BENCH_DEFAULT = {
    "r1": (0.70, 0.90),  # LPV/Cliques   (70% a 90%)
    "r2": (0.12, 0.20),  # Checkout/LPV  (12% a 20%)
    "r3": (0.30, 0.40),  # Compra/Checkout (30% a 40%)
}

def _bench_state_init():
    """Inicializa os valores dos benchmarks em st.session_state (em %), caso não existam."""
    if "bench_r1_lo" not in st.session_state:
        st.session_state.update({
            "bench_r1_lo": int(BENCH_DEFAULT["r1"][0] * 100),
            "bench_r1_hi": int(BENCH_DEFAULT["r1"][1] * 100),
            "bench_r2_lo": int(BENCH_DEFAULT["r2"][0] * 100),
            "bench_r2_hi": int(BENCH_DEFAULT["r2"][1] * 100),
            "bench_r3_lo": int(BENCH_DEFAULT["r3"][0] * 100),
            "bench_r3_hi": int(BENCH_DEFAULT["r3"][1] * 100),
        })

def get_bench():
    """Retorna dict com faixas **em fração** (0–1) a partir dos valores da Sidebar."""
    return {
        "r1": (st.session_state["bench_r1_lo"] / 100.0, st.session_state["bench_r1_hi"] / 100.0),
        "r2": (st.session_state["bench_r2_lo"] / 100.0, st.session_state["bench_r2_hi"] / 100.0),
        "r3": (st.session_state["bench_r3_lo"] / 100.0, st.session_state["bench_r3_hi"] / 100.0),
    }

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

                ic  = _sum_actions_exact(actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS)
                api = _sum_actions_exact(actions, ["add_payment_info"], allowed_keys=ATTR_KEYS)
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

                ic   = _sum_actions_exact(actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS)
                api_ = _sum_actions_exact(actions, ["add_payment_info"], allowed_keys=ATTR_KEYS)
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
                ic   = _sum_actions_exact(actions, ["initiate_checkout"], allowed_keys=ATTR_KEYS)
                api_ = _sum_actions_exact(actions, ["add_payment_info"], allowed_keys=ATTR_KEYS)
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
    since = st.sidebar.date_input("Desde", value=_since_auto, key="since_custom")
    until = st.sidebar.date_input("Até",   value=_until_auto, key="until_custom")
else:
    # ✅ NÃO usar date_input aqui (evita estado preso)
    since, until = _since_auto, _until_auto
    st.sidebar.caption(f"**Desde:** {since}  \n**Até:** {until}")

# Benchmarks editáveis na UI
_bench_state_init()
with st.sidebar.expander("Benchmarks do funil (editar faixas %)", expanded=False):
    st.caption("Ajuste as faixas ideais das taxas (em **%**).")
    r1_lo, r1_hi = st.sidebar.slider(
        "LPV / Cliques (%)",
        min_value=0, max_value=100,
        value=(st.session_state["bench_r1_lo"], st.session_state["bench_r1_hi"]),
        step=1, key="slider_r1"
    )
    st.session_state["bench_r1_lo"], st.session_state["bench_r1_hi"] = r1_lo, r1_hi

    r2_lo, r2_hi = st.sidebar.slider(
        "Checkout / LPV (%)",
        min_value=0, max_value=100,
        value=(st.session_state["bench_r2_lo"], st.session_state["bench_r2_hi"]),
        step=1, key="slider_r2"
    )
    st.session_state["bench_r2_lo"], st.session_state["bench_r2_hi"] = r2_lo, r2_hi

    r3_lo, r3_hi = st.sidebar.slider(
        "Compra / Checkout (%)",
        min_value=0, max_value=100,
        value=(st.session_state["bench_r3_lo"], st.session_state["bench_r3_hi"]),
        step=1, key="slider_r3"
    )
    st.session_state["bench_r3_lo"], st.session_state["bench_r3_hi"] = r3_lo, r3_hi

    # OBS: use st.sidebar.button porque o expander está na Sidebar.
    if st.sidebar.button("↺ Resetar para padrão", key="btn_reset_bench"):
        st.session_state["bench_r1_lo"] = int(BENCH_DEFAULT["r1"][0] * 100)
        st.session_state["bench_r1_hi"] = int(BENCH_DEFAULT["r1"][1] * 100)
        st.session_state["bench_r2_lo"] = int(BENCH_DEFAULT["r2"][0] * 100)
        st.session_state["bench_r2_hi"] = int(BENCH_DEFAULT["r2"][1] * 100)
        st.session_state["bench_r3_lo"] = int(BENCH_DEFAULT["r3"][0] * 100)
        st.session_state["bench_r3_hi"] = int(BENCH_DEFAULT["r3"][1] * 100)

        # Streamlit >= 1.30: st.rerun(); fallback para versões antigas
        (getattr(st, "rerun", st.experimental_rerun))()

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

    # Taxas do funil (para o Guia de Ação)
    r1 = _safe_div(values_total[1], values_total[0])  # LPV / Cliques
    r2 = _safe_div(values_total[2], values_total[1])  # Checkout / LPV
    r3 = _safe_div(values_total[4], values_total[2])  # Compra / Checkout

    # Quedas absolutas por etapa (para priorização)
    drop1 = max(0, values_total[0] - values_total[1])  # Cliques -> LPV
    drop2 = max(0, values_total[1] - values_total[2])  # LPV -> Checkout
    drop3 = max(0, values_total[2] - values_total[4])  # Checkout -> Compra


    # ===== 🧭 GUIA DE AÇÃO — metas, leitura rápida e prioridade =====
    st.subheader("🧭 Guia de Ação — metas & prioridade")

    # Metas de referência (e-commerce)
    bench = get_bench()

    def _band(val, lo, hi):
        if not pd.notnull(val): return "sem"
        if val < lo:  return "abaixo"
        if val > hi:  return "acima"
        return "dentro"

    status = {
        "LPV/Cliques":     (_fmt_pct_br(r1), _band(r1, *bench["r1"]), bench["r1"]),
        "Checkout/LPV":    (_fmt_pct_br(r2), _band(r2, *bench["r2"]), bench["r2"]),
        "Compra/Checkout": (_fmt_pct_br(r3), _band(r3, *bench["r3"]), bench["r3"]),
    }

    # Tabela-resumo das metas vs atual
    st.markdown("**Metas de referência (boas práticas)**")
    tbl = pd.DataFrame([
        ["LPV / Cliques",     status["LPV/Cliques"][0],     f"{int(bench['r1'][0]*100)}–{int(bench['r1'][1]*100)}%"],
        ["Checkout / LPV",    status["Checkout/LPV"][0],    f"{int(bench['r2'][0]*100)}–{int(bench['r2'][1]*100)}%"],
        ["Compra / Checkout", status["Compra/Checkout"][0], f"{int(bench['r3'][0]*100)}–{int(bench['r3'][1]*100)}%"],
    ], columns=["Taxa", "Seu valor", "Faixa saudável"])
    st.dataframe(tbl, use_container_width=True, height=150)

    # Leitura rápida (semáforo)
    def _chip_sem(color, txt): 
        return f"<span style='padding:3px 8px;border-radius:999px;background:{color};color:#fff;font-size:12px'>{txt}</span>"

    def _badge(name):
        val, band, (lo, hi) = status[name]
        if band == "abaixo": color = "#dc2626"; lab = "Abaixo"
        elif band == "dentro": color = "#16a34a"; lab = "OK"
        elif band == "acima": color = "#059669"; lab = "Acima"
        else: color = "#6b7280"; lab = "–"
        return f"**{name}** {val} {_chip_sem(color, lab)}"

    st.markdown("**Leitura em 30s**", unsafe_allow_html=True)
    st.markdown(
        "- " + _badge("LPV/Cliques") + "<br>" +
        "- " + _badge("Checkout/LPV") + "<br>" +
        "- " + _badge("Compra/Checkout"),
        unsafe_allow_html=True
    )

    # Priorização — escolhe onde atuar primeiro
    # regra: onde estiver "abaixo" e onde a QUEDA absoluta é maior
    abaixo_map = {
        "Teste de criativo (Cliques→LPV)":      ("LPV/Cliques", drop1),
        "Interesse/Oferta (LPV→Checkout)":      ("Checkout/LPV", drop2),
        "Pagamento & RMK (Checkout→Compra)":    ("Compra/Checkout", drop3),
    }
    candidatos = {k: v for k, v in abaixo_map.items() if status[v[0]][1] == "abaixo"}

    if candidatos:
        foco = max(candidatos.items(), key=lambda kv: kv[1][1])[0]
    else:
        # se nada abaixo das metas e há volume, o foco vira escala
        foco = "Escala" if pd.notnull(r1) and pd.notnull(r2) and pd.notnull(r3) else "Diagnóstico"

    # Ações recomendadas por foco
    acoes = {
        "Teste de criativo (Cliques→LPV)": [
            "Alinhar headline/visual do anúncio com a primeira dobra da LP.",
            "Testar 3 ângulos (dor, ganho, prova) + 3 CTAs diferentes.",
            "Melhorar velocidade (LCP < 2,5s) e remover distrações acima da dobra."
        ],
        "Interesse/Oferta (LPV→Checkout)": [
            "Evidenciar benefício + prova social + preço/parcelamento antes do CTA.",
            "Responder objeções (troca, prazo, garantia) na LP.",
            "Testar bundle/brinde e CTA ('Comprar agora' vs 'Adicionar ao carrinho')."
        ],
        "Pagamento & RMK (Checkout→Compra)": [
            "Checkout curto (1–2 passos), 'convidado', autofill e erros claros.",
            "Destacar Pix/Cartão e incentivo (ex.: Pix -10% / frete grátis).",
            "RMK por etapa: D1 prova/garantia • D3 objeções • D7 urgência/cupom."
        ],
        "Escala": [
            "Aumentar 10–30% nas campanhas/horários com melhor ROAS nos últimos 7–14 dias.",
            "Duplicar conjuntos vencedores e expandir públicos gradualmente.",
            "Manter rotatividade de criativos vencedores."
        ],
        "Diagnóstico": [
            "Checar tracking (duplicidade de pixel/UTM) e integridade de eventos.",
            "Rever experiência mobile (velocidade, checkout, meios de pagamento).",
            "Validar oferta vs concorrência e percepção de valor."
        ]
    }

    # Metas táticas para a semana (puxa seus valores e aponta próximo alvo)
    def _meta_alvo(cur, lo, step=0.02):
        if not pd.notnull(cur): return None
        if cur < lo: 
            # define alvo mínimo + 'step'
            return max(lo, round((cur + step), 4))
        return round(cur, 4)

    alvo_r3 = _meta_alvo(r3, bench["r3"][0], 0.02)  # +2 p.p. ou até o piso
    alvo_r2 = _meta_alvo(r2, bench["r2"][0], 0.02)

    st.markdown("---")
    if foco == "Escala":
        st.success("**✅ Prioridade da semana: Escala** — taxas dentro/above das metas e volume ok.")
    else:
        st.warning(f"**⚠️ Prioridade da semana: {foco}**")

    st.markdown("**Ações imediatas (execute nesta semana):**")
    for tip in acoes.get(foco, []):
        st.markdown(f"- {tip}")

    # Metas numéricas claras (se existirem)
    metas = []
    if alvo_r3 and r3 < bench["r3"][0]:
        metas.append(f"Levar **Compra/Checkout** de { _fmt_pct_br(r3) } → **{ _fmt_pct_br(alvo_r3) }**.")
    if alvo_r2 and r2 < bench["r2"][0]:
        metas.append(f"Levar **Checkout/LPV** de { _fmt_pct_br(r2) } → **{ _fmt_pct_br(alvo_r2) }**.")
    if r1 < bench["r1"][0]:
        metas.append(f"Elevar **LPV/Cliques** até **{int(bench['r1'][0]*100)}%+**.")

    if metas:
        st.markdown("**Metas táticas (7 dias):**")
        for m in metas:
            st.markdown(f"- {m}")

    # Mini-calculadora de impacto (aproximada)
    ticket_medio = st.number_input("Ticket médio estimado (R$)", 0.0, 100000.0, 120.0, 1.0, key="ticket_meta")
    # impacto de subir Compra/Checkout até o alvo
    if alvo_r3 and pd.notnull(r3) and values_total[2] > 0:
        delta_conv = int(round(values_total[2] * max(0.0, alvo_r3 - r3)))
        if delta_conv > 0:
            st.caption(f"🎯 Subir **Compra/Checkout** até { _fmt_pct_br(alvo_r3) } ⇒ ~**{_fmt_int_br(delta_conv)}** compras extras (≈ {_fmt_money_br(delta_conv*ticket_medio)}).")

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
        
# -------------------- ABA DE HORÁRIOS (Heatmap no topo) --------------------
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

    # Agora aplicamos o filtro por produto no campaign_name (como no modelo antigo)
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
        xaxis_title="Hora do dia",
        yaxis_title="Compras",
        height=380,
        template="plotly_white",
        margin=dict(l=10, r=10, t=48, b=10),
        separators=",."
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos.")

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

                    # ===== Gráfico do Período A =====
                    fig_A = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_A.add_trace(go.Bar(name="Gasto (A)", x=x, y=merged["spend (A)"], legendgroup="A", offsetgroup="A"))
                    fig_A.add_trace(go.Bar(name="Faturamento (A)", x=x, y=merged["revenue (A)"], legendgroup="A", offsetgroup="A"))
                    fig_A.add_trace(
                        go.Scatter(name=f"Compras (A) — {period_sinceA} a {period_untilA}",
                                   x=x, y=merged["purchases (A)"], mode="lines+markers", legendgroup="A"),
                        secondary_y=True
                    )
                    fig_A.update_layout(
                        title=f"Período A — {period_sinceA} a {period_untilA} (Gasto + Faturamento + Compras)",
                        barmode="stack", bargap=0.15, bargroupgap=0.12,
                        template="plotly_white", height=460,
                        margin=dict(l=10, r=10, t=48, b=10), legend_title_text="", separators=",."
                    )
                    fig_A.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
                    fig_A.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
                    fig_A.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])
                    st.plotly_chart(fig_A, use_container_width=True)

                    # ===== Gráfico do Período B =====
                    fig_B = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_B.add_trace(go.Bar(name="Gasto (B)", x=x, y=merged["spend (B)"], legendgroup="B", offsetgroup="B"))
                    fig_B.add_trace(go.Bar(name="Faturamento (B)", x=x, y=merged["revenue (B)"], legendgroup="B", offsetgroup="B"))
                    fig_B.add_trace(
                        go.Scatter(name=f"Compras (B) — {period_sinceB} a {period_untilB}",
                                   x=x, y=merged["purchases (B)"], mode="lines+markers", legendgroup="B"),
                        secondary_y=True
                    )
                    fig_B.update_layout(
                        title=f"Período B — {period_sinceB} a {period_untilB} (Gasto + Faturamento + Compras)",
                        barmode="stack", bargap=0.15, bargroupgap=0.12,
                        template="plotly_white", height=460,
                        margin=dict(l=10, r=10, t=48, b=10), legend_title_text="", separators=",."
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
    st.caption("Explore por dimensão: Idade, Gênero, Idade+Gênero, País, Plataforma, Posicionamento, Dia e Hora. Há um modo 'Populares' com os TOP 5.")

    colf1, colf2 = st.columns([2,1])
    with colf1:
        produto_sel_det = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="det_produto")
    with colf2:
        min_spend_det = st.slider("Gasto mínimo para considerar (R$)", 0.0, 2000.0, 0.0, 10.0, key="det_min_spend")

    dimensao = st.radio(
        "Dimensão",
        ["Populares","Idade","Gênero","Idade + Gênero", "Região", "País","Plataforma","Posicionamento"],
        index=0, horizontal=True
    )

    # ========= Helpers locais =========
    def _apply_prod_filter(df_base: pd.DataFrame) -> pd.DataFrame:
        if produto_sel_det and produto_sel_det != "(Todos)":
            mask = df_base["campaign_name"].str.contains(produto_sel_det, case=False, na=False)
            return df_base[mask].copy()
        return df_base

    def _ensure_cols_exist(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]:
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

        agg_cols = ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
        g = df2.groupby(group_cols, dropna=False, as_index=False)[agg_cols].sum()
        g["ROAS"] = np.where(g["spend"]>0, g["revenue"]/g["spend"], np.nan)

        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]

        if not g.empty:
            g = g.sort_values(["purchases","ROAS"], ascending=[False, False])

        gf = g.copy()
        if not gf.empty:
            gf["Valor usado"] = gf["spend"].apply(_fmt_money_br)
            gf["Valor de conversão"] = gf["revenue"].apply(_fmt_money_br)
            gf["ROAS"] = gf["ROAS"].map(_fmt_ratio_br)
            gf = gf.drop(columns=["spend","revenue"])
        return g, gf

    def _bar_chart(x_labels, y_values, title, x_title, y_title):
        fig = go.Figure(go.Bar(x=x_labels, y=y_values))
        fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title,
            height=420, template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10), separators=",."
        )
        st.plotly_chart(fig, use_container_width=True)

    # ========= POPULARES =========
    if dimensao == "Populares":
        # Base: usa o diário consolidado no período
        base = df_daily.copy()
        base = _apply_prod_filter(base)
        base = _ensure_cols_exist(base)

        # Agrega por campanha
        g = (base.groupby(["campaign_id","campaign_name"], as_index=False)
                  [["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]]
                  .sum())
        g["ROAS"] = np.where(g["spend"]>0, g["revenue"]/g["spend"], np.nan)

        # Aplica gasto mínimo (detalhe)
        if min_spend_det and float(min_spend_det) > 0:
            g = g[g["spend"] >= float(min_spend_det)]

        # TOP 5
        top_comp = g.sort_values(["purchases","ROAS"], ascending=[False,False]).head(5).copy()
        top_roas = g[g["spend"]>0].sort_values("ROAS", ascending=False).head(5).copy()

        # Formata
        def _fmt_disp(df_):
            out = df_.copy()
            out["Valor usado"] = out["spend"].apply(_fmt_money_br)
            out["Valor de conversão"] = out["revenue"].apply(_fmt_money_br)
            out["ROAS"] = out["ROAS"].map(_fmt_ratio_br)
            out.rename(columns={
                "campaign_name":"Campanha", "purchases":"Compras",
                "link_clicks":"Cliques", "lpv":"LPV",
                "init_checkout":"Checkout", "add_payment":"Add Pagto"
            }, inplace=True)
            return out

        disp_comp = _fmt_disp(top_comp)[["Campanha","Compras","Valor usado","Valor de conversão","ROAS","Cliques","LPV","Checkout","Add Pagto"]]
        disp_roas = _fmt_disp(top_roas)[["Campanha","ROAS","Compras","Valor usado","Valor de conversão","Cliques","LPV","Checkout","Add Pagto"]]

        st.subheader("TOP 5 — Campanhas")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Por Compras**")
            st.dataframe(disp_comp, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV — TOP 5 por Compras",
                data=disp_comp.to_csv(index=False).encode("utf-8-sig"),
                file_name="top5_campanhas_por_compras.csv",
                mime="text/csv"
            )
        with c2:
            st.markdown("**Por ROAS** (com gasto > 0)")
            st.dataframe(disp_roas, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV — TOP 5 por ROAS",
                data=disp_roas.to_csv(index=False).encode("utf-8-sig"),
                file_name="top5_campanhas_por_roas.csv",
                mime="text/csv"
            )

        st.stop()  # evita executar o restante desta aba quando Populares estiver selecionado

    # ========= DEMAIS DIMENSÕES =========
    dim_to_breakdowns = {
        "Idade": ["age"],
        "Gênero": ["gender"],
        "Idade + Gênero": ["age","gender"],
        "Região": ["region"],
        "País": ["country"],
        "Plataforma": ["publisher_platform"],
        "Posicionamento": ["publisher_platform","platform_position"],
    }

    # Proteção para Posicionamento em nível errado
    level_bd = level
    if dimensao == "Posicionamento" and level_bd not in ["adset", "ad"]:
        level_bd = "adset"

    if dimensao in dim_to_breakdowns:
        bks = dim_to_breakdowns[dimensao]
        df_bd = fetch_insights_breakdown(
            act_id, token, api_version, str(since), str(until), bks, level_bd,
            product_name=st.session_state.get("det_produto")
        )

        if df_bd.empty:
            st.info(f"Sem dados para {dimensao} no período/filtro.")
            st.stop()

        rename_map = {
            "age":"Idade", "gender":"Gênero", "region":"Região", "country":"País",
            "publisher_platform":"Plataforma", "platform_position":"Posicionamento"
        }
        group_cols = [rename_map.get(c, c) for c in bks]

        raw, disp = _agg_and_format(df_bd.rename(columns=rename_map), group_cols)
        if disp.empty:
            st.info(f"Sem dados para {dimensao} após aplicar filtros.")
            st.stop()

        st.subheader(f"Desempenho por {dimensao}")

        base_cols = group_cols + ["Compras","ROAS","Valor usado","Valor de conversão",
                                  "Cliques","LPV","Checkout","Add Pagto"]
        disp = disp.rename(columns={
            "purchases":"Compras","link_clicks":"Cliques","lpv":"LPV",
            "init_checkout":"Checkout","add_payment":"Add Pagto"
        })
        cols_presentes = [c for c in base_cols if c in disp.columns]

        st.dataframe(disp[cols_presentes], use_container_width=True, height=520)

        if len(group_cols) == 1:
            xlab = group_cols[0]
            _bar_chart(raw[xlab], raw["purchases"], f"Compras por {xlab}", xlab, "Compras")
        else:
            idx, col = group_cols
            pvt = raw.pivot_table(index=idx, columns=col, values="purchases", aggfunc="sum").fillna(0)
            fig = go.Figure(data=go.Heatmap(
                z=pvt.values, x=pvt.columns.astype(str), y=pvt.index.astype(str),
                colorbar=dict(title="Compras"),
                hovertemplate=f"{idx}: "+"%{y}<br>"+f"{col}: "+"%{x}<br>Compras: %{z}<extra></extra>"
            ))
            fig.update_layout(
                title=f"Heatmap — Compras por {idx} × {col}",
                height=460, template="plotly_white",
                margin=dict(l=10, r=10, t=48, b=10), separators=",."
            )
            st.plotly_chart(fig, use_container_width=True)

        st.stop()
