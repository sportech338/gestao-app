# ============================ Imports ============================
# Built-in
import json
import time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ============================ Config ============================
APP_TZ = ZoneInfo("America/Sao_Paulo")
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")

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

# ============================ Constantes ============================
ATTR_KEYS = ["7d_click", "1d_view"]  # Janelas de atribuição
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"
MAX_WORKERS = 5

# ============================ Sessão HTTP ============================
_session = None

def _get_session():
    """Sessão requests com keep-alive e compressão."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session

# ============================ Utils ============================

def _retry_call(fn, max_retries: int = 5, base_wait: float = 1.2):
    """Executa uma função com backoff exponencial para erros/transientes comuns."""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "retry", "temporarily unavailable", "timeout", "timed out"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")


def _ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"


def _to_float(x) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


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

# ---------- Somatórios de actions ----------

def _sum_item(item, allowed_keys=None) -> float:
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
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def _pick_checkout_totals(rows, allowed_keys=None) -> float:
    """Soma Initiate Checkout priorizando omni; senão pega o MAIOR entre variantes."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type", "")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] in ("omni_initiated_checkout", "omni_initiate_checkout")]
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
    """Soma Add Payment Info com suporte a omni/onsite/offsite."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type", "")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] in ("omni_add_payment_info", "add_payment_info.omni")]
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

# ---------- Formatadores ----------

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_pct_br(x):
    return (
        f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else ""
    )


def _fmt_ratio_br(x):  # ROAS "1,23x"
    return (
        f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "—"
    )


def _fmt_int_br(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return ""


def _fmt_int_signed_br(x):
    try:
        v = int(round(float(x)))
        s = f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v > 0 else (f"-{s}" if v < 0 else "0")
    except Exception:
        return ""


def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return (n / d) if d > 0 else np.nan

# ---------- Taxas do funil (helpers) ----------
def _daily_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.groupby("date", as_index=False)[["clicks", "lpv", "init_checkout", "purchases", "spend", "impressions"]].sum()
    g["rate_lpv_click"] = np.where(g["clicks"]>0, g["lpv"]/g["clicks"], np.nan)
    g["rate_chk_lpv"]   = np.where(g["lpv"]>0, g["init_checkout"]/g["lpv"], np.nan)
    g["rate_buy_chk"]   = np.where(g["init_checkout"]>0, g["purchases"]/g["init_checkout"], np.nan)
    g["cpc"]  = np.where(g["clicks"]>0, g["spend"]/g["clicks"], np.nan)
    g["ctr"]  = np.where(g["impressions"]>0, g["clicks"]/g["impressions"], np.nan)
    g["dow"]  = g["date"].dt.dayofweek
    return g.sort_values("date")

def _pct_inside_band(series, low, high):
    s = pd.Series(series).dropna()
    if s.empty: return 0.0
    return float(((s>=low) & (s<=high)).mean())

def _mk_rate_card(label, avg, inside_pct):
    st.markdown(
        f'''<div class="kpi-card"><div class="small-muted">{label} (média)</div>
        <div class="big-number">{_fmt_pct_br(avg)}</div>
        <div class="small-muted">Dias dentro da banda: {_fmt_pct_br(inside_pct)}</div></div>''',
        unsafe_allow_html=True,
    )

def _band_color(val, low, high):
    if pd.isna(val): return "⚪"
    if val < low:    return "🔴"
    if val > high:   return "🔵"
    return "🟢"

def _recommendation_from_rates(lpvc, chk, buy, bands):
    recs = []
    def flag(name, v, lo, hi):
        if pd.isna(v): return "sem_dado"
        if v < lo: return "baixo"
        if v > hi: return "alto"
        return "ok"
    f1 = flag("LPV/Cliques", lpvc, *bands["lpv_click"])
    f2 = flag("Checkout/LPV", chk,  *bands["chk_lpv"])
    f3 = flag("Compra/Checkout", buy, *bands["buy_chk"])

    # regras simples
    if f1 == "baixo":
        recs.append(("Topo (tráfego/LP)", "Fortaleça criativos e promessa; teste públicos; melhore velocidade/UX da LP."))
    if f2 == "baixo":
        recs.append(("Meio (intenção/UX)", "Revisar aderência LP→oferta, clareza de preço/benefícios, prova social, atritos do formulário."))
    if f3 == "baixo":
        recs.append(("Fundo (checkout/pagto)", "Simplificar checkout, métodos de pagamento, frete, confiança (selos, garantia)."))
    if not recs:
        recs.append(("Escala", "Bandeiras saudáveis — aumente orçamento nas melhores campanhas e reforce criativos vencedores."))
    return recs

# ===== Recomendação de foco (intensidade + motivo) =====
def _intensity_label(gap_frac: float) -> str:
    """Classifica a intensidade pela distância até a banda mínima (em pontos percentuais)."""
    g = abs(gap_frac or 0.0)  # ex.: 0.06 = 6pp
    if g >= 0.08:
        return "Alta"
    if g >= 0.04:
        return "Média"
    return "Baixa"

def _decide_focus(avg_lpv: float, avg_chk: float, avg_buy: float, bands: dict, purchases_total: int) -> dict:
    """
    Decide foco tático com base nas médias e bandas.
    Retorna: {"foco", "intensidade", "porque", "acoes":[...]}
    """
    def inside(v, lo, hi):
        return (pd.notnull(v)) and (lo <= v <= hi)

    # 1) Caso saudável com volume -> Escala
    if purchases_total >= 20 and all([
        inside(avg_lpv, *bands["lpv_click"]),
        inside(avg_chk,  *bands["chk_lpv"]),
        inside(avg_buy,  *bands["buy_chk"]),
    ]):
        return {
            "foco": "Escala",
            "intensidade": "Média",
            "porque": "Taxas médias dentro das bandas e volume de compras razoável.",
            "acoes": [
                "Aumentar orçamento nas campanhas com melhor ROAS.",
                "Duplicar criativos vencedores (formatos/ângulos).",
                "Expandir audiências (lookalike, broad com exclusões).",
            ],
        }

    # 2) Encontrar o maior 'gap' para baixo (quanto falta até o mínimo da banda)
    gaps = {
        "Topo (Criativo/LP)": bands["lpv_click"][0] - (avg_lpv or 0.0),
        "Meio (Aderência/UX)": bands["chk_lpv"][0] - (avg_chk or 0.0),
        "Fundo (Checkout/Pagto)": bands["buy_chk"][0] - (avg_buy or 0.0),
    }
    foco, gap = max(gaps.items(), key=lambda x: x[1])

    # 3) Se ninguém está abaixo da banda mínima, mas sem volume -> Escala cautelosa
    if gap <= 0:
        return {
            "foco": "Escala",
            "intensidade": "Baixa",
            "porque": "Taxas ok na média, porém volume de compras baixo.",
            "acoes": [
                "Escalar com incrementos de 10–20%/dia.",
                "Abrir novos públicos e testar mais criativos.",
                "Checar tracking (Purchase value/currency).",
            ],
        }

    # 4) Checklists por etapa
    tips = {
        "Topo (Criativo/LP)": [
            "Rotacionar ângulos e hooks dos criativos.",
            "Melhorar promessa/título e velocidade da LP.",
            "Testar públicos (broad, LALs, interesses).",
        ],
        "Meio (Aderência/UX)": [
            "Alinhar LP→oferta (benefícios, preço claro).",
            "Adicionar prova social (reviews, UGC).",
            "Reduzir atritos do formulário (campos, passos).",
        ],
        "Fundo (Checkout/Pagto)": [
            "Simplificar checkout (menos etapas).",
            "Adicionar métodos de pagamento e frete claro.",
            "Selos de confiança e garantia visíveis.",
        ],
    }

    return {
        "foco": foco,
        "intensidade": _intensity_label(gap),
        "porque": f"Maior desvio abaixo da banda mínima nessa etapa ({gap*100:.1f} pp).",
        "acoes": tips.get(foco, []),
    }

# ---------- Gráficos ----------

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
        separators=",.",
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

# ---------- Tempo ----------

def _chunks_by_days(since_str: str, until_str: str, max_days: int = 30):
    """Divide [since, until] em janelas de até max_days (inclusive)."""
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days - 1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)


def _filter_by_product(df: pd.DataFrame, produto: str) -> pd.DataFrame:
    """Filtra pelo nome do produto dentro de campaign_name."""
    if not isinstance(df, pd.DataFrame) or df.empty or not produto or produto == "(Todos)":
        return df
    mask = df["campaign_name"].str.contains(produto, case=False, na=False)
    return df[mask].copy()

# ============================ Coletas ============================

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
    try_extra_fields: bool = True,
    product_name: str | None = None,
) -> pd.DataFrame:
    """Coleta diária com paridade a Ads Manager (conversion + ATTR_KEYS)."""
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
            params["filtering"] = json.dumps(
                [{"field": "campaign.name", "operator": "CONTAIN", "value": product_name}]
            )

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
                raise RuntimeError(
                    f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}"
                )

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
                        lpv = _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS) or _sum_actions_contains(
                            actions, ["landing_page"], allowed_keys=ATTR_KEYS
                        )

                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                revenue_val = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows_local.append(
                    {
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
                        "add_payment": _to_float(api_),
                        "purchases": _to_float(purchases_cnt),
                        "revenue": _to_float(revenue_val),
                    }
                )

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
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u, try_extra_fields) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

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

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df = df.sort_values("date")
    return df


@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
) -> pd.DataFrame:
    """Coleta por hora (dayparting) com fallback para impression se conversion falhar."""
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
                    lpv = (
                        _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                        or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                        or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS)
                    )

                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
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

        return pd.DataFrame(rows)

    # Tenta por "conversion"; se falhar, cai para "impression"
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

    # Campos para a aba de Daypart
    df["dow"] = df["date"].dt.dayofweek
    order = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    df["dow_label"] = df["dow"].map(order)

    # Métrica auxiliar (não obrigatória para a aba)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)

    return df.sort_values(["date", "hour"])


@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_breakdown(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    breakdowns: list[str],
    level: str = "campaign",
    product_name: str | None = None,
) -> pd.DataFrame:
    """Coleta insights com até 2 breakdowns (paridade conversion + ATTR_KEYS)."""
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
            "breakdowns": ",".join(breakdowns[:2]),
        }
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps(
                [{"field": "campaign.name", "operator": "CONTAIN", "value": product_name}]
            )

        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            payload = resp.json()
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                raise RuntimeError(
                    f"Graph API error breakdown | code={err.get('code')} sub={err.get('error_subcode')} | {err.get('message')}"
                )

            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []

                link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)
                lpv = (
                    _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                    or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                    or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS)
                )
                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
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
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df


# ============================ Sidebar ============================

st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign"], index=0)

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
    st.sidebar.caption(f"**Desde:** {since}  \n**Até:** {until}")

st.sidebar.subheader("Bandas de referência (taxas)")
col_b1, col_b2 = st.sidebar.columns(2)
lpv_low  = col_b1.number_input("LPV/Clique (mín)", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
lpv_high = col_b2.number_input("LPV/Clique (máx)", min_value=0.0, max_value=1.0, value=0.60, step=0.01, format="%.2f")
col_b3, col_b4 = st.sidebar.columns(2)
chk_low  = col_b3.number_input("Checkout/LPV (mín)", min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f")
chk_high = col_b4.number_input("Checkout/LPV (máx)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
col_b5, col_b6 = st.sidebar.columns(2)
buy_low  = col_b5.number_input("Compra/Checkout (mín)", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f")
buy_high = col_b6.number_input("Compra/Checkout (máx)", min_value=0.0, max_value=1.0, value=0.55, step=0.01, format="%.2f")

st.sidebar.subheader("Filtros das taxas diárias")
min_clicks_day = st.sidebar.number_input("Ignorar dias com < cliques", min_value=0, value=50, step=10)
mark_weekends  = st.sidebar.checkbox("Marcar fins de semana no gráfico", value=True)

_BANDS = {
    "lpv_click": (lpv_low, lpv_high),
    "chk_lpv":   (chk_low,  chk_high),
    "buy_chk":   (buy_low,  buy_high),
}


ready = bool(act_id and token)

# ============================ Tela ============================
st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Checkout → Add Pagamento → Compra. Tudo alinhado ao período selecionado.")

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

# ============================ Coleta Inicial ============================
with st.spinner("Buscando dados da Meta…"):
    df_daily = fetch_insights_daily(
        act_id=act_id,
        token=token,
        api_version=api_version,
        since_str=str(since),
        until_str=str(until),
        level=level,
        product_name=st.session_state.get("daily_produto"),  # pode ser None
    )

df_hourly = None  # carregado sob demanda

if df_daily.empty and (df_hourly is None or df_hourly.empty):
    st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
    st.stop()

# ============================ Funções auxiliares extras ============================

def _kpis_sum(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(spend=0.0, purchases=0.0, revenue=0.0, clicks=0.0, lpv=0.0,
                    init_checkout=0.0, add_payment=0.0, roas=np.nan, cpa=np.nan)
    spend = float(df["spend"].sum())
    revenue = float(df["revenue"].sum())
    purchases = float(df["purchases"].sum())
    clicks = float(df["clicks"].sum())
    lpv = float(df["lpv"].sum())
    ic = float(df["init_checkout"].sum())
    api_ = float(df["add_payment"].sum())
    roas = revenue / spend if spend > 0 else np.nan
    cpa = spend / purchases if purchases > 0 else np.nan
    return dict(spend=spend, purchases=purchases, revenue=revenue, clicks=clicks, lpv=lpv,
                init_checkout=ic, add_payment=api_, roas=roas, cpa=cpa)

def _diff(a, b):
    """retorna (abs, pct) com sinal relativo a A->B"""
    if pd.isna(a) and pd.isna(b):
        return 0.0, np.nan
    if a == 0:
        return (b - a), np.nan
    return (b - a), (b - a) / a

def _fmt_diff(abs_v, pct_v, money=False, ratio=False, pct=False):
    if pct:
        abs_s = _fmt_pct_br(abs_v)
    elif ratio:
        abs_s = _fmt_ratio_br(abs_v)
    elif money:
        abs_s = _fmt_money_br(abs_v)
    else:
        abs_s = _fmt_int_signed_br(abs_v)
    pct_s = _fmt_pct_br(pct_v) if pd.notnull(pct_v) else "—"
    sign = "🔼" if abs_v > 0 else ("🔽" if abs_v < 0 else "⟲")
    return f"{sign} {abs_s} ({pct_s})"

# ============================ Abas ============================
tab_daily, tab_daypart, tab_detail, tab_compare, tab_budget = st.tabs(
    ["📅 Visão diária", "⏱️ Horários (principal)", "📊 Detalhamento", "⚖️ A vs B", "💸 Recomendação de Verba"]
)

# -------------------- ABA 1: VISÃO DIÁRIA --------------------
with tab_daily:
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
            st.caption("⚠️ Exibindo com símbolo **R$** apenas para **formatação visual**. Os valores permanecem na moeda da conta.")

    st.caption(f"Moeda da conta detectada: **{currency_detected}** — Exibindo como: **{currency_label}**")

    produto_sel_daily = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daily_produto")
    df_daily_view = _filter_by_product(df_daily, produto_sel_daily)

    if df_daily_view.empty:
        st.info("Sem dados para o produto selecionado nesse período.")
        st.stop()

    if produto_sel_daily != "(Todos)":
        st.caption(f"🔎 Filtrando por produto: **{produto_sel_daily}**")

    tot = _kpis_sum(df_daily_view)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Valor usado</div>'
            f'<div class="big-number">{_fmt_money_br(tot["spend"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Vendas</div>'
            f'<div class="big-number">{int(round(tot["purchases"])):,}</div></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">Valor de conversão</div>'
            f'<div class="big-number">{_fmt_money_br(tot["revenue"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="kpi-card"><div class="small-muted">ROAS</div>'
            f'<div class="big-number">{_fmt_ratio_br(tot["roas"])}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Série diária — Investimento e Conversão")
    daily = df_daily_view.groupby("date", as_index=False)[["spend", "revenue", "purchases"]].sum()
    daily_pt = daily.rename(columns={"spend": "Gasto", "revenue": "Faturamento"})
    st.line_chart(daily_pt.set_index("date")[["Faturamento", "Gasto"]])
    st.caption("Linhas diárias de Receita e Gasto. Vendas na tabela abaixo.")

    st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")
    values_total = [
        int(round(df_daily_view["link_clicks"].sum())),
        int(round(df_daily_view["lpv"].sum())),
        int(round(df_daily_view["init_checkout"].sum())),
        int(round(df_daily_view["add_payment"].sum())),
        int(round(df_daily_view["purchases"].sum())),
    ]
    labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
    force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
    values_plot = enforce_monotonic(values_total) if force_shape else values_total
    st.plotly_chart(funnel_fig(labels_total, values_plot, title="Funil do período"), use_container_width=True)

    core_rows = [
        ("LPV / Cliques", _safe_div(values_total[1], values_total[0])),
        ("Checkout / LPV", _safe_div(values_total[2], values_total[1])),
        ("Compra / Checkout", _safe_div(values_total[4], values_total[2])),
    ]
    extras_def = {
        "Add Pagto / Checkout": _safe_div(values_total[3], values_total[2]),
        "Compra / Add Pagto": _safe_div(values_total[4], values_total[3]),
        "Compra / LPV": _safe_div(values_total[4], values_total[1]),
        "Compra / Cliques": _safe_div(values_total[4], values_total[0]),
        "Checkout / Cliques": _safe_div(values_total[2], values_total[0]),
        "Add Pagto / LPV": _safe_div(values_total[3], values_total[1]),
    }
    with st.expander("Comparar outras taxas (opcional)"):
        extras_selected = st.multiselect("Escolha métricas adicionais para visualizar:", options=list(extras_def.keys()), default=[])
    rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
    sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
    sr["Valor"] = sr["Valor"].map(lambda x: _fmt_pct_br(x) if pd.notnull(x) else "")
    st.dataframe(sr, use_container_width=True, height=160 + 36 * len(extras_selected))

    st.subheader("Taxas do funil por dia (com bandas)")

    daily_all = _daily_rates(df_daily_view)
    if daily_all.empty:
        st.info("Sem série diária suficiente para taxas.")
    else:
        # aplica filtro de qualidade por dia
        daily_f = daily_all.copy()
        daily_f.loc[daily_f["clicks"] < min_clicks_day, ["rate_lpv_click","rate_chk_lpv","rate_buy_chk"]] = np.nan

        # cartões-resumo
        cA, cB, cC = st.columns(3)
        with cA:
            _mk_rate_card(
                "LPV / Cliques",
                daily_f["rate_lpv_click"].mean(),
                _pct_inside_band(daily_f["rate_lpv_click"], *_BANDS["lpv_click"])
            )
        with cB:
            _mk_rate_card(
                "Checkout / LPV",
                daily_f["rate_chk_lpv"].mean(),
                _pct_inside_band(daily_f["rate_chk_lpv"], *_BANDS["chk_lpv"])
            )
        with cC:
            _mk_rate_card(
                "Compra / Checkout",
                daily_f["rate_buy_chk"].mean(),
                _pct_inside_band(daily_f["rate_buy_chk"], *_BANDS["buy_chk"])
            )

        # --- Comparativo rápido vs período anterior ---
        with st.expander("Comparativo das taxas vs período anterior"):
            # Janela anterior com mesmo tamanho de dias
            _days = (until - since).days + 1
            prev_since = since - timedelta(days=_days)
            prev_until = since - timedelta(days=1)

            with st.spinner("Calculando período anterior…"):
                df_prev = fetch_insights_daily(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(prev_since),
                    until_str=str(prev_until),
                    level=level,
                    product_name=st.session_state.get("daily_produto"),
                )

            if df_prev is None or df_prev.empty:
                st.caption("Sem dados no período anterior.")
            else:
                prev_all = _daily_rates(_filter_by_product(df_prev, produto_sel_daily))
                # aplicar o mesmo filtro de qualidade
                prev_f = prev_all.copy()
                prev_f.loc[prev_f["clicks"] < min_clicks_day, ["rate_lpv_click","rate_chk_lpv","rate_buy_chk"]] = np.nan

                # Médias atuais vs anteriores
                cur = {
                    "lpv_click": daily_f["rate_lpv_click"].mean(),
                    "chk_lpv":   daily_f["rate_chk_lpv"].mean(),
                    "buy_chk":   daily_f["rate_buy_chk"].mean(),
                }
                prv = {
                    "lpv_click": prev_f["rate_lpv_click"].mean(),
                    "chk_lpv":   prev_f["rate_chk_lpv"].mean(),
                    "buy_chk":   prev_f["rate_buy_chk"].mean(),
                }

                def _pp(x):  # em pontos percentuais
                    return f"{(x*100):.2f} pp" if pd.notnull(x) else "—"

                cA2, cB2, cC2 = st.columns(3)
                with cA2:
                    delta = cur["lpv_click"] - prv["lpv_click"] if pd.notnull(prv["lpv_click"]) else np.nan
                    st.markdown(
                        f"**LPV/Cliques**  \nAtual: {_fmt_pct_br(cur['lpv_click'])}  \nAnterior: {_fmt_pct_br(prv['lpv_click'])}  \nΔ: {_pp(delta)}"
                    )
                with cB2:
                    delta = cur["chk_lpv"] - prv["chk_lpv"] if pd.notnull(prv["chk_lpv"]) else np.nan
                    st.markdown(
                        f"**Checkout/LPV**  \nAtual: {_fmt_pct_br(cur['chk_lpv'])}  \nAnterior: {_fmt_pct_br(prv['chk_lpv'])}  \nΔ: {_pp(delta)}"
                    )
                with cC2:
                    delta = cur["buy_chk"] - prv["buy_chk"] if pd.notnull(prv["buy_chk"]) else np.nan
                    st.markdown(
                        f"**Compra/Checkout**  \nAtual: {_fmt_pct_br(cur['buy_chk'])}  \nAnterior: {_fmt_pct_br(prv['buy_chk'])}  \nΔ: {_pp(delta)}"
                    )

        # função de gráfico com banda
        def rate_fig(dates, values, low, high, title):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dates, y=values, mode="lines+markers", name="Taxa",
                    hovertemplate="%{y:.2%}<extra></extra>"
                )
            )
            fig.add_hrect(y0=low, y1=high, fillcolor="LightGreen", opacity=0.35, line_width=0)
            if mark_weekends:
                for d, dow in zip(daily_f["date"], daily_f["dow"]):
                    if dow >= 5:  # sábado/domingo
                        fig.add_vrect(x0=d, x1=d + pd.Timedelta(days=1), fillcolor="LightGray", opacity=0.15, line_width=0)
            fig.update_layout(
                title=title, template="plotly_white", height=320,
                yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=48, b=10),
                separators=",."
            )
            return fig

        colg1, colg2 = st.columns(2)
        with colg1:
            st.plotly_chart(
                rate_fig(daily_f["date"], daily_f["rate_lpv_click"], *_BANDS["lpv_click"], "LPV / Cliques"),
                use_container_width=True
            )
            st.plotly_chart(
                rate_fig(daily_f["date"], daily_f["rate_buy_chk"], *_BANDS["buy_chk"], "Compra / Checkout"),
                use_container_width=True
            )
        with colg2:
            st.plotly_chart(
                rate_fig(daily_f["date"], daily_f["rate_chk_lpv"], *_BANDS["chk_lpv"], "Checkout / LPV"),
                use_container_width=True
            )

        # linha rápida de status + recomendação
        avg_lpv = daily_f["rate_lpv_click"].mean()
        avg_chk = daily_f["rate_chk_lpv"].mean()
        avg_buy = daily_f["rate_buy_chk"].mean()
        s1 = _band_color(avg_lpv, *_BANDS["lpv_click"])
        s2 = _band_color(avg_chk, *_BANDS["chk_lpv"])
        s3 = _band_color(avg_buy, *_BANDS["buy_chk"])
        st.markdown(f"**Status médio:** LPV/Clique {s1} • Checkout/LPV {s2} • Compra/Checkout {s3}")

        recs = _recommendation_from_rates(avg_lpv, avg_chk, avg_buy, _BANDS)
        with st.expander("🔔 Recomendações automáticas"):
            for area, tip in recs:
                st.markdown(f"- **{area}:** {tip}")

        # --- Para onde vai a verba? (automático) ---
        with st.expander("🔔 Para onde vai a verba? (automático)"):
            total_pur = int(round(df_daily_view["purchases"].sum()))
            foco = _decide_focus(avg_lpv, avg_chk, avg_buy, _BANDS, total_pur)

            st.markdown(f"**Foco sugerido:** {foco['foco']}  \n**Intensidade:** {foco['intensidade']}")
            st.caption(f"Motivo: {foco['porque']}")
            if foco.get("acoes"):
                st.markdown("**Ações recomendadas (checklist):**")
                for a in foco["acoes"]:
                    st.markdown(f"- {a}")



# -------------------- ABA 2: HORÁRIOS (DAYPART) --------------------
with tab_daypart:
    st.subheader("Desempenho por horário e dia da semana")
    st.caption("Carregaremos esta visão somente quando você abrir a aba, para economizar requisições.")
    load_hourly = st.checkbox("Carregar dados por hora", value=False)
    if load_hourly:
        with st.spinner("Carregando horário por horário…"):
            df_hourly = fetch_insights_hourly(
                act_id=act_id,
                token=token,
                api_version=api_version,
                since_str=str(since),
                until_str=str(until),
                level=level,
            )

        if df_hourly is None or df_hourly.empty:
            st.info("Sem dados horários no período.")
        else:
            produto_sel_hour = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="hourly_produto")
            df_hour_view = _filter_by_product(df_hourly, produto_sel_hour)
            if df_hour_view.empty:
                st.info("Sem dados para o produto nesta aba.")
            else:
                pivot = df_hour_view.groupby(["dow_label", "hour"], as_index=False)[["spend", "revenue"]].sum()
                pivot["roas"] = np.where(pivot["spend"] > 0, pivot["revenue"] / pivot["spend"], np.nan)
                dows = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
                mat = np.full((7, 24), np.nan)
                for _, r in pivot.iterrows():
                    i = dows.index(r["dow_label"]) if r["dow_label"] in dows else None
                    j = int(r["hour"]) if pd.notnull(r["hour"]) else None
                    if i is not None and j is not None:
                        mat[i, j] = r["roas"]

                hm = go.Figure(
                    data=go.Heatmap(
                        z=mat, x=list(range(24)), y=dows, coloraxis="coloraxis",
                        hovertemplate="Dia %{y} • %{x}:00<br>ROAS: %{z:.2f}x<extra></extra>",
                    )
                )
                hm.update_layout(
                    title="ROAS por hora e dia da semana", xaxis_title="Hora do dia", yaxis_title="Dia da semana",
                    coloraxis=dict(colorscale="Blues"), height=520, template="plotly_white",
                    margin=dict(l=10, r=10, t=48, b=10), separators=",.",
                )
                st.plotly_chart(hm, use_container_width=True)
                st.caption("Dica: priorize janelas com **ROAS mais alto** e volume razoável.")

# -------------------- ABA 3: DETALHAMENTO --------------------
with tab_detail:
    st.subheader("Detalhamento por breakdown")
    st.caption("Selecione até 2 dimensões.")
    BREAKDOWN_OPTIONS = [
        "age", "gender", "country", "region", "dma", "publisher_platform",
        "impression_device", "device_platform", "placement", "product_id",
    ]
    try:
        bks = st.multiselect("Breakdowns", BREAKDOWN_OPTIONS, default=["age", "gender"], max_selections=2)
    except TypeError:
        bks = st.multiselect("Breakdowns", BREAKDOWN_OPTIONS, default=["age", "gender"])
        bks = bks[:2]
    produto_sel_detail = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="detail_prod")
    if not bks:
        st.info("Escolha pelo menos 1 breakdown.")
    else:
        with st.spinner("Coletando breakdown…"):
            df_bk = fetch_insights_breakdown(
                act_id=act_id, token=token, api_version=api_version,
                since_str=str(since), until_str=str(until),
                breakdowns=bks, level=level, product_name=produto_sel_detail,
            )
        if df_bk.empty:
            st.info("Sem dados para o(s) breakdown(s) selecionado(s).")
        else:
            cols = ["campaign_name", *bks, "spend", "revenue", "purchases", "clicks", "lpv", "init_checkout", "add_payment"]
            cols = [c for c in cols if c in df_bk.columns]
            df_view = df_bk[cols].copy().sort_values("revenue", ascending=False)
            st.dataframe(df_view, use_container_width=True, height=480)

            group_cols = [c for c in bks if c in df_bk.columns]
            if group_cols:
                agg = df_bk.groupby(group_cols, as_index=False)[["spend", "revenue", "purchases"]].sum()
                agg["ROAS"] = np.where(agg["spend"] > 0, agg["revenue"] / agg["spend"], np.nan)
                st.markdown("**Top segmentos (por ROAS)**")
                st.dataframe(agg.sort_values("ROAS", ascending=False), use_container_width=True, height=360)

# -------------------- ABA 4: COMPARATIVO A vs B --------------------
with tab_compare:
    st.subheader("Comparativo de Períodos — A vs B")
    colA, colB = st.columns(2)
    with colA:
        a_since = st.date_input("A — Desde", value=since)
        a_until = st.date_input("A — Até", value=until)
    with colB:
        b_since = st.date_input("B — Desde", value=since - timedelta(days=(until - since).days + 1))
        b_until = st.date_input("B — Até", value=a_since - timedelta(days=1))

    produto_cmp = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="cmp_prod")

    go_btn = st.button("Comparar períodos")
    if go_btn:
        with st.spinner("Buscando A e B…"):
            dfA = fetch_insights_daily(act_id, token, api_version, str(a_since), str(a_until), level=level,
                                       product_name=produto_cmp if produto_cmp != "(Todos)" else None)
            dfB = fetch_insights_daily(act_id, token, api_version, str(b_since), str(b_until), level=level,
                                       product_name=produto_cmp if produto_cmp != "(Todos)" else None)
        if dfA.empty or dfB.empty:
            st.info("Sem dados em A ou B para comparar.")
        else:
            def _row(label, a, b, money=False, ratio=False, pct=False):
                abs_v, pct_v = _diff(a, b)

                if pct:
                    fmtA = _fmt_pct_br(a)
                    fmtB = _fmt_pct_br(b)
                elif money:
                    fmtA = _fmt_money_br(a)
                    fmtB = _fmt_money_br(b)
                elif ratio:
                    fmtA = _fmt_ratio_br(a)
                    fmtB = _fmt_ratio_br(b)
                else:
                    fmtA = _fmt_int_br(a)
                    fmtB = _fmt_int_br(b)

                return [label, fmtA, fmtB, _fmt_diff(abs_v, pct_v, money=money, ratio=ratio, pct=pct)]

            # KPIs agregados de A e B
            kA, kB = _kpis_sum(dfA), _kpis_sum(dfB)

            # métricas adicionais
            cpcA = (dfA["spend"].sum() / dfA["clicks"].sum()) if dfA["clicks"].sum() > 0 else np.nan
            cpcB = (dfB["spend"].sum() / dfB["clicks"].sum()) if dfB["clicks"].sum() > 0 else np.nan
            ctrA = (dfA["clicks"].sum() / dfA["impressions"].sum()) if dfA["impressions"].sum() > 0 else np.nan
            ctrB = (dfB["clicks"].sum() / dfB["impressions"].sum()) if dfB["impressions"].sum() > 0 else np.nan

            rows = []
            rows.append(_row("Gasto",     kA["spend"],        kB["spend"], money=True))
            rows.append(_row("Receita",   kA["revenue"],      kB["revenue"], money=True))
            rows.append(_row("Vendas",    kA["purchases"],    kB["purchases"]))
            rows.append(_row("ROAS",      kA["roas"],         kB["roas"], ratio=True))
            rows.append(_row("CPA",       kA["cpa"],          kB["cpa"], money=True))
            rows.append(_row("CPC",       cpcA,               cpcB, money=True))
            rows.append(_row("CTR",       ctrA,               ctrB, pct=True))
            rows.append(_row("Cliques",   kA["clicks"],       kB["clicks"]))
            rows.append(_row("LPV",       kA["lpv"],          kB["lpv"]))
            rows.append(_row("Checkout",  kA["init_checkout"], kB["init_checkout"]))
            rows.append(_row("Add Pagto", kA["add_payment"],   kB["add_payment"]))

            df_cmp = pd.DataFrame(rows, columns=["Métrica", "Período A", "Período B", "Δ B vs A"])
            st.dataframe(df_cmp, use_container_width=True, height=520)


# -------------------- ABA 5: RECOMENDAÇÃO DE VERBA --------------------
with tab_budget:
    st.subheader("Sugestão de Distribuição de Verba Diária")
    st.caption("Baseado no desempenho recente por campanha (ROAS e CPA).")
    produto_budget = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="budget_prod")
    df_budget = _filter_by_product(df_daily, produto_budget)

    if df_budget.empty:
        st.info("Sem dados para recomendar verba.")
    else:
        # KPIs por campanha
        k = df_budget.groupby("campaign_name", as_index=False)[["spend", "revenue", "purchases"]].sum()
        k["ROAS"] = np.where(k["spend"] > 0, k["revenue"] / k["spend"], np.nan)
        k["CPA"] = np.where(k["purchases"] > 0, k["spend"] / k["purchases"], np.nan)

        # Pesos: metade por ROAS normalizado, metade por conversões (sinal de escala)
        roas_norm = (k["ROAS"] / k["ROAS"].max()).fillna(0)
        conv_norm = (k["purchases"] / k["purchases"].max()).fillna(0)
        k["score"] = 0.5 * roas_norm + 0.5 * conv_norm

        total_daily_budget = st.number_input("Verba total diária (R$)", min_value=0.0, value=float(k["spend"].mean() * 1.2) if len(k) else 0.0, step=50.0)
        if total_daily_budget <= 0:
            st.info("Informe uma verba total diária maior que 0.")
        else:
            # Evita divisão por zero se todos os scores forem 0
            if k["score"].sum() == 0:
                k["score"] = 1.0

            k["budget_sugesto"] = total_daily_budget * (k["score"] / k["score"].sum())
            k = k.sort_values("budget_sugesto", ascending=False)

            show = k[["campaign_name", "ROAS", "CPA", "purchases", "budget_sugesto"]].copy()
            show.rename(columns={"campaign_name": "Campanha", "purchases": "Vendas", "budget_sugesto": "Sugestão (R$/dia)"}, inplace=True)
            show["Sugestão (R$/dia)"] = show["Sugestão (R$/dia)"].map(_fmt_money_br)
            show["ROAS"] = show["ROAS"].map(_fmt_ratio_br)
            show["CPA"] = show["CPA"].map(_fmt_money_br)

            st.dataframe(show, use_container_width=True, height=420)
            st.caption("Regra simples: 50% peso ROAS + 50% peso volume de vendas. Ajuste manualmente conforme estratégia.")
