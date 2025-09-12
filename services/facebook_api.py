from __future__ import annotations

import json
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from config.constants import ATTR_KEYS, HOUR_BREAKDOWN
from utils.helpers import (
    get_session,
    retry_call,
    ensure_act_prefix,
    to_float,
    chunks_by_days,
)
from utils.parsing import parse_hour_bucket


# ========================= Helpers de actions =========================

def _sum_item(item, allowed_keys: Optional[List[str]] = None) -> float:
    """Usa 'value' quando existir; senão soma SOMENTE as chaves permitidas (ex.: 7d_click, 1d_view)."""
    if not isinstance(item, dict):
        return to_float(item)
    if "value" in item:
        return to_float(item.get("value"))
    keys = allowed_keys or ATTR_KEYS
    s = 0.0
    for k in keys:
        s += to_float(item.get(k))
    return s


def _sum_actions_exact(rows: List[Dict], exact_names: List[str], allowed_keys: Optional[List[str]] = None) -> float:
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


def _sum_actions_contains(rows: List[Dict], substrs: List[str], allowed_keys: Optional[List[str]] = None) -> float:
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


def _pick_purchase_totals(rows: List[Dict], allowed_keys: Optional[List[str]] = None) -> float:
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
    grp: Dict[str, float] = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def _pick_checkout_totals(rows: List[Dict], allowed_keys: Optional[List[str]] = None) -> float:
    """
    Soma Initiate Checkout priorizando omni; senão pega o MAIOR entre variantes (sem duplicar janelas).
    Aceita initiated/initiate e offsite/onsite.
    """
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

    grp: Dict[str, float] = {}
    for r in rows:
        if "initiate" in r["action_type"] and "checkout" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def _pick_add_payment_totals(rows: List[Dict], allowed_keys: Optional[List[str]] = None) -> float:
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

    grp: Dict[str, float] = {}
    for r in rows:
        if "add_payment" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


# ========================= Fetchers principais =========================

def fetch_insights_daily(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
    try_extra_fields: bool = True,
    product_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único ('campaign' recomendado)
    - Usa action_report_time=conversion e action_attribution_windows fixos (paridade com Ads Manager)
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
    - Paraleliza chunks de 30d e usa requests.Session para keep-alive.
    - Opcional: filtering por nome da campanha (product_name) direto na API.
    """
    act_id = ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    base_fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name",
    ]
    extra_fields = ["link_clicks", "landing_page_views"]

    def _fetch_range(_since: str, _until: str, _try_extra: bool) -> List[Dict]:
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
                "field": "campaign.name",
                "operator": "CONTAIN",
                "value": product_name
            }])

        rows_local: List[Dict] = []
        next_url, next_params = base_url, params.copy()
        while next_url:
            sess = get_session()
            resp = retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
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

                # link_clicks / lpv com fallback via actions
                link_clicks = rec.get("link_clicks", None)
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

                lpv = rec.get("landing_page_views", None)
                if lpv is None:
                    lpv = _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS) or \
                          _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS)

                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                revenue_val = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows_local.append({
                    "date": pd.to_datetime(rec.get("date_start")),
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend": to_float(rec.get("spend")),
                    "impressions": to_float(rec.get("impressions")),
                    "clicks": to_float(rec.get("clicks")),
                    "link_clicks": to_float(link_clicks),
                    "lpv": to_float(lpv),
                    "init_checkout": to_float(ic),
                    "add_payment": to_float(api),
                    "purchases": to_float(purchases_cnt),
                    "revenue": to_float(revenue_val),
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
    chunks = list(chunks_by_days(since_str, until_str, max_days=30))
    all_rows: List[Dict] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u, try_extra_fields) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    num_cols = ["spend", "impressions", "clicks", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df.sort_values("date")


def fetch_insights_hourly(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    level: str = "campaign",
) -> pd.DataFrame:
    """
    Coleta por hora em janelas menores (default 30 dias) para evitar code=1 e concatena o resultado.
    Tenta 'conversion' por chunk; se falhar, tenta 'impression'.
    """
    act_id = ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name",
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
            sess = get_session()
            resp = retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
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
                hour_bucket = parse_hour_bucket(rec.get(HOUR_BREAKDOWN))

                link_clicks = rec.get("link_clicks")
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

                lpv = rec.get("landing_page_views")
                if lpv is None:
                    lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS) or
                           _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS) or
                           _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))

                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows.append({
                    "date": pd.to_datetime(rec.get("date_start")),
                    "hour": hour_bucket,
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend": to_float(rec.get("spend")),
                    "impressions": to_float(rec.get("impressions")),
                    "clicks": to_float(rec.get("clicks")),
                    "link_clicks": to_float(link_clicks),
                    "lpv": to_float(lpv),
                    "init_checkout": to_float(ic),
                    "add_payment": to_float(api),
                    "purchases": to_float(pur),
                    "revenue": to_float(rev),
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

        return pd.DataFrame(rows)

    dfs: List[pd.DataFrame] = []
    for s_chunk, u_chunk in chunks_by_days(since_str, until_str, max_days=30):
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
    order = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    df["dow_label"] = df["dow"].map(order)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df.sort_values(["date", "hour"])


def fetch_insights_breakdown(
    act_id: str,
    token: str,
    api_version: str,
    since_str: str,
    until_str: str,
    breakdowns: List[str],
    level: str = "campaign",
    product_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Coleta insights com 1 ou 2 breakdowns. Mantém paridade (conversion + ATTR_KEYS),
    chunking 30d, requests.Session e paralelismo.
    Opcional: filtering por campanha (product_name) direto na API.
    """
    act_id = ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name",
    ]

    def _fetch_range(_since: str, _until: str) -> List[Dict]:
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
            params["filtering"] = json.dumps([{
                "field": "campaign.name", "operator": "CONTAIN", "value": product_name
            }])

        rows: List[Dict] = []
        next_url, next_params = base_url, params.copy()
        while next_url:
            sess = get_session()
            resp = retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
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
                lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS) or
                       _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS) or
                       _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))
                ic = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                base = {
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend": to_float(rec.get("spend")),
                    "impressions": to_float(rec.get("impressions")),
                    "clicks": to_float(rec.get("clicks")),
                    "link_clicks": to_float(link_clicks),
                    "lpv": to_float(lpv),
                    "init_checkout": to_float(ic),
                    "add_payment": to_float(api_),
                    "purchases": to_float(pur),
                    "revenue": to_float(rev),
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
    all_rows: List[Dict] = []
    chunks = list(chunks_by_days(since_str, until_str, max_days=30))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df
