# lib/meta_helpers.py
import time, json, numpy as np, pandas as pd, requests
from datetime import timedelta, datetime, date
from zoneinfo import ZoneInfo

# ==== Constantes ====
APP_TZ = ZoneInfo("America/Sao_Paulo")
ATTR_KEYS = ["7d_click", "1d_view"]
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"

# ==== Sessão HTTP reutilizável ====
_session = None
def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session

# ==== Helpers genéricos ====  (cole os seus aqui)
# === PASTE: _retry_call -> renomeie para retry_call ===
def retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit","retry","temporarily unavailable","timeout","timed out"]):
                time.sleep(base_wait * (2 ** i)); continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

# === PASTE: _ensure_act_prefix -> ensure_act_prefix ===
def ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

# === PASTE: _to_float/_sum_item/_sum_actions_exact/_sum_actions_contains/_pick_purchase_totals ===
def to_float(x):
    try: return float(x or 0)
    except: return 0.0

def sum_item(item, allowed_keys=None):
    if not isinstance(item, dict):
        return to_float(item)
    if "value" in item:
        return to_float(item.get("value"))
    keys = allowed_keys or ATTR_KEYS
    s = 0.0
    for k in keys: s += to_float(item.get(k))
    return s

def sum_actions_exact(rows, exact_names, allowed_keys=None) -> float:
    if not rows: return 0.0
    names = {str(n).lower() for n in exact_names}
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type","")).lower()
        if at in names: tot += sum_item(r, allowed_keys)
    return float(tot)

def sum_actions_contains(rows, substrs, allowed_keys=None) -> float:
    if not rows: return 0.0
    ss = [str(s).lower() for s in substrs]
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type","")).lower()
        if any(s in at for s in ss): tot += sum_item(r, allowed_keys)
    return float(tot)

def pick_purchase_totals(rows, allowed_keys=None) -> float:
    if not rows: return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni: return float(sum(sum_item(r, allowed_keys) for r in omni))
    candidates = {"purchase":0.0,"onsite_conversion.purchase":0.0,"offsite_conversion.fb_pixel_purchase":0.0}
    for r in rows:
        at = r["action_type"]
        if at in candidates: candidates[at] += sum_item(r, allowed_keys)
    if any(v>0 for v in candidates.values()):
        return float(max(candidates.values()))
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"],0.0)
            grp[r["action_type"]] += sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

# === PASTE: formatadores / utilitários ===
def fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def enforce_monotonic(values):
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

def safe_div(n, d):
    n = float(n or 0); d = float(d or 0)
    return (n / d) if d > 0 else np.nan

def fmt_pct_br(x):
    import pandas as pd
    return (f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x) else "")

def fmt_ratio_br(x):
    import pandas as pd
    return (f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x) else "")

def fmt_int_br(x):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
    except: return ""

def chunks_by_days(since_str: str, until_str: str, max_days: int = 30):
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days-1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)

def parse_hour_bucket(h):
    if h is None: return None
    try:
        s = str(h).strip()
        val = int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except:
        return None
