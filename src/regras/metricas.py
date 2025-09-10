import numpy as np

def to_float(x):
    try: return float(x or 0)
    except: return 0.0

def safe_div(n, d):
    n = float(n or 0); d = float(d or 0)
    return (n / d) if d > 0 else np.nan

def rate(a, b): return (a / b) if b and b > 0 else np.nan

# ----- somatórios de actions -----
def _sum_item(item, allowed_keys):
    if not isinstance(item, dict): return to_float(item)
    if "value" in item: return to_float(item.get("value"))
    s = 0.0
    for k in (allowed_keys or []):
        s += to_float(item.get(k))
    return s

def sum_actions_exact(rows, names, allowed_keys):
    if not rows: return 0.0
    names = {str(n).lower() for n in names}
    return float(sum(_sum_item(r, allowed_keys) for r in rows
                     if str(r.get("action_type","")).lower() in names))

def sum_actions_contains(rows, substrs, allowed_keys):
    if not rows: return 0.0
    ss = [str(s).lower() for s in substrs]
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type","")).lower()
        if any(s in at for s in ss):
            tot += _sum_item(r, allowed_keys)
    return float(tot)

def pick_purchase_totals(rows, allowed_keys):
    if not rows: return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni: return float(sum(_sum_item(r, allowed_keys) for r in omni))
    candidates = {"purchase":0.0,"onsite_conversion.purchase":0.0,"offsite_conversion.fb_pixel_purchase":0.0}
    for r in rows:
        at = r["action_type"]
        if at in candidates: candidates[at] += _sum_item(r, allowed_keys)
    if any(v>0 for v in candidates.values()): return float(max(candidates.values()))
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)
