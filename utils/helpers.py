import time, requests
from datetime import datetime, timedelta
import pandas as pd

_session = None
def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding":"gzip, deflate"})
        _session = s
    return _session

def retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try: return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout","timed out"]):
                time.sleep(base_wait*(2**i)); continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def ensure_act_prefix(ad_account_id: str) -> str:
    s=(ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def to_float(x):
    try: return float(x or 0)
    except: return 0.0

def chunks_by_days(since_str: str, until_str: str, max_days:int=30):
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur=s
    while cur<=u:
        end=min(cur+timedelta(days=max_days-1), u)
        yield str(cur), str(end)
        cur=end+timedelta(days=1)

def filter_by_product(df: pd.DataFrame, produto: str):
    if df is None or df.empty or not produto or produto=="(Todos)": return df
    return df[df["campaign_name"].str.contains(produto, case=False, na=False)].copy()
