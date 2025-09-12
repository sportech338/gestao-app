import numpy as np, pandas as pd

def fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_ratio_br(x):
    return f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "—"

def fmt_pct_br(x):
    return f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "—"

def fmt_int_br(x):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
    except: return "—"

def fmt_int_signed_br(x):
    try:
        v=int(round(float(x))); s=f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v>0 else (f"-{s}" if v<0 else "0")
    except: return "—"

def safe_div(n,d):
    n=float(n or 0); d=float(d or 0)
    return (n/d) if d>0 else np.nan
