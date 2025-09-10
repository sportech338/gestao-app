import numpy as np
import pandas as pd

def dinheiro_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def pct_br(x):
    return f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X",".") if pd.notnull(x) else ""

def ratio_br(x):
    return f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X",".") if pd.notnull(x) else ""

def int_br(x):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
    except: return ""

def int_signed_br(x):
    try:
        v = int(round(float(x))); s = f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v>0 else (f"-{s}" if v<0 else "0")
    except: return ""
