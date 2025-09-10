import numpy as np
import pandas as pd

def _intensity_label(share):
    if not np.isfinite(share): return "Baixa"
    if share > 0.60: return "Alta"
    if share >= 0.30: return "Média"
    return "Baixa"

def decide_foco(r1, r2, r3, clicks, lpv, co, addpay, purch,
                bm_r1, bm_r2, bm_r3, min_clicks, min_lpv, min_co, min_purch,
                split_rmk=True):
    healthy = (
        (pd.notnull(r1) and r1 >= bm_r1/100.0) and
        (pd.notnull(r2) and r2 >= bm_r2/100.0) and
        (pd.notnull(r3) and r3 >= bm_r3/100.0) and
        (float(purch or 0) >= float(min_purch or 0))
    )
    low_volume_guard = (float(clicks or 0) < float(min_clicks or 0)) or \
                       (float(lpv or 0)    < float(min_lpv or 0))    or \
                       (float(co or 0)     < float(min_co or 0))

    drop1 = max(0.0, float(clicks or 0) - float(lpv or 0))
    drop2 = max(0.0, float(lpv or 0)    - float(co or 0))
    drop3a= max(0.0, float(co or 0)     - float(addpay or 0))
    drop3b= max(0.0, float(addpay or 0) - float(purch or 0))

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
        gaps = {"Teste de criativo": drop1, "Teste de interesse": drop2, "Remarketing (fundo do funil)": drop3a + drop3b}

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
""")
