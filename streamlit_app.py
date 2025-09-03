# app.py — Meta Ads com Funil completo
import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta
import plotly.graph_objects as go

# =============== Config & Estilos ===============
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# =============== Helpers de rede/parse ===============
def _retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit","retry","temporarily unavailable","timeout"]):
                time.sleep(base_wait * (2 ** i)); continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def _ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def _to_float(x):
    try: return float(x or 0)
    except: return 0.0

def _sum_item(item: dict) -> float:
    """Usa 'value' quando existir; senão soma chaves numéricas (ex.: 7d_click, 1d_view...)."""
    if not isinstance(item, dict): return _to_float(item)
    if "value" in item: return _to_float(item.get("value"))
    s = 0.0
    for k, v in item.items():
        if k not in {"action_type","action_target","action_destination","action_device","action_channel",
                     "action_canvas_component_name","action_reaction","value"}:
            s += _to_float(v)
    return s

def _sum_actions_exact(rows: list, exact_names: list) -> float:
    """Soma totals de actions pelos nomes exatos (case-insensitive)."""
    if not rows: return 0.0
    names = {n.lower() for n in exact_names}
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type","")).lower()
        if at in names:
            tot += _sum_item(r)
    return float(tot)

def _sum_actions_contains(rows: list, substrs: list) -> float:
    """Soma totals de actions que CONTENHAM qualquer substring (fallback)."""
    if not rows: return 0.0
    ss = [s.lower() for s in substrs]
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type","")).lower()
        if any(s in at for s in ss):
            tot += _sum_item(r)
    return float(tot)

def _pick_purchase_totals(rows: list) -> float:
    """Prioriza omni_purchase; se ausente, pega o MAIOR entre tipos específicos para evitar duplicação."""
    if not rows: return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni:
        return float(sum(_sum_item(r) for r in omni))
    candidates = {
        "purchase": 0.0,
        "onsite_conversion.purchase": 0.0,
        "offsite_conversion.fb_pixel_purchase": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r)
    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))
    # fallback bem amplo
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r)
    return float(max(grp.values()) if grp else 0.0)

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def funnel_fig(labels, values, title=None):
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value",             # <<< mostra só os valores
            textposition="inside",
            opacity=0.95,
            connector={"line": {"dash": "dot", "width": 1}},
        )
    )
    fig.update_layout(
        title=title or "",
        margin=dict(l=12, r=12, t=40, b=12),
        height=440,
        template="plotly_white",
        separators=",.",                  # pt-BR
    )
    return fig



def enforce_monotonic(values):
    """Garante formato de funil: cada etapa <= etapa anterior (só para o desenho)."""
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

# ==== Helpers de comparação ====
def _rate(a, b):
    return (a / b) if b and b > 0 else np.nan

def _fmt_pct_br(x):
    return (f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x) else "")

def _fmt_ratio_br(x):  # ROAS "1,23x"
    return (f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x) else "")

def _safe_div(n, d):
    n = float(n or 0); d = float(d or 0)
    return (n / d) if d > 0 else np.nan

def _drivers_decomp(clicksA, lpvA, chkA, purA, revA,
                    clicksB, lpvB, chkB, purB, revB):
    """
    Decompõe ΔReceita (B - A) em 5 fatores multiplicativos:
    Receita = Cliques * (LPV/Cliques) * (Checkout/LPV) * (Compra/Checkout) * (Receita/Compra)
    Usa média de duas ordens (A→B e B→A) para reduzir viés.
    """
    # Fatores A
    r1A = _safe_div(lpvA, clicksA)
    r2A = _safe_div(chkA, lpvA)
    r3A = _safe_div(purA, chkA)
    aovA = _safe_div(revA, purA)

    # Fatores B
    r1B = _safe_div(lpvB, clicksB)
    r2B = _safe_div(chkB, lpvB)
    r3B = _safe_div(purB, chkB)
    aovB = _safe_div(revB, purB)

    fA = [clicksA, r1A, r2A, r3A, aovA]
    fB = [clicksB, r1B, r2B, r3B, aovB]
    labels = ["Cliques", "LPV/Cliques", "Checkout/LPV", "Compra/Checkout", "Ticket médio"]

    def _step_contrib(f_from, f_to, order):
        cur = f_from[:]
        contrib = [0.0]*5
        for idx in order:
            before = np.prod([v for v in cur if pd.notnull(v)]) if all(pd.notnull(v) for v in cur) else 0.0
            cur[idx] = f_to[idx]
            after  = np.prod([v for v in cur if pd.notnull(v)]) if all(pd.notnull(v) for v in cur) else 0.0
            contrib[idx] += (after - before)
        return contrib

    order1 = [0,1,2,3,4]
    order2 = [4,3,2,1,0]
    c1 = _step_contrib(fA[:], fB[:], order1)
    c2 = _step_contrib(fA[:], fB[:], order2)
    contrib = [(c1[i]+c2[i])/2 for i in range(5)]
    return labels, contrib

# =============== Coleta (com fallback de campos extras) ===============
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único ('campaign' recomendado)
    - NÃO envia action_types/action_attribution_windows
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    base_fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]
    extra_fields = ["link_clicks","landing_page_views"]  # alguns setups dão #100; se der, refazemos sem eles

    fields = base_fields + (extra_fields if try_extra_fields else [])
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({"since": since_str, "until": until_str}),
        "time_increment": 1,
        "fields": ",".join(fields),
        "limit": 500,
    }

    rows, next_url, next_params = [], base_url, params.copy()
    while next_url:
        resp = _retry_call(lambda: requests.get(next_url, params=next_params, timeout=90))
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("Resposta inválida da Graph API.")
        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
            if code == 100 and try_extra_fields:
                # refaz sem extras
                return fetch_insights_daily(act_id, token, api_version, since_str, until_str, level, try_extra_fields=False)
            raise RuntimeError(f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}")

        for rec in payload.get("data", []):
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            # ---- Cliques (preferir field; fallback via actions: 'link_click')
            link_clicks = rec.get("link_clicks", None)
            if link_clicks is None:
                link_clicks = _sum_actions_exact(actions, ["link_click"])

            # ---- LPV (preferir field; fallback via actions: 'landing_page_view' ou 'view_content')
            lpv = rec.get("landing_page_views", None)
            if lpv is None:
                lpv = _sum_actions_exact(actions, ["landing_page_view"])
                if lpv == 0:
                    lpv = _sum_actions_exact(actions, ["view_content"]) or _sum_actions_contains(actions, ["landing_page"])

            # ---- Initiate Checkout (finalização de compra)
            ic = _sum_actions_exact(actions, ["initiate_checkout"])

            # ---- Add Payment Info (add informações de pagamento)
            api = _sum_actions_exact(actions, ["add_payment_info"])

            # ---- Purchase (qtd) e Revenue (valor)
            purchases_cnt = _pick_purchase_totals(actions)
            revenue_val   = _pick_purchase_totals(action_values)

            rows.append({
                "date":           pd.to_datetime(rec.get("date_start")),
                "currency":       rec.get("account_currency", "BRL"),
                "campaign_id":    rec.get("campaign_id", ""),
                "campaign_name":  rec.get("campaign_name", ""),

                # métricas básicas
                "spend":          _to_float(rec.get("spend")),
                "impressions":    _to_float(rec.get("impressions")),
                "clicks":         _to_float(rec.get("clicks")),

                # funil
                "link_clicks":    _to_float(link_clicks),
                "lpv":            _to_float(lpv),
                "init_checkout":  _to_float(ic),
                "add_payment":    _to_float(api),
                "purchases":      _to_float(purchases_cnt),
                "revenue":        _to_float(revenue_val),
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy()
            next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if df.empty: return df

    num_cols = ["spend","impressions","clicks","link_clicks","lpv","init_checkout","add_payment","purchases","revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Métricas derivadas (do período/dia; taxas serão calculadas em agregações)
    df["roas"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
    df = df.sort_values("date")
    return df

# =============== Sidebar (filtros) ===============
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign","account"], index=0)
today = date.today()
since = st.sidebar.date_input("Desde", value=today - timedelta(days=7))
until = st.sidebar.date_input("Até", value=today)
ready = bool(act_id and token)

# =============== Tela ===============
st.title("📊 Meta Ads — Paridade com Filtro + Funil")
st.caption("KPIs + Funil: Cliques → LPV → Finalização → Add Pagamento → Compra. Tudo alinhado ao período selecionado.")

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
    st.stop()

with st.spinner("Buscando dados da Meta…"):
    df = fetch_insights_daily(
        act_id=act_id, token=token, api_version=api_version,
        since_str=str(since), until_str=str(until), level=level
    )

if df.empty:
    st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
    st.stop()

# ========= KPIs do período =========
tot_spend = float(df["spend"].sum())
tot_purch = float(df["purchases"].sum())
tot_rev   = float(df["revenue"].sum())
roas_g    = (tot_rev/tot_spend) if tot_spend>0 else 0.0

c1,c2,c3,c4 = st.columns(4)
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
    st.markdown('<div class="kpi-card"><div class="small-muted">ROAS</div>'
                f'<div class="big-number">{roas_g:,.2f}</div></div>'.replace(",", "X").replace(".", ",").replace("X", "."),
                unsafe_allow_html=True)

st.divider()

# ========= Série diária =========
st.subheader("Série diária — Investimento e Conversão")
daily = df.groupby("date", as_index=False)[["spend","revenue","purchases"]].sum()
st.line_chart(daily.set_index("date")[["spend","revenue"]])
st.caption("Linhas diárias de Valor usado e Valor de conversão. Vendas na tabela abaixo.")

# ========= COMPARATIVOS (Período A vs Período B) =========
st.subheader("Comparativos — descubra o que mudou e por quê")

# Deduzi o tamanho do período atual para propor um período anterior de mesmo tamanho
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
        dfA = fetch_insights_daily(act_id, token, api_version, str(sinceA), str(untilA), level)
        dfB = fetch_insights_daily(act_id, token, api_version, str(sinceB), str(untilB), level)

    if dfA.empty or dfB.empty:
        st.info("Sem dados em um dos períodos selecionados.")
    else:
        # Agregados por período
        def _agg(d):
            return {
                "spend": d["spend"].sum(),
                "revenue": d["revenue"].sum(),
                "purchases": d["purchases"].sum(),
                "clicks": d["link_clicks"].sum(),
                "lpv": d["lpv"].sum(),
                "checkout": d["init_checkout"].sum(),
            }
        A = _agg(dfA); B = _agg(dfB)

        # KPIs lado a lado
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Valor usado A", _fmt_money_br(A["spend"]))
        c2.metric("Valor usado B", _fmt_money_br(B["spend"]),
                  delta=_fmt_money_br(B["spend"]-A["spend"]))
        c3.metric("Vendas A", f"{int(A['purchases']):,}".replace(",", "."))
        c4.metric("Vendas B", f"{int(B['purchases']):,}".replace(",", "."),
                  delta=f"{int(B['purchases']-A['purchases']):,}".replace(",", "."))
        c5.metric("Faturamento A", _fmt_money_br(A["revenue"]))
        c6.metric("Faturamento B", _fmt_money_br(B["revenue"]),
                  delta=_fmt_money_br(B["revenue"]-A["revenue"]))

        c7,c8,c9 = st.columns(3)
        roasA = _safe_div(A["revenue"], A["spend"])
        roasB = _safe_div(B["revenue"], B["spend"])
        cpaA  = _safe_div(A["spend"], A["purchases"])
        cpaB  = _safe_div(B["spend"], B["purchases"])
        cpcA  = _safe_div(A["spend"], A["clicks"])
        cpcB  = _safe_div(B["spend"], B["clicks"])
        c7.metric("ROAS A", _fmt_ratio_br(roasA))
        c8.metric("ROAS B", _fmt_ratio_br(roasB),
                  delta=_fmt_ratio_br(roasB-roasA) if pd.notnull(roasA) and pd.notnull(roasB) else "")
        c9.metric("CPA B vs A", _fmt_money_br(cpaB) if pd.notnull(cpaB) else "",
                  delta=_fmt_money_br(cpaB-cpaA) if pd.notnull(cpaA) and pd.notnull(cpaB) else "")

        st.markdown("---")

        # Taxas do funil (as 3 principais)
        rates = pd.DataFrame({
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
        rates["Δ"] = rates["Período B"] - rates["Período A"]
        rates_fmt = rates.copy()
        for col in ["Período A","Período B","Δ"]:
            rates_fmt[col] = rates_fmt[col].map(_fmt_pct_br)
        st.markdown("**Taxas do funil (A vs B)**")
        st.dataframe(rates_fmt, use_container_width=True, height=180)

        st.markdown("---")

        # Driver analysis (waterfall) — por que a receita mudou?
        labels_drv, contrib = _drivers_decomp(
            A["clicks"], A["lpv"], A["checkout"], A["purchases"], A["revenue"],
            B["clicks"], B["lpv"], B["checkout"], B["purchases"], B["revenue"]
        )
        revA = A["revenue"]; revB = B["revenue"]

        fig = go.Figure(go.Waterfall(
            x=["Período A"] + labels_drv + ["Período B"],
            measure=["absolute"] + ["relative"]*len(labels_drv) + ["total"],
            y=[revA] + contrib + [revB],
            connector={"line": {"dash": "dot", "width": 1}},
            textposition="outside"
        ))
        fig.update_layout(
            title="O que explicou a mudança de receita?",
            template="plotly_white",
            margin=dict(l=20,r=20,t=50,b=20),
            separators=",.",
            yaxis=dict(tickprefix="R$ ")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretação: barras positivas puxaram a receita; negativas derrubaram. "
                   "Os fatores são volume (Cliques), qualidade/aderência (LPV/Cliques), avanço no funil "
                   "(Checkout/LPV, Compra/Checkout) e ticket médio (Receita/Compra).")


# ========= FUNIL (Período) — FUNIL VISUAL =========
st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")

# Totais
f_clicks = float(df["link_clicks"].sum())
f_lpv    = float(df["lpv"].sum())
f_ic     = float(df["init_checkout"].sum())
f_api    = float(df["add_payment"].sum())
f_pur    = float(df["purchases"].sum())

# RÓTULOS curtos e VALORES inteiros (legibilidade)
labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)),
                int(round(f_api)), int(round(f_pur))]

# Toggle: manter forma de funil sempre decrescente só no desenho
force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
values_plot = enforce_monotonic(values_total) if force_shape else values_total

# Gráfico (com valores dentro das faixas)
st.plotly_chart(
    funnel_fig(labels_total, values_plot, title="Funil do período"),
    use_container_width=True
)

# Tabela de taxas — principais sempre visíveis + extras opcionais
def _rate(a, b):
    return (a / b) if b and b > 0 else np.nan

# ---- Principais (sempre)
core_rows = [
    ("LPV / Cliques",      _rate(values_total[1], values_total[0])),
    ("Checkout / LPV",     _rate(values_total[2], values_total[1])),
    ("Compra / Checkout",  _rate(values_total[4], values_total[2])),
]

# ---- Extras (opc.) — excluímos as três principais da lista
extras_def = {
    "Add Pagto / Checkout": (_rate(values_total[3], values_total[2])),
    "Compra / Add Pagto":   (_rate(values_total[4], values_total[3])),
    "Compra / LPV":         (_rate(values_total[4], values_total[1])),
    "Compra / Cliques":     (_rate(values_total[4], values_total[0])),
    "Checkout / Cliques":   (_rate(values_total[2], values_total[0])),
    "Add Pagto / LPV":      (_rate(values_total[3], values_total[1])),
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

# Altura ajusta conforme nº de extras
base_h = 160
row_h  = 36
height = base_h + row_h * len(extras_selected)
st.dataframe(sr, use_container_width=True, height=height)

# ========= FUNIL por CAMPANHA =========
st.subheader("Campanhas — Funil e Taxas (somatório no período)")

# Agregados necessários
agg_cols = ["spend","link_clicks","lpv","init_checkout","add_payment","purchases","revenue"]
camp = df.groupby(["campaign_id","campaign_name"], as_index=False)[agg_cols].sum()

# Taxas PRINCIPAIS (mesmas do funil geral)
camp["LPV/Cliques"]      = np.where(camp["link_clicks"]>0, camp["lpv"]/camp["link_clicks"], np.nan)
camp["Checkout/LPV"]     = np.where(camp["lpv"]>0, camp["init_checkout"]/camp["lpv"], np.nan)
camp["Compra/Checkout"]  = np.where(camp["init_checkout"]>0, camp["purchases"]/camp["init_checkout"], np.nan)

# ROAS continua útil
camp["ROAS"]             = np.where(camp["spend"]>0, camp["revenue"]/camp["spend"], np.nan)

# ---- Extras (só quando você quiser ver)
extras_cols = {
    "Add Pagto / Checkout":  np.where(camp["init_checkout"]>0, camp["add_payment"]/camp["init_checkout"], np.nan),
    "Compra / Add Pagto":    np.where(camp["add_payment"]>0, camp["purchases"]/camp["add_payment"], np.nan),
    "Compra / LPV":          np.where(camp["lpv"]>0, camp["purchases"]/camp["lpv"], np.nan),
    "Compra / Cliques":      np.where(camp["link_clicks"]>0, camp["purchases"]/camp["link_clicks"], np.nan),
    "Checkout / Cliques":    np.where(camp["link_clicks"]>0, camp["init_checkout"]/camp["link_clicks"], np.nan),
    "Add Pagto / LPV":       np.where(camp["lpv"]>0, camp["add_payment"]/camp["lpv"], np.nan),
}

with st.expander("Comparar outras taxas (opcional)"):
    extras_selected = st.multiselect(
        "Escolha métricas adicionais:",
        options=list(extras_cols.keys()),
        default=[],
    )

# Monta a visão final (apenas as colunas principais + extras escolhidas)
cols_base = [
    "campaign_id","campaign_name",
    "spend","revenue","ROAS",           # dinheiro/eficiência
    "link_clicks","lpv","init_checkout","purchases",  # etapas do funil
    "LPV/Cliques","Checkout/LPV","Compra/Checkout"    # taxas principais
]
camp_view = camp[cols_base].copy()

# Anexa extras selecionadas
for name in extras_selected:
    camp_view[name] = extras_cols[name]

# Formatação
money_cols = ["spend","revenue"]
for c in money_cols:
    camp_view[c] = camp_view[c].apply(_fmt_money_br)

# Percentuais (APENAS as taxas do funil)
pct_cols = ["LPV/Cliques","Checkout/LPV","Compra/Checkout"] + list(extras_selected)
for c in pct_cols:
    camp_view[c] = camp_view[c].map(lambda x: (f"{x*100:,.2f}%" if pd.notnull(x) else "")
                                    .replace(",", "X").replace(".", ",").replace("X", "."))

# ROAS como valor (razão), ex.: 1,23x
camp_view["ROAS"] = camp_view["ROAS"].map(
    lambda x: (f"{x:,.2f}x" if pd.notnull(x) else "").replace(",", "X").replace(".", ",").replace("X", ".")
)

# Renomeia para exibição
camp_view = camp_view.rename(columns={
    "campaign_id": "ID campanha",
    "campaign_name": "Campanha",
    "spend": "Valor usado",
    "revenue": "Valor de conversão",
    "link_clicks": "Cliques",
    "lpv": "LPV",
    "init_checkout": "Checkout",
    "purchases": "Compras",
})

# Ordenação padrão por investimento
camp_view = camp_view.sort_values("Valor usado", ascending=False)

# Altura adaptativa (linha base + extras)
base_h = 520
st.dataframe(camp_view, use_container_width=True, height=base_h)

with st.expander("Dados diários (detalhe por campanha)"):
    dd = df.copy()
    dd["date"] = dd["date"].dt.date
    cols = ["date","campaign_name","spend","link_clicks","lpv","init_checkout","add_payment","purchases","revenue","roas"]
    dd_fmt = dd[cols].copy()
    dd_fmt["spend"] = dd_fmt["spend"].apply(_fmt_money_br)
    dd_fmt["revenue"] = dd_fmt["revenue"].apply(_fmt_money_br)
    dd_fmt["roas"] = dd_fmt["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
    st.dataframe(dd_fmt.rename(columns={
        "date":"Data","campaign_name":"Campanha","spend":"Valor usado",
        "link_clicks":"Cliques","lpv":"LPV","init_checkout":"Finalização",
        "add_payment":"Add Pagamento","purchases":"Compras","revenue":"Valor de conversão","roas":"ROAS"
    }), use_container_width=True)
