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
    ("Compra / LPV",       _rate(values_total[4], values_total[1])),
]

# ---- Extras (somente se você quiser ver)
extras_def = {
    "Add Pagto / Checkout": (_rate(values_total[3], values_total[2])),
    "Compra / Add Pagto":   (_rate(values_total[4], values_total[3])),
    "Compra / Cliques":     (_rate(values_total[4], values_total[0])),
    "Checkout / Cliques":   (_rate(values_total[2], values_total[0])),
    "Add Pagto / LPV":      (_rate(values_total[3], values_total[1])),
}

# UI para escolher comparações extras
with st.expander("Comparar outras taxas (opcional)"):
    extras_selected = st.multiselect(
        "Escolha métricas adicionais para visualizar:",
        options=list(extras_def.keys()),
        default=[],
    )

# Monta a tabela final
rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
sr["Valor"] = sr["Valor"].map(lambda x: f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                              if pd.notnull(x) else "")

# Altura ajustada: 3 linhas fixas + nº de extras
base_h = 160
row_h  = 36
height = base_h + row_h * len(extras_selected)

st.dataframe(sr, use_container_width=True, height=height)

# ========= FUNIL por CAMPANHA =========
st.subheader("Campanhas — Funil e Taxas (somatório no período)")
agg_cols = ["spend","link_clicks","lpv","init_checkout","add_payment","purchases","revenue"]
camp = df.groupby(["campaign_id","campaign_name"], as_index=False)[agg_cols].sum()

# taxas por campanha (sequenciais)
camp["LPV/Clique"]     = np.where(camp["link_clicks"]>0, camp["lpv"]/camp["link_clicks"], np.nan)
camp["Fin/LPV"]        = np.where(camp["lpv"]>0, camp["init_checkout"]/camp["lpv"], np.nan)
camp["AddPg/Fin"]      = np.where(camp["init_checkout"]>0, camp["add_payment"]/camp["init_checkout"], np.nan)
camp["Compra/AddPg"]   = np.where(camp["add_payment"]>0, camp["purchases"]/camp["add_payment"], np.nan)
camp["Compra/Clique"]  = np.where(camp["link_clicks"]>0, camp["purchases"]/camp["link_clicks"], np.nan)
camp["ROAS"]           = np.where(camp["spend"]>0, camp["revenue"]/camp["spend"], np.nan)

camp_show = camp.copy()
money_cols = ["spend","revenue"]
for c in money_cols: camp_show[c] = camp_show[c].apply(_fmt_money_br)
pct_cols = ["LPV/Clique","Fin/LPV","AddPg/Fin","Compra/AddPg","Compra/Clique","ROAS"]
for c in pct_cols:
    camp_show[c] = camp_show[c].map(lambda x: (f"{x*100:,.2f}%" if pd.notnull(x) else "")\
                                    .replace(",", "X").replace(".", ",").replace("X", "."))

camp_show = camp_show.rename(columns={
    "campaign_id":"ID campanha",
    "campaign_name":"Campanha",
    "spend":"Valor usado",
    "link_clicks":"Cliques",
    "lpv":"LPV",
    "init_checkout":"Finalização",
    "add_payment":"Add Pagamento",
    "purchases":"Compras",
    "revenue":"Valor de conversão",
})
st.dataframe(camp_show.sort_values("Valor usado", ascending=False), use_container_width=True, height=520)

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
