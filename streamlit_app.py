import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta

# ---------------- Config da página ----------------
st.set_page_config(page_title="Meta Ads — Paridade com Manager", page_icon="📊", layout="wide")

st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def _retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["rate limit", "retry", "temporarily unavailable", "timeout"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def _ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def _to_float(x):
    try: return float(x or 0)
    except: return 0.0

def _sum_item(item: dict) -> float:
    """
    Para entradas de actions/action_values:
    - Se houver 'value', usa (Meta já agrega).
    - Senão, soma somente valores numéricos (p.ex.: '7d_click','1d_view', etc.).
    """
    if not isinstance(item, dict): return _to_float(item)
    if "value" in item: return _to_float(item.get("value"))
    s = 0.0
    for k, v in item.items():
        if k not in {"action_type","action_target","action_destination","action_device",
                     "action_channel","action_reaction","action_canvas_component_name","value"}:
            s += _to_float(v)
    return s

def _pick_purchase_totals(rows: list) -> float:
    """
    Política de paridade com o Manager:
    1) Se existir omni_purchase -> usar somente omni (deduplicado).
    2) Caso contrário, pegar o melhor 'purchase' específico sem somar múltiplos tipos (evita duplicar).
       Preferência: purchase > onsite_conversion.purchase > offsite_conversion.fb_pixel_purchase.
    """
    if not rows: return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]

    # 1) omni_purchase
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni:
        return float(sum(_sum_item(r) for r in omni))

    # 2) específicos (pega o MAIOR entre eles, não soma)
    cands = {
        "purchase": 0.0,
        "onsite_conversion.purchase": 0.0,
        "offsite_conversion.fb_pixel_purchase": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in cands:
            cands[at] += _sum_item(r)

    if any(v > 0 for v in cands.values()):
        return float(max(cands.values()))

    # 3) fallback amplo: qualquer coisa contendo 'purchase' (pega o maior grupo)
    groups = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            groups.setdefault(r["action_type"], 0.0)
            groups[r["action_type"]] += _sum_item(r)
    return float(max(groups.values()) if groups else 0.0)

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign") -> pd.DataFrame:
    """
    Busca insights diários, alinhado ao Ads Manager (jun/2025+):
      - time_range: since/until (datas que você escolhe no filtro)
      - time_increment=1 (série diária)
      - NÃO usa 'action_types', 'action_attribution_windows' ou 'action_report_time'
      - level único (campaign recomendado)
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]
    params = {
        "access_token": token,
        "level": level,                           # 'campaign' recomendado
        "time_range": json.dumps({"since": since_str, "until": until_str}),
        "time_increment": 1,                      # diário
        "fields": ",".join(fields),
        "limit": 500,
        # Mudanças de 10/jun/2025: API já alinha com Manager; não setamos unified/ART aqui.
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
            raise RuntimeError(f"Graph API error {resp.status_code} | code={err.get('code')} "
                               f"subcode={err.get('error_subcode')} | {err.get('message')}")

        for rec in payload.get("data", []):
            spend = _to_float(rec.get("spend"))
            actions = rec.get("actions") or []
            action_values = rec.get("action_values") or []

            purchases_cnt = _pick_purchase_totals(actions)
            revenue_val   = _pick_purchase_totals(action_values)

            rows.append({
                "date":           pd.to_datetime(rec.get("date_start")),
                "spend":          spend,
                "purchases":      purchases_cnt,
                "revenue":        revenue_val,
                "impressions":    _to_float(rec.get("impressions")),
                "clicks":         _to_float(rec.get("clicks")),
                "currency":       rec.get("account_currency", "BRL"),
                "campaign_id":    rec.get("campaign_id", ""),
                "campaign_name":  rec.get("campaign_name", ""),
            })

        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if after:
            next_url, next_params = base_url, params.copy()
            next_params["after"] = after
        else:
            break

    df = pd.DataFrame(rows)
    if df.empty: return df

    for c in ["spend","purchases","revenue","impressions","clicks"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["roas"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
    df = df.sort_values("date")
    return df

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------- Sidebar — Filtro de data ----------------
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", placeholder="ex.: act_1234567890")
token = st.sidebar.text_input("Access Token", type="password")
api_version = st.sidebar.text_input("API Version", value="v23.0")
level = st.sidebar.selectbox("Nível (recomendado: campaign)", ["campaign","account"], index=0)

today = date.today()
since = st.sidebar.date_input("Desde", value=today - timedelta(days=7))
until = st.sidebar.date_input("Até", value=today)

# Atualiza automaticamente ao mudar filtros (sem botão)
ready = bool(act_id and token)

st.title("📊 Meta Ads — Paridade por Data (Manager)")
st.caption("Mostra Valor usado, Vendas, Valor de conversão e ROAS exatamente pelo intervalo selecionado.")

if not ready:
    st.info("Informe **Ad Account ID** e **Access Token** para iniciar.")
else:
    with st.spinner("Buscando dados da Meta…"):
        df = fetch_insights_daily(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(since),
            until_str=str(until),
            level=level
        )

    if df.empty:
        st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
        st.stop()

    # KPIs do período
    tot_spend = float(df["spend"].sum())
    tot_purch = float(df["purchases"].sum())
    tot_rev   = float(df["revenue"].sum())
    roas_g    = (tot_rev / tot_spend) if tot_spend > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
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

    # Série diária
    daily = df.groupby("date", as_index=False)[["spend","revenue","purchases"]].sum()
    st.subheader("Série diária (Respeita o filtro de data)")
    st.line_chart(daily.set_index("date")[["spend","revenue"]])
    st.caption("Linha: Valor usado e Valor de conversão por dia. As Vendas estão na tabela abaixo.")

    # Campanhas no período
    st.subheader("Campanhas (somatório no período)")
    agg_cols = ["spend","purchases","revenue","impressions","clicks"]
    by_campaign = df.groupby(["campaign_id","campaign_name"], as_index=False)[agg_cols].sum()
    by_campaign["roas"] = np.where(by_campaign["spend"]>0, by_campaign["revenue"]/by_campaign["spend"], np.nan)

    # Formatar
    show = by_campaign.copy()
    show["spend"] = show["spend"].apply(_fmt_money_br)
    show["revenue"] = show["revenue"].apply(_fmt_money_br)
    show["roas"] = show["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
    st.dataframe(
        show.rename(columns={
            "campaign_id":"ID campanha","campaign_name":"Campanha",
            "spend":"Valor usado","purchases":"Vendas","revenue":"Valor de conversão",
            "roas":"ROAS","impressions":"Impressões","clicks":"Cliques"
        }).sort_values("Valor usado", ascending=False),
        use_container_width=True, height=520
    )

    with st.expander("Ver dados diários (detalhe)"):
        dd = df.copy()
        dd["date"] = dd["date"].dt.date
        dd_fmt = dd[["date","campaign_name","spend","purchases","revenue","roas"]].copy()
        dd_fmt["spend"] = dd_fmt["spend"].apply(_fmt_money_br)
        dd_fmt["revenue"] = dd_fmt["revenue"].apply(_fmt_money_br)
        dd_fmt["roas"] = dd_fmt["roas"].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "")
        st.dataframe(dd_fmt.rename(columns={
            "date":"Data","campaign_name":"Campanha","spend":"Valor usado",
            "purchases":"Vendas","revenue":"Valor de conversão","roas":"ROAS"
        }), use_container_width=True)
