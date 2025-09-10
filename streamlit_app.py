import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração de fuso horário
APP_TZ = ZoneInfo("America/Sao_Paulo")

# Sessão de requests para reutilização de conexão
_session = None
def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session

# =============== Config & Estilos ===============
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# Nomes de eventos para agregação (padrão da API)
CHECKOUT_NAMES = [
    "omni_initiated_checkout", "initiate_checkout",
    "onsite_conversion.initiated_checkout", "offsite_conversion.fb_pixel_initiate_checkout"
]
ADDPAY_NAMES = [
    "omni_add_payment_info", "add_payment_info",
    "onsite_conversion.add_payment_info", "offsite_conversion.fb_pixel_add_payment_info"
]
PURCHASE_EVENTS = [
    "omni_purchase", "purchase",
    "offsite_conversion.fb_pixel_purchase", "onsite_conversion.purchase"
]
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"

# =============== Helpers genéricos ===============
def _parse_hour_bucket(h):
    if h is None: return None
    try:
        s = str(h).strip()
        val = int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except: return None

def _retry_call(fn, max_retries=5, base_wait=1.2):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "retry", "temporarily unavailable", "timeout", "timed out"]):
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

def _sum_item(item):
    if not isinstance(item, dict): return _to_float(item)
    return _to_float(item.get("value"))

def _sum_actions_exact(rows, exact_names) -> float:
    if not rows: return 0.0
    names = {str(n).lower() for n in exact_names}
    tot = 0.0
    for r in rows:
        if str(r.get("action_type", "")).lower() in names:
            tot += _sum_item(r)
    return float(tot)

def _pick_purchase_totals(rows) -> float:
    if not rows: return 0.0
    total = 0.0
    for r in rows:
        if str(r.get("action_type", "")).lower() in PURCHASE_EVENTS:
            total += _sum_item(r)
    return float(total)

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def funnel_fig(labels, values, title=None):
    fig = go.Figure(go.Funnel(
        y=labels, x=values, textinfo="value+percent initial", textposition="inside",
        texttemplate="<b>%{value}</b>  
(%{percentInitial:.2%})",
        textfont=dict(size=16), opacity=0.9, marker={"color": "#636EFA"}
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=48, b=10), height=540, template="plotly_white", separators=",.")
    return fig

def enforce_monotonic(values):
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

def _safe_div(n, d):
    n, d = _to_float(n), _to_float(d)
    return (n / d) if d > 0 else 0.0

def _fmt_pct_br(x):
    if pd.isnull(x): return ""
    return f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_ratio_br(x):
    if pd.isnull(x): return ""
    return f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_int_br(x):
    try: return f"{int(round(_to_float(x))):,}".replace(",", ".")
    except: return ""

def _chunks_by_days(since_str: str, until_str: str, max_days: int = 30):
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days - 1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)

def _filter_by_product(df: pd.DataFrame, produto: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or not produto or produto == "(Todos)":
        return df
    return df[df["campaign_name"].str.contains(produto, case=False, na=False)].copy()

# =============== Funções de Coleta de Dados ===============
@st.cache_data(ttl=600, show_spinner="Buscando dados...")
def fetch_insights(act_id: str, token: str, api_version: str, since_str: str, until_str: str, level: str, breakdowns: list = None, product_name: str | None = None) -> pd.DataFrame:
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    
    fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name",
        "link_clicks", "landing_page_views"
    ]

    def _fetch_range(_since: str, _until: str ) -> list[dict]:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1 if not breakdowns or "hourly" in "".join(breakdowns) else 0,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": "conversion",
        }
        if breakdowns:
            params["breakdowns"] = ",".join(breakdowns)
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps([{"field": "campaign.name", "operator": "CONTAIN", "value": product_name}])

        rows_local, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=120))
            payload = resp.json()
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                raise RuntimeError(f"Graph API Error: {err.get('message')} (Code: {err.get('code')})")

            for rec in payload.get("data", []):
                actions = rec.get("actions", []) or []
                action_values = rec.get("action_values", []) or []
                
                base_rec = {
                    "date": pd.to_datetime(rec.get("date_start")),
                    "currency": rec.get("account_currency", "BRL"),
                    "campaign_id": rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend": _to_float(rec.get("spend")),
                    "impressions": _to_float(rec.get("impressions")),
                    "clicks": _to_float(rec.get("clicks")),
                    "link_clicks": _to_float(rec.get("link_clicks")),
                    "lpv": _to_float(rec.get("landing_page_views")) or _sum_actions_exact(actions, ["landing_page_view", "view_content"]),
                    "init_checkout": _sum_actions_exact(actions, CHECKOUT_NAMES),
                    "add_payment": _sum_actions_exact(actions, ADDPAY_NAMES),
                    "purchases": _pick_purchase_totals(actions),
                    "revenue": _pick_purchase_totals(action_values),
                }
                if breakdowns:
                    for b in breakdowns:
                        base_rec[b.split(':')[0]] = rec.get(b) # Handle hourly breakdown name
                rows_local.append(base_rec)

            paging = payload.get("paging", {})
            if paging.get("next"):
                next_url, next_params = paging["next"], None
            else:
                break
        return rows_local

    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    if not all_rows: return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    num_cols = ["spend", "impressions", "clicks", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    
    df["roas"] = df.apply(lambda row: _safe_div(row["revenue"], row["spend"]), axis=1)
    return df.sort_values("date").reset_index(drop=True)

# =============== Sidebar (Filtros) ===============
st.sidebar.header("Configuração")
act_id = st.sidebar.text_input("Ad Account ID", st.secrets.get("META_AD_ACCOUNT_ID", ""), placeholder="act_1234567890")
token = st.sidebar.text_input("Access Token", st.secrets.get("META_ACCESS_TOKEN", ""), type="password", placeholder="Cole seu token aqui")
api_version = st.sidebar.text_input("API Version", value="v20.0")
level = "campaign" # Fixo para simplificar

preset = st.sidebar.radio(
    "Período rápido",
    ["Hoje", "Ontem", "Últimos 7 dias", "Últimos 14 dias", "Últimos 30 dias", "Personalizado"],
    index=2, horizontal=True
)

def _range_from_preset(p):
    today = datetime.now(APP_TZ).date()
    if p == "Hoje": return today, today
    if p == "Ontem":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    if p == "Últimos 7 dias": return today - timedelta(days=6), today
    if p == "Últimos 14 dias": return today - timedelta(days=13), today
    if p == "Últimos 30 dias": return today - timedelta(days=29), today
    return today - timedelta(days=6), today

if preset == "Personalizado":
    since = st.sidebar.date_input("Desde", value=_range_from_preset("Últimos 7 dias")[0], key="since_custom")
    until = st.sidebar.date_input("Até", value=datetime.now(APP_TZ).date(), key="until_custom")
else:
    since, until = _range_from_preset(preset)
    st.sidebar.caption(f"**Período:** {since.strftime('%d/%m/%Y')} a {until.strftime('%d/%m/%Y')}")

ready = bool(act_id and token)

# =============== Tela Principal ===============
st.title("📊 Meta Ads — Análise de Funil e Desempenho")

if not ready:
    st.info("👈 Informe seu **Ad Account ID** e **Access Token** na barra lateral para começar.")
    st.stop()

# ===================== Coleta Principal =====================
df_daily = fetch_insights(act_id, token, api_version, str(since), str(until), level)

if df_daily.empty:
    st.warning("Nenhum dado encontrado para o período selecionado. Verifique as permissões do token, o ID da conta e se há campanhas ativas.")
    st.stop()

# ===================== Abas =====================
tab_daily, tab_daypart, tab_detail = st.tabs(["📅 Visão Geral", "⏱️ Análise por Hora", "📊 Detalhamento"])

# -------------------- ABA 1: VISÃO GERAL --------------------
with tab_daily:
    produto_sel_daily = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daily_produto")
    df_view = _filter_by_product(df_daily, produto_sel_daily)

    if df_view.empty:
        st.info("Nenhum dado encontrado para o produto selecionado neste período.")
        st.stop()

    if produto_sel_daily != "(Todos)":
        st.caption(f"🔎 Exibindo dados apenas para campanhas contendo **{produto_sel_daily}**")

    # ========= KPIs do Período =========
    kpis = {
        "Gasto": df_view["spend"].sum(),
        "Vendas": df_view["purchases"].sum(),
        "Receita": df_view["revenue"].sum(),
        "ROAS": _safe_div(df_view["revenue"].sum(), df_view["spend"].sum())
    }
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor Gasto", _fmt_money_br(kpis["Gasto"]))
    c2.metric("Vendas", _fmt_int_br(kpis["Vendas"]))
    c3.metric("Receita", _fmt_money_br(kpis["Receita"]))
    c4.metric("ROAS", _fmt_ratio_br(kpis["ROAS"]))

    st.divider()

    # ========= Funil Visual do Período =========
    st.subheader("Funil de Conversão do Período")
    funnel_values = {
        "Cliques no Link": df_view["link_clicks"].sum(),
        "Visualizações da Página": df_view["lpv"].sum(),
        "Checkouts Iniciados": df_view["init_checkout"].sum(),
        "Info de Pagamento": df_view["add_payment"].sum(),
        "Compras": df_view["purchases"].sum()
    }
    
    labels = list(funnel_values.keys())
    values = [int(round(v)) for v in funnel_values.values()]
    
    force_shape = st.checkbox("Forçar formato de funil (visual)", value=True, help="Garante que cada etapa seja menor ou igual à anterior, apenas para a visualização.")
    plot_values = enforce_monotonic(values) if force_shape else values
    
    st.plotly_chart(funnel_fig(labels, plot_values), use_container_width=True)

    # ========= Tabela de Taxas do Funil =========
    st.markdown("**Taxas de Conversão do Funil**")
    rates_data = [
        ("Visualização / Clique", _safe_div(funnel_values["Visualizações da Página"], funnel_values["Cliques no Link"])),
        ("Checkout / Visualização", _safe_div(funnel_values["Checkouts Iniciados"], funnel_values["Visualizações da Página"])),
        ("Info Pagto / Checkout", _safe_div(funnel_values["Info de Pagamento"], funnel_values["Checkouts Iniciados"])),
        ("Compra / Info Pagto", _safe_div(funnel_values["Info de Pagamento"], funnel_values["Checkouts Iniciados"])),
        ("Compra / Checkout", _safe_div(funnel_values["Compras"], funnel_values["Checkouts Iniciados"])),
        ("Compra / Visualização", _safe_div(funnel_values["Compras"], funnel_values["Visualizações da Página"])),
    ]
    df_rates = pd.DataFrame(rates_data, columns=["Métrica", "Taxa"])
    df_rates["Taxa"] = df_rates["Taxa"].apply(_fmt_pct_br)
    st.dataframe(df_rates, use_container_width=True)

# -------------------- ABA 2: ANÁLISE POR HORA --------------------
with tab_daypart:
    st.subheader("Desempenho por Hora do Dia")
    produto_sel_hr = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daypart_produto")

    df_hourly_raw = fetch_insights(act_id, token, api_version, str(since), str(until), level, breakdowns=[HOUR_BREAKDOWN], product_name=produto_sel_hr)

    if df_hourly_raw.empty:
        st.info("Nenhum dado por hora encontrado para o período/filtro.")
        st.stop()

    df_hourly = df_hourly_raw.copy()
    df_hourly["hour"] = df_hourly[HOUR_BREAKDOWN].apply(_parse_hour_bucket)
    df_hourly.dropna(subset=["hour"], inplace=True)
    df_hourly["hour"] = df_hourly["hour"].astype(int)

    # Agregação por hora
    hourly_agg = df_hourly.groupby("hour").agg({
        "spend": "sum", "revenue": "sum", "purchases": "sum", "link_clicks": "sum"
    }).reset_index()
    hourly_agg["roas"] = hourly_agg.apply(lambda row: _safe_div(row["revenue"], row["spend"]), axis=1)

    metric_hr = st.selectbox("Métrica para análise por hora", ["Compras", "Receita", "Gasto", "ROAS"], key="hr_metric")
    metric_map = {"Compras": "purchases", "Receita": "revenue", "Gasto": "spend", "ROAS": "roas"}
    selected_metric = metric_map[metric_hr]

    fig_hr = go.Figure(go.Bar(x=hourly_agg["hour"], y=hourly_agg[selected_metric], marker_color="#636EFA"))
    fig_hr.update_layout(
        title=f"{metric_hr} por Hora do Dia",
        xaxis_title="Hora do Dia (fuso da conta)",
        yaxis_title=metric_hr,
        template="plotly_white",
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("**Tabela de Dados por Hora**")
    hourly_display = hourly_agg.rename(columns={
        "hour": "Hora", "spend": "Gasto", "revenue": "Receita", "purchases": "Compras", "roas": "ROAS"
    })
    hourly_display["Gasto"] = hourly_display["Gasto"].apply(_fmt_money_br)
    hourly_display["Receita"] = hourly_display["Receita"].apply(_fmt_money_br)
    hourly_display["ROAS"] = hourly_display["ROAS"].apply(_fmt_ratio_br)
    st.dataframe(hourly_display[["Hora", "Compras", "Receita", "Gasto", "ROAS"]], use_container_width=True)

# -------------------- ABA 3: DETALHAMENTO --------------------
with tab_detail:
    st.subheader("Detalhamento por Dimensão")
    
    dimensao = st.selectbox(
        "Escolha uma dimensão para detalhar",
        ["Campanha", "Idade", "Gênero", "Idade e Gênero", "Plataforma", "Posicionamento"]
    )
    
    produto_sel_det = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="det_produto")

    breakdown_map = {
        "Campanha": [],
        "Idade": ["age"],
        "Gênero": ["gender"],
        "Idade e Gênero": ["age", "gender"],
        "Plataforma": ["publisher_platform"],
        "Posicionamento": ["publisher_platform", "platform_position"],
    }
    
    group_cols = breakdown_map[dimensao]
    if dimensao == "Campanha":
        df_detail = _filter_by_product(df_daily, produto_sel_det)
        group_by_cols = ["campaign_name"]
    else:
        df_detail = fetch_insights(act_id, token, api_version, str(since), str(until), level, breakdowns=group_cols, product_name=produto_sel_det)
        group_by_cols = group_cols

    if df_detail.empty:
        st.info(f"Nenhum dado encontrado para a dimensão '{dimensao}'.")
        st.stop()

    agg_detail = df_detail.groupby(group_by_cols).agg({
        "spend": "sum", "revenue": "sum", "purchases": "sum", "link_clicks": "sum"
    }).reset_index()
    agg_detail["roas"] = agg_detail.apply(lambda row: _safe_div(row["revenue"], row["spend"]), axis=1)
    agg_detail = agg_detail.sort_values("purchases", ascending=False)

    st.markdown(f"**Desempenho por {dimensao}**")
    
    display_detail = agg_detail.rename(columns={
        "campaign_name": "Campanha", "age": "Idade", "gender": "Gênero",
        "publisher_platform": "Plataforma", "platform_position": "Posicionamento",
        "spend": "Gasto", "revenue": "Receita", "purchases": "Compras", "roas": "ROAS"
    })
    
    display_detail["Gasto"] = display_detail["Gasto"].apply(_fmt_money_br)
    display_detail["Receita"] = display_detail["Receita"].apply(_fmt_money_br)
    display_detail["ROAS"] = display_detail["ROAS"].apply(_fmt_ratio_br)
    
    st.dataframe(display_detail, use_container_width=True)
