
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go 
import time, json
import requests
# =========================
# Config
# =========================
st.set_page_config(
    page_title="Metas & Performance — Simples",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📊 Metas & Performance — Simples")
st.caption("Defina a META MENSAL e o app reparte automaticamente a META SEMANAL da semana selecionada.")

# =========================
# Tema / Paleta (cores consistentes)
# =========================
ETAPAS_COLORS = {
    "Teste de Criativo":  "#7C3AED",  # roxo
    "Teste de Interesse": "#06B6D4",  # ciano
    "Escala":             "#22C55E",  # verde
    "Remarketing":        "#F59E0B",  # âmbar
    "Outros":             "#94A3B8",  # cinza
}

# Paleta base p/ linhas e séries sem mapeamento explícito
COLORWAY = ["#7C3AED", "#06B6D4", "#22C55E", "#F59E0B", "#94A3B8", "#0EA5E9", "#EF4444", "#10B981", "#3B82F6"]

def style_fig(fig, title=None):
    """Tema padrão do dashboard."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial", size=13),
        title=dict(text=title or fig.layout.title.text, x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)"),
        colorway=COLORWAY,
    )
    return fig

def style_fig_pdf(fig, title=None):
    """Mesma estética + fundo branco (Kaleido) para PDF."""
    style_fig(fig, title)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return fig

def _darken(hex_color, factor=0.85):
    hex_color = hex_color.lstrip("#")
    r = max(0, min(255, int(int(hex_color[0:2],16)*factor)))
    g = max(0, min(255, int(int(hex_color[2:4],16)*factor)))
    b = max(0, min(255, int(int(hex_color[4:6],16)*factor)))
    return f"rgb({r},{g},{b})"

def make_pie_3dish(df, names_col, values_col, color_map, title):
    fig = px.pie(
        df, names=names_col, values=values_col, hole=0.45,
        color=names_col, color_discrete_map=color_map
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label", pull=0.02,
        marker=dict(line=dict(color="rgba(0,0,0,0.12)", width=2))
    )
    return style_fig(fig, title)

def make_bar_3dish(df, x, y, color=None, color_map=None, title="", barmode="group"):
    if color:
        fig = px.bar(
            df, x=x, y=y, color=color, barmode=barmode,
            color_discrete_map=color_map if color_map else None,
            text_auto=".0f"
        )
    else:
        fig = px.bar(df, x=x, y=y, text_auto=".0f")

    # borda suave
    fig.update_traces(marker_line_color="rgba(0,0,0,0.18)", marker_line_width=1.2)

    return style_fig(fig, title)

def make_line_glow(df, x, y_cols, title=""):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        base_color = COLORWAY[i % len(COLORWAY)]
        glow1 = go.Scatter(
            x=df[x], y=df[col], mode="lines",
            line=dict(width=14, color=_darken(base_color, 0.85)), opacity=0.10,
            hoverinfo="skip", showlegend=False
        )
        glow2 = go.Scatter(
            x=df[x], y=df[col], mode="lines",
            line=dict(width=8, color=_darken(base_color, 0.92)), opacity=0.18,
            hoverinfo="skip", showlegend=False
        )
        main = go.Scatter(
            x=df[x], y=df[col], mode="lines+markers",
            line=dict(width=3, color=base_color),
            marker=dict(size=6, line=dict(width=1, color="white")),
            name=str(col)
        )
        fig.add_traces([glow1, glow2, main])
    style_fig(fig, title)
    return fig

def make_funnelarea_3dish(df, stage_col, value_col, color_map, title):
    fig = px.funnel_area(
        df, names=stage_col, values=value_col,
        color=stage_col, color_discrete_map=color_map
    )
    fig.update_traces(marker_line=dict(color="rgba(0,0,0,0.15)", width=1))
    return style_fig(fig, title)

# =========================
# Sidebar — Parâmetros essenciais
# =========================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    target_roas = st.number_input("ROAS alvo", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.subheader("📅 Referências de Tempo")
    week_start = st.date_input(
        "Início da semana (segunda)",
        value=(datetime.today() - timedelta(days=datetime.today().weekday())).date(),
    )
    include_weekends = st.checkbox("Metas consideram finais de semana", value=True, help="Se desmarcar, metas diárias ignoram sábados e domingos.")

    month_ref = st.date_input("Qualquer dia do mês da meta", value=datetime.today().date())

        # =========================
    # 🔌 Meta Marketing API
    # =========================
    st.subheader("🔌 Meta Marketing API")
    act_id = st.text_input(
        "Ad Account ID (ex.: act_1234567890)",
        value="",
        help="Formato: act_XXXX"
    )
    access_token = st.text_input(
        "Access Token",
        type="password",
        value=st.secrets.get("FB_TOKEN", "") if hasattr(st, "secrets") else "",
        help="Token com ads_read / ads_management"
    )
    api_version = st.text_input("API version", value="v23.0")

    st.subheader("🗂️ Nível & Quebra")
    level = st.selectbox("Level (agregação)", ["account","campaign","adset","ad"], index=1)
    breakdowns = st.multiselect(
        "Breakdowns (opcional)",
        ["publisher_platform","platform_position","device_platform","age","gender","region"],
        default=[]
    )

    st.subheader("📅 Intervalo de Datas (para a API)")
    since_api = st.date_input("Desde (API)", value=(datetime.today() - timedelta(days=7)).date())
    until_api = st.date_input("Até (API)", value=datetime.today().date())

    st.subheader("🎯 META MENSAL (base de tudo)")
    goal_type_m = st.radio("Definir por", ["Faturamento", "Compras"], index=0, horizontal=True)
    monthly_goal_value = st.number_input("Valor da meta mensal (R$ se Faturamento; nº se Compras)", value=40000.0, min_value=0.0, step=500.0)

    st.subheader("📥 CSV do Gerenciador")

    st.markdown("**Envie 1 ou mais arquivos (opcional)** — se você enviar mais de um, o app unifica tudo:")
    uploaded_daily   = st.file_uploader("CSV **DIÁRIO** (ex.: export por dia)", type=["csv"], key="csv_diario")
    uploaded_weekly  = st.file_uploader("CSV **SEMANAL** (ex.: export por semana)", type=["csv"], key="csv_semana")
    uploaded_monthly = st.file_uploader("CSV **MENSAL** (ex.: export por mês)", type=["csv"], key="csv_mes")

    st.caption("Formato flexível: o leitor normaliza nomes/acentos e tenta detectar colunas (gasto, faturamento, compras, data etc.).")

def _safe_to_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.normalize()

def unify_frames(frames):
    """Une dataframes (somando métricas em duplicatas por data+campanha)."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # garante colunas básicas
    for c in ["campanha","gasto","faturamento","compras"]:
        if c not in df_all.columns:
            df_all[c] = 0.0 if c != "campanha" else ""

    # achar/normalizar coluna de data
    if "data" not in df_all.columns:
        for alt in ["dia","date","data_inicio"]:
            if alt in df_all.columns:
                df_all["data"] = df_all[alt]
                break

    if "data" in df_all.columns:
        df_all["_date"] = _safe_to_date(df_all["data"])
    else:
        df_all["_date"] = pd.NaT

    # somar números quando repetir (mesma data + mesma campanha)
    num_cols = [c for c in df_all.columns if c not in ["campanha","status","veiculacao","data","_date"]]
    agg = {c: "sum" for c in num_cols}
    if "status" in df_all.columns: agg["status"] = "first"
    if "veiculacao" in df_all.columns: agg["veiculacao"] = "first"

    df_all = (df_all
              .groupby(["_date","campanha"], dropna=False, as_index=False)
              .agg(agg))

    df_all["data"] = df_all["_date"]
    df_all.drop(columns=["_date"], inplace=True)
    return df_all

# =========================
# Helpers
# =========================
def daterange(start_date, end_date, include_weekends=True):
    days, d = [], start_date
    while d <= end_date:
        if include_weekends or d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

@st.cache_data(show_spinner=False)
def read_csv_flex(file):
    import re, unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("(brl)", "").replace(" r$", "").strip()
        return s

    def _read(file):
        for enc in ["utf-8", "latin-1", "utf-16", "cp1252"]:
            try:
                return pd.read_csv(file, sep=None, engine="python", encoding=enc)
            except Exception:
                file.seek(0)
                continue
        file.seek(0)
        return pd.read_csv(file)

    raw = _read(file)

    ALIASES = {
        # Identificação / status
        "campanha": ("nome da campanha", "campanha", "campaign name", "nome da campanha (id)"),
        "status": ("desativado/ativado", "ativado/desativado", "estado", "status da campanha"),
        "veiculacao": ("veiculação", "veiculacao", "veiculação da campanha", "posicionamento"),

        # Investimento / receita / roas
        "gasto": ("valor usado", "valor gasto", "amount spent", "spend", "valor usado brl"),
        "faturamento": (
            "valor de conversão da compra", "valor de conversao da compra",
            "purchase conversion value", "receita"
        ),
        "roas": ("retorno sobre o investimento em publicidade (roas) das compras", "roas"),
        "orcamento": ("orçamento", "budget"),

        # Eventos do funil
        "cliques": ("cliques no link", "link clicks", "clicks"),
        "lp_views": ("visualizações da página de destino", "visualizacoes da pagina de destino", "landing page views"),
        "add_cart": ("adições ao carrinho", "adicoes ao carrinho", "add to cart"),
        "ck_init": (
            "finalizações de compra iniciadas", "finalizacoes de compra iniciadas",
            "checkout iniciado", "conversão checkout", "conversao checkout"
        ),
        "entrega": ("entrega",),
        "pay_info": ("inclusões de informações de pagamento", "inclusoes de informacoes de pagamento", "info. pagamento / entrega"),
        "compras": ("compras", "compras / inf. pagamento", "compras / informações de pagamento", "purchases"),

        # Métricas de alcance / mid-funnel
        "alcance": ("alcance",),
        "impressoes": ("impressões", "impressoes", "impressions"),
        "frequencia": ("frequência", "frequencia", "frequency"),
        "cpm": ("cpm (custo por 1.000 impressões)", "cpm"),
        "cpc": ("cpc (custo por clique no link)", "cpc"),
        "ctr": ("ctr (taxa de cliques no link)", "ctr"),

        # Datas
    "data": (
        "data", "date", "dia",
        "início dos relatórios",   # com acento
        "inicio dos relatorios",   # sem acento
        "término dos relatórios",  # opcional
        "termino dos relatorios"   # opcional sem acento
    ),
}

    norm_map = { _norm(c): c for c in raw.columns }
    rename = {}
    for final, choices in ALIASES.items():
        for cand in choices:
            key = _norm(cand)
            if key in norm_map:
                rename[norm_map[key]] = final
                break

    df = raw.rename(columns=rename).copy()

    def _to_num(x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).lower().replace("r$", " ").replace("brl", " ")
        s = s.replace(".", "").replace(",", ".")
        s = "".join(ch for ch in s if ch.isdigit() or ch in ".-eE")
        try:
            return float(s)
        except:
            return 0.0

    for col in [
        "gasto","orcamento","faturamento","roas",
        "compras","impressoes","alcance","frequencia",
        "cliques","lp_views","add_cart","ck_init","entrega","pay_info",
        "cpc","ctr","cpm"
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # percentuais
    if "ctr" in df.columns:
        df["ctr"] = df["ctr"].apply(lambda v: v/100.0 if v > 1.5 else v)

    return df

def _retry_call(fn, max_retries=5, base_wait=1.5):
    """Retry exponencial simples para lidar com rate limit/transientes."""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "retry", "temporarily unavailable", "timeout"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

@st.cache_data(show_spinner=True, ttl=60*10)
def pull_meta_insights_http(act_id: str, token: str, api_version: str, level: str,
                            since: datetime, until: datetime, breakdowns: list,
                            daily_granularity: bool = True) -> pd.DataFrame:
    """
    Busca insights direto no Graph API (HTTP), pagina com 'after',
    e normaliza para colunas do app: data, gasto, faturamento, compras, campanha, status, veiculacao (+breakdowns).
    """
    if not act_id or not token:
        return pd.DataFrame()

    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    params = {
        "access_token": token,
        "level": level,
        "time_range": json.dumps({
            "since": since.strftime("%Y-%m-%d"),
            "until": until.strftime("%Y-%m-%d"),
        }),
        "limit": 500,
        "fields": ",".join([
            "spend",
            "purchases",
            "purchase_conversion_value",
            "date_start","date_stop",
            "campaign_name","campaign_id",
            "adset_name","adset_id",
            "ad_name","ad_id",
            "account_name","account_id",
            "effective_status"
        ]),
    }
    if breakdowns:
        params["breakdowns"] = ",".join(breakdowns)
    if daily_granularity:
        params["time_increment"] = 1

    rows = []
    next_url = base_url
    next_params = params.copy()

    while next_url:
        def _do():
            return requests.get(next_url, params=next_params, timeout=60)

        resp = _retry_call(_do)

        # Tenta decodificar a resposta
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        # Tratamento de erro com mensagem clara na tela
        if resp.status_code != 200:
            err = (payload or {}).get("error", {})
            code = err.get("code")
            subcode = err.get("error_subcode")
            msg = err.get("message")
            fbtrace = err.get("fbtrace_id")
            st.error(
                f"Graph API error {resp.status_code} | code={code} subcode={subcode}\n"
                f"message: {msg}\nfbtrace_id: {fbtrace}"
            )
            raise RuntimeError(
                f"Graph API error {resp.status_code}: {msg} (code={code}, subcode={subcode})"
            )

        # Se chegou aqui, deu 200 OK — processa os registros
        data = payload.get("data", [])
        for rec in data:
            linha = {}
            linha["data"] = pd.to_datetime(rec.get("date_start"))
            linha["gasto"] = float(rec.get("spend", 0) or 0)
            linha["faturamento"] = float(rec.get("purchase_conversion_value", 0) or 0)
            linha["compras"] = float(rec.get("purchases", 0) or 0)

            if level == "campaign":
                linha["campanha"] = rec.get("campaign_name", "")
            elif level == "adset":
                linha["campanha"] = rec.get("adset_name", "")
            elif level == "ad":
                linha["campanha"] = rec.get("ad_name", "")
            else:
                linha["campanha"] = rec.get("account_name", "")

            linha["status"] = rec.get("effective_status", "")

            if "publisher_platform" in rec:
                linha["veiculacao"] = rec["publisher_platform"]
            else:
                linha["veiculacao"] = ""

            for k in ["campaign_id","adset_id","ad_id","account_id","date_start","date_stop"]:
                if k in rec:
                    linha[k] = rec[k]

            for b in breakdowns or []:
                if b in rec:
                    linha[b] = rec[b]

            rows.append(linha)

        # Paginação
        paging = payload.get("paging", {})
        cursors = paging.get("cursors", {})
        after = cursors.get("after")
        if after:
            next_url = base_url
            next_params = params.copy()
            next_params["after"] = after
        else:
            next_url = None


    df = pd.DataFrame(rows)
    if not df.empty and "data" in df.columns:
        df = df.sort_values("data")
    # garante tipos
    for c in ["gasto","faturamento","compras"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# =========================
# Carregamento de Dados (API Meta → fallback CSV)
# =========================
df_daily = read_csv_flex(uploaded_daily)   if uploaded_daily   else pd.DataFrame()
df_week  = read_csv_flex(uploaded_weekly)  if uploaded_weekly  else pd.DataFrame()
df_month = read_csv_flex(uploaded_monthly) if uploaded_monthly else pd.DataFrame()
df_csv_all = unify_frames([df_daily, df_week, df_month])

df_api = pd.DataFrame()
if act_id and access_token:
    with st.spinner("Conectando ao Meta Ads e coletando insights..."):
        df_api = pull_meta_insights_http(
            act_id=act_id,
            token=access_token,
            api_version=api_version,
            level=level,
            since=datetime.combine(since_api, datetime.min.time()),
            until=datetime.combine(until_api, datetime.min.time()),
            breakdowns=breakdowns,
            daily_granularity=True  # um ponto por dia
        )

# Preferir API se disponível; senão, CSV
df_all = df_api if not df_api.empty else df_csv_all

any_api = not df_api.empty
any_csv = not df_csv_all.empty
any_data = not df_all.empty

if act_id and access_token and df_api.empty:
    st.warning("Conectei, mas não vieram dados para o intervalo/level escolhido. Verifique permissões, datas, level e se há eventos atribuídos.")


def classificar_funil(nome):
    nome = str(nome).lower()
    if "[teste/público]" in nome or "[teste/publico]" in nome:
        return "Teste de Interesse"
    elif "[teste/criativo]" in nome:
        return "Teste de Criativo"
    elif "cbo" in nome or "escala" in nome:
        return "Escala"
    elif "remarketing" in nome:
        return "Remarketing"
    return "Outros"

# =========================
# Metas — MÊS ➜ Semana (consistente por semanas do mês)
# =========================
month_first = month_ref.replace(day=1)
next_month_first = (month_first.replace(year=month_first.year+1, month=1)
                    if month_first.month == 12 else month_first.replace(month=month_first.month+1))
month_last = next_month_first - timedelta(days=1)

def weeks_in_month_alinhada(month_first, month_last, include_weekends=True):
    """
    Cria semanas alinhadas à 2ª feira (seg→dom) dentro do mês.
    A primeira e/ou última podem ser “quebradas” nas bordas do mês.
    """
    # acha a primeira segunda >= month_first
    offset = (0 - month_first.weekday()) % 7  # 0=segunda
    first_monday = month_first + timedelta(days=offset)

    weeks = []

    # bloco parcial antes da primeira segunda (se existir)
    if month_first < first_monday <= month_last:
        w_start = month_first
        w_end = min(first_monday - timedelta(days=1), month_last)
        days_considered = len(daterange(w_start, w_end, include_weekends=include_weekends))
        weeks.append({"start": w_start, "end": w_end, "days_considered": days_considered})

    # semanas cheias seg→dom
    cur = first_monday
    while cur <= month_last:
        w_start = cur
        w_end = min(cur + timedelta(days=6), month_last)
        days_considered = len(daterange(w_start, w_end, include_weekends=include_weekends))
        weeks.append({"start": w_start, "end": w_end, "days_considered": days_considered})
        cur = w_end + timedelta(days=1)

    total_considered = sum(w["days_considered"] for w in weeks) or 1
    for w in weeks:
        w["share"] = w["days_considered"] / total_considered

    return weeks


# Semana escolhida pelo usuário
week_start_dt = datetime.combine(week_start, datetime.min.time())
week_end_dt = week_start_dt + timedelta(days=6)
week_days_all = daterange(week_start_dt.date(), week_end_dt.date(), include_weekends=include_weekends)

# Distribuição de metas: primeiro definimos a meta MENSAL
if goal_type_m == "Faturamento":
    goal_rev_month = float(monthly_goal_value)
    goal_pur_month = goal_rev_month / aov if aov > 0 else 0.0
else:
    goal_pur_month = float(monthly_goal_value)
    goal_rev_month = goal_pur_month * aov

budget_goal_month = goal_rev_month / target_roas if target_roas > 0 else 0.0

# Agora distribuímos por semanas do mês
weeks = weeks_in_month_alinhada(month_first, month_last, include_weekends=include_weekends)

# Acrescenta metas semanais a cada semana
for w in weeks:
    w["goal_rev_week"] = goal_rev_month * w["share"]
    w["goal_pur_week"] = goal_pur_month * w["share"]
    w["budget_goal_week"] = budget_goal_month * w["share"]

# Seleciona a semana ativa (a que o usuário escolheu)
def pick_week(weeks, week_start_dt):
    ws = week_start_dt.date()
    for w in weeks:
        if w["start"] == ws:
            return w
        # fallback: caso o usuário selecione um dia que caia dentro da semana
        if w["start"] <= ws <= w["end"]:
            return w
    return None

sel_week = pick_week(weeks, week_start_dt)
# Se por algum motivo não encontrar, cai na última semana válida
if sel_week is None and weeks:
    sel_week = weeks[-1]

# Estes são usados no restante do app
goal_rev_week  = sel_week["goal_rev_week"]  if sel_week else 0.0
goal_pur_week  = sel_week["goal_pur_week"]  if sel_week else 0.0
budget_goal_week = sel_week["budget_goal_week"] if sel_week else 0.0

# =========================
# Layout por Abas
# =========================
tab_plan, tab_goals, tab_perf, tab_funnel, tab_campaigns, tab_daily, tab_pdf = st.tabs([
    "Planejamento", "Metas", "Performance", "Funil", "Campanhas", "Acompanhamento Diário", "Relatório (PDF)"
])

# =========================
# 💰 Planejamento de Verba por Etapa (%)
# =========================
with tab_plan:
    st.subheader("💰 Planejamento de Verba por Etapa (%)")

    colm1, colm2 = st.columns([2,1])
    with colm1:
        escala_mode = st.radio(
            "Como definir Escala?",
            ["Automático (restante)", "Manual"],
            index=0, horizontal=True,
            help="Automático: Escala recebe o restante após Teste de Interesse, Teste de Criativo e Remarketing."
        )
    with colm2:
        auto_norm = st.checkbox(
            "Normalizar p/ 100%",
            value=True,
            help="Se ligado, ajusta proporcionalmente para que a soma feche em 100%."
        )

    col1, col2, col3, col4 = st.columns(4)
    pct_teste_interesse = col1.number_input("Teste de Interesse (%)", value=20.0, step=1.0, min_value=0.0, max_value=100.0)
    pct_teste_criativo  = col2.number_input("Teste de Criativo (%)",  value=15.0, step=1.0, min_value=0.0, max_value=100.0)
    pct_remarketing     = col3.number_input("Remarketing (%)",        value=15.0, step=1.0, min_value=0.0, max_value=100.0)

    if escala_mode == "Manual":
        pct_escala = col4.number_input("Escala (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0)
    else:
        pct_escala = max(0.0, 100.0 - (pct_teste_interesse + pct_teste_criativo + pct_remarketing))
        col4.metric("Escala (%) (auto)", f"{pct_escala:.1f}")

    total_pct = pct_teste_interesse + pct_teste_criativo + pct_remarketing + pct_escala

    if auto_norm and total_pct > 0:
        fator = 100.0 / total_pct
        n_interesse   = pct_teste_interesse * fator
        n_criativo    = pct_teste_criativo  * fator
        n_remarketing = pct_remarketing     * fator
        n_escala      = pct_escala          * fator
    else:
        n_interesse, n_criativo, n_remarketing, n_escala = (
            pct_teste_interesse, pct_teste_criativo, pct_remarketing, pct_escala
        )

    st.caption(
        f"Total informado: {total_pct:.1f}% "
        + ("(normalizado para 100%)" if auto_norm and abs(total_pct-100.0) > 0.01 else "")
    )

    if not auto_norm and total_pct > 100.0:
        st.error(f"As etapas somam {total_pct:.1f}%. Reduza para 100% ou ative a normalização.")
    elif not auto_norm and total_pct < 100.0 and escala_mode == "Manual":
        st.warning(f"As etapas somam {total_pct:.1f}%. Há {100.0 - total_pct:.1f}% sem alocação.")

    planejado_funil = {
        "Teste de Criativo":  (n_criativo/100.0)    * budget_goal_week,
        "Teste de Interesse": (n_interesse/100.0)   * budget_goal_week,
        "Escala":             (n_escala/100.0)      * budget_goal_week,
        "Remarketing":        (n_remarketing/100.0) * budget_goal_week,
    }

    mix_plot_df = pd.DataFrame({
        "Etapa": list(planejado_funil.keys()),
        "Valor (R$)": list(planejado_funil.values())
    }).sort_values("Valor (R$)", ascending=False)

    fig_mix = make_pie_3dish(
        mix_plot_df, names_col="Etapa", values_col="Valor (R$)",
        color_map=ETAPAS_COLORS, title="Mix Planejado da Verba (R$)"
    )
    st.plotly_chart(fig_mix, use_container_width=True)

    st.markdown("### 💵 Distribuição Planejada da Verba (por dia)")

    week_days_considered_list = [
        d for d in week_days_all
        if (month_first <= d <= month_last) and (include_weekends or d.weekday() < 5)
    ]
    dias = max(1, len(week_days_considered_list))

    rows = []
    for etapa, valor_semana in planejado_funil.items():
        valor_dia = valor_semana / dias
        for d in week_days_considered_list:
            rows.append({
                "Data": d.strftime("%d/%m/%Y"),
                "Etapa": etapa,
                "Valor Diário (R$)": valor_dia
            })
    df_planejado_dia = pd.DataFrame(rows)

    st.dataframe(df_planejado_dia, use_container_width=True)

    fig_stack = make_bar_3dish(
        df_planejado_dia,
        x="Data", y="Valor Diário (R$)",
        color="Etapa", color_map=ETAPAS_COLORS,
        title="Distribuição Planejada da Verba por Dia (R$)",
        barmode="stack"
    )
    fig_stack.update_xaxes(type="category")
    st.plotly_chart(fig_stack, use_container_width=True)


# =========================
# Bloco 1 — Metas (planejado)
# =========================
with tab_goals:
    st.markdown("## 🎯 Metas (Planejado)")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Meta MENSAL — Faturamento", f"R$ {goal_rev_month:,.0f}".replace(",","."))
    with c2:
        st.metric("Meta MENSAL — Compras", f"{goal_pur_month:,.0f}".replace(",","."))
    with c3:
        st.metric("Orçamento MENSAL p/ ROAS", f"R$ {budget_goal_month:,.0f}".replace(",","."))

    # --- Tabela com a meta de CADA semana do mês (consistente com a meta mensal) ---
    if weeks:
        metas_semanas_df = pd.DataFrame([{
            "Semana #": i+1,
            "Início (seg)": w["start"].strftime("%d/%m"),
            "Fim (dom)": w["end"].strftime("%d/%m"),
            "Dias considerados": w["days_considered"],
            "Meta Faturamento (R$)": round(w["goal_rev_week"], 2),
            "Meta Compras (un)": round(w["goal_pur_week"], 2),
            "Orçamento p/ ROAS (R$)": round(w["budget_goal_week"], 2),
        } for i, w in enumerate(weeks)])

        st.markdown("### 📅 Metas por semana do mês (proporcionais aos dias considerados)")
        st.dataframe(metas_semanas_df, use_container_width=True)

    st.markdown("---")


    s1,s2,s3 = st.columns(3)
    with s1:
        st.metric("Meta SEMANAL (derivada) — Faturamento", f"R$ {goal_rev_week:,.0f}".replace(",","."), help="Proporcional aos dias da semana que caem dentro do mês de referência")
    with s2:
        st.metric("Meta SEMANAL (derivada) — Compras", f"{goal_pur_week:,.0f}".replace(",","."))
    with s3:
        st.metric("Orçamento SEMANAL (derivada)", f"R$ {budget_goal_week:,.0f}".replace(",","."))

    # --- Tabela com a meta de CADA semana do mês (consistente com a meta mensal) ---

# =========================
# Bloco 2 — Performance Real
# =========================
with tab_perf:
    st.markdown("## 📥 Performance Real")
    if not any_data:
        st.info("Conecte o Meta Ads (ID da conta + token) ou envie um CSV para ver os KPIs.")
    else:
        df = df_all.copy()

        # --- Filtros ---
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            sel_camp = st.selectbox("Campanha", ["(Todas)"] + sorted(df.get("campanha", pd.Series([""])).dropna().astype(str).unique().tolist())) if "campanha" in df.columns else "(Todas)"
        with colf2:
            sel_status = st.selectbox("Status", ["(Todos)"] + sorted(df.get("status", pd.Series([""])).dropna().astype(str).unique().tolist())) if "status" in df.columns else "(Todos)"
        with colf3:
            sel_veic = st.selectbox("Veiculação", ["(Todas)"] + sorted(df.get("veiculacao", pd.Series([""])).dropna().astype(str).unique().tolist())) if "veiculacao" in df.columns else "(Todas)"

        filt = pd.Series(True, index=df.index)
        if "campanha" in df.columns and sel_camp != "(Todas)":
            filt &= (df["campanha"].astype(str) == sel_camp)
        if "status" in df.columns and sel_status != "(Todos)":
            filt &= (df["status"].astype(str) == sel_status)
        if "veiculacao" in df.columns and sel_veic != "(Todas)":
            filt &= (df["veiculacao"].astype(str) == sel_veic)

        dff = df.loc[filt].copy()

        invest_total = float(dff.get("gasto", pd.Series([0])).sum())
        fatur_total = float(dff.get("faturamento", pd.Series([0])).sum())
        compras_total = float(dff.get("compras", pd.Series([0])).sum())

        # --- KPIs DIÁRIAS (primeiro) ---
        st.markdown("### 📅 KPIs Diárias")
        base_date = (
            dff["data"] if "data" in dff.columns else
            dff["dia"] if "dia" in dff.columns else
            dff["date"] if "date" in dff.columns else
            dff["data_inicio"] if "data_inicio" in dff.columns else
            pd.Series(pd.NaT, index=dff.index)
        )
        dff["_date"] = pd.to_datetime(base_date, errors="coerce", dayfirst=True).dt.normalize()
        daily = (
            dff.dropna(subset=["_date"])
               .groupby("_date", as_index=False)[["gasto","faturamento","compras"]]
               .sum()
               .sort_values("_date")
        )
        if daily.empty:
            st.info("⚠️ Não foi possível identificar dados diários (coluna de data) para calcular KPIs.")
        else:
            daily["ROAS"] = np.where(daily["gasto"] > 0, daily["faturamento"]/daily["gasto"], np.nan)
            daily["CPA"]  = np.where(daily["compras"] > 0, daily["gasto"]/daily["compras"], np.nan)

            for _, row in daily.iterrows():
                st.markdown(f"#### 📆 {row['_date'].strftime('%d/%m/%Y')}")
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("💰 Investimento", f"R$ {row['gasto']:,.0f}".replace(",","."))
                c2.metric("🏪 Faturamento", f"R$ {row['faturamento']:,.0f}".replace(",","."))
                c3.metric("🛒 Compras", f"{row['compras']:,.0f}".replace(",","."))
                c4.metric("📈 ROAS", f"{row['ROAS']:,.2f}".replace(",","."))
                c5.metric("🎯 CPA", f"R$ {row['CPA']:,.2f}".replace(",","."))
                
                # 🚀 Barra de progresso em relação à meta diária
                dias_considerados = daterange(month_first, month_last, include_weekends=include_weekends)
                meta_dia = goal_rev_month / max(1, len(dias_considerados))

                perc = min(1.0, row['faturamento'] / max(1.0, meta_dia))
                st.progress(
                    perc,
                    text=f"Meta diária: R$ {row['faturamento']:,.0f} / R$ {meta_dia:,.0f}".replace(",", ".")
                )

                st.markdown("---")


        # --- Agora KPIs Semanais ---
        dd = dff.copy()
        date_col = next((c for c in ["data","dia","date","data_inicio"] if c in dff.columns), None)
        if date_col:
            dd["_date"] = pd.to_datetime(dd[date_col], errors="coerce", dayfirst=True).dt.normalize()
            week_mask  = (dd["_date"] >= pd.to_datetime(week_start_dt)) & (dd["_date"] <= pd.to_datetime(week_end_dt))
            today = datetime.today().date()  # pega apenas a data de hoje
            month_mask = (dd["_date"] >= pd.to_datetime(month_first)) & (dd["_date"] <= pd.to_datetime(today))
            w = dd.loc[week_mask]
            m = dd.loc[month_mask]
            invest_w = float(w.get("gasto", pd.Series([0])).sum())
            fatur_w  = float(w.get("faturamento", pd.Series([0])).sum())
            compras_w= float(w.get("compras", pd.Series([0])).sum())
            invest_m = float(m.get("gasto", pd.Series([0])).sum())
            fatur_m  = float(m.get("faturamento", pd.Series([0])).sum())
            compras_m= float(m.get("compras", pd.Series([0])).sum())
        else:
            dd["_date"] = pd.NaT
            invest_w = invest_m = invest_total
            fatur_w  = fatur_m  = fatur_total
            compras_w= compras_m= compras_total

        roas_w = (fatur_w/invest_w) if invest_w>0 else 0.0
        cpa_w  = (invest_w/compras_w) if compras_w>0 else 0.0
        st.markdown("### 📌 KPIs Semanais (Reais)")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("💰 Investimento — Semana", f"R$ {invest_w:,.0f}".replace(",","."))
        k2.metric("🏪 Faturamento — Semana", f"R$ {fatur_w:,.0f}".replace(",","."))
        k3.metric("📈 ROAS — Semana", f"{roas_w:,.2f}".replace(",","."))
        k4.metric("🎯 CPA — Semana", f"R$ {cpa_w:,.2f}".replace(",","."))
        k5.metric("🛒 Compras — Semana", f"{compras_w:,.0f}".replace(",","."))
        st.progress(min(1.0, fatur_w/max(1.0, goal_rev_week)), text=f"Semana: R$ {fatur_w:,.0f} / R$ {goal_rev_week:,.0f}".replace(",","."))

        # --- KPIs Mensais ---
        roas_m = (fatur_m/invest_m) if invest_m>0 else 0.0
        cpa_m  = (invest_m/compras_m) if compras_m>0 else 0.0
        st.markdown("### 📌 KPIs Mensais (Reais)")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Investimento — Mês", f"R$ {invest_m:,.0f}".replace(",","."))
        m2.metric("Faturamento — Mês", f"R$ {fatur_m:,.0f}".replace(",","."))
        m3.metric("ROAS — Mês", f"{roas_m:,.2f}".replace(",","."))
        m4.metric("CPA — Mês", f"R$ {cpa_m:,.2f}".replace(",","."))
        m5.metric("Compras — Mês", f"{compras_m:,.0f}".replace(",", "."))

        # 🚀 Barra de progresso simples em relação à meta mensal inteira
        perc_mensal = min(1.0, fatur_m / max(1.0, goal_rev_month))
        st.progress(
            perc_mensal,
            text=f"Mês: R$ {fatur_m:,.0f} / R$ {goal_rev_month:,.0f}".replace(",", ".")
        )


        # --- ROAS Diário (gráfico linha) ---
        st.markdown("### 📅 ROAS Diário")
        if not daily.empty:
            fig_roas = make_line_glow(daily, x="_date", y_cols=["ROAS"], title="ROAS Diário")
            st.plotly_chart(fig_roas, use_container_width=True)
        else:
           st.info("⚠️ Não foi possível identificar dados diários (coluna de data) para calcular o ROAS.")
            
# =========================
# Funil (volumes + taxas)
# =========================
with tab_funnel:
    st.markdown("### 🧭 Funil (volumes do filtro)")

    if any_data:
        if 'dff' not in locals():
            dff = df_all.copy()

        def _sum(col):
            return float(dff[col].sum()) if col in dff.columns else 0.0

        clicks  = _sum("cliques")
        lp      = _sum("lp_views")
        ck      = _sum("ck_init")
        compras = _sum("compras")

        funil = pd.DataFrame({
            "Etapa": ["Cliques","LP Views","Checkout","Compras"],
            "Volume": [clicks, lp, ck, compras]
        })
        funil = funil[funil["Volume"] > 0]

        if not funil.empty:
            st.dataframe(funil, use_container_width=True)
            fig_funil = make_funnelarea_3dish(
                funil, stage_col="Etapa", value_col="Volume",
                color_map=ETAPAS_COLORS, title="Funil de Conversão (Volume)"
            )
            st.plotly_chart(fig_funil, use_container_width=True)

            st.markdown("### 📈 Taxas do Funil (sem AddToCart)")
            def _rate(num, den): return (num / den) if den > 0 else np.nan
            taxas = [
                {"De→Para": "Cliques → LP",      "Taxa": _rate(lp, clicks)},
                {"De→Para": "LP → Checkout",     "Taxa": _rate(ck, lp)},
                {"De→Para": "Checkout → Compra", "Taxa": _rate(compras, ck)},
            ]
            df_taxas = pd.DataFrame(taxas)
            df_taxas["Taxa (%)"] = (df_taxas["Taxa"] * 100).round(2)
            st.dataframe(df_taxas[["De→Para","Taxa (%)"]], use_container_width=True)

            df_taxas_plot = df_taxas.dropna(subset=["Taxa"])
            if not df_taxas_plot.empty:
                fig_taxas = px.bar(
                    df_taxas_plot, x="Taxa", y="De→Para", orientation="h",
                    text="Taxa", range_x=[0, 1]
                )
                fig_taxas.update_traces(texttemplate="%{x:.0%}", textposition="outside")
                fig_taxas.update_layout(xaxis_tickformat=".0%")
                st.plotly_chart(style_fig(fig_taxas, "Taxas por Etapa (Cliques→LP→Checkout→Compra)"), use_container_width=True)
        else:
            st.info("⚠️ Não há volumes para montar o funil no filtro atual.")
    else:
        st.warning("⚠️ Sem dados. Conecte o Meta Ads ou envie um CSV para visualizar o funil.")

    st.markdown("---")


# =========================
# Ranking de campanhas
# =========================
with tab_campaigns:
    st.markdown("### 🏆 Campanhas (Top 10 por ROAS)")
    if any_data:
        if 'dff' not in locals():
            dff = df_all.copy()

        if "campanha" in dff.columns:
            grp = dff.groupby("campanha").agg({
                **({"gasto":"sum"} if "gasto" in dff.columns else {}),
                **({"faturamento":"sum"} if "faturamento" in dff.columns else {}),
                **({"compras":"sum"} if "compras" in dff.columns else {}),
            }).reset_index()

            if "gasto" in grp.columns and "faturamento" in grp.columns:
                grp["ROAS"] = grp["faturamento"] / grp["gasto"].replace(0, np.nan)
            if "gasto" in grp.columns and "compras" in grp.columns:
                grp["CPA"] = grp["gasto"] / grp["compras"].replace(0, np.nan)

            order_cols = [c for c in ["ROAS","faturamento","gasto"] if c in grp.columns]
            if order_cols:
                grp = grp.sort_values(order_cols, ascending=[False, False, True]).head(10)

            friendly = grp.rename(columns={
                "campanha":"Campanha",
                "gasto":"Investimento (R$)",
                "faturamento":"Faturamento (R$)"
            })
            st.dataframe(friendly, use_container_width=True)
        else:
            st.info("⚠️ A fonte atual não contém a coluna 'campanha'. Ajuste o nível da API (ex.: campaign/adset/ad) ou verifique o CSV.")
    else:
        st.warning("⚠️ Sem dados. Conecte o Meta Ads ou envie um CSV para visualizar o ranking.")

    st.markdown("---")

    st.markdown("### 💵 Orçamento por Etapa (Planejado vs Realizado)")
    if any_data:
        if 'dff' not in locals():
            dff = df_all.copy()

        dff["etapa_funil"] = dff["campanha"].apply(classificar_funil) if "campanha" in dff.columns else "Outros"
        realizado_funil = dff.groupby("etapa_funil")["gasto"].sum().to_dict() if "gasto" in dff.columns else {}

        etapas = ["Teste de Criativo", "Teste de Interesse", "Escala", "Remarketing"]
        comp = pd.DataFrame({
            "Etapa": etapas,
            "Planejado (R$)": [planejado_funil.get(e,0) for e in etapas],
            "Realizado (R$)": [realizado_funil.get(e,0) for e in etapas],
        })
        comp["Diferença (R$)"] = comp["Realizado (R$)"] - comp["Planejado (R$)"]

        st.dataframe(comp, use_container_width=True)

        st.markdown("### 📌 Orientações de Verba")
        for _, row in comp.iterrows():
            etapa = row["Etapa"]
            diff = row["Diferença (R$)"]
            if diff < 0:
                st.warning(f"➡️ Falta investir **R$ {abs(diff):,.0f}** em **{etapa}** para bater o planejado.".replace(",","."))
            elif diff > 0:
                st.info(f"✅ Já investiu **R$ {diff:,.0f}** a mais do que o planejado em **{etapa}**.".replace(",","."))
            else:
                st.success(f"⚖️ A etapa **{etapa}** está exatamente alinhada com o planejado.")

        col1, col2 = st.columns(2)
        with col1:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comp["Etapa"], y=comp["Planejado (R$)"],
                name="Planejado (R$)",
                marker_color="#CBD5E1",
                text=comp["Planejado (R$)"], textposition="outside"
            ))
            fig_comp.add_trace(go.Bar(
                x=comp["Etapa"], y=comp["Realizado (R$)"],
                name="Realizado (R$)",
                marker_color="#0EA5E9",
                text=comp["Realizado (R$)"], textposition="outside"
            ))
            fig_comp.update_layout(
                barmode="group",
                title="Orçamento — Planejado vs Realizado",
                font=dict(size=12),
                xaxis=dict(title="Etapa"),
                yaxis=dict(title="Valor (R$)"),
                margin=dict(l=40, r=20, t=60, b=40)
            )
            fig_comp.update_traces(texttemplate="%{y:.0f}", textposition="outside")
            fig_comp.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig_comp, use_container_width=True)

        with col2:
            fig_real = px.pie(
                comp, values="Realizado (R$)", names="Etapa",
                hole=0.45, color="Etapa", color_discrete_map=ETAPAS_COLORS
            )
            fig_real.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(style_fig(fig_real, "Distribuição Realizada por Etapa"), use_container_width=True)
    else:
        st.warning("⚠️ Sem dados. Conecte o Meta Ads ou envie um CSV para visualizar o orçamento por etapa.")

# =========================
# Bloco 3 — Acompanhamento Diário (enxuto, derivado da meta mensal)
# =========================
st.markdown("---")
with tab_daily:
    st.subheader("✅ Acompanhamento Diário — Semana (meta derivada do mês)")

    week_days_considered_list = [
        d for d in week_days_all
        if (month_first <= d <= month_last) and (include_weekends or d.weekday() < 5)
    ]

    meta_dia_rev = (goal_rev_week / max(1,len(week_days_considered_list))) if week_days_considered_list else 0.0
    budget_dia = (budget_goal_week / max(1,len(week_days_considered_list))) if week_days_considered_list else 0.0

    base = pd.DataFrame({
        "data": week_days_considered_list if week_days_considered_list else week_days_all,
        "meta_faturamento": meta_dia_rev,
        "meta_investimento": budget_dia,
        "real_faturamento": 0.0,
        "real_investimento": 0.0,
    })

    edited = st.data_editor(
        base,
        column_config={
            "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY", step=1),
            "meta_faturamento": st.column_config.NumberColumn("Meta Faturamento (R$)"),
            "meta_investimento": st.column_config.NumberColumn("Meta Investimento (R$)"),
            "real_faturamento": st.column_config.NumberColumn("Real Faturamento (R$)"),
            "real_investimento": st.column_config.NumberColumn("Real Investimento (R$)"),
        },
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        df_plot = edited.copy()
        df_plot["data"] = pd.to_datetime(df_plot["data"])
        df_plot = df_plot.sort_values("data")
        df_plot["meta_fat_cum"] = df_plot["meta_faturamento"].cumsum()
        df_plot["real_fat_cum"] = df_plot["real_faturamento"].cumsum()
        st.plotly_chart(px.line(df_plot, x="data", y=["meta_fat_cum","real_fat_cum"], title="Faturamento Acumulado — Meta vs Real (Semana)"), use_container_width=True)
    with col_b:
        df_inv = edited.copy()
        df_inv["data"] = pd.to_datetime(df_inv["data"])
        df_inv = df_inv.sort_values("data")
        df_inv["meta_inv_cum"] = df_inv["meta_investimento"].cumsum()
        df_inv["real_inv_cum"] = df_inv["real_investimento"].cumsum()
        st.plotly_chart(px.line(df_inv, x="data", y=["meta_inv_cum","real_inv_cum"], title="Investimento Acumulado — Meta vs Real (Semana)"), use_container_width=True)

    out = edited.copy()
    out["data"] = pd.to_datetime(out["data"]).dt.strftime("%Y-%m-%d")
    st.download_button("⬇️ Baixar plano semanal derivado (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="plano_semana_derivado.csv", mime="text/csv")

# =========================
# 📦 Relatório para Sócios — Download (PDF)
# =========================
st.markdown("---")

# --- Funções auxiliares para o PDF ---
from io import BytesIO
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors

def df_to_table(df, col_widths=None):
    """Converte DataFrame em tabela do ReportLab com estilo básico."""
    if df is None or df.empty:
        return None
    data = [list(df.columns)] + df.values.tolist()
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F2F6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#333333")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DDDDDD")),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FBFBFD")]),
    ]))
    return tbl

def fig_to_rl_image(fig, width=500):
    """Converte figura Plotly em Image (ReportLab) usando kaleido."""
    if fig is None:
        return None
    try:
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=40, r=20, t=50, b=40)
        )
        png = fig.to_image(format="png", scale=2)  # requer 'kaleido'
    except Exception:
        st.error("Não foi possível exportar o gráfico para o PDF. Verifique se 'kaleido' está no requirements.txt.")
        return None
    bio = BytesIO(png)
    img = Image(bio)
    img._restrictSize(width, 9999)
    return img


with tab_pdf:
    st.header("📦 Relatório para Sócios — PDF")

    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import plotly.express as px
    import plotly.io as pio

    # ----------------------------
    # Tabelas-base para o PDF
    # ----------------------------
    resumo_df = pd.DataFrame({
        "Item": [
            "Mês de Referência",
            "Semana (início → fim)",
            "Ticket Médio (AOV)",
            "ROAS Alvo",
            "Meta Mensal — Faturamento",
            "Meta Mensal — Compras",
            "Orçamento Mensal p/ ROAS",
            "Meta Semanal — Faturamento (derivada)",
            "Meta Semanal — Compras (derivada)",
            "Orçamento Semanal (derivada)",
        ],
        "Valor": [
            month_first.strftime("%m/%Y"),
            f"{week_start_dt.date().strftime('%d/%m/%Y')} → {week_end_dt.date().strftime('%d/%m/%Y')}",
            f"R$ {aov:,.2f}".replace(",", "."),
            f"{target_roas:,.2f}".replace(",", "."),
            f"R$ {goal_rev_month:,.0f}".replace(",", "."),
            f"{goal_pur_month:,.0f}".replace(",", "."),
            f"R$ {budget_goal_month:,.0f}".replace(",", "."),
            f"R$ {goal_rev_week:,.0f}".replace(",", "."),
            f"{goal_pur_week:,.0f}".replace(",", "."),
            f"R$ {budget_goal_week:,.0f}".replace(",", "."),
        ]
    })

    mix_df = pd.DataFrame({
        "Etapa": list(planejado_funil.keys()),
        "Planejado (R$)": list(planejado_funil.values())
    }).sort_values("Planejado (R$)", ascending=False)

    plano_df = df_planejado_dia.copy() if 'df_planejado_dia' in locals() else pd.DataFrame()

    funil_df = pd.DataFrame()
    if 'clicks' in locals():
        funil_df = pd.DataFrame({
            "Etapa": ["Cliques", "LP Views", "Checkout", "Compras"],
            "Volume": [clicks, lp, ck, compras]
        })

    taxas_df = df_taxas[["De→Para","Taxa (%)"]].copy() if 'df_taxas' in locals() else pd.DataFrame()
    comp_df = comp.copy() if 'comp' in locals() else pd.DataFrame()

    kpis_sem_df = pd.DataFrame([{
        "Investimento — Semana (R$)": invest_w if 'invest_w' in locals() else np.nan,
        "Faturamento — Semana (R$)": fatur_w if 'fatur_w' in locals() else np.nan,
        "ROAS — Semana": roas_w if 'roas_w' in locals() else np.nan,
        "CPA — Semana (R$)": cpa_w if 'cpa_w' in locals() else np.nan,
        "Compras — Semana (nº)": compras_w if 'compras_w' in locals() else np.nan,
    }])

    kpis_mes_df = pd.DataFrame([{
        "Investimento — Mês (R$)": invest_m if 'invest_m' in locals() else np.nan,
        "Faturamento — Mês (R$)": fatur_m if 'fatur_m' in locals() else np.nan,
        "ROAS — Mês": roas_m if 'roas_m' in locals() else np.nan,
        "CPA — Mês (R$)": cpa_m if 'cpa_m' in locals() else np.nan,
        "Compras — Mês (nº)": compras_m if 'compras_m' in locals() else np.nan,
    }])

    # ----------------------------
    # Gráficos (cores alinhadas ao dashboard)
    # ----------------------------
    # 1) Pizza do mix planejado
    fig_mix_pdf = px.pie(
        mix_df,
        values="Planejado (R$)",
        names="Etapa",
        title="Mix Planejado da Verba (R$)",
        color="Etapa",
        color_discrete_map=ETAPAS_COLORS
    )
    fig_mix_pdf = style_fig_pdf(fig_mix_pdf)

    # 2) Barras empilhadas por dia (planejado)
    fig_dia = None
    if not plano_df.empty:
        fig_dia = px.bar(
            plano_df, x="Data", y="Valor Diário (R$)",
            color="Etapa",
            title="Distribuição Planejada por Dia (R$)",
            barmode="stack",
            color_discrete_map=ETAPAS_COLORS
        )
        fig_dia = style_fig_pdf(fig_dia)

    # 3) Funil de volumes
    fig_funil_pdf = None
    if not funil_df.empty:
        fig_funil_pdf = px.funnel(
            funil_df, x="Volume", y="Etapa",
            title="Funil de Conversão (Volume)",
            color="Etapa",
            color_discrete_map=ETAPAS_COLORS
        )
        fig_funil_pdf = style_fig_pdf(fig_funil_pdf)

    # 4) Planejado vs Realizado por etapa
    fig_comp_pdf = None
    if not comp_df.empty:
        fig_comp_pdf = px.bar(
            comp_df, x="Etapa", y=["Planejado (R$)", "Realizado (R$)"],
            barmode="group", title="Orçamento — Planejado vs Realizado",
            color_discrete_sequence=["#CBD5E1", "#0EA5E9"]
        )
        fig_comp_pdf.update_traces(texttemplate="%{y:.0f}", textposition="outside")
        fig_comp_pdf = style_fig_pdf(fig_comp_pdf)

    # 5) ROAS diário — aplica tema/fundo branco e colorway
    fig_roas_pdf = None
    if 'daily' in locals() and not daily.empty and "ROAS" in daily.columns:
        fig_roas_pdf = px.line(daily, x="_date", y="ROAS", title="ROAS Diário")
        fig_roas_pdf.update_traces(mode="lines+markers")
        fig_roas_pdf.update_yaxes(title="ROAS", tickformat=".2f")
        fig_roas_pdf.update_xaxes(title="Data")
        fig_roas_pdf = style_fig_pdf(fig_roas_pdf)

    # ----------------------------
    # Montagem do PDF
    # ----------------------------
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]; H1.fontSize = 16
    H2 = styles["Heading2"]; H2.fontSize = 13
    N = styles["BodyText"]; N.fontSize = 10

    story = []
    story.append(Paragraph("Relatório para Sócios — Metas & Performance", H1))
    sub = f"Mês: {month_first.strftime('%m/%Y')} | Semana: {week_start_dt.date().strftime('%d/%m')}–{week_end_dt.date().strftime('%d/%m')} | Gerado em: {datetime.today().strftime('%d/%m/%Y %H:%M')}"
    story.append(Paragraph(sub, N))
    story.append(Spacer(1, 10))

    # Resumo
    story.append(Paragraph("Resumo", H2))
    tbl_resumo = df_to_table(resumo_df, col_widths=[180, 330])
    if tbl_resumo: story.append(tbl_resumo)
    story.append(Spacer(1, 12))

    # Mix Planejado
    story.append(Paragraph("Mix Planejado da Verba", H2))
    tbl_mix = df_to_table(mix_df, col_widths=[200, 150])
    if tbl_mix:
        story.append(tbl_mix); story.append(Spacer(1, 6))
    img_mix = fig_to_rl_image(fig_mix_pdf, width=460)
    if img_mix: story.append(img_mix); story.append(Spacer(1, 12))

    # Distribuição por dia
    if fig_dia is not None:
        story.append(Paragraph("Distribuição Planejada por Dia", H2))
        img_dia = fig_to_rl_image(fig_dia, width=460)
        if img_dia: story.append(img_dia); story.append(Spacer(1, 12))

    # Funil + Taxas
    if fig_funil_pdf is not None:
        story.append(Paragraph("Funil de Conversão (Volume)", H2))
        img_funil = fig_to_rl_image(fig_funil_pdf, width=460)
        if img_funil: story.append(img_funil); story.append(Spacer(1, 6))
        if not taxas_df.empty:
            story.append(Paragraph("Taxas do Funil", H2))
            tbl_taxas = df_to_table(taxas_df, col_widths=[220, 80])
            if tbl_taxas: story.append(tbl_taxas); story.append(Spacer(1, 12))

    # KPIs Semanais
    if not kpis_sem_df.empty:
        story.append(Paragraph("KPIs Semanais", H2))
        tbl_kpi_w = df_to_table(kpis_sem_df.round(2))
        if tbl_kpi_w: story.append(tbl_kpi_w); story.append(Spacer(1, 10))

    # KPIs Mensais
    if not kpis_mes_df.empty:
        story.append(Paragraph("KPIs Mensais", H2))
        tbl_kpi_m = df_to_table(kpis_mes_df.round(2))
        if tbl_kpi_m: story.append(tbl_kpi_m); story.append(Spacer(1, 12))

    # Comparativo por etapa
    if fig_comp_pdf is not None:
        story.append(Paragraph("Orçamento por Etapa — Comparativo", H2))
        img_comp = fig_to_rl_image(fig_comp_pdf, width=460)
        if img_comp: story.append(img_comp); story.append(Spacer(1, 12))
        if not comp_df.empty:
            tbl_comp = df_to_table(comp_df.round(2))
            if tbl_comp: story.append(tbl_comp); story.append(Spacer(1, 12))

    # ROAS Diário
    if fig_roas_pdf is not None:
        story.append(Paragraph("ROAS Diário", H2))
        img_roas = fig_to_rl_image(fig_roas_pdf, width=460)
        if img_roas: story.append(img_roas); story.append(Spacer(1, 12))

    # Finaliza PDF
    doc.build(story)
    buffer.seek(0)

    file_pdf_name = f"Relatorio_Socios_{datetime.today().strftime('%Y-%m-%d')}.pdf"
    st.download_button(
        "⬇️ Baixar Relatório para Sócios (PDF)",
        data=buffer.getvalue(),
        file_name=file_pdf_name,
        mime="application/pdf",
        help="PDF com resumo, tabelas e gráficos (mix, diário, funil, KPIs, comparativos)."
    )

    st.caption("Dica: envie este PDF no grupo dos sócios. Ele já vem com resumo, metas e comparativos — didático e objetivo.")


st.info("Esta versão deriva toda a semana a partir da META MENSAL, proporcional aos dias da semana que caem no mês selecionado e respeitando a opção de incluir/excluir finais de semana.")
