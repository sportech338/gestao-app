import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Metas & Performance — Tráfego Pago", layout="wide")

st.title("📊 Metas & Performance — Tráfego Pago")
st.caption("Defina metas semanais/mensais e analise a performance real do Gerenciador (funil, mídia, vídeo, ranking).")

# =========================
# Sidebar — Parâmetros gerais
# =========================
with st.sidebar:
    st.header("⚙️ Parâmetros de Meta")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    to_checkout = st.number_input("Taxa Sessão → Checkout (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, format="%.2f") / 100.0
    checkout_conv = st.number_input("Taxa Checkout → Compra (%)", value=40.0, min_value=0.0, max_value=100.0, step=1.0, format="%.2f") / 100.0
    target_roas = st.number_input("ROAS alvo (ex: 2.0 = 200%)", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.markdown("---")
    st.subheader("📆 Janela Semanal")
    week_start = st.date_input(
        "Início da semana (segunda)",
        value=(datetime.today() - timedelta(days=datetime.today().weekday())).date(),
    )
    include_weekends = st.checkbox("Incluir finais de semana", value=True)

    st.subheader("📅 Janela Mensal")
    month_ref = st.date_input("Ref. do mês (qualquer dia do mês)", value=datetime.today().date())

    st.markdown("---")
    st.subheader("🎯 Tipo de Meta")
    goal_type = st.radio("Defina por:", options=["Faturamento", "Compras"], index=0, horizontal=True)
    weekly_goal_value = st.number_input("Meta SEMANAL (R$ se Faturamento; nº se Compras)", value=10000.0, min_value=0.0, step=100.0)
    monthly_goal_value = st.number_input("Meta MENSAL (R$ se Faturamento; nº se Compras)", value=40000.0, min_value=0.0, step=500.0)

    st.markdown("---")
    st.subheader("🚦 Alertas")
    alert_roas_min = st.number_input("ROAS mínimo aceitável", value=1.5, step=0.1)
    alert_cpa_max = st.number_input("CPA máximo aceitável (R$)", value=40.0, step=1.0)

    st.markdown("---")
    st.subheader("📥 Dados do Gerenciador (CSV)")
    uploaded = st.file_uploader("Envie o CSV exportado do Gerenciador de Anúncios (separador vírgula).", type=["csv"])

# =========================
# Helpers
# =========================
def daterange(start_date, end_date, include_weekends=True):
    days = []
    d = start_date
    while d <= end_date:
        if include_weekends or d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def build_targets(dates, period_goal_value, goal_type, aov, to_checkout, checkout_conv, target_roas):
    if goal_type == "Faturamento":
        rev_goal = float(period_goal_value)
        purchases_goal = rev_goal / aov if aov > 0 else 0.0
    else:
        purchases_goal = float(period_goal_value)
        rev_goal = purchases_goal * aov

    sessions_needed = purchases_goal / (to_checkout * checkout_conv) if (to_checkout > 0 and checkout_conv > 0) else 0.0
    spend_budget = rev_goal / target_roas if target_roas > 0 else 0.0
    roi = (rev_goal - spend_budget) / spend_budget if spend_budget > 0 else 0.0

    n_days = max(1, len(dates))
    daily_targets = pd.DataFrame({
        "data": dates,
        "meta_faturamento": rev_goal / n_days,
        "meta_compras": purchases_goal / n_days,
        "meta_sessoes": sessions_needed / n_days,
        "meta_investimento": spend_budget / n_days
    })
    totals = {
        "faturamento": rev_goal,
        "compras": purchases_goal,
        "sessoes": sessions_needed,
        "investimento": spend_budget,
        "roi_estimado": roi
    }
    return daily_targets, totals

def pct(a, b):
    return (a/b) if b and b != 0 else 0.0

# =========================
# Metas (semana e mês)
# =========================
week_start_dt = datetime.combine(week_start, datetime.min.time())
week_end_dt = week_start_dt + timedelta(days=6)
week_days = daterange(week_start_dt.date(), week_end_dt.date(), include_weekends)

weekly_df, weekly_totals = build_targets(
    week_days, weekly_goal_value, goal_type, aov, to_checkout, checkout_conv, target_roas
)

month_first = month_ref.replace(day=1)
if month_first.month == 12:
    next_month_first = month_first.replace(year=month_first.year+1, month=1)
else:
    next_month_first = month_first.replace(month=month_first.month+1)
month_last = next_month_first - timedelta(days=1)

month_days = daterange(month_first, month_last, include_weekends=True)
monthly_df, monthly_totals = build_targets(
    month_days, monthly_goal_value, goal_type, aov, to_checkout, checkout_conv, target_roas
)

# =========================
# Seção 1 — KPIs de Meta (planejado)
# =========================
st.markdown("## 🎯 Metas Planejadas")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Meta Semana (R$)", f"R$ {weekly_totals['faturamento']:,.0f}".replace(",", "."))
k2.metric("Meta Semana (Compras)", f"{weekly_totals['compras']:,.0f}".replace(",", "."))
k3.metric("Sessões Necessárias (semana)", f"{weekly_totals['sessoes']:,.0f}".replace(",", "."))
k4.metric("Orçamento p/ ROAS alvo (semana)", f"R$ {weekly_totals['investimento']:,.0f}".replace(",", "."))
k5.metric("ROI Estimado (semana)", f"{weekly_totals['roi_estimado']*100:,.0f}%".replace(",", "."))

# =========================
# Seção 2 — Dados do Gerenciador (upload)
# =========================
st.markdown("---")
st.markdown("## 📥 Performance Real (Gerenciador)")

if uploaded:
    # =========================
    # Leitura flexível + normalização de cabeçalhos
    # =========================
    import re, unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("(brl)", "").replace(" r$", "").strip()
        return s

    def _read_flex(file):
        for enc in ["utf-8", "latin-1", "utf-16", "cp1252"]:
            try:
                return pd.read_csv(file, sep=None, engine="python", encoding=enc)
            except Exception:
                continue
        return pd.read_csv(file)

    raw = _read_flex(uploaded)

    # Mapa de aliases -> nomes finais que seu dashboard usa
    ALIASES = {
        "Desativado/Ativado": ("desativado/ativado","ativado/desativado","status da campanha","estado"),
        "campanha": ("nome da campanha","campanha","nome da campanha (id)"),
        "Veiculação": ("veiculacao da campanha","veiculacao","posicionamento"),
        "Resultados": ("resultados",),
        "Custo por resultado": ("custo por resultado","custo por resultados"),
        "Orçamento": ("orcamento","orcamento do conjunto de anuncios","orcamento do conjunto de anúncios"),
        "Valor usado": ("valor usado","valor usado brl","valor gasto","valor gasto brl"),
        "Retorno sobre o investimento em publicidade (ROAS) das compras": ("roas das compras","retorno sobre o investimento em publicidade (roas) das compras","roas"),
        "Valor de conversão da compra": ("valor de conversao da compra","valor de conversão da compra","receita","faturamento"),
        "Custo por finalização de compra iniciada": ("custo por finalizacao de compra iniciada","custo por inic. checkout","custo por checkout iniciado"),
        "Alcance": ("alcance",),
        "Impressões": ("impressoes","impressões"),
        "Frequência": ("frequencia","frequência"),
        "CPM (custo por 1.000 impressões)": ("cpm (custo por 1.000 impressoes)","cpm"),
        "Conexão": ("conexao","conexão"),
        "Conversão Página": ("conversao pagina","conversao de pagina","conversão pagina","conversão de página","visualizacoes da pagina de destino"),
        "Entrega": ("entrega","entrega.1"),
        "Info. Pagamento / Entrega": ("info. pagamento / entrega","informacoes de pagamento / entrega","informações de pagamento / entrega"),
        "Compras / Inf. Pagamento": ("compras / inf. pagamento","compras/inf. pagamento"),
        "Conversão Checkout": ("conversao checkout","conversão checkout"),
        "Cliques no link": ("cliques no link","cliques"),
        "Visualizações da página de destino": ("visualizacoes da pagina de destino","visualizações da página de destino","page views"),
        "Adições ao carrinho": ("adicoes ao carrinho","adições ao carrinho","add to cart"),
        "Finalizações de compra iniciadas": ("finalizacoes de compra iniciadas","finalizações de compra iniciadas","checkout iniciado"),
        "Inclusões de informações de pagamento": ("inclusoes de informacoes de pagamento","inclusões de informações de pagamento","pagamento info"),
        "Compras": ("compras","purchases"),
        "CPC (custo por clique no link)": ("cpc (custo por clique no link)","cpc"),
        "CTR (taxa de cliques no link)": ("ctr (taxa de cliques no link)","ctr"),
        "Reproduções de 25% do vídeo": ("reproducoes de 25% do video","reproduções de 25% do vídeo"),
        "Reproduções de 50% do vídeo": ("reproducoes de 50% do video","reproduções de 50% do vídeo"),
        "Reproduções de 75% do vídeo": ("reproducoes de 75% do video","reproduções de 75% do vídeo"),
        "Reproduções de 95% do vídeo": ("reproducoes de 95% do video","reproduções de 95% do vídeo"),
        "Tempo médio de reprodução do vídeo": ("tempo medio de reproducao do video","tempo médio de reprodução do vídeo"),
    }

    norm_map = {_norm(c): c for c in raw.columns}
    rename_dict = {}
    for final_name, choices in ALIASES.items():
        for cand in choices:
            if cand in norm_map:
                rename_dict[norm_map[cand]] = final_name
                break

    df = raw.rename(columns=rename_dict).copy()
    df = df.loc[:, ~df.columns.duplicated()]

    # Conversão numérica robusta (formato BR)
    def _to_num(x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().lower()
        s = s.replace("r$", "").replace("brl", "")
        s = s.replace(".", "").replace(",", ".")
        s = re.sub(r"[^0-9\.\-eE]", "", s)
        try:
            return float(s)
        except:
            return 0.0

    for col in [
        "Valor usado","Valor de conversão da compra","Compras",
        "CPC (custo por clique no link)","CTR (taxa de cliques no link)",
        "CPM (custo por 1.000 impressões)","Impressões","Alcance","Frequência",
        "Cliques no link","Visualizações da página de destino","Adições ao carrinho",
        "Finalizações de compra iniciadas","Inclusões de informações de pagamento",
        "Custo por resultado","Custo por finalização de compra iniciada",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # =========================
    # Filtros (APENAS UMA VEZ)
    # =========================
    st.markdown("### 🔎 Filtros")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        campanhas = ["(Todas)"] + sorted(df["campanha"].astype(str).unique().tolist()) if "campanha" in df.columns else ["(Todas)"]
        sel_campanha = st.selectbox("Campanha", campanhas)
    with colf2:
        status = ["(Todos)"] + sorted(df["Desativado/Ativado"].astype(str).unique().tolist()) if "Desativado/Ativado" in df.columns else ["(Todos)"]
        sel_status = st.selectbox("Status", status)
    with colf3:
        veics = ["(Todas)"] + sorted(df["Veiculação"].astype(str).unique().tolist()) if "Veiculação" in df.columns else ["(Todas)"]
        sel_veic = st.selectbox("Veiculação", veics)

    filt = pd.Series(True, index=df.index)
    if "campanha" in df.columns and sel_campanha != "(Todas)":
        filt &= (df["campanha"].astype(str) == sel_campanha)
    if "Desativado/Ativado" in df.columns and sel_status != "(Todos)":
        filt &= (df["Desativado/Ativado"].astype(str) == sel_status)
    if "Veiculação" in df.columns and sel_veic != "(Todas)":
        filt &= (df["Veiculação"].astype(str) == sel_veic)

    dff = df.loc[filt].copy()
    # 🔸 Só analisa linhas com gasto > 0
    if "Valor usado" in dff.columns:
        dff = dff[dff["Valor usado"] > 0].copy()

    # =========================
    # KPIs principais (real)
    # =========================
    invest = dff["Valor usado"].sum() if "Valor usado" in dff.columns else 0.0
    fatur = dff["Valor de conversão da compra"].sum() if "Valor de conversão da compra" in dff.columns else 0.0
    compras = dff["Compras"].sum() if "Compras" in dff.columns else 0.0
    roas = (fatur / invest) if invest > 0 else 0.0
    cpa = (invest / compras) if compras > 0 else 0.0
    cpc = dff["CPC (custo por clique no link)"].mean() if "CPC (custo por clique no link)" in dff.columns and len(dff)>0 else 0.0
    ctr = (dff["CTR (taxa de cliques no link)"].mean() / 100.0) if "CTR (taxa de cliques no link)" in dff.columns and len(dff)>0 else 0.0
# 🔸 CVR (cliques → compra) com fallback para coluna "Cliques"
clicks_sum = dff["Cliques no link"].sum() if "Cliques no link" in dff.columns else 0.0
# fallback (algumas exports vêm só como "Cliques")
if clicks_sum == 0 and "Cliques" in dff.columns:
    clicks_sum = dff["Cliques"].sum()
cvr = (compras / clicks_sum) if clicks_sum > 0 else 0.0

    st.markdown("### 📌 KPIs — Performance Real")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
    kpi1.metric("Investimento", f"R$ {invest:,.0f}".replace(",", "."))
    kpi2.metric("Faturamento", f"R$ {fatur:,.0f}".replace(",", "."))
    kpi3.metric("ROAS", f"{roas:,.2f}".replace(",", "."))
    kpi4.metric("CPA", f"R$ {cpa:,.2f}".replace(",", "."))
    kpi5.metric("CTR", f"{ctr*100:,.2f}%".replace(",", "."))
    kpi6.metric("CPC", f"R$ {cpc:,.2f}".replace(",", "."))
    kpi7.metric("CVR (Cliques→Compra)", f"{cvr*100:,.2f}%".replace(",", "."))

    # Alertas
    alerts = []
    if roas < alert_roas_min:
        alerts.append(f"ROAS abaixo do mínimo ( {roas:.2f} < {alert_roas_min:.2f} )")
    if cpa > alert_cpa_max:
        alerts.append(f"CPA acima do máximo ( R$ {cpa:.2f} > R$ {alert_cpa_max:.2f} )")

    if alerts:
        st.error("🚨 " + " | ".join(alerts))
    else:
        st.success("✅ Dentro dos limites definidos de ROAS/CPA.")


    st.markdown("---")

    # =========================
    # Funil de conversão
    # =========================
    st.markdown("### 🧭 Funil de Conversão")
    def safe_sum(col): return dff[col].sum() if col in dff.columns else 0
    clicks   = safe_sum("Cliques no link")
    lpviews  = safe_sum("Visualizações da página de destino")
    addcart  = safe_sum("Adições ao carrinho")
    cko_init = safe_sum("Finalizações de compra iniciadas")
    pay_info = safe_sum("Inclusões de informações de pagamento")
    purchases= safe_sum("Compras")

    funil = pd.DataFrame({"etapa": ["Cliques","LP Views","Add to Cart","Checkout Iniciado","Info. Pagamento","Compras"],
                          "valor": [clicks, lpviews, addcart, cko_init, pay_info, purchases]})
    st.plotly_chart(px.funnel(funil, x="valor", y="etapa", title="Funil — volume por etapa"), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    def pct(a,b): return (a/b) if b else 0.0
    c1.metric("Cliques → LP", f"{pct(lpviews, clicks)*100:,.1f}%".replace(",", "."))
    c2.metric("LP → AddCart", f"{pct(addcart, lpviews)*100:,.1f}%".replace(",", "."))
    c3.metric("AddCart → Checkout", f"{pct(cko_init, addcart)*100:,.1f}%".replace(",", "."))
    c4.metric("Checkout → Pagamento", f"{pct(pay_info, cko_init)*100:,.1f}%".replace(",", "."))
    c5.metric("Pagamento → Compra", f"{pct(purchases, pay_info)*100:,.1f}%".replace(",", "."))

    st.markdown("---")

    # =========================
    # Eficiência de mídia (por campanha)
    # =========================
    st.markdown("### 📈 Eficiência de Mídia (por Campanha)")
    if "campanha" in dff.columns:
        # refiltra por gasto > 0 (garantia)
        if "Valor usado" in dff.columns:
            dff = dff[dff["Valor usado"] > 0].copy()

        grp = dff.groupby("campanha").agg({
            "Valor usado":"sum",
            "Valor de conversão da compra":"sum",
            "Compras":"sum",
            "Impressões":"sum",
            "Cliques no link":"sum",
            "CPM (custo por 1.000 impressões)":"mean",
            "CPC (custo por clique no link)":"mean",
            "CTR (taxa de cliques no link)":"mean",
        }).reset_index()

        grp["ROAS"] = grp["Valor de conversão da compra"] / grp["Valor usado"].replace(0, np.nan)
        grp["CPA"]  = grp["Valor usado"] / grp["Compras"].replace(0, np.nan)
        grp["CPC_calc"] = grp["Valor usado"] / grp["Cliques no link"].replace(0, np.nan)
        grp["CPM_calc"] = (grp["Valor usado"] / grp["Impressões"].replace(0, np.nan)) * 1000.0

        tabs = st.tabs(["CPA", "CPC", "CPM", "ROAS"])
        with tabs[0]:
            st.plotly_chart(px.bar(grp, x="campanha", y="CPA", title="CPA por campanha"), use_container_width=True)
        with tabs[1]:
            st.plotly_chart(px.bar(grp, x="campanha", y=grp["CPC (custo por clique no link)"].fillna(grp["CPC_calc"]), title="CPC por campanha"), use_container_width=True)
        with tabs[2]:
            st.plotly_chart(px.bar(grp, x="campanha", y=grp["CPM (custo por 1.000 impressões)"].fillna(grp["CPM_calc"]), title="CPM por campanha"), use_container_width=True)
        with tabs[3]:
            st.plotly_chart(px.bar(grp, x="campanha", y="ROAS", title="ROAS por campanha"), use_container_width=True)
    else:
        st.info("A coluna 'campanha' não foi encontrada para agrupar eficiência de mídia.")


    st.markdown("---")

    # =========================
    # Engajamento de vídeo
    # =========================
    st.markdown("### 🎥 Engajamento de Vídeo")
    vid_cols = ["Reproduções de 25% do vídeo","Reproduções de 50% do vídeo","Reproduções de 75% do vídeo","Reproduções de 95% do vídeo"]
    has_video = all([c in dff.columns for c in vid_cols])
    if has_video:
        totals = dff[vid_cols].sum()
        df_vid = pd.DataFrame({"etapa":["25%","50%","75%","95%"], "reproducoes":[totals[0], totals[1], totals[2], totals[3]]})
        st.plotly_chart(px.line(df_vid, x="etapa", y="reproducoes", markers=True, title="Retenção de Vídeo (volume total)"), use_container_width=True)
        if "Tempo médio de reprodução do vídeo" in dff.columns:
            tempo_med = dff["Tempo médio de reprodução do vídeo"].apply(_to_num).mean()  # <- aqui troquei para _to_num
            st.metric("⏱️ Tempo médio de reprodução", f"{tempo_med:,.1f} s".replace(",", "."))
    else:
        st.info("Para retenção, inclua no CSV as colunas de reprodução em 25/50/75/95%.")

    st.markdown("---")

        # =========================
    # Ranking de campanhas (robusto a colunas ausentes)
    # =========================
    st.markdown("### 🏆 Ranking de Campanhas")

    if "campanha" not in dff.columns:
        st.info("A coluna 'campanha' não foi encontrada para exibir o ranking.")
    else:
        # Quais métricas temos de fato?
        have = set(dff.columns)
        metrics_defs = {
            "Desativado/Ativado": ("first", "Status"),
            "Veiculação": ("first", "Veiculação"),
            "Valor usado": ("sum", "Investimento (R$)"),
            "Valor de conversão da compra": ("sum", "Faturamento (R$)"),
            "Compras": ("sum", "Compras"),
        }

        # Monta dict de agregações só com as colunas disponíveis
        agg_dict = {col: func for col, (func, _) in metrics_defs.items() if col in have}

        if not agg_dict:
            st.info("Nenhuma métrica disponível para ranquear campanhas (faltam colunas como Valor usado, Faturamento, Compras…).")
        else:
            rank = dff.groupby("campanha").agg(agg_dict).reset_index()

            # KPIs derivados se possíveis
            if "Valor usado" in rank.columns and "Valor de conversão da compra" in rank.columns:
                rank["ROAS"] = rank["Valor de conversão da compra"] / rank["Valor usado"].replace(0, np.nan)
            if "Valor usado" in rank.columns and "Compras" in rank.columns:
                rank["CPA"] = rank["Valor usado"] / rank["Compras"].replace(0, np.nan)

            # Opções de ordenação só para colunas que existem
            order_opts = []
            if "ROAS" in rank.columns: order_opts.append("ROAS desc")
            if "CPA"  in rank.columns: order_opts.append("CPA asc")
            if "Valor de conversão da compra" in rank.columns: order_opts.append("Faturamento desc")
            if "Valor usado" in rank.columns: order_opts.append("Investimento desc")
            if not order_opts:
                order_opts = ["Campanha A→Z"]

            order_by = st.selectbox("Ordenar ranking por:", order_opts)
            if order_by == "ROAS desc":
                rank = rank.sort_values("ROAS", ascending=False)
            elif order_by == "CPA asc":
                rank = rank.sort_values("CPA", ascending=True)
            elif order_by == "Faturamento desc":
                rank = rank.sort_values("Valor de conversão da compra", ascending=False)
            elif order_by == "Investimento desc":
                rank = rank.sort_values("Valor usado", ascending=False)
            else:
                rank = rank.sort_values("campanha", ascending=True)

            # Renomeia colunas amigáveis (só as que existem)
            rename_map = {col: label for col, (_, label) in metrics_defs.items() if col in rank.columns}
            rename_map.update({"campanha": "Campanha"})
            st.dataframe(rank.rename(columns=rename_map), use_container_width=True)


    st.markdown("---")

    # =========================
    # Visão Temporal (se houver coluna de Data)
    # =========================
    st.markdown("### 📅 Visão Temporal")
    date_col = next((c for c in ["Data","data","date","Dia","dia"] if c in dff.columns), None)
    if date_col:
        dff["_date"] = pd.to_datetime(dff[date_col], errors="coerce", dayfirst=True)
        t = dff.dropna(subset=["_date"]).groupby("_date").agg({
            "Valor usado":"sum",
            "Valor de conversão da compra":"sum",
            "Compras":"sum"
        }).reset_index().sort_values("_date")
        t["ROAS"] = t["Valor de conversão da compra"] / t["Valor usado"].replace(0, np.nan)

        tabs_t = st.tabs(["ROAS diário","Investimento diário","Compras diárias"])
        with tabs_t[0]: st.plotly_chart(px.line(t, x="_date", y="ROAS", title="ROAS diário"), use_container_width=True)
        with tabs_t[1]: st.plotly_chart(px.line(t, x="_date", y="Valor usado", title="Investimento diário"), use_container_width=True)
        with tabs_t[2]: st.plotly_chart(px.line(t, x="_date", y="Compras", title="Compras diárias"), use_container_width=True)
    else:
        st.info("Para visão temporal, inclua uma coluna de data no CSV (ex.: 'Data').")


else:
    st.info("Envie o CSV do Gerenciador para liberar os painéis de performance real.")

# =========================
# Metas — Acompanhamento Diário (semana)
# =========================
st.markdown("---")
st.subheader("✅ Acompanhamento Diário — Semana (Metas & Real)")
st.caption("Edite os campos de realizado. Baixe a planilha, atualize no Excel/Sheets e reenvie quando quiser.")

week_live = weekly_df.copy()
week_live["real_investimento"] = 0.0
week_live["real_sessoes"] = 0.0
week_live["real_checkouts"] = 0.0
week_live["real_compras"] = 0.0
week_live["real_faturamento"] = 0.0

edited_week = st.data_editor(
    week_live,
    column_config={
        "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY", step=1),
        "meta_faturamento": st.column_config.NumberColumn("Meta Faturamento (R$)"),
        "meta_compras": st.column_config.NumberColumn("Meta Compras (nº)"),
        "meta_sessoes": st.column_config.NumberColumn("Meta Sessões (nº)"),
        "meta_investimento": st.column_config.NumberColumn("Meta Investimento (R$)"),
        "real_investimento": st.column_config.NumberColumn("Real Investimento (R$)"),
        "real_sessoes": st.column_config.NumberColumn("Real Sessões (nº)"),
        "real_checkouts": st.column_config.NumberColumn("Real Checkouts (nº)"),
        "real_compras": st.column_config.NumberColumn("Real Compras (nº)"),
        "real_faturamento": st.column_config.NumberColumn("Real Faturamento (R$)"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
)

st.markdown("### 📌 KPIs Semanais (Meta)")
kpi_cols = st.columns(5)
kpi_cols[0].metric("Meta Faturamento", f"R$ {weekly_totals['faturamento']:,.0f}".replace(",", "."))
kpi_cols[1].metric("Meta Compras", f"{weekly_totals['compras']:,.0f}".replace(",", "."))
kpi_cols[2].metric("Meta Sessões", f"{weekly_totals['sessoes']:,.0f}".replace(",", "."))
kpi_cols[3].metric("Orçamento (ROAS alvo)", f"R$ {weekly_totals['investimento']:,.0f}".replace(",", "."))
kpi_cols[4].metric("ROI Estimado", f"{weekly_totals['roi_estimado']*100:,.0f}%".replace(",", "."))

realized = edited_week[["real_investimento","real_sessoes","real_checkouts","real_compras","real_faturamento"]].sum()
realized_cols = st.columns(5)
realized_cols[0].metric("Investimento (Real)", f"R$ {realized['real_investimento']:,.0f}".replace(",", "."))
realized_cols[1].metric("Sessões (Real)", f"{realized['real_sessoes']:,.0f}".replace(",", "."))
realized_cols[2].metric("Checkouts (Real)", f"{realized['real_checkouts']:,.0f}".replace(",", "."))
realized_cols[3].metric("Compras (Real)", f"{realized['real_compras']:,.0f}".replace(",", "."))
realized_cols[4].metric("Faturamento (Real)", f"R$ {realized['real_faturamento']:,.0f}".replace(",", "."))

st.markdown("---")
left, right = st.columns(2)
with left:
    df_plot = edited_week.copy()
    df_plot["data"] = pd.to_datetime(df_plot["data"])
    df_plot = df_plot.sort_values("data")
    df_plot["meta_fat_cum"] = df_plot["meta_faturamento"].cumsum()
    df_plot["real_fat_cum"] = df_plot["real_faturamento"].cumsum()
    fig = px.line(df_plot, x="data", y=["meta_fat_cum","real_fat_cum"], title="Faturamento Acumulado — Meta vs Real")
    st.plotly_chart(fig, use_container_width=True)

with right:
    df_inv = edited_week.copy()
    df_inv["data"] = pd.to_datetime(df_inv["data"])
    df_inv = df_inv.sort_values("data")
    df_inv["meta_inv_cum"] = df_inv["meta_investimento"].cumsum()
    df_inv["real_inv_cum"] = df_inv["real_investimento"].cumsum()
    fig2 = px.line(df_inv, x="data", y=["meta_inv_cum","real_inv_cum"], title="Investimento Acumulado — Meta vs Real")
    st.plotly_chart(fig2, use_container_width=True)

# Downloads
st.markdown("### ⬇️ Exportar Planos de Meta")
wcsv = weekly_df.copy()
wcsv["data"] = pd.to_datetime(wcsv["data"]).dt.strftime("%Y-%m-%d")
mcsv = monthly_df.copy()
mcsv["data"] = pd.to_datetime(mcsv["data"]).dt.strftime("%Y-%m-%d")
st.download_button("Baixar Plano Semanal (CSV)", data=wcsv.to_csv(index=False).encode("utf-8"), file_name="plano_semanal.csv", mime="text/csv")
st.download_button("Baixar Plano Mensal (CSV)", data=mcsv.to_csv(index=False).encode("utf-8"), file_name="plano_mensal.csv", mime="text/csv")

st.info("💾 Para persistência real (Google Sheets/Supabase/DB), adapte o código de leitura/gravação. O upload lê o CSV exportado do Gerenciador.")

# =========================
# 🧪 Teste de Criativos
# =========================
st.markdown("---")
st.header("🧪 Teste de Criativos — Análise e Campeões")

with st.expander("Como usar"):
    st.write(
        "- Exporte do Gerenciador o CSV da **campanha de teste de criativos** (nível anúncio).\n"
        "- Colunas úteis (qualquer variação é aceita): Campanha, Conjunto, Anúncio/ID, Valor gasto, Impressões, Cliques, Compras, Receita/Valor de conversão.\n"
        "- Opcional: Views da LP, thumbnails/links da peça para pré-visualização."
    )

uploaded_creatives = st.file_uploader("📥 Suba o CSV do teste de criativos", type=["csv"], key="upload_creatives")

if uploaded_creatives:
    import re, unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("(brl)", "").replace(" r$", "").strip()
        return s

    def _read_flex(file):
        for enc in ["utf-8", "latin-1", "utf-16", "cp1252"]:
            try:
                return pd.read_csv(file, sep=None, engine="python", encoding=enc)
            except Exception:
                continue
        return pd.read_csv(file)

    def _to_num(x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).lower().replace("r$", "").replace("brl", "")
        s = s.replace(".", "").replace(",", ".")
        s = re.sub(r"[^0-9\.\-eE]", "", s)
        try: return float(s)
        except: return 0.0

    rawc = _read_flex(uploaded_creatives)
    norm_map = {_norm(c): c for c in rawc.columns}

    # Aliases flexíveis
    A = {
        "campanha": ("campanha","campaign name","nome da campanha"),
        "conjunto": ("conjunto","ad set name","conjunto de anúncios","conjunto de anuncios"),
        "anuncio": ("anúncio","anuncio","ad name","nome do anúncio","nome do anuncio"),
        "ad_id": ("id do anúncio","id do anuncio","ad id","id"),
        "gasto": ("valor usado","valor gasto","spend","amount spent","valor usado brl","gasto"),
        "imp": ("impressões","impressoes","impressions"),
        "clicks": (
    "cliques no link","clicks","link clicks",
    "cliques (todos)","cliques","cliques no link (todos)"
),
        "lpv": ("visualizações da página de destino","visualizacoes da pagina de destino","landing page views","lp views"),
        "compras": ("compras","purchases"),
        "receita": ("valor de conversão da compra","valor de conversao da compra","purchase conversion value","revenue","faturamento"),
        "thumb": ("thumbnail","image url","creativo","criação","link da midia","link da mídia"),
        "data": ("data","date","dia"),
    }

    def pick(*keys, default=None):
        for k in keys:
            nk = _norm(k)
            if nk in norm_map: return norm_map[nk]
        return default

    cols = {k: pick(*v) for k, v in A.items()}
    dfc = rawc.rename(columns={v: k for k, v in cols.items() if v}).copy()

    # Converte numéricos
    for c in ["gasto","imp","clicks","lpv","compras","receita"]:
        if c in dfc.columns: dfc[c] = dfc[c].apply(_to_num)

    # Filtros básicos
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        sel_camp = st.selectbox("Campanha", ["(Todas)"] + sorted(dfc.get("campanha", pd.Series([""])).dropna().astype(str).unique().tolist())) if "campanha" in dfc.columns else "(Todas)"
    with colf2:
        sel_set = st.selectbox("Conjunto", ["(Todos)"] + sorted(dfc.get("conjunto", pd.Series([""])).dropna().astype(str).unique().tolist())) if "conjunto" in dfc.columns else "(Todos)"
    with colf3:
        min_spend = st.number_input("Gasto mínimo p/ considerar (R$)", value=10.0, step=5.0)

    filt = pd.Series(True, index=dfc.index)
    if "campanha" in dfc.columns and sel_camp != "(Todas)":
        filt &= (dfc["campanha"].astype(str) == sel_camp)
    if "conjunto" in dfc.columns and sel_set != "(Todos)":
        filt &= (dfc["conjunto"].astype(str) == sel_set)

    dfc = dfc.loc[filt].copy()

    if "gasto" in dfc.columns:
        dfc = dfc[dfc["gasto"] >= float(min_spend)].copy()

    if dfc.empty:
        st.warning("Nenhum criativo com os filtros atuais (ou acima do gasto mínimo).")
        st.stop()

    # Chave do criativo (anúncio ou id)
    key_col = "anuncio" if "anuncio" in dfc.columns else ("ad_id" if "ad_id" in dfc.columns else None)
    if key_col is None:
        st.error("Não achei coluna de identificação do criativo (ex.: 'Anúncio' ou 'Ad ID').")
        st.stop()

    # Agregação por criativo
    grp = dfc.groupby(key_col).agg({
        **({ "gasto":"sum" } if "gasto" in dfc.columns else {}),
        **({ "imp":"sum" } if "imp" in dfc.columns else {}),
        **({ "clicks":"sum" } if "clicks" in dfc.columns else {}),
        **({ "lpv":"sum" } if "lpv" in dfc.columns else {}),
        **({ "compras":"sum" } if "compras" in dfc.columns else {}),
        **({ "receita":"sum" } if "receita" in dfc.columns else {}),
    }).reset_index().rename(columns={key_col:"Criativo"})

    # Métricas derivadas
    eps = 1e-9
    if "imp" in grp.columns and "clicks" in grp.columns:
        grp["CTR"] = grp["clicks"] / (grp["imp"] + eps)
    if "clicks" in grp.columns and "compras" in grp.columns:
        grp["CVR"] = grp["compras"] / (grp["clicks"] + eps)
    if "gasto" in grp.columns and "clicks" in grp.columns:
        grp["CPC"] = grp["gasto"] / (grp["clicks"] + eps)
    if "gasto" in grp.columns and "imp" in grp.columns:
        grp["CPM"] = (grp["gasto"] / (grp["imp"] + eps)) * 1000.0
    if "gasto" in grp.columns and "compras" in grp.columns:
        grp["CPA"] = grp["gasto"] / (grp["compras"] + eps)
    if "receita" in grp.columns and "gasto" in grp.columns:
        grp["ROAS"] = grp["receita"] / (grp["gasto"] + eps)

    # Significância simples (proporção z) vs. média da amostra
    def prop_z_test(success_a, n_a, p_pool):
        # z = (p_a - p_pool) / sqrt(p_pool*(1-p_pool)/n_a)
        if n_a <= 0 or p_pool<=0 or p_pool>=1: return np.nan
        p_a = success_a / n_a
        se = np.sqrt(p_pool*(1-p_pool)/(n_a+eps))
        if se==0: return np.nan
        z = (p_a - p_pool)/se
        # p-value bicaudal aproximado
        from math import erf, sqrt
        p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
        return p

    # p-valor para CTR e CVR (se existirem)
    if "CTR" in grp.columns and "clicks" in grp.columns and "imp" in grp.columns:
        p_ctr_base = (dfc["clicks"].sum())/max(dfc["imp"].sum(), eps)
        grp["p_CTR"] = grp.apply(lambda r: prop_z_test(r["clicks"], r["imp"], p_ctr_base), axis=1)
    if "CVR" in grp.columns and "compras" in grp.columns and "clicks" in grp.columns:
        p_cvr_base = (dfc["compras"].sum())/max(dfc["clicks"].sum(), eps)
        grp["p_CVR"] = grp.apply(lambda r: prop_z_test(r["compras"], r["clicks"], p_cvr_base), axis=1)

    # Regras de campeão
    st.subheader("🏆 Campeões")
    colr1, colr2, colr3 = st.columns(3)
    with colr1:
        rank_metric = st.selectbox("Métrica para ranquear", [m for m in ["ROAS","CPA","CTR","CVR","Compras","Receita","Gasto"] if m in grp.columns] or ["ROAS"])
    with colr2:
        top_n = st.slider("Quantos destacar", 1, min(10, len(grp)), value=min(3, len(grp)))
    with colr3:
        p_thresh = st.selectbox("Significância (p-value)", ["(ignorar)","0.10","0.05","0.01"], index=2)

    # Ordenação
    asc = True if rank_metric in ["CPA"] else False
    if rank_metric.lower() == "compras" and "compras" in grp.columns:
        grp["_sort"] = grp["compras"]
        asc = False
    elif rank_metric.lower() == "receita" and "receita" in grp.columns:
        grp["_sort"] = grp["receita"]; asc = False
    elif rank_metric.lower() == "gasto" and "gasto" in grp.columns:
        grp["_sort"] = grp["gasto"]; asc = False
    else:
        grp["_sort"] = grp.get(rank_metric, pd.Series(np.zeros(len(grp))))

    board = grp.sort_values("_sort", ascending=asc).drop(columns=["_sort"])

    # Aplica critério de p-valor se selecionado
    if p_thresh != "(ignorar)":
        cut = float(p_thresh)
        # se ranqueando por CTR usa p_CTR, se CVR usa p_CVR; caso contrário, aceita se qualquer um for significativo
        if rank_metric == "CTR" and "p_CTR" in board.columns:
            board = board[(board["p_CTR"] <= cut) | (board["p_CTR"].isna())]
        elif rank_metric == "CVR" and "p_CVR" in board.columns:
            board = board[(board["p_CVR"] <= cut) | (board["p_CVR"].isna())]
        elif {"p_CTR","p_CVR"}.issubset(board.columns):
            board = board[(board["p_CTR"] <= cut) | (board["p_CVR"] <= cut) | (board["p_CTR"].isna() & board["p_CVR"].isna())]

    # Marcação de campeões
    board["Campeão"] = False
    board.loc[board.index[:top_n], "Campeão"] = True

    # Formatação amigável
    def _fmt_pct(x): return "" if pd.isna(x) else f"{x*100:,.2f}%".replace(",", ".")
    def _fmt_moeda(x): return "" if pd.isna(x) else f"R$ {x:,.2f}".replace(",", ".")
    def _fmt_p(x): return "" if pd.isna(x) else f"{x:.3f}"

    show = board.copy()
    rename = {
        "gasto":"Gasto","imp":"Impressões","clicks":"Cliques","lpv":"LP Views",
        "compras":"Compras","receita":"Receita","CTR":"CTR","CVR":"CVR",
        "CPC":"CPC","CPM":"CPM","CPA":"CPA","ROAS":"ROAS","p_CTR":"p(CTR)","p_CVR":"p(CVR)"
    }
    show = show.rename(columns=rename)

    # Apresentação
    st.markdown("#### Ranking de Criativos")
    st.dataframe(
        show.style
            .format({
                "CTR": _fmt_pct, "CVR": _fmt_pct,
                "CPC": _fmt_moeda, "CPM": _fmt_moeda, "CPA": _fmt_moeda, "ROAS": "{:,.2f}".format,
                "Gasto": _fmt_moeda, "Receita": _fmt_moeda,
                "p(CTR)": _fmt_p, "p(CVR)": _fmt_p
            })
            .apply(lambda s: ["background-color:#e6ffe6" if (show.loc[idx, "Campeão"] is True) else "" for idx in s.index], axis=0),
        use_container_width=True
    )

    # Galeria opcional dos campeões (se tiver thumbnail/link)
    thumbs_col = None
    for cand in ["thumb","image url","thumbnail","link da midia","link da mídia"]:
        if cand in [c.lower() for c in rawc.columns]:
            thumbs_col = norm_map.get(cand, None)
            break

    if thumbs_col and thumbs_col in rawc.columns:
        st.markdown("#### Galeria — Campeões")
        champs = board[board["Campeão"]].merge(dfc[[key_col, thumbs_col]].drop_duplicates(), left_on="Criativo", right_on=key_col if key_col!="anuncio" else "anuncio", how="left")
        cols = st.columns(min(4, len(champs)))
        i = 0
        for _, r in champs.iterrows():
            with cols[i % len(cols)]:
                st.caption(str(r["Criativo"]))
                st.image(r.get(thumbs_col), use_container_width=True)
            i += 1

    # Download do ranking
    st.download_button(
        "📤 Baixar ranking de criativos (CSV)",
        data=board.to_csv(index=False).encode("utf-8"),
        file_name="ranking_criativos.csv",
        mime="text/csv"
    )
else:
    st.info("Para analisar criativos, suba o CSV específico da campanha de teste.")
