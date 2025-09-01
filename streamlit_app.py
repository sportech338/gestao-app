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

    st.markdown("### 📌 KPIs — Performance Real")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Investimento", f"R$ {invest:,.0f}".replace(",", "."))
    kpi2.metric("Faturamento", f"R$ {fatur:,.0f}".replace(",", "."))
    kpi3.metric("ROAS", f"{roas:,.2f}".replace(",", "."))
    kpi4.metric("CPA", f"R$ {cpa:,.2f}".replace(",", "."))
    kpi5.metric("CTR", f"{ctr*100:,.2f}%".replace(",", "."))
    kpi6.metric("CPC", f"R$ {cpc:,.2f}".replace(",", "."))

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
        with tabs[0]: st.plotly_chart(px.bar(grp, x="campanha", y="CPA", title="CPA por campanha"), use_container_width=True)
        with tabs[1]: st.plotly_chart(px.bar(grp, x="campanha", y=grp["CPC (custo por clique no link)"].fillna(grp["CPC_calc"]), title="CPC por campanha"), use_container_width=True)
        with tabs[2]: st.plotly_chart(px.bar(grp, x="campanha", y=grp["CPM (custo por 1.000 impressões)"].fillna(grp["CPM_calc"]), title="CPM por campanha"), use_container_width=True)
        with tabs[3]: st.plotly_chart(px.bar(grp, x="campanha", y="ROAS", title="ROAS por campanha"), use_container_width=True)
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
    # Ranking de campanhas
    # =========================
    st.markdown("### 🏆 Ranking de Campanhas")
    if "campanha" in dff.columns:
        rank = dff.groupby("campanha").agg({
            "Desativado/Ativado":"first",
            "Veiculação":"first",
            "Valor usado":"sum",
            "Valor de conversão da compra":"sum",
            "Compras":"sum",
        }).reset_index()
        rank["ROAS"] = rank["Valor de conversão da compra"] / rank["Valor usado"].replace(0, np.nan)
        rank["CPA"]  = rank["Valor usado"] / rank["Compras"].replace(0, np.nan)

        order_by = st.selectbox("Ordenar ranking por:", ["ROAS desc","CPA asc","Faturamento desc","Investimento desc"])
        if order_by == "ROAS desc":              rank = rank.sort_values("ROAS", ascending=False)
        elif order_by == "CPA asc":              rank = rank.sort_values("CPA",  ascending=True)
        elif order_by == "Faturamento desc":     rank = rank.sort_values("Valor de conversão da compra", ascending=False)
        else:                                    rank = rank.sort_values("Valor usado", ascending=False)

        st.dataframe(
            rank.rename(columns={
                "campanha":"Campanha",
                "Desativado/Ativado":"Status",
                "Veiculação":"Veiculação",
                "Valor usado":"Investimento (R$)",
                "Valor de conversão da compra":"Faturamento (R$)"
            }),
            use_container_width=True
        )
    else:
        st.info("A coluna 'campanha' não foi encontrada para exibir o ranking.")

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
