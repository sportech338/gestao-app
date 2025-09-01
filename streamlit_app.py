import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# =========================
# Config
# =========================
st.set_page_config(page_title="Metas & Performance — Simples", layout="wide")
st.title("📊 Metas & Performance — Simples")
st.caption("Defina a META MENSAL e o app reparte automaticamente a META SEMANAL da semana selecionada.")

# =========================
# Sidebar — Parâmetros essenciais
# =========================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    target_roas = st.number_input("ROAS alvo", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.subheader("📅 Referências de Tempo")
    # Semana baseia-se nesta data (segunda a domingo)
    week_start = st.date_input(
        "Início da semana (segunda)",
        value=(datetime.today() - timedelta(days=datetime.today().weekday())).date(),
    )
    include_weekends = st.checkbox("Metas consideram finais de semana", value=True, help="Se desmarcar, metas diárias ignoram sábados e domingos.")

    # Mês de referência para a META MENSAL
    month_ref = st.date_input("Qualquer dia do mês da meta", value=datetime.today().date())

    st.subheader("🎯 META MENSAL (base de tudo)")
    goal_type_m = st.radio("Definir por", ["Faturamento", "Compras"], index=0, horizontal=True)
    monthly_goal_value = st.number_input("Valor da meta mensal (R$ se Faturamento; nº se Compras)", value=40000.0, min_value=0.0, step=500.0)

    st.subheader("📥 CSV do Gerenciador")
    uploaded = st.file_uploader("Envie o CSV (separador vírgula)", type=["csv"]) 

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
        s = s.replace("\n"," ").replace("\r"," ")
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

    # mapa simples de aliases
    ALIASES = {
        "campanha": ("nome da campanha", "campanha", "campaign name", "nome da campanha (id)"),
        "status": ("desativado/ativado", "ativado/desativado", "estado", "status da campanha"),
        "veiculacao": ("veiculacao", "veiculacao da campanha", "posicionamento"),
        "gasto": ("valor usado", "valor gasto", "amount spent", "spend", "valor usado brl"),
        "faturamento": ("valor de conversao da compra", "valor de conversão da compra", "purchase conversion value", "receita"),
        "compras": ("compras", "purchases"),
        "impressoes": ("impressões", "impressoes", "impressions"),
        "cliques": ("cliques no link", "link clicks", "clicks"),
        "lp_views": ("visualizações da página de destino", "visualizacoes da pagina de destino", "landing page views"),
        "add_cart": ("adições ao carrinho", "adicoes ao carrinho", "add to cart"),
        "ck_init": ("finalizações de compra iniciadas", "finalizacoes de compra iniciadas", "checkout iniciado"),
        "pay_info": ("inclusões de informações de pagamento", "inclusoes de informacoes de pagamento"),
        "cpc": ("cpc (custo por clique no link)", "cpc"),
        "ctr": ("ctr (taxa de cliques no link)", "ctr"),
        "cpm": ("cpm (custo por 1.000 impressões)", "cpm"),
        "data": ("data", "date", "dia"),
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
        if isinstance(x,(int,float)): return float(x)
        s = str(x).lower().replace("r$"," ").replace("brl"," ")
        s = s.replace(".", "").replace(",", ".")
        s = "".join(ch for ch in s if ch.isdigit() or ch in ".-eE")
        try: return float(s)
        except: return 0.0

    for col in ["gasto","faturamento","compras","cpc","ctr","cpm","impressoes","cliques","lp_views","add_cart","ck_init","pay_info"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # percentuais
    if "ctr" in df.columns:
        df["ctr"] = df["ctr"].apply(lambda v: v/100.0 if v>1.5 else v)  # se vier em % converte para proporção

    return df

# =========================
# Metas — MÊS ➜ Semana
# =========================
# Mês
month_first = month_ref.replace(day=1)
next_month_first = (month_first.replace(year=month_first.year+1, month=1) if month_first.month==12 else month_first.replace(month=month_first.month+1))
month_last = next_month_first - timedelta(days=1)
month_days = daterange(month_first, month_last, include_weekends=True)  # metas mensais sempre olham o calendário completo; a granularidade dia/semana usa o checkbox

# Semana
week_start_dt = datetime.combine(week_start, datetime.min.time())
week_end_dt = week_start_dt + timedelta(days=6)
week_days_all = daterange(week_start_dt.date(), week_end_dt.date(), include_weekends=True)  # semana cheia

# Intersecção SEMANA x MÊS (para repartir a meta mensal só pela fração que cai no mês)
week_days_in_month = [d for d in week_days_all if (month_first <= d <= month_last)]

# Contagem de dias “considerados” para rateio diário
def count_considered_days(days, consider_weekends):
    if consider_weekends:
        return len(days)
    return sum(1 for d in days if datetime.strptime(str(d), "%Y-%m-%d").weekday() < 5)

month_days_considered = count_considered_days(month_days, include_weekends)
week_days_considered = count_considered_days(week_days_in_month, include_weekends)

# Evitar divisão por zero
month_days_considered = max(1, month_days_considered)
week_days_considered = max(1, week_days_considered)

# Meta mensal em faturamento/compras e orçamento
if goal_type_m == "Faturamento":
    goal_rev_month = float(monthly_goal_value)
    goal_pur_month = goal_rev_month / aov if aov>0 else 0.0
else:
    goal_pur_month = float(monthly_goal_value)
    goal_rev_month = goal_pur_month * aov

budget_goal_month = goal_rev_month / target_roas if target_roas>0 else 0.0

# Proporção da semana dentro do mês (ajustada por dias considerados)
week_share = week_days_considered / month_days_considered

# Metas semanais DERIVADAS da meta mensal
goal_rev_week = goal_rev_month * week_share
goal_pur_week = goal_pur_month * week_share
budget_goal_week = budget_goal_month * week_share

# =========================
# Bloco 1 — Metas (planejado)
# =========================
st.markdown("## 🎯 Metas (Planejado)")
c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Meta MENSAL — Faturamento", f"R$ {goal_rev_month:,.0f}".replace(",","."))
with c2:
    st.metric("Meta MENSAL — Compras", f"{goal_pur_month:,.0f}".replace(",","."))
with c3:
    st.metric("Orçamento MENSAL p/ ROAS", f"R$ {budget_goal_month:,.0f}".replace(",","."))

s1,s2,s3 = st.columns(3)
with s1:
    st.metric("Meta SEMANAL (derivada) — Faturamento", f"R$ {goal_rev_week:,.0f}".replace(",","."), help="Proporcional aos dias da semana que caem dentro do mês de referência")
with s2:
    st.metric("Meta SEMANAL (derivada) — Compras", f"{goal_pur_week:,.0f}".replace(",","."))
with s3:
    st.metric("Orçamento SEMANAL (derivada)", f"R$ {budget_goal_week:,.0f}".replace(",","."))

st.markdown("---")

# =========================
# Bloco 2 — Performance Real
# =========================
st.markdown("## 📥 Performance Real")
if not uploaded:
    st.info("Envie o CSV para ver os KPIs reais, funil e ranking de campanhas — com progresso SEMANAL e MENSAL.")
else:
    df = read_csv_flex(uploaded)

    # filtros simples
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

    # KPIs gerais (arquivo inteiro)
    invest_total = float(dff.get("gasto", pd.Series([0])).sum())
    fatur_total = float(dff.get("faturamento", pd.Series([0])).sum())
    compras_total = float(dff.get("compras", pd.Series([0])).sum())

    # Se houver datas: separar MÊS e SEMANA para progresso
    date_col = next((c for c in ["data"] if c in dff.columns), None)
    if date_col:
        dd = dff.copy()
        dd["_date"] = pd.to_datetime(dd[date_col], errors="coerce", dayfirst=True)
        # filtros por janela
        week_mask = (dd["_date"] >= pd.to_datetime(week_start_dt)) & (dd["_date"] <= pd.to_datetime(week_end_dt))
        month_mask = (dd["_date"] >= pd.to_datetime(month_first)) & (dd["_date"] <= pd.to_datetime(month_last))
        w = dd.loc[week_mask]
        m = dd.loc[month_mask]
        invest_w = float(w.get("gasto", pd.Series([0])).sum())
        fatur_w = float(w.get("faturamento", pd.Series([0])).sum())
        compras_w = float(w.get("compras", pd.Series([0])).sum())
        invest_m = float(m.get("gasto", pd.Series([0])).sum())
        fatur_m = float(m.get("faturamento", pd.Series([0])).sum())
        compras_m = float(m.get("compras", pd.Series([0])).sum())
    else:
        # Sem datas: usa totais do CSV como "atual"
        invest_w = invest_m = invest_total
        fatur_w = fatur_m = fatur_total
        compras_w = compras_m = compras_total

    # KPIs essenciais (SEMANA atual)
    roas_w = (fatur_w/invest_w) if invest_w>0 else 0.0
    cpa_w = (invest_w/compras_w) if compras_w>0 else 0.0

    st.markdown("### 📌 KPIs Semanais (Reais)")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Investimento — Semana", f"R$ {invest_w:,.0f}".replace(",","."))
    k2.metric("Faturamento — Semana", f"R$ {fatur_w:,.0f}".replace(",","."))
    k3.metric("ROAS — Semana", f"{roas_w:,.2f}".replace(",","."), delta=f"alvo {target_roas:,.2f}")
    k4.metric("CPA — Semana", f"R$ {cpa_w:,.2f}".replace(",","."))
    k5.metric("Compras — Semana", f"{compras_w:,.0f}".replace(",","."))

    # Progresso SEMANA vs meta derivada
    st.progress(min(1.0, fatur_w/max(1.0, goal_rev_week)), text=f"Semana: R$ {fatur_w:,.0f} / R$ {goal_rev_week:,.0f}".replace(",","."))

    # KPIs MENSAL (Reais)
    roas_m = (fatur_m/invest_m) if invest_m>0 else 0.0
    cpa_m = (invest_m/compras_m) if compras_m>0 else 0.0

    st.markdown("### 📌 KPIs Mensais (Reais)")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Investimento — Mês", f"R$ {invest_m:,.0f}".replace(",","."))
    m2.metric("Faturamento — Mês", f"R$ {fatur_m:,.0f}".replace(",","."))
    m3.metric("ROAS — Mês", f"{roas_m:,.2f}".replace(",","."), delta=f"alvo {target_roas:,.2f}")
    m4.metric("CPA — Mês", f"R$ {cpa_m:,.2f}".replace(",","."))
    m5.metric("Compras — Mês", f"{compras_m:,.0f}".replace(",","."))

    # Progresso MÊS vs meta mensal
    st.progress(min(1.0, fatur_m/max(1.0, goal_rev_month)), text=f"Mês: R$ {fatur_m:,.0f} / R$ {goal_rev_month:,.0f}".replace(",","."))

    st.markdown("---")

    # Funil simplificado
    st.markdown("### 🧭 Funil (volume total do filtro)")
    def safe_sum(col):
        return dff[col].sum() if col in dff.columns else 0
    funil = pd.DataFrame({
        "etapa": ["Cliques","LP Views","Add to Cart","Checkout","Pagamento","Compras"],
        "valor": [safe_sum("cliques"), safe_sum("lp_views"), safe_sum("add_cart"), safe_sum("ck_init"), safe_sum("pay_info"), safe_sum("compras")]
    })
    funil = funil[funil["valor"]>0]
    if funil.empty:
        st.info("Para ver o funil, inclua no CSV colunas de cliques, LP views, add to cart, checkout, pagamento e compras.")
    else:
        st.plotly_chart(px.funnel(funil, x="valor", y="etapa", title="Funil de Conversão"), use_container_width=True)

    st.markdown("---")

    # Ranking por campanha
    st.markdown("### 🏆 Campanhas (Top 10 por ROAS)")
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
        friendly = grp.rename(columns={"campanha":"Campanha","gasto":"Investimento (R$)","faturamento":"Faturamento (R$)"})
        st.dataframe(friendly, use_container_width=True)
    else:
        st.info("Coluna 'campanha' não encontrada para ranquear.")

    st.markdown("---")

    # ROAS diário (se houver data)
    st.markdown("### 📅 ROAS diário (arquivo filtrado)")
    if date_col:
        dd = dff.copy()
        dd["_date"] = pd.to_datetime(dd[date_col], errors="coerce", dayfirst=True)
        t = dd.dropna(subset=["_date"]).groupby("_date").agg({"gasto":"sum","faturamento":"sum"}).reset_index().sort_values("_date")
        if not t.empty:
            t["ROAS"] = t["faturamento"] / t["gasto"].replace(0, np.nan)
            st.plotly_chart(px.line(t, x="_date", y="ROAS", title="ROAS diário"), use_container_width=True)
        else:
            st.info("Sem datas válidas para série temporal.")
    else:
        st.info("Inclua uma coluna de data no CSV para ver a série temporal.")

# =========================
# Bloco 3 — Acompanhamento Diário (enxuto, derivado da meta mensal)
# =========================
st.markdown("---")
st.subheader("✅ Acompanhamento Diário — Semana (meta derivada do mês)")

# Tabela mínima: data, metas derivadas (dia) e realizados
week_days_considered_list = []
for d in week_days_all:
    # mantém apenas dias que caem dentro do mês selecionado
    if (month_first <= pd.to_datetime(d) <= month_last):
        # aplica regra de finais de semana
        if include_weekends or datetime.strptime(str(d), "%Y-%m-%d").weekday() < 5:
            week_days_considered_list.append(d)

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

# Download semanal
out = edited.copy()
out["data"] = pd.to_datetime(out["data"]).dt.strftime("%Y-%m-%d")
st.download_button("⬇️ Baixar plano semanal derivado (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="plano_semana_derivado.csv", mime="text/csv")

st.info("Esta versão deriva toda a semana a partir da META MENSAL, proporcional aos dias da semana que caem no mês selecionado e respeitando a opção de incluir/excluir finais de semana.")
