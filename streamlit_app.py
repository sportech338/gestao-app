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
st.caption("Foque no essencial: metas da semana, KPIs reais e 2 visões (Funil e Campanhas).")

# =========================
# Sidebar — Parâmetros essenciais
# =========================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    aov = st.number_input("Ticket médio (R$)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    target_roas = st.number_input("ROAS alvo", value=2.0, min_value=0.1, step=0.1, format="%.2f")

    st.subheader("📆 Semana de referência")
    week_start = st.date_input(
        "Início (segunda)",
        value=(datetime.today() - timedelta(days=datetime.today().weekday())).date(),
    )
    include_weekends = st.checkbox("Incluir finais de semana", value=True)

    st.subheader("🎯 Meta da Semana")
    goal_type = st.radio("Definir por", ["Faturamento", "Compras"], index=0, horizontal=True)
    weekly_goal_value = st.number_input("Valor da meta (R$ se Faturamento; nº se Compras)", value=10000.0, min_value=0.0, step=100.0)

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
# Metas — Semana
# =========================
week_start_dt = datetime.combine(week_start, datetime.min.time())
week_end_dt = week_start_dt + timedelta(days=6)

n_days = len(daterange(week_start_dt.date(), week_end_dt.date(), include_weekends)) or 1
if goal_type == "Faturamento":
    goal_rev = float(weekly_goal_value)
    goal_pur = goal_rev / aov if aov>0 else 0.0
else:
    goal_pur = float(weekly_goal_value)
    goal_rev = goal_pur * aov

budget_goal = goal_rev / target_roas if target_roas>0 else 0.0

# =========================
# Bloco 1 — Metas (planejado)
# =========================
st.markdown("## 🎯 Metas da Semana (Planejado)")
c1,c2,c3 = st.columns(3)
c1.metric("Faturamento alvo", f"R$ {goal_rev:,.0f}".replace(",","."))
c2.metric("Compras alvo", f"{goal_pur:,.0f}".replace(",","."))
c3.metric("Orçamento p/ ROAS alvo", f"R$ {budget_goal:,.0f}".replace(",","."))

st.markdown("---")

# =========================
# Bloco 2 — Performance Real
# =========================
st.markdown("## 📥 Performance Real")
if not uploaded:
    st.info("Envie o CSV para ver os KPIs reais, funil e ranking de campanhas.")
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

    # KPIs essenciais
    invest = float(dff.get("gasto", pd.Series([0])).sum())
    fatur = float(dff.get("faturamento", pd.Series([0])).sum())
    compras = float(dff.get("compras", pd.Series([0])).sum())
    roas = (fatur/invest) if invest>0 else 0.0
    cpa = (invest/compras) if compras>0 else 0.0

    st.markdown("### 📌 KPIs Reais")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Investimento", f"R$ {invest:,.0f}".replace(",","."))
    k2.metric("Faturamento", f"R$ {fatur:,.0f}".replace(",","."))
    k3.metric("ROAS", f"{roas:,.2f}".replace(",","."), delta=f"alvo {target_roas:,.2f}")
    k4.metric("CPA", f"R$ {cpa:,.2f}".replace(",","."))
    k5.metric("Compras", f"{compras:,.0f}".replace(",","."))

    # Aderência à meta
    st.progress(min(1.0, fatur/max(1.0, goal_rev)), text=f"Progresso de faturamento: R$ {fatur:,.0f} / R$ {goal_rev:,.0f}".replace(",","."))

    st.markdown("---")

    # Funil simplificado
    st.markdown("### 🧭 Funil (volume)")
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
        # order by ROAS desc
        order_cols = [c for c in ["ROAS","faturamento","gasto"] if c in grp.columns]
        if order_cols:
            grp = grp.sort_values(order_cols, ascending=[False, False, True]).head(10)
        friendly = grp.rename(columns={"campanha":"Campanha","gasto":"Investimento (R$)","faturamento":"Faturamento (R$)"})
        st.dataframe(friendly, use_container_width=True)
    else:
        st.info("Coluna 'campanha' não encontrada para ranquear.")

    st.markdown("---")

    # ROAS diário (se houver data)
    st.markdown("### 📅 ROAS diário")
    date_col = next((c for c in ["data"] if c in dff.columns), None)
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
# Bloco 3 — Acompanhamento Diário (opcional e enxuto)
# =========================
st.markdown("---")
st.subheader("✅ Acompanhamento Diário — Semana (enxuto)")

# tabela mínima: data, meta_faturamento, meta_investimento, real_faturamento, real_investimento
week_days = daterange(week_start_dt.date(), week_end_dt.date(), include_weekends)
meta_dia = (goal_rev / max(1,len(week_days))) if week_days else 0.0
budget_dia = (budget_goal / max(1,len(week_days))) if week_days else 0.0

base = pd.DataFrame({
    "data": week_days,
    "meta_faturamento": meta_dia,
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
    st.plotly_chart(px.line(df_plot, x="data", y=["meta_fat_cum","real_fat_cum"], title="Faturamento Acumulado — Meta vs Real"), use_container_width=True)
with col_b:
    df_inv = edited.copy()
    df_inv["data"] = pd.to_datetime(df_inv["data"])
    df_inv = df_inv.sort_values("data")
    df_inv["meta_inv_cum"] = df_inv["meta_investimento"].cumsum()
    df_inv["real_inv_cum"] = df_inv["real_investimento"].cumsum()
    st.plotly_chart(px.line(df_inv, x="data", y=["meta_inv_cum","real_inv_cum"], title="Investimento Acumulado — Meta vs Real"), use_container_width=True)

# Download único (semanal)
out = edited.copy()
out["data"] = pd.to_datetime(out["data"]).dt.strftime("%Y-%m-%d")
st.download_button("⬇️ Baixar plano semanal (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="plano_semana_simples.csv", mime="text/csv")

st.info("Versão simples: CSV único do Gerenciador ➜ KPIs + Funil + Campanhas. Metas apenas semanais. Sem módulos avançados (vídeo, teste de criativos, etc.).")
