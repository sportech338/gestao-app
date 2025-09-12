import streamlit as st, numpy as np, pandas as pd
from datetime import date, timedelta  # para comparativos/periodização
from typing import Optional

from config.constants import PRODUTOS
from utils.formatting import fmt_money_br, fmt_ratio_br, fmt_int_br
from utils.metrics import funnel_fig, enforce_monotonic
from utils.helpers import filter_by_product

# fetch_insights_daily é opcional (pode estar em outro módulo do seu app)
try:
    from fetchers.meta import fetch_insights_daily  # ajuste o caminho se necessário
except Exception:  # noqa: BLE001
    fetch_insights_daily = None  # type: ignore

# ---- helpers locais (sem dependências externas) ----
def _safe_div(n, d):
    n = float(n or 0); d = float(d or 0)
    return (n / d) if d > 0 else np.nan

def _fmt_pct_br(x):
    return (f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
            if pd.notnull(x) else "")

def _fmt_int_signed_br(x):
    try:
        v = int(round(float(x)))
        s = f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v > 0 else (f"-{s}" if v < 0 else "0")
    except Exception:  # noqa: BLE001
        return ""


def render_daily_tab(df_daily, act_id, token, api_version, level, since, until):
    # ---- garantias mínimas de colunas ----
    needed = ["date","currency","spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
    df_daily = df_daily.copy()
    for c in needed:
        if c not in df_daily.columns:
            df_daily[c] = 0.0 if c not in ("date","currency") else ("BRL" if c=="currency" else pd.NaT)

    # moeda detectada
    currency_detected = (df_daily["currency"].dropna().iloc[0]
                         if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty
                         else "BRL")
    colA, colB = st.columns([1,2])
    with colA:
        use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True)
    with colB:
        if use_brl_display and currency_detected != "BRL":
            st.caption("⚠️ Símbolo **R$** só para formatação visual. Valores permanecem na moeda da conta.")
        st.caption(f"Moeda da conta: **{currency_detected}**")

    # filtro por produto
    produto_sel = st.selectbox("Filtrar por produto (opcional)", ["(Todos)"] + PRODUTOS, key="daily_produto")
    df_view = filter_by_product(df_daily, produto_sel)
    if df_view.empty:
        st.info("Sem dados para o produto selecionado nesse período.")
        return
    df_view = df_view.copy()

    # KPIs totais
    tot_spend = float(df_view["spend"].sum())
    tot_purch = float(df_view["purchases"].sum())
    tot_rev = float(df_view["revenue"].sum())
    roas_g = (tot_rev / tot_spend) if tot_spend > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="small-muted">Valor usado</div><div class="big-number">{fmt_money_br(tot_spend)}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="small-muted">Vendas</div><div class="big-number">{fmt_int_br(tot_purch)}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="small-muted">Valor de conversão</div><div class="big-number">{fmt_money_br(tot_rev)}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="small-muted">ROAS</div><div class="big-number">{fmt_ratio_br(roas_g)}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Série diária
    st.subheader("Série diária — Investimento e Conversão")
    df_view["date"] = pd.to_datetime(df_view["date"], errors="coerce")
    daily = (df_view.dropna(subset=["date"])
                   .groupby("date", as_index=False)[["spend", "revenue"]].sum()
                   .rename(columns={"spend":"Gasto", "revenue":"Faturamento"})
                   .sort_values("date"))
    if daily.empty:
        st.info("Sem datas válidas para plotar a série diária.")
    else:
        st.line_chart(daily.set_index("date")[["Faturamento", "Gasto"]])
        st.caption("Linhas diárias de Receita e Gasto.")

    # FUNIL (Período)
    st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")
    f_clicks = float(df_view["link_clicks"].sum())
    f_lpv    = float(df_view["lpv"].sum())
    f_ic     = float(df_view["init_checkout"].sum())
    f_api    = float(df_view["add_payment"].sum())
    f_pur    = float(df_view["purchases"].sum())

    labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
    values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)), int(round(f_api)), int(round(f_pur))]
    force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
    values_plot = enforce_monotonic(values_total) if force_shape else values_total
    st.plotly_chart(funnel_fig(labels_total, values_plot, title="Funil do período"), use_container_width=True)

    # ===== Comparar outras taxas (agora com defs) =====
    core_rows = [
        ("LPV / Cliques",     _safe_div(values_total[1], values_total[0])),
        ("Checkout / LPV",    _safe_div(values_total[2], values_total[1])),
        ("Compra / Checkout", _safe_div(values_total[4], values_total[2])),
    ]
    extras_def = {
        "Add Pagto / Checkout": _safe_div(values_total[3], values_total[2]),
        "Compra / Add Pagto":   _safe_div(values_total[4], values_total[3]),
        "Compra / LPV":         _safe_div(values_total[4], values_total[1]),
        "Compra / Cliques":     _safe_div(values_total[4], values_total[0]),
        "Checkout / Cliques":   _safe_div(values_total[2], values_total[0]),
        "Add Pagto / LPV":      _safe_div(values_total[3], values_total[1]),
    }
    with st.expander("Comparar outras taxas (opcional)"):
        extras_selected = st.multiselect(
            "Escolha métricas adicionais para visualizar:",
            options=list(extras_def.keys()),
            default=[],
        )
    rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
    sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
    sr["Valor"] = sr["Valor"].map(_fmt_pct_br)
    base_h, row_h = 160, 36
    height = base_h + row_h * len(extras_selected)
    st.dataframe(sr, use_container_width=True, height=height)

    # ========= TAXAS POR DIA (e bandas) =========
    st.markdown("### Taxas por dia — evolução e leitura guiada")
    from datetime import timedelta  # garante no escopo

    with st.expander("Ajustes de exibição", expanded=True):
        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            min_clicks_day = st.slider("Ignorar dias com menos de X cliques", 0, 500, 30, 10)
            mark_weekends = st.checkbox("Marcar fins de semana no fundo", value=True)
            show_band = st.checkbox("Mostrar banda saudável (faixa alvo)", value=True)
        with col_cfg2:
            st.caption("Faixas saudáveis (%)")
            lpv_cli_low, lpv_cli_high = st.slider("LPV / Cliques", 0, 100, (70, 85), 1, key="tx_lpv_cli_band")
            co_lpv_low,  co_lpv_high  = st.slider("Checkout / LPV", 0, 100, (10, 20), 1, key="tx_co_lpv_band")
            buy_co_low,  buy_co_high  = st.slider("Compra / Checkout", 0, 100, (25, 40), 1, key="tx_buy_co_band")

    # agrega por dia
    df_view["date"] = pd.to_datetime(df_view["date"], errors="coerce")
    daily_conv = (
        df_view.dropna(subset=["date"])
               .groupby("date", as_index=False)[["link_clicks","lpv","init_checkout","add_payment","purchases"]]
               .sum()
               .rename(columns={"link_clicks":"clicks","init_checkout":"checkout","add_payment":"addpay"})
    )

    daily_conv = daily_conv[daily_conv["clicks"] >= min_clicks_day].copy()
    if daily_conv.empty:
        st.info("Sem dias suficientes após o filtro de cliques mínimos.")
    else:
        daily_conv["LPV/Cliques"]     = daily_conv.apply(lambda r: _safe_div(r["lpv"],       r["clicks"]),   axis=1)
        daily_conv["Checkout/LPV"]    = daily_conv.apply(lambda r: _safe_div(r["checkout"],  r["lpv"]),      axis=1)
        daily_conv["Compra/Checkout"] = daily_conv.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)

        # período anterior com mesma duração (se tivermos fetch_insights_daily)
        daily_prev = pd.DataFrame()
        if fetch_insights_daily is not None:
            period_len = (until - since).days + 1
            prev_since = since - timedelta(days=period_len)
            prev_until = since - timedelta(days=1)
            try:
                df_prev = fetch_insights_daily(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(prev_since),
                    until_str=str(prev_until),
                    level=level,
                    product_name=st.session_state.get("daily_produto")
                )
                if df_prev is not None and not df_prev.empty:
                    daily_prev = (
                        df_prev.groupby("date", as_index=False)[
                            ["link_clicks","lpv","init_checkout","purchases"]
                        ].sum()
                        .rename(columns={"link_clicks":"clicks","init_checkout":"checkout"})
                    )
                    daily_prev["LPV/Cliques"]     = daily_prev.apply(lambda r: _safe_div(r["lpv"],       r["clicks"]),   axis=1)
                    daily_prev["Checkout/LPV"]    = daily_prev.apply(lambda r: _safe_div(r["checkout"],  r["lpv"]),      axis=1)
                    daily_prev["Compra/Checkout"] = daily_prev.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)
            except Exception:
                daily_prev = pd.DataFrame()

        def _line_pct_banded(df, col, lo_pct, hi_pct, title):
            import plotly.graph_objects as go
            x = df["date"]; y = (df[col] * 100).round(2)

            def _status(v):
                if not pd.notnull(v): return "sem"
                v_pct = float(v) * 100.0
                return "abaixo" if v_pct < lo_pct else ("acima" if v_pct > hi_pct else "dentro")

            status = df[col].map(_status).tolist()
            colors = [{"abaixo":"#dc2626","dentro":"#16a34a","acima":"#0ea5e9","sem":"#9ca3af"}[s] for s in status]

            fig = go.Figure()
            if show_band and not x.empty:
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=x.min(), x1=x.max(), y0=lo_pct, y1=hi_pct,
                              fillcolor="rgba(34,197,94,0.08)", line=dict(width=0), layer="below")
            if mark_weekends:
                for d in x:
                    if pd.to_datetime(d).weekday() >= 5:
                        fig.add_shape(type="rect", xref="x", yref="paper",
                                      x0=d, x1=pd.to_datetime(d)+pd.Timedelta(days=1),
                                      y0=0, y1=1, line=dict(width=0), fillcolor="rgba(2,132,199,0.06)")

            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Diário",
                                     marker=dict(size=7, color=colors), line=dict(width=1.5)))
            if not daily_prev.empty and col in daily_prev.columns:
                x_aligned = df["date"].values[:len(daily_prev)]
                y_prev = (daily_prev[col] * 100).round(2)
                fig.add_trace(go.Scatter(x=x_aligned, y=y_prev, mode="lines",
                                         name="Período anterior (sobreposto)",
                                         line=dict(width=2.2, dash="dot")))
            fig.update_layout(title=title, yaxis_title="%", xaxis_title="Data",
                              height=340, template="plotly_white",
                              margin=dict(l=10, r=10, t=48, b=10), separators=“,.”,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                              hovermode="x unified")
            fig.update_yaxes(range=[max(0, min(y.min(), lo_pct) - 5), min(100, max(y.max(), hi_pct) + 5)],
                             ticksuffix="%")
            return fig

        st.markdown("**Resumo das taxas (período filtrado)**")
        def _resume_box(df_rates, col, lo_pct, hi_pct, label):
            vals = df_rates[col].dropna()
            if vals.empty:
                a,b,c = st.columns(3); a.metric(label,"—"); b.metric("% dias dentro","—"); c.metric("Tendência (período)","—"); return
            mean_pct = float(vals.mean() * 100.0)
            inside = float(((vals * 100.0 >= lo_pct) & (vals * 100.0 <= hi_pct)).mean() * 100.0)
            # tendência vs período anterior (se disponível)
            delta_pp: Optional[float] = None
            if not daily_prev.empty and col in daily_prev.columns:
                prev_mean = float(daily_prev[col].mean() * 100.0)
                delta_pp = mean_pct - prev_mean
            a,b,c = st.columns(3)
            a.metric(label, f"{mean_pct:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
            b.metric("% dias dentro", f"{inside:,.0f}%".replace(",", "X").replace(".", ",").replace("X", "."))
            c.metric("Tendência (período)", "—" if delta_pp is None
                     else (("+" if delta_pp>=0 else "") + f"{delta_pp:,.2f} pp").replace(",", "X").replace(".", ",").replace("X", "."))

        _resume_box(daily_conv, "LPV/Cliques",     lpv_cli_low, lpv_cli_high, "LPV/Cliques (média)")
        _resume_box(daily_conv, "Checkout/LPV",    co_lpv_low,  co_lpv_high,  "Checkout/LPV (média)")
        _resume_box(daily_conv, "Compra/Checkout", buy_co_low,  buy_co_high,  "Compra/Checkout (média)")

        left, mid, right = st.columns(3)
        with left:
            st.plotly_chart(_line_pct_banded(daily_conv, "LPV/Cliques", lpv_cli_low, lpv_cli_high, "LPV/Cliques"), use_container_width=True)
        with mid:
            st.plotly_chart(_line_pct_banded(daily_conv, "Checkout/LPV", co_lpv_low, co_lpv_high, "Checkout/LPV"), use_container_width=True)
        with right:
            st.plotly_chart(_line_pct_banded(daily_conv, "Compra/Checkout", buy_co_low, buy_co_high, "Compra/Checkout"), use_container_width=True)
        st.caption("Pontos verdes = dentro da banda; vermelhos = abaixo; azuis = acima. Área verde = alvo. Linha tracejada = período anterior (se disponível).")

    # === RECOMENDAÇÃO DE VERBA ===
    st.subheader("🔔 Para onde vai a verba? (recomendação automática)")
    min_purchases_to_scale = st.number_input("Compras mínimas para sugerir Escala (volume)", min_value=0, value=50, step=1)

    r1 = _safe_div(values_total[1], values_total[0])   # LPV/Cliques
    r2 = _safe_div(values_total[2], values_total[1])   # Checkout/LPV
    r3 = _safe_div(values_total[4], values_total[2])   # Compra/Checkout

    drop1 = max(0, values_total[0] - values_total[1])
    drop2 = max(0, values_total[1] - values_total[2])
    drop3 = max(0, values_total[2] - values_total[4])

    def _band_status(val, lo, hi):
        if not pd.notnull(val): return "sem_dado"
        v = val * 100
        return "abaixo" if v < lo else ("acima" if v > hi else "dentro")

    def _chip(label, val, lo, hi):
        s = _band_status(val, lo, hi)
        if s == "abaixo": return f"❌ **{label}** — {_fmt_pct_br(val)} (alvo {lo}–{hi}%)"
        if s == "dentro": return f"✅ **{label}** — {_fmt_pct_br(val)} (dentro de {lo}–{hi}%)"
        if s == "acima":  return f"🟢 **{label}** — {_fmt_pct_br(val)} (acima de {hi}%)"
        return f"⛔ **{label}** — sem dados suficientes"

    stages = {
        "Teste de criativo": {"rate": r1, "lo": lpv_cli_low, "hi": lpv_cli_high, "drop": drop1,
                              "explain": "Perda entre Cliques → LPV (qualidade do clique, criativo, velocidade e UX da landing).",
                              "todo": ["Testar variações de criativo (ângulo, thumb, 3s iniciais, CTA).",
                                       "Melhorar tempo de carregamento e primeira dobra da LP.",
                                       "Revisar promessa/título para alinhar com o anúncio."]},
        "Teste de interesse": {"rate": r2, "lo": co_lpv_low, "hi": co_lpv_high, "drop": drop2,
                               "explain": "Perda entre LPV → Checkout (público/segmentação e proposta de valor).",
                               "todo": ["Refinar públicos/lookalikes e excluir desinteressados.",
                                        "Evidenciar prova social e benefícios acima do CTA.",
                                        "Harmonizar oferta (preço/parcelas/bundle) com o público certo."]},
        "Remarketing": {"rate": r3, "lo": buy_co_low, "hi": buy_co_high, "drop": drop3,
                        "explain": "Perda entre Checkout → Compra (confiança, meios de pagamento, follow-up).",
                        "todo": ["RMK dinâmico com objeções, frete e garantia claros.",
                                 "Oferecer alternativas de pagamento (pix/boleto/parcelas).",
                                 "Recuperar carrinhos (e-mail/SMS/Whats) em até 24h."]},
    }

    abaixos = {k: v for k, v in stages.items() if _band_status(v["rate"], v["lo"], v["hi"]) == "abaixo"}
    if abaixos:
        foco, foco_dat = max(abaixos.items(), key=lambda kv: kv[1]["drop"])
    else:
        total_purch = values_total[4]
        todas_ok = all(_band_status(v["rate"], v["lo"], v["hi"]) in ["dentro", "acima"] for v in stages.values())
        if todas_ok and total_purch >= min_purchases_to_scale:
            foco, foco_dat = "Escala", {"rate": None, "lo": None, "hi": None, "drop": 0,
                                        "explain": "Taxas saudáveis e volume suficiente. Hora de aumentar alcance nas melhores campanhas."}
        else:
            foco, foco_dat = max(stages.items(), key=lambda kv: kv[1]["drop"])

    total_drop = max(1, drop1 + drop2 + drop3)
    share = foco_dat["drop"] / total_drop
    if share > 0.60: intensidade, faixa_verba = "Alta", "↑ realocar **20–30%** do budget"
    elif share >= 0.30: intensidade, faixa_verba = "Média", "↑ realocar **10–15%** do budget"
    else: intensidade, faixa_verba = "Baixa", "↑ realocar **5–10%** do budget"

    st.markdown("---")
    colL, colR = st.columns([1, 2])
    with colL:
        st.markdown("**Taxas do período**")
        st.markdown(_chip("LPV/Cliques", r1, lpv_cli_low, lpv_cli_high))
        st.markdown(_chip("Checkout/LPV", r2, co_lpv_low,  co_lpv_high))
        st.markdown(_chip("Compra/Checkout", r3, buy_co_low,  buy_co_high))
    with colR:
        if foco == "Escala":
            st.success(f"**✅ Recomendação: Escala**\n\n- Motivo: {foco_dat['explain']}\n- Compras no período: **{fmt_int_br(values_total[4])}** "
                       f"(mín. para escalar: **{fmt_int_br(min_purchases_to_scale)}**)\n- Ação: aumentar orçamento nas campanhas com melhor ROAS; manter horários e públicos vencedores.")
        else:
            st.warning(f"**⚠️ Recomendação: {foco}**\n\n- Motivo: {foco_dat['explain']}\n- Queda concentrada nessa etapa: **{_fmt_int_signed_br(foco_dat['drop'])}** pessoas "
                       f"(intensidade **{intensidade}** → {faixa_verba}).\n\n**O que fazer agora**")
            for tip in foco_dat["todo"]:
                st.markdown(f"- {tip}")

    with st.expander("ℹ️ Como interpretar"):
        st.markdown("""- **LPV/Cliques** baixo → **Criativo/LP**.
- **Checkout/LPV** baixo → **Interesse/Oferta**.
- **Compra/Checkout** baixo → **Remarketing/Pagamento**.
- Se tudo saudável **e** há volume → **Escala**.""")

    # ========= COMPARATIVOS (A vs B) =========
    with st.expander("Comparativos — Período A vs Período B (opcional)", expanded=False):
        if fetch_insights_daily is None:
            st.info("Para usar os comparativos, importe `fetch_insights_daily` neste módulo.")
            return

        st.subheader("Comparativos — descubra o que mudou e por quê")
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
            return

        with st.spinner("Comparando períodos…"):
            dfA = fetch_insights_daily(act_id, token, api_version, str(sinceA), str(untilA), level,
                                       product_name=produto_sel)
            dfB = fetch_insights_daily(act_id, token, api_version, str(sinceB), str(untilB), level,
                                       product_name=produto_sel)

        if dfA.empty or dfB.empty:
            st.info("Sem dados em um dos períodos selecionados.")
            return

        def _agg(d):
            return {
                "spend": d["spend"].sum(),
                "revenue": d["revenue"].sum(),
                "purchases": d["purchases"].sum(),
                "clicks": d["link_clicks"].sum(),
                "lpv": d["lpv"].sum(),
                "checkout": d["init_checkout"].sum(),
                "add_payment": d["add_payment"].sum(),
            }

        A = _agg(dfA); B = _agg(dfB)
        roasA = _safe_div(A["revenue"], A["spend"])
        roasB = _safe_div(B["revenue"], B["spend"])
        cpaA  = _safe_div(A["spend"], A["purchases"]); cpaB  = _safe_div(B["spend"], B["purchases"])
        cpcA  = _safe_div(A["spend"], A["clicks"]);    cpcB  = _safe_div(B["spend"], B["clicks"])

        dir_map = {"Valor usado":"neutral","Faturamento":"higher","Vendas":"higher","ROAS":"higher","CPC":"lower","CPA":"lower"}
        delta_map = {
            "Valor usado":  B["spend"] - A["spend"],
            "Faturamento":  B["revenue"] - A["revenue"],
            "Vendas":       B["purchases"] - A["purchases"],
            "ROAS":         (roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else np.nan,
            "CPC":          (cpcB - cpcA)   if pd.notnull(cpcA) and pd.notnull(cpcB) else np.nan,
            "CPA":          (cpaB - cpaA)   if pd.notnull(cpaA) and pd.notnull(cpaB) else np.nan,
        }

        kpi_rows = [
            ("Valor usado", fmt_money_br(A["spend"]),   fmt_money_br(B["spend"]),   fmt_money_br(B["spend"] - A["spend"])),
            ("Faturamento", fmt_money_br(A["revenue"]), fmt_money_br(B["revenue"]), fmt_money_br(B["revenue"] - A["revenue"])),
            ("Vendas",      fmt_int_br(A["purchases"]), fmt_int_br(B["purchases"]), fmt_int_br(B["purchases"] - A["purchases"])),
            ("ROAS",        fmt_ratio_br(roasA),        fmt_ratio_br(roasB),
                             (fmt_ratio_br(roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else "")),
            ("CPC",         fmt_money_br(cpcA) if pd.notnull(cpcA) else "",
                             fmt_money_br(cpcB) if pd.notnull(cpcB) else "",
                             fmt_money_br(cpcB - cpcA) if pd.notnull(cpcA) and pd.notnull(cpcB) else ""),
            ("CPA",         fmt_money_br(cpaA) if pd.notnull(cpaA) else "",
                             fmt_money_br(cpaB) if pd.notnull(cpaB) else "",
                             fmt_money_br(cpaB - cpaA) if pd.notnull(cpaA) and pd.notnull(cpaB) else ""),
        ]
        kpi_df_disp = pd.DataFrame(kpi_rows, columns=["Métrica", "Período A", "Período B", "Δ (B - A)"])

        def _style_kpi(row):
            metric = row["Métrica"]; d = delta_map.get(metric, np.nan); rule = dir_map.get(metric, "neutral")
            styles = [""] * len(row)
            try:
                idxB = list(row.index).index("Período B"); idxD = list(row.index).index("Δ (B - A)")
            except Exception:
                return styles
            if pd.isna(d) or rule == "neutral" or d == 0: return styles
            better = (d > 0) if rule == "higher" else (d < 0)
            color  = "#16a34a" if better else "#dc2626"; weight = "700"
            styles[idxB] = f"color:{color}; font-weight:{weight};"; styles[idxD] = f"color:{color}; font-weight:{weight};"
            return styles

        st.markdown("**KPIs do período (A vs B)**")
        st.dataframe(kpi_df_disp.style.apply(_style_kpi, axis=1), use_container_width=True, height=260)

        st.markdown("---")

        # Taxas do funil
        rates_num = pd.DataFrame({
            "Taxa": ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout"],
            "Período A": [_safe_div(A["lpv"], A["clicks"]), _safe_div(A["checkout"], A["lpv"]), _safe_div(A["purchases"], A["checkout"])],
            "Período B": [_safe_div(B["lpv"], B["clicks"]), _safe_div(B["checkout"], B["lpv"]), _safe_div(B["purchases"], B["checkout"])],
        })
        rates_num["Δ"] = rates_num["Período B"] - rates_num["Período A"]
        rates_disp = rates_num.copy()
        for col in ["Período A", "Período B", "Δ"]:
            rates_disp[col] = rates_disp[col].map(_fmt_pct_br)

        delta_by_taxa = dict(zip(rates_num["Taxa"], rates_num["Δ"]))

        def _style_rate(row):
            taxa = row["Taxa"]; d = delta_by_taxa.get(taxa, np.nan)
            styles = [""] * len(row)
            try:
                idxB = list(row.index).index("Período B"); idxD = list(row.index).index("Δ")
            except Exception:
                return styles
            if pd.isna(d) or d == 0: return styles
            better = d > 0; color  = "#16a34a" if better else "#dc2626"; weight = "700"
            styles[idxB] = f"color:{color}; font-weight:{weight};"; styles[idxD] = f"color:{color}; font-weight:{weight};"
            return styles

        st.markdown("**Taxas do funil (A vs B)**")
        st.dataframe(rates_disp.style.apply(_style_rate, axis=1), use_container_width=True, height=180)

        # Funis lado a lado
        labels_funnel = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
        valsA = [int(round(A["clicks"])), int(round(A["lpv"])), int(round(A["checkout"])), int(round(A["add_payment"])), int(round(A["purchases"]))]
        valsB = [int(round(B["clicks"])), int(round(B["lpv"])), int(round(B["checkout"])), int(round(B["add_payment"])), int(round(B["purchases"]))]
        valsA_plot = enforce_monotonic(valsA); valsB_plot = enforce_monotonic(valsB)

        cA, cB = st.columns(2)
        with cA:
            st.plotly_chart(funnel_fig(labels_funnel, valsA_plot, title=f"Funil — Período A ({sinceA} a {untilA})"), use_container_width=True)
        with cB:
            st.plotly_chart(funnel_fig(labels_funnel, valsB_plot, title=f"Funil — Período B ({sinceB} a {untilB})"), use_container_width=True)

        # Δ por etapa
        delta_counts = [b - a for a, b in zip(valsA, valsB)]
        delta_df = pd.DataFrame({"Etapa": labels_funnel, "Período A": valsA, "Período B": valsB, "Δ (B - A)": delta_counts})
        delta_disp = delta_df.copy()
        delta_disp["Período A"]  = delta_disp["Período A"].map(fmt_int_br)
        delta_disp["Período B"]  = delta_disp["Período B"].map(fmt_int_br)
        delta_disp["Δ (B - A)"]  = delta_disp["Δ (B - A)"].map(_fmt_int_signed_br)
        st.markdown("**Pessoas a mais/menos em cada etapa (B − A)**")
        st.dataframe(delta_disp, use_container_width=True, height=240)
