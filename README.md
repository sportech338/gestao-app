
# Dashboard de Metas — Tráfego Pago (Streamlit)

Este é um app **Streamlit** para planejar e acompanhar metas **semanais e mensais** de tráfego pago.
- Define metas por **Faturamento** ou **Compras**
- Calcula **sessões necessárias** usando as taxas *Sessão→Checkout* e *Checkout→Compra*
- Sugere **orçamento** com base no **ROAS alvo**
- Exibe **ROI estimado**
- Gera **planilhas CSV** de metas (semanal e mensal)
- Permite editar os **realizados** (investimento, sessões, checkouts, compras, faturamento) direto no app e baixar o CSV

## Como usar com GitHub + Streamlit Cloud

1. Crie um repositório com estes arquivos:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. Suba para o GitHub.
3. Vá ao **Streamlit Community Cloud** (share.streamlit.io) e conecte ao seu repositório.
4. Selecione o arquivo principal `app.py` e implante.
5. Use os botões para **baixar** os CSVs de plano, atualize no seu Excel/Sheets e **reenvie** no app quando quiser (no momento, o exemplo não persiste escrita em disco no Cloud).

## Fórmulas principais

Supondo:
- `AOV` = ticket médio (R$)
- `t1` = taxa Sessão→Checkout (ex.: 5% = 0.05)
- `t2` = taxa Checkout→Compra (ex.: 40% = 0.40)
- `ROAS_alvo` (ex.: 2.0)

**Se a meta for Faturamento (R$):**
- Compras meta = `Faturamento_meta / AOV`
- Sessões necessárias = `Compras_meta / (t1 * t2)`
- Orçamento sugerido = `Faturamento_meta / ROAS_alvo`
- ROI estimado = `(Faturamento_meta - Orçamento) / Orçamento`

**Se a meta for Compras (nº):**
- Faturamento meta = `Compras_meta * AOV`
- Sessões necessárias = `Compras_meta / (t1 * t2)`
- Orçamento sugerido = `Faturamento_meta / ROAS_alvo`

## Observações
- Para persistência real (ex.: Google Sheets, Supabase, etc.), adapte o código para gravar/ler de uma fonte externa.
- Este projeto usa `st.data_editor` para facilitar a edição diária.
