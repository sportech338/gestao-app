import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
APP_TZ = ZoneInfo("America/Sao_Paulo")
# =====================================================
# 🔐 CREDENCIAIS GLOBAIS DO FACEBOOK (META ADS)
# =====================================================
facebook_secrets = st.secrets.get("facebook", {})

act_id = (
    facebook_secrets.get("ad_account_id")
    or facebook_secrets.get("DEFAULT_ACT_ID")
    or None
)

token = (
    facebook_secrets.get("access_token")
    or facebook_secrets.get("FB_TOKEN")
    or None
)

api_version = "v23.0"
level = "campaign"

# 🚨 Alerta opcional de debug
if not act_id or not token:
    st.warning("⚠️ Credenciais do Facebook não encontradas em st.secrets. Verifique o [facebook] no secrets.toml.")


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional



st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")

# =====================================================
# 🎛️ MENU PRINCIPAL NA SIDEBAR
# =====================================================
st.sidebar.markdown("## SELECIONE O DASHBOARD:")
menu = st.sidebar.radio(
    "",
    ["📊 Dashboard – Tráfego Pago", "📦 Dashboard – Logística"],
    index=0
)

# =====================================================
# 🧠 Verificação de segredos (Shopify)
# =====================================================
if "shopify" not in st.secrets:
    st.error("❌ Configurações da Shopify ausentes. Vá em Settings → Secrets e adicione:")
    st.code("""
[shopify]
shop_name = "nomedaloja.myshopify.com"
access_token = "seu_token_privado_aqui"
    """, language="toml")
    st.stop()

# =====================================================
# 🔐 Integração com Shopify API
# =====================================================
SHOP_NAME = st.secrets["shopify"]["shop_name"]
ACCESS_TOKEN = st.secrets["shopify"]["access_token"]
API_VERSION = "2024-10"

BASE_URL = f"https://{SHOP_NAME}/admin/api/{API_VERSION}"
HEADERS = {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}

# =====================================================
# Sessão persistente (para otimizar requisições)
# =====================================================
_session = None
def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session

# =====================================================
# Produtos
# =====================================================
@st.cache_data(ttl=600)
def get_products_with_variants(limit=250):
    s = _get_session()
    url = f"{BASE_URL}/products.json?limit={limit}"
    all_products = []

    while url:
        r = s.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        all_products.extend(data.get("products", []))
        url = r.links.get("next", {}).get("url")

    rows = []
    for p in all_products:
        for v in p.get("variants", []):
            rows.append({
                "product_id": p["id"],
                "product_title": p["title"],
                "variant_id": v["id"],
                "variant_title": v["title"],
                "sku": v.get("sku"),
                "price": float(v.get("price") or 0),
                "compare_at_price": float(v.get("compare_at_price") or 0),
                "inventory": v.get("inventory_quantity"),
                "inventory_item_id": v.get("inventory_item_id"),
            })

    df_variants = pd.DataFrame(rows)

    if not df_variants.empty and "inventory_item_id" in df_variants.columns:
        inventory_ids = df_variants["inventory_item_id"].dropna().unique().tolist()
        costs_rows = []
        for i in range(0, len(inventory_ids), 100):
            batch = inventory_ids[i:i + 100]
            ids_param = ",".join(str(x) for x in batch)
            inv_url = f"{BASE_URL}/inventory_items.json?ids={ids_param}"
            inv_resp = s.get(inv_url, headers=HEADERS, timeout=60)
            inv_resp.raise_for_status()
            items = inv_resp.json().get("inventory_items", [])
            for it in items:
                costs_rows.append({
                    "inventory_item_id": it["id"],
                    "cost": float(it.get("cost") or 0)
                })

        df_costs = pd.DataFrame(costs_rows)
        df_variants = df_variants.merge(df_costs, on="inventory_item_id", how="left")
    else:
        df_variants["cost"] = np.nan

    return df_variants

# =====================================================
# Pedidos
# =====================================================
@st.cache_data(ttl=600)
def get_orders(start_date=None, end_date=None, only_paid=True, limit=250):
    """Baixa pedidos da Shopify com filtro dinâmico de período, garantindo o e-mail do cliente e puxando CPF, telefone e endereço."""
    hoje = datetime.now(APP_TZ).date()
    if start_date is None:
        start_date = hoje
    if end_date is None:
        end_date = hoje

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=APP_TZ)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=APP_TZ)

    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    url = f"{BASE_URL}/orders.json?limit={limit}&status=any&created_at_min={start_str}&created_at_max={end_str}"
    all_rows = []

    s = _get_session()
    while url:
        r = s.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        orders = data.get("orders", [])

        for o in orders:
            if only_paid and o.get("financial_status") not in ["paid", "partially_paid"]:
                continue

            customer = o.get("customer") or {}
            shipping = o.get("shipping_address") or {}
            shipping_lines = o.get("shipping_lines") or [{}]

            nome_cliente = (
                f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
                or "(Cliente não informado)"
            )

            email_cliente = (
                customer.get("email")
                or o.get("email")
                or o.get("contact_email")
                or "(sem email)"
            )

            # 📞 Telefone e CPF
            telefone = (
                shipping.get("phone")
                or customer.get("phone")
                or o.get("phone")
                or "(sem telefone)"
            )

            # 🧾 CPF pode vir de vários lugares
            cpf = None
            # 1️⃣ Em note_attributes (checkout customizado)
            for attr in o.get("note_attributes", []):
                if "cpf" in str(attr.get("name", "")).lower():
                    cpf = attr.get("value")
                    break
            # 2️⃣ Às vezes vem no campo company (Shopify padrão)
            if not cpf:
                cpf = shipping.get("company") or "(sem cpf)"

            # 🏠 Endereço completo
            endereco = shipping.get("address1", "(sem endereço)")
            bairro = shipping.get("address2", "(sem bairro)")
            cidade = shipping.get("city", "(sem cidade)")
            estado = shipping.get("province", "(sem estado)")
            cep = shipping.get("zip", "(sem cep)")
            pais = shipping.get("country", "(sem país)")

            for it in o.get("line_items", []):
                preco = float(it.get("price") or 0)
                qtd = int(it.get("quantity", 0))
                all_rows.append({
                    "order_id": o.get("id"),
                    "order_number": o.get("order_number"),
                    "created_at": o.get("created_at"),
                    "financial_status": o.get("financial_status"),
                    "fulfillment_status": o.get("fulfillment_status"),
                    "customer_name": nome_cliente,
                    "customer_email": email_cliente,
                    "customer_phone": telefone,
                    "customer_cpf": cpf,
                    "product_title": it.get("title"),
                    "variant_title": it.get("variant_title"),
                    "variant_id": it.get("variant_id"),
                    "sku": it.get("sku"),
                    "quantity": qtd,
                    "price": preco,
                    "line_revenue": preco * qtd,
                    "forma_entrega": shipping_lines[0].get("title", "N/A"),
                    "endereco": endereco,
                    "bairro": bairro,
                    "cidade": cidade,
                    "estado": estado,
                    "cep": cep,
                    "pais": pais,
                })

        url = r.links.get("next", {}).get("url")

    return pd.DataFrame(all_rows)


# =====================================================
# Buscar clientes/pedidos na Shopify (nome, e-mail ou número do pedido)
# =====================================================
@st.cache_data(ttl=300)
def search_orders_shopify(query: str, limit=100):
    """
    Busca pedidos diretamente na Shopify com base em nome, e-mail ou número do pedido.
    Ignora outros campos e evita baixar todos os pedidos.
    """
    if not query.strip():
        return pd.DataFrame()

    search_q = query.strip()

    # 🔍 Caso pareça número (pedido ou ID)
    if search_q.isdigit():
        url = f"{BASE_URL}/orders.json?limit={limit}&status=any&name={search_q}"
    else:
        # 🔍 Caso seja e-mail ou nome (Shopify aceita filtro direto por e-mail)
        if "@" in search_q:
            url = f"{BASE_URL}/orders.json?limit={limit}&status=any&email={search_q}"
        else:
            # 🔍 Caso seja nome (fazemos busca genérica mas apenas pelo campo de cliente)
            url = f"{BASE_URL}/orders.json?limit={limit}&status=any&query=customer_name:{search_q}"

    s = _get_session()
    r = s.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json().get("orders", [])

    all_rows = []
    for o in data:
        customer = o.get("customer") or {}
        shipping = o.get("shipping_address") or {}
        shipping_lines = o.get("shipping_lines") or [{}]
        nome_cliente = (
            f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
            or "(Cliente não informado)"
        )

        # 🔎 Filtra novamente localmente, só para garantir correspondência precisa
        if (
            search_q.lower() not in nome_cliente.lower()
            and search_q.lower() not in str(o.get("order_number", "")).lower()
            and search_q.lower() not in str(customer.get("email", "")).lower()
        ):
            continue

        # ✅ Captura de e-mail mais robusta
        email_cliente = (
            customer.get("email")
            or o.get("email")
            or o.get("contact_email")
            or "(sem email)"
        )

        for it in o.get("line_items", []):
            preco = float(it.get("price") or 0)
            qtd = int(it.get("quantity", 0))
            all_rows.append({
                "order_id": o.get("id"),
                "order_number": o.get("order_number"),
                "created_at": o.get("created_at"),
                "financial_status": o.get("financial_status"),
                "fulfillment_status": o.get("fulfillment_status"),
                "customer_name": nome_cliente,
                "customer_email": email_cliente,
                "product_title": it.get("title"),
                "variant_title": it.get("variant_title"),
                "variant_id": it.get("variant_id"),
                "sku": it.get("sku"),
                "quantity": qtd,
                "price": preco,
                "line_revenue": preco * qtd,
                "forma_entrega": shipping_lines[0].get("title", "N/A"),
                "estado": shipping.get("province", "N/A"),
                "cidade": shipping.get("city", "N/A"),
            })

    return pd.DataFrame(all_rows)


# =====================================================
# Fulfillment (processar pedido) — versão com fallback 422 DSers
# =====================================================
def create_fulfillment(order_id, tracking_number=None, tracking_company="Correios"):
    """
    Cria fulfillment pela API da Shopify, mesmo para pedidos DSers.
    Se ocorrer erro 422, reatribui automaticamente o fulfillment_order e tenta novamente.
    """

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": ACCESS_TOKEN
    }

    # 1️⃣ Buscar fulfillment orders do pedido
    f_orders_url = f"{BASE_URL}/orders/{order_id}/fulfillment_orders.json"
    f_orders_resp = requests.get(f_orders_url, headers=headers)
    f_orders_resp.raise_for_status()
    f_orders = f_orders_resp.json()

    if "fulfillment_orders" not in f_orders or not f_orders["fulfillment_orders"]:
        raise Exception(f"⚠️ Nenhum fulfillment order encontrado para o pedido {order_id}")

    fulfillment_orders = f_orders["fulfillment_orders"]

    # 2️⃣ Tentar criar fulfillment para cada fulfillment_order encontrado
    for f_order in fulfillment_orders:
        fulfillment_order_id = f_order["id"]

        payload = {
            "fulfillment": {
                "line_items_by_fulfillment_order": [
                    {"fulfillment_order_id": fulfillment_order_id}
                ],
                "tracking_info": {
                    "number": tracking_number or "",
                    "company": tracking_company
                },
                "notify_customer": True
            }
        }

        url_fulfillment = f"{BASE_URL}/fulfillments.json"
        resp = requests.post(url_fulfillment, headers=headers, json=payload)

        # 3️⃣ Se der erro 422 → reatribui o fulfillment order para o seu location e tenta novamente
        if resp.status_code == 422:
            try:
                new_location_id = f_order.get("assigned_location_id")
                if new_location_id:
                    move_url = f"{BASE_URL}/fulfillment_orders/{fulfillment_order_id}/move.json"
                    move_payload = {"fulfillment_order": {"new_location_id": new_location_id}}
                    move_resp = requests.post(move_url, headers=headers, json=move_payload)
                    if move_resp.status_code in [200, 201]:
                        # tenta novamente após mover
                        resp = requests.post(url_fulfillment, headers=headers, json=payload)
            except Exception as err:
                print(f"⚠️ Erro ao reatribuir fulfillment order: {err}")

        # 4️⃣ Se funcionar, retorna o resultado
        if resp.status_code in [200, 201]:
            return resp.json()
        else:
            print(f"❌ Falha ao criar fulfillment (Pedido {order_id}): {resp.status_code} → {resp.text}")

    # 5️⃣ Se nenhuma tentativa funcionar, lança erro
    raise Exception(f"❌ Não foi possível criar fulfillment para o pedido {order_id}")

# =====================================================
# 🚀 LOGGING (opcional)
# =====================================================
def log_fulfillment(order_id):
    with open("fulfillment_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(APP_TZ)} - Pedido {order_id} processado\n")

# =============== Config & Estilos ===============
st.markdown("""
<style>
.small-muted { color:#6b7280; font-size:12px; }
.kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# Janelas de atribuição (paridade com Ads Manager)
ATTR_KEYS = ["7d_click", "1d_view"]
PRODUTOS = ["Flexlive", "KneePro", "NasalFlex", "Meniscus"]

# --- Constantes e parser para breakdown por hora
HOUR_BREAKDOWN = "hourly_stats_aggregated_by_advertiser_time_zone"

def _parse_hour_bucket(h):
    """Normaliza bucket de hora ('0'..'23' ou '00:00'..'23:00') para int [0..23]."""
    if h is None:
        return None
    try:
        s = str(h).strip()
        val = int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except Exception:
        return None

# =============== Helpers genéricos ===============
def _retry_call(fn, max_retries=5, base_wait=1.2):
    """Executa uma função com backoff exponencial para erros/transientes comuns."""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit","retry","temporarily unavailable","timeout","timed out"]):
                time.sleep(base_wait * (2 ** i))
                continue
            raise
    raise RuntimeError("Falha após múltiplas tentativas.")

def _ensure_act_prefix(ad_account_id: str) -> str:
    s = (ad_account_id or "").strip()
    return s if s.startswith("act_") else f"act_{s}"

def _to_float(x):
    try:
        return float(x or 0)
    except:
        return 0.0

def _sum_item(item, allowed_keys=None):
    """Usa 'value' quando existir; senão soma SOMENTE as chaves permitidas (ex.: 7d_click, 1d_view)."""
    if not isinstance(item, dict):
        return _to_float(item)
    if "value" in item:
        return _to_float(item.get("value"))
    keys = allowed_keys or ATTR_KEYS
    s = 0.0
    for k in keys:
        s += _to_float(item.get(k))
    return s

def _sum_actions_exact(rows, exact_names, allowed_keys=None) -> float:
    """Soma totals de actions pelos nomes exatos (case-insensitive), respeitando a janela (allowed_keys)."""
    if not rows:
        return 0.0
    names = {str(n).lower() for n in exact_names}
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type", "")).lower()
        if at in names:
            tot += _sum_item(r, allowed_keys)
    return float(tot)

def _sum_actions_contains(rows, substrs, allowed_keys=None) -> float:
    """Soma totals de actions que CONTENHAM qualquer substring, respeitando a janela (allowed_keys)."""
    if not rows:
        return 0.0
    ss = [str(s).lower() for s in substrs]
    tot = 0.0
    for r in rows:
        at = str(r.get("action_type", "")).lower()
        if any(s in at for s in ss):
            tot += _sum_item(r, allowed_keys)
    return float(tot)

def _pick_purchase_totals(rows, allowed_keys=None) -> float:
    """Prioriza omni_purchase; senão pega o MAIOR entre tipos específicos (sem duplicar janelas)."""
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type", "")).lower()} for r in rows]
    omni = [r for r in rows if r["action_type"] == "omni_purchase"]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))
    candidates = {
        "purchase": 0.0,
        "onsite_conversion.purchase": 0.0,
        "offsite_conversion.fb_pixel_purchase": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)
    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))
    # fallback amplo (respeitando allowed_keys)
    grp = {}
    for r in rows:
        if "purchase" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)

def _fmt_money_br(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def funnel_fig(labels, values, title=None):
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value",
            textposition="inside",
            texttemplate="<b>%{value}</b>",
            textfont=dict(size=35),
            opacity=0.95,
            connector={"line": {"dash": "dot", "width": 1}},
        )
    )
    fig.update_layout(
        title=title or "",
        margin=dict(l=10, r=10, t=48, b=10),
        height=540,
        template="plotly_white",
        separators=",.",  # pt-BR
        uniformtext=dict(minsize=12, mode="show")
    )
    return fig

def _pick_checkout_totals(rows, allowed_keys=None) -> float:
    """
    Soma Initiate Checkout priorizando omni; senão pega o MAIOR entre variantes
    (sem duplicar janelas). Aceita initiated/initiate e offsite/onsite.
    """
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]

    # 1) preferir omni
    omni = [r for r in rows if r["action_type"] in ("omni_initiated_checkout","omni_initiate_checkout")]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))

    # 2) variantes conhecidas
    candidates = {
        "initiate_checkout": 0.0,
        "initiated_checkout": 0.0,
        "onsite_conversion.initiated_checkout": 0.0,
        "offsite_conversion.fb_pixel_initiate_checkout": 0.0,
        "offsite_conversion.fb_pixel_initiated_checkout": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)

    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))

    # 3) fallback amplo por substring
    grp = {}
    for r in rows:
        if "initiate" in r["action_type"] and "checkout" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def _pick_add_payment_totals(rows, allowed_keys=None) -> float:
    """
    Soma Add Payment Info com suporte a omni/onsite/offsite.
    """
    if not rows:
        return 0.0
    rows = [{**r, "action_type": str(r.get("action_type","")).lower()} for r in rows]

    omni = [r for r in rows if r["action_type"] in ("omni_add_payment_info","add_payment_info.omni")]
    if omni:
        return float(sum(_sum_item(r, allowed_keys) for r in omni))

    candidates = {
        "add_payment_info": 0.0,
        "onsite_conversion.add_payment_info": 0.0,
        "offsite_conversion.fb_pixel_add_payment_info": 0.0,
    }
    for r in rows:
        at = r["action_type"]
        if at in candidates:
            candidates[at] += _sum_item(r, allowed_keys)

    if any(v > 0 for v in candidates.values()):
        return float(max(candidates.values()))

    grp = {}
    for r in rows:
        if "add_payment" in r["action_type"]:
            grp.setdefault(r["action_type"], 0.0)
            grp[r["action_type"]] += _sum_item(r, allowed_keys)
    return float(max(grp.values()) if grp else 0.0)


def enforce_monotonic(values):
    """Garante formato de funil: cada etapa <= etapa anterior (só para o desenho)."""
    out, cur = [], None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

# ==== Helpers de comparação/formatos ====
def _rate(a, b):
    return (a / b) if b and b > 0 else np.nan

def _fmt_pct_br(x):
    return (
        f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_ratio_br(x):  # ROAS "1,23x"
    return (
        f"{x:,.2f}x".replace(",", "X").replace(".", ",").replace("X", ".")
        if pd.notnull(x) else ""
    )

def _fmt_int_br(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except:
        return ""

def _fmt_int_signed_br(x):
    try:
        v = int(round(float(x)))
        s = f"{abs(v):,}".replace(",", ".")
        return f"+{s}" if v > 0 else (f"-{s}" if v < 0 else "0")
    except:
        return ""

def _safe_div(n, d):
    n = float(n or 0)
    d = float(d or 0)
    return (n / d) if d > 0 else np.nan

def _pct(a, b):
    a = float(a or 0); b = float(b or 0)
    return (a / b) if b > 0 else np.nan

def _intensity_label(share):
    # share = fração do maior drop sobre a soma de todos os drops (>0)
    if not np.isfinite(share):
        return "Baixa"
    if share > 0.60: 
        return "Alta"
    if share >= 0.30:
        return "Média"
    return "Baixa"

def _decide_focus(r1, r2, r3, clicks, lpv, co, addpay, purch,
                  bm_r1, bm_r2, bm_r3, min_clicks, min_lpv, min_co, min_purch,
                  split_rmk=True):
    # saúde para Escala
    healthy = (
        (pd.notnull(r1) and r1 >= bm_r1/100.0) and
        (pd.notnull(r2) and r2 >= bm_r2/100.0) and
        (pd.notnull(r3) and r3 >= bm_r3/100.0) and
        (float(purch or 0) >= float(min_purch or 0))
    )
    low_volume_guard = (float(clicks or 0) < float(min_clicks or 0)) or \
                       (float(lpv or 0)    < float(min_lpv or 0))    or \
                       (float(co or 0)     < float(min_co or 0))

    # drops por volume
    drop1 = max(0.0, float(clicks or 0) - float(lpv or 0))        # Criativo
    drop2 = max(0.0, float(lpv or 0)    - float(co or 0))         # Interesse
    drop3a= max(0.0, float(co or 0)     - float(addpay or 0))     # RMK (checkout->pagto)
    drop3b= max(0.0, float(addpay or 0) - float(purch or 0))      # RMK (pagto->compra)

    if healthy and not low_volume_guard:
        return "Escala", "Taxas ≥ benchmarks e volume OK — elegível a escalar.", "Média", False, drop1, drop2, drop3a, drop3b

    if split_rmk:
        gaps = {
            "Teste de criativo": drop1,
            "Teste de interesse": drop2,
            "Remarketing (checkout→pagto)": drop3a,
            "Remarketing (pagto→compra)": drop3b,
        }
    else:
        gaps = {
            "Teste de criativo": drop1,
            "Teste de interesse": drop2,
            "Remarketing (fundo do funil)": drop3a + drop3b,
        }

    major = max(gaps, key=gaps.get)
    max_drop = gaps[major]
    total_drop = sum(v for v in gaps.values() if v > 0)
    share = (max_drop / total_drop) if total_drop > 0 else np.nan
    intensity = _intensity_label(share)

    if max_drop <= 0 and not healthy:
        return "Diagnóstico", "Sem queda dominante; revisar tracking/UX/oferta.", "Baixa", low_volume_guard, drop1, drop2, drop3a, drop3b

    reason = f"Maior perda em **{major}** (Δ={int(round(max_drop))} pessoas)."
    if low_volume_guard:
        return f"{major} (c/ cautela)", reason, intensity, True, drop1, drop2, drop3a, drop3b
    return major, reason, intensity, False, drop1, drop2, drop3a, drop3b


def _chunks_by_days(since_str: str, until_str: str, max_days: int = 30):
    """Divide [since, until] em janelas de até max_days (inclusive)."""
    s = datetime.fromisoformat(str(since_str)).date()
    u = datetime.fromisoformat(str(until_str)).date()
    cur = s
    while cur <= u:
        end = min(cur + timedelta(days=max_days - 1), u)
        yield str(cur), str(end)
        cur = end + timedelta(days=1)

# --- Filtro simples por produto no nome da campanha ---
def _filter_by_product(df: pd.DataFrame, produto: str) -> pd.DataFrame:
    """
    Filtra pelo nome do produto dentro de campaign_name.
    Se produto == "(Todos)", retorna o df original.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not produto or produto == "(Todos)":
        return df
    mask = df["campaign_name"].str.contains(produto, case=False, na=False)
    return df[mask].copy()


# =============== Coleta (com fallback de campos extras) ===============
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_daily(act_id: str, token: str, api_version: str,
                         since_str: str, until_str: str,
                         level: str = "campaign",
                         try_extra_fields: bool = True,
                         product_name: str | None = None) -> pd.DataFrame:
    """
    - time_range (since/until) + time_increment=1
    - level único ('campaign' recomendado)
    - Usa action_report_time=conversion e action_attribution_windows fixos (paridade com Ads Manager)
    - Traz fields extras (link_clicks, landing_page_views) e faz fallback se houver erro #100.
    - Paraleliza chunks de 30d e usa requests.Session para keep-alive.
    - Opcional: filtering por nome da campanha (product_name) direto na API.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"

    base_fields = [
        "spend", "impressions", "clicks", "actions", "action_values",
        "account_currency", "date_start", "campaign_id", "campaign_name"
    ]
    extra_fields = ["link_clicks", "landing_page_views"]

    def _fetch_range(_since: str, _until: str, _try_extra: bool) -> list[dict]:
        fields = base_fields + (extra_fields if _try_extra else [])
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": "conversion",
            "action_attribution_windows": ",".join(ATTR_KEYS),
        }
        # filtro por campanha (produto) direto na API, quando aplicável
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps([{
                "field": "campaign.name", "operator": "CONTAIN", "value": product_name
            }])

        rows_local, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API.")
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
                if code == 100 and _try_extra:
                    # refaz sem extras só para ESTA janela
                    return _fetch_range(_since, _until, _try_extra=False)
                raise RuntimeError(f"Graph API error {resp.status_code} | code={code} subcode={sub} | {msg}")

            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []

                link_clicks = rec.get("link_clicks", None)
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

                lpv = rec.get("landing_page_views")
                if lpv in (None, "", "null"):
                    lpv = _sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)

                ic  = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                purchases_cnt = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                revenue_val   = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows_local.append({
                    "date":           pd.to_datetime(rec.get("date_start")),
                    "currency":       rec.get("account_currency", "BRL"),
                    "campaign_id":    rec.get("campaign_id", ""),
                    "campaign_name":  rec.get("campaign_name", ""),
                    "spend":          _to_float(rec.get("spend")),
                    "impressions":    _to_float(rec.get("impressions")),
                    "clicks":         _to_float(rec.get("clicks")),
                    "link_clicks":    _to_float(link_clicks),
                    "lpv":            _to_float(lpv),
                    "init_checkout":  _to_float(ic),
                    "add_payment":    _to_float(api),
                    "purchases":      _to_float(purchases_cnt),
                    "revenue":        _to_float(revenue_val),
                })

            paging = (payload or {}).get("paging", {})
            if paging.get("next"):
                next_url, next_params = paging.get("next"), None
            else:
                after = (paging.get("cursors") or {}).get("after")
                if after:
                    next_url, next_params = base_url, params.copy()
                    next_params["after"] = after
                else:
                    break

        return rows_local

    # Busca em paralelo (3-5 workers é seguro)
    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u, try_extra_fields) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    num_cols = ["spend", "impressions", "clicks", "link_clicks",
                "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df = df.sort_values("date")
    return df

# --- Coleta por hora (dayparting)
@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_hourly(act_id: str, token: str, api_version: str,
                          since_str: str, until_str: str,
                          level: str = "campaign") -> pd.DataFrame:
    """
    Coleta por hora em janelas menores (default 30 dias) para evitar code=1
    e concatena o resultado. Tenta 'conversion' por chunk; se falhar, tenta 'impression'.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]

    def _fetch_range(_since: str, _until: str, action_rt: str) -> pd.DataFrame:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": action_rt,
            "breakdowns": HOUR_BREAKDOWN,
        }
        if action_rt == "conversion":
            params["action_attribution_windows"] = ",".join(ATTR_KEYS)

        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError("Resposta inválida da Graph API (hourly).")
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                code, sub, msg = err.get("code"), err.get("error_subcode"), err.get("message")
                raise RuntimeError(f"Graph API error hourly | code={code} subcode={sub} | {msg}")

            for rec in (payload.get("data") or []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []
                hour_bucket = _parse_hour_bucket(rec.get(HOUR_BREAKDOWN))

                link_clicks = rec.get("link_clicks")
                if link_clicks is None:
                    link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)

                lpv = rec.get("landing_page_views")
                if lpv is None:
                    lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                           or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                           or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))

                ic   = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur  = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev  = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                rows.append({
                    "date":          pd.to_datetime(rec.get("date_start")),
                    "hour":          hour_bucket,
                    "currency":      rec.get("account_currency", "BRL"),
                    "campaign_id":   rec.get("campaign_id", ""),
                    "campaign_name": rec.get("campaign_name", ""),
                    "spend":         _to_float(rec.get("spend")),
                    "impressions":   _to_float(rec.get("impressions")),
                    "clicks":        _to_float(rec.get("clicks")),
                    "link_clicks":   _to_float(link_clicks),
                    "lpv":           _to_float(lpv),
                    "init_checkout": _to_float(ic),
                    "add_payment":   _to_float(api_),
                    "purchases":     _to_float(pur),
                    "revenue":       _to_float(rev),
                })

            paging = (payload.get("paging") or {})
            if paging.get("next"):
                next_url, next_params = paging.get("next"), None
            else:
                after = (paging.get("cursors") or {}).get("after")
                if after:
                    next_url, next_params = base_url, params.copy()
                    next_params["after"] = after
                else:
                    break

        df = pd.DataFrame(rows)
        return df

    # Agrega chunks menores (use 30 dias por segurança no hourly)
    dfs = []
    for s_chunk, u_chunk in _chunks_by_days(since_str, until_str, max_days=30):
        try:
            df_chunk = _fetch_range(s_chunk, u_chunk, "conversion")
        except Exception:
            # fallback silencioso por chunk
            try:
                df_chunk = _fetch_range(s_chunk, u_chunk, "impression")
            except Exception:
                df_chunk = pd.DataFrame()
        if df_chunk is not None and not df_chunk.empty:
            dfs.append(df_chunk)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int).clip(0, 23)
    df["dow"] = df["date"].dt.dayofweek
    order = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
    df["dow_label"] = df["dow"].map(order)
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    return df.sort_values(["date", "hour"])

@st.cache_data(ttl=600, show_spinner=True)
def fetch_insights_breakdown(act_id: str, token: str, api_version: str,
                             since_str: str, until_str: str,
                             breakdowns: list[str],
                             level: str = "campaign",
                             product_name: str | None = None) -> pd.DataFrame:
    """
    Coleta insights com 1 ou 2 breakdowns.
    Mantém paridade (conversion + ATTR_KEYS), chunking 30d, requests.Session e paralelismo.
    Opcional: filtering por campanha (product_name) direto na API.
    """
    act_id = _ensure_act_prefix(act_id)
    base_url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "spend","impressions","clicks","actions","action_values",
        "account_currency","date_start","campaign_id","campaign_name"
    ]

    def _fetch_range(_since: str, _until: str) -> list[dict]:
        params = {
            "access_token": token,
            "level": level,
            "time_range": json.dumps({"since": _since, "until": _until}),
            "time_increment": 1,
            "fields": ",".join(fields),
            "limit": 500,
            "action_report_time": "conversion",
            "action_attribution_windows": ",".join(ATTR_KEYS),
            "breakdowns": ",".join(breakdowns[:2])
        }
        if level == "campaign" and product_name and product_name != "(Todos)":
            params["filtering"] = json.dumps([{
                "field": "campaign.name", "operator": "CONTAIN", "value": product_name
            }])

        rows, next_url, next_params = [], base_url, params.copy()
        while next_url:
            sess = _get_session()
            resp = _retry_call(lambda: sess.get(next_url, params=next_params, timeout=90))
            payload = resp.json()
            if resp.status_code != 200:
                err = (payload or {}).get("error", {})
                raise RuntimeError(
                    f"Graph API error breakdown | code={err.get('code')} "
                    f"sub={err.get('error_subcode')} | {err.get('message')}"
                )

            for rec in payload.get("data", []):
                actions = rec.get("actions") or []
                action_values = rec.get("action_values") or []

                link_clicks = _sum_actions_exact(actions, ["link_click"], allowed_keys=ATTR_KEYS)
                lpv = (_sum_actions_exact(actions, ["landing_page_view"], allowed_keys=ATTR_KEYS)
                       or _sum_actions_exact(actions, ["view_content"], allowed_keys=ATTR_KEYS)
                       or _sum_actions_contains(actions, ["landing_page"], allowed_keys=ATTR_KEYS))
                ic   = _pick_checkout_totals(actions, allowed_keys=ATTR_KEYS)
                api_ = _pick_add_payment_totals(actions, allowed_keys=ATTR_KEYS)
                pur  = _pick_purchase_totals(actions, allowed_keys=ATTR_KEYS)
                rev  = _pick_purchase_totals(action_values, allowed_keys=ATTR_KEYS)

                base = {
                    "currency":      rec.get("account_currency","BRL"),
                    "campaign_id":   rec.get("campaign_id",""),
                    "campaign_name": rec.get("campaign_name",""),
                    "spend":         _to_float(rec.get("spend")),
                    "impressions":   _to_float(rec.get("impressions")),
                    "clicks":        _to_float(rec.get("clicks")),
                    "link_clicks":   _to_float(link_clicks),
                    "lpv":           _to_float(lpv),
                    "init_checkout": _to_float(ic),
                    "add_payment":   _to_float(api_),
                    "purchases":     _to_float(pur),
                    "revenue":       _to_float(rev),
                }
                for b in breakdowns[:2]:
                    base[b] = rec.get(b)
                rows.append(base)

            paging = (payload or {}).get("paging", {})
            if paging.get("next"):
                next_url, next_params = paging.get("next"), None
            else:
                after = (paging.get("cursors") or {}).get("after")
                if after:
                    next_url, next_params = base_url, params.copy()
                    next_params["after"] = after
                else:
                    break

        return rows

    # 🔑 Junta em blocos de até 30 dias — em paralelo
    chunks = list(_chunks_by_days(since_str, until_str, max_days=30))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as ex:
        futs = [ex.submit(_fetch_range, s, u) for s, u in chunks]
        for f in as_completed(futs):
            all_rows.extend(f.result() or [])

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["ROAS"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
    return df

def _range_from_preset(p):
    """Define intervalos automáticos de data com base no período rápido selecionado."""
    local_today = datetime.now(APP_TZ).date()
    base_end = local_today - timedelta(days=1)  # evita incluir dia parcial

    if p == "Hoje":
        return local_today, local_today
    if p == "Ontem":
        return local_today - timedelta(days=1), local_today - timedelta(days=1)
    if p == "Últimos 7 dias":
        return base_end - timedelta(days=6), base_end
    if p == "Últimos 14 dias":
        return base_end - timedelta(days=13), base_end
    if p == "Últimos 30 dias":
        return base_end - timedelta(days=29), base_end
    if p == "Últimos 90 dias":
        return base_end - timedelta(days=89), base_end
    if p == "Esta semana":
        start_week = local_today - timedelta(days=local_today.weekday())
        return start_week, local_today
    if p == "Este mês":
        start_month = local_today.replace(day=1)
        return start_month, local_today
    if p == "Máximo":
        return date(2017, 1, 1), base_end
    # fallback padrão
    return base_end - timedelta(days=6), base_end

# =====================================================
# ================== DASHBOARD TRÁFEGO PAGO ============
# =====================================================
if menu == "📊 Dashboard – Tráfego Pago":
    st.title("📈 Dashboard — Tráfego Pago")
    st.caption("Análise completa de campanhas e funil de conversão.")

    # ================= EXIBE STATUS NO SIDEBAR =================
    with st.sidebar:
        st.info(f"**Ad Account ID:** {act_id or '❌ Não encontrado'}")
        st.markdown(f"**API Version:** `{api_version}`")
        st.markdown(f"**Nível:** `{level}`")

        if act_id and token:
            st.success("✅ Credenciais do Facebook carregadas via `st.secrets`")
        else:
            st.error("⚠️ Falha ao carregar credenciais do Facebook.")

    # ================= CONTINUA ANÁLISE NORMAL =================

    # -------------------------------------------------
    # 🧭 SIDEBAR — Filtro lateral de período
    # -------------------------------------------------
    st.sidebar.header("📅 Período rápido")

    hoje = datetime.now(APP_TZ).date()

    opcoes_periodo = [
        "Hoje", "Ontem", "Últimos 7 dias", "Últimos 14 dias",
        "Últimos 30 dias", "Últimos 90 dias", "Esta semana",
        "Este mês", "Máximo", "Personalizado"
    ]

    escolha_periodo = st.sidebar.radio("Selecione:", opcoes_periodo, index=0)

    if escolha_periodo == "Hoje":
        start_date, end_date = hoje, hoje
    elif escolha_periodo == "Ontem":
        start_date, end_date = hoje - timedelta(days=1), hoje - timedelta(days=1)
    elif escolha_periodo == "Últimos 7 dias":
        start_date, end_date = hoje - timedelta(days=7), hoje - timedelta(days=1)
    elif escolha_periodo == "Últimos 14 dias":
        start_date, end_date = hoje - timedelta(days=14), hoje - timedelta(days=1)
    elif escolha_periodo == "Últimos 30 dias":
        start_date, end_date = hoje - timedelta(days=30), hoje - timedelta(days=1)
    elif escolha_periodo == "Últimos 90 dias":
        start_date, end_date = hoje - timedelta(days=90), hoje - timedelta(days=1)
    elif escolha_periodo == "Esta semana":
        start_date, end_date = hoje - timedelta(days=hoje.weekday()), hoje
    elif escolha_periodo == "Este mês":
        start_date = hoje.replace(day=1)
        end_date = hoje
    elif escolha_periodo == "Máximo":
        start_date = date(2020, 1, 1)
        end_date = hoje
    else:
        periodo = st.sidebar.date_input("📆 Selecione o intervalo:", (hoje, hoje), format="DD/MM/YYYY")
        if isinstance(periodo, tuple) and len(periodo) == 2:
            start_date, end_date = periodo
        else:
            st.sidebar.warning("🟡 Selecione o fim do período.")
            st.stop()

    st.sidebar.markdown(f"**Desde:** {start_date}  \n**Até:** {end_date}")

    # 🔁 Compatibilidade com variáveis antigas (evita NameError no restante do código)
    since = start_date
    until = end_date

    # ================= VALIDAÇÃO E COLETA DE DADOS =================
    if not act_id or not token:
        st.error("⚠️ Credenciais ausentes — verifique o arquivo secrets.toml.")
        st.stop()

    with st.spinner("Buscando dados da Meta…"):
        df_daily = fetch_insights_daily(
            act_id=act_id,
            token=token,
            api_version=api_version,
            since_str=str(since),
            until_str=str(until),
            level=level,
            product_name=st.session_state.get("daily_produto")
        )

    df_hourly = None

    if df_daily.empty and (df_hourly is None or df_hourly.empty):
        st.warning("Sem dados para o período. Verifique permissões, conta e se há eventos de Purchase (value/currency).")
        st.stop()


    tab_daily, tab_funnel, tab_daypart, tab_detail = st.tabs([
        "📅 Visão diária",
        "🔄 Funil",
        "⏱️ Horários (principal)",
        "📊 Detalhamento"
    ])

    # -------------------- ABA 1: VISÃO DIÁRIA --------------------

    with tab_daily:
        st.subheader("Comparativo de Saídas e Custos por Variante:")
        
        # =====================================================
        # 🔄 Carregamento de produtos e pedidos
        # =====================================================
        produtos = st.session_state.get("produtos", get_products_with_variants())
        if produtos.empty:
            st.warning("⚠️ Nenhum produto encontrado.")
            st.stop()

        pedidos_cached = st.session_state.get("pedidos", pd.DataFrame())
        produtos_unicos = sorted(
            pedidos_cached.get("product_title", pd.Series(dtype=str)).dropna().unique().tolist()
            or produtos["product_title"].dropna().unique().tolist()
        )

        # =====================================================
        # 🧾 Seleção de produto (com opção "(Todos)" pré-selecionada)
        # =====================================================
        lista_produtos = ["(Todos)"] + sorted(produtos_unicos)

        produto_escolhido = st.selectbox(
            "🧾 Selecione o produto:",
            lista_produtos,
            index=0  # 🔹 Sempre começa com "(Todos)" selecionado
        )

        # =====================================================
        # 🗓️ Seleção de períodos para comparação (hoje vs ontem)
        # =====================================================
        hoje = datetime.now(APP_TZ).date()
        ontem = hoje - timedelta(days=1)

        col1, col2 = st.columns(2)
        with col1:
            periodo_a = st.date_input(
                "📅 Período A (mais recente):",
                (hoje, hoje),
                format="DD/MM/YYYY"
            )
        with col2:
            periodo_b = st.date_input(
                "📅 Período B (comparar):",
                (ontem, ontem),
                format="DD/MM/YYYY"
            )

        if not (isinstance(periodo_a, tuple) and len(periodo_a) == 2):
            st.warning("⚠️ Selecione um intervalo completo para o Período A.")
            st.stop()
        if not (isinstance(periodo_b, tuple) and len(periodo_b) == 2):
            st.warning("⚠️ Selecione um intervalo completo para o Período B.")
            st.stop()

        inicio_a, fim_a = periodo_a
        inicio_b, fim_b = periodo_b

        # =====================================================
        # 📦 Garantir pedidos atualizados
        # =====================================================
        def ensure_orders_for_range(start_date, end_date):
            loaded_range = st.session_state.get("periodo_atual")
            pedidos_cached = st.session_state.get("pedidos", pd.DataFrame())

            def range_covers(loaded, start_, end_):
                return loaded and (loaded[0] <= start_ and loaded[1] >= end_)

            if pedidos_cached.empty or (not range_covers(loaded_range, start_date, end_date)):
                with st.spinner(f"🔄 Carregando pedidos da Shopify de {start_date:%d/%m/%Y} a {end_date:%d/%m/%Y}..."):
                    pedidos_new = get_orders(start_date=start_date, end_date=end_date, only_paid=True)
                    st.session_state["pedidos"] = pedidos_new
                    st.session_state["periodo_atual"] = (start_date, end_date)
                    return pedidos_new
            return pedidos_cached

        periodo_min = min(inicio_a, inicio_b)
        periodo_max = max(fim_a, fim_b)
        pedidos = ensure_orders_for_range(periodo_min, periodo_max)

        # =====================================================
        # 🧩 Normaliza nomes de variantes — unifica por quantidade de peças
        # =====================================================
        import re

        def extrair_qtd_pecas(nome):
            """Extrai o número de peças do nome da variante (ex: '60 peças (Mais Vendido)' -> 60)."""
            if not isinstance(nome, str):
                return None
            match = re.search(r"(\d+)\s*(peças|pçs?|unidades|unid|uni)", nome.lower())
            return int(match.group(1)) if match else None

        # Cria nova coluna auxiliar (só pra conferência)
        pedidos["Qtd Base"] = pedidos["variant_title"].apply(extrair_qtd_pecas)

        # Substitui o próprio variant_title por nome padronizado "XX peças"
        pedidos["variant_title"] = pedidos["Qtd Base"].apply(
            lambda x: f"{int(x)} peças" if pd.notna(x) else None
        )

        st.info("✅ Variantes normalizadas — nomes antigos e novos agora são tratados como iguais.")
        
        if pedidos.empty:
            st.warning("⚠️ Nenhum pedido encontrado no intervalo selecionado.")
            st.stop()

        # =====================================================
        # 🧮 Análise de saídas por produto
        # =====================================================
        pedidos["created_at"] = pd.to_datetime(pedidos["created_at"], utc=True, errors="coerce")\
                                   .dt.tz_convert(APP_TZ).dt.tz_localize(None)

        # =====================================================
        # 🧩 Base dinâmica — produto específico → variantes | (Todos) → produtos
        # =====================================================
        if produto_escolhido == "(Todos)":
            base_prod = pedidos.copy()
            nivel_agrupamento = "product_title"
            label_nivel = "Produto"
        else:
            base_prod = pedidos[pedidos["product_title"] == produto_escolhido].copy()
            nivel_agrupamento = "variant_title"
            label_nivel = "Variante"

        def filtrar_periodo(df, ini, fim):
            return df[(df["created_at"].dt.date >= ini) & (df["created_at"].dt.date <= fim)].copy()

        df_a = filtrar_periodo(base_prod, inicio_a, fim_a)
        df_b = filtrar_periodo(base_prod, inicio_b, fim_b)

        # =====================================================
        # 📊 Consolida vendas por produto ou variante sem duplicar linhas
        # =====================================================
        resumo_a = (
            df_a.groupby(nivel_agrupamento, as_index=False)["quantity"]
            .sum()
            .rename(columns={"quantity": "Qtd A"})
        )
        resumo_b = (
            df_b.groupby(nivel_agrupamento, as_index=False)["quantity"]
            .sum()
            .rename(columns={"quantity": "Qtd B"})
        )

        # 🔗 Faz merge sem duplicar nomes idênticos
        comparativo = (
            pd.merge(resumo_a, resumo_b, on=nivel_agrupamento, how="outer")
            .groupby(nivel_agrupamento, as_index=False)
            .sum(numeric_only=True)
            .fillna(0)
        )

        comparativo["Diferença (unid.)"] = comparativo["Qtd A"] - comparativo["Qtd B"]
        comparativo["A-B(Qtd.%)"] = np.where(
            comparativo["Qtd B"] > 0,
            (comparativo["Qtd A"] - comparativo["Qtd B"]) / comparativo["Qtd B"] * 100,
            np.nan
        )
        comparativo["Part.A (%)"] = np.where(
            comparativo["Qtd A"].sum() > 0,
            comparativo["Qtd A"] / comparativo["Qtd A"].sum() * 100,
            0
        )
        comparativo["Part.B (%)"] = np.where(
            comparativo["Qtd B"].sum() > 0,
            comparativo["Qtd B"] / comparativo["Qtd B"].sum() * 100,
            0
        )
        comparativo["A-B(Part. | p.p)"] = comparativo["Part.A (%)"] - comparativo["Part.B (%)"]

        comparativo.rename(columns={nivel_agrupamento: label_nivel}, inplace=True)
        comparativo = comparativo.sort_values("Qtd A", ascending=False).reset_index(drop=True)

        # =====================================================
        # 💰 Carregar planilha de custos (Google Sheets)
        # =====================================================
        import gspread
        from google.oauth2.service_account import Credentials

        def get_gsheet_client():
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            gcp_info = dict(st.secrets["gcp_service_account"])
            if isinstance(gcp_info.get("private_key"), str):
                gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(gcp_info, scopes=scopes)
            client = gspread.authorize(creds)
            return client

        @st.cache_data(ttl=600)
        def carregar_planilha_custos():
            client = get_gsheet_client()
            sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"]).sheet1
            df = pd.DataFrame(sheet.get_all_records())
            df.columns = df.columns.str.strip()
            mapa_colunas = {
                "Produto": "Produto",
                "Variantes": "Variante",
                "Custo | Aliexpress": "Custo AliExpress (R$)",
                "Custo | Estoque": "Custo Estoque (R$)",
            }
            df.rename(columns=mapa_colunas, inplace=True)
            return df

        try:
            df_custos = carregar_planilha_custos()
        except Exception as e:
            st.error(f"❌ Erro ao carregar planilha de custos: {e}")
            st.stop()

        for col in ["Custo AliExpress (R$)", "Custo Estoque (R$)"]:
            if col in df_custos.columns:
                df_custos[col] = (
                    df_custos[col]
                    .astype(str)
                    .str.replace("R$", "", regex=False)
                    .str.replace(",", ".")
                    .str.strip()
                    .replace(["inexistente", ""], np.nan)
                    .astype(float)
                )

        # -------------------------------------------------
        # 💼 Análise de Custos e Lucros por Fornecedor
        # -------------------------------------------------
        st.subheader("💼 Análise de Custos e Lucros por Fornecedor")

        fornecedor = st.radio(
            "Selecione o fornecedor para cálculo de custos e lucros:",
            ["AliExpress", "Estoque"],
            horizontal=True
        )

        col_custo = "Custo AliExpress (R$)" if fornecedor == "AliExpress" else "Custo Estoque (R$)"

        # 🔍 Detecta itens realmente existentes em cada período (variante ou produto)
        itens_a = base_prod[base_prod["created_at"].dt.date.between(inicio_a, fim_a)][nivel_agrupamento].unique().tolist()
        itens_b = base_prod[base_prod["created_at"].dt.date.between(inicio_b, fim_b)][nivel_agrupamento].unique().tolist()

        # 🔧 Cria base de custos separada para cada período
        custos_base_A = df_custos[df_custos["Variante"].isin(itens_a) | df_custos["Produto"].isin(itens_a)].copy()
        custos_base_B = df_custos[df_custos["Variante"].isin(itens_b) | df_custos["Produto"].isin(itens_b)].copy()

        # 🔗 Adiciona colunas de quantidade correspondentes
        custos_base_A = custos_base_A.merge(comparativo[[label_nivel, "Qtd A"]], left_on=label_nivel, right_on=label_nivel, how="left")
        custos_base_B = custos_base_B.merge(comparativo[[label_nivel, "Qtd B"]], left_on=label_nivel, right_on=label_nivel, how="left")

        # 🔢 Ajusta custos unitários para cada base (dinâmico)
        if not df_custos.empty:
            # Escolhe a coluna base para indexar de forma segura
            col_index = label_nivel if label_nivel in df_custos.columns else "Variante"

            # Remove duplicados antes de indexar para evitar InvalidIndexError
            df_custos_indexed = df_custos.drop_duplicates(subset=[col_index]).set_index(col_index)

            # --- Período A ---
            custos_base_A["Custo Unitário"] = custos_base_A[label_nivel].map(df_custos_indexed[col_custo])
            custos_base_A["Custo Unitário"] = pd.to_numeric(custos_base_A["Custo Unitário"], errors="coerce").fillna(0)

            # --- Período B ---
            custos_base_B["Custo Unitário"] = custos_base_B[label_nivel].map(df_custos_indexed[col_custo])
            custos_base_B["Custo Unitário"] = pd.to_numeric(custos_base_B["Custo Unitário"], errors="coerce").fillna(0)

        # -------------------------------------------------
        # 💵 Adiciona preço médio real (se não existir)
        # -------------------------------------------------
        def add_preco_medio(custos_df):
            if "Preço Médio" not in custos_df.columns:
                if "price" in base_prod.columns:
                    precos = base_prod.groupby(nivel_agrupamento)["price"].mean().reset_index()
                    precos.rename(columns={nivel_agrupamento: label_nivel, "price": "Preço Médio"}, inplace=True)
                    custos_df = custos_df.merge(precos, on=label_nivel, how="left")
                else:
                    custos_df["Preço Médio"] = (custos_df["Custo Unitário"] * 2.5).round(2)
            return custos_df

        custos_base_A = add_preco_medio(custos_base_A)
        custos_base_B = add_preco_medio(custos_base_B)

        # -------------------------------------------------
        # 🧮 Cálculos por período (independentes)
        # -------------------------------------------------
        def calc_periodo(df, periodo_label, qtd_col):
            df = df.copy()
            df[f"Custo {periodo_label}"] = df["Custo Unitário"] * df[qtd_col]
            df[f"Receita {periodo_label}"] = (
                df[qtd_col] * df["Preço Médio"] if "Preço Médio" in df.columns else np.nan
            )
            df[f"Lucro Bruto {periodo_label}"] = df[f"Receita {periodo_label}"] - df[f"Custo {periodo_label}"]

            # =====================================================
            # 📊 Correção: participação global no modo "(Todos)"
            # =====================================================
            if produto_escolhido == "(Todos)":
                # Calcula o total de receita global de todos os produtos
                total_receita_global = (
                    pedidos.assign(receita_total=pedidos["price"] * pedidos["quantity"])
                    .groupby("product_title")["receita_total"]
                    .sum()
                    .sum()
                )

                # Se houver receita global válida, calcula a % global
                df[f"Part.{periodo_label} (%)"] = np.where(
                    total_receita_global > 0,
                    df[f"Receita {periodo_label}"] / total_receita_global * 100,
                    0
                )

            else:
                # Mantém o comportamento atual (por variantes dentro do produto)
                total_receita = df[f"Receita {periodo_label}"].sum() if df[f"Receita {periodo_label}"].notna().any() else 0
                df[f"Part.{periodo_label} (%)"] = np.where(
                    total_receita > 0,
                    df[f"Receita {periodo_label}"] / total_receita * 100,
                    0
                )

            return df[[label_nivel, qtd_col, f"Custo {periodo_label}", f"Receita {periodo_label}",
                       f"Lucro Bruto {periodo_label}", f"Part.{periodo_label} (%)"]]


        # =====================================================
        # 💡 Consolidação correta — evita duplicar produtos no modo "(Todos)"
        # =====================================================
        if produto_escolhido != "(Todos)":
            # Produto específico → mantém cálculo normal por variante
            df_a = calc_periodo(custos_base_A, "A", "Qtd A")
            df_b = calc_periodo(custos_base_B, "B", "Qtd B")
        else:
            # "(Todos)" → consolida direto da base de pedidos (por produto)
            def consolidar_por_produto_todos(pedidos, ini, fim, periodo_label):
                """
                Consolida pedidos por produto (sem duplicar variantes).
                Usa quantidades reais da Shopify e calcula custos, receita e lucro por produto.
                """
                df_filtro = pedidos[
                    (pedidos["created_at"].dt.date >= ini)
                    & (pedidos["created_at"].dt.date <= fim)
                ].copy()

                # 🔹 Normaliza nome do produto
                df_filtro["product_title"] = (
                    df_filtro["product_title"]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s{2,}", " ", regex=True)
                )

                # 🔹 Garante que o mesmo produto dentro de um mesmo pedido não duplique
                df_filtro = (
                    df_filtro
                    .sort_values(["order_id", "product_title"])
                    .drop_duplicates(subset=["order_id", "product_title"], keep="first")
                    .reset_index(drop=True)
                )

                # 🔹 Agrupa direto por produto
                agrup = (
                    df_filtro.groupby("product_title", as_index=False)
                    .agg({
                        "quantity": "sum",
                        "price": lambda x: np.average(
                            x, weights=df_filtro.loc[x.index, "quantity"]
                        ),
                    })
                    .rename(columns={
                        "product_title": "Produto",
                        "quantity": f"Qtd {periodo_label}",
                        "price": "Preço Médio"
                    })
                )

                # 🔗 Junta custo real por produto
                agrup = agrup.merge(
                    df_custos[["Produto", col_custo]].drop_duplicates("Produto"),
                    on="Produto", how="left"
                )

                agrup["Custo Unitário"] = pd.to_numeric(agrup[col_custo], errors="coerce").fillna(0)
                agrup[f"Custo {periodo_label}"] = agrup["Custo Unitário"] * agrup[f"Qtd {periodo_label}"]
                agrup[f"Receita {periodo_label}"] = agrup[f"Qtd {periodo_label}"] * agrup["Preço Médio"]
                agrup[f"Lucro Bruto {periodo_label}"] = (
                    agrup[f"Receita {periodo_label}"] - agrup[f"Custo {periodo_label}"]
                )

                # 💡 Investimento será distribuído mais adiante, no bloco Meta Ads
                return agrup

            # 👉 Cria df_a / df_b direto dos pedidos (sem investimento ainda)
            df_a = consolidar_por_produto_todos(pedidos, inicio_a, fim_a, "A")
            df_b = consolidar_por_produto_todos(pedidos, inicio_b, fim_b, "B")

            # =====================================================
            # 📊 Calcula participação global correta no modo "(Todos)"
            # =====================================================
            total_receita_global_a = df_a["Receita A"].sum()
            total_receita_global_b = df_b["Receita B"].sum()

            df_a["Part.A (%)"] = np.where(
                total_receita_global_a > 0,
                df_a["Receita A"] / total_receita_global_a * 100,
                0
            )
            df_b["Part.B (%)"] = np.where(
                total_receita_global_b > 0,
                df_b["Receita B"] / total_receita_global_b * 100,
                0
            )


        # =====================================================
        # 💸 Vincular investimento Meta Ads automaticamente
        # =====================================================
        from datetime import datetime

        # Datas formatadas para API Meta
        since_a, until_a = inicio_a.strftime("%Y-%m-%d"), fim_a.strftime("%Y-%m-%d")
        since_b, until_b = inicio_b.strftime("%Y-%m-%d"), fim_b.strftime("%Y-%m-%d")

        # Nome base do produto (ex.: "Flexlive", "KneePro", etc.)
        produto_nome = produto_escolhido.split(" - ")[0]

        try:
            # Busca investimento Meta por produto e período (usando credenciais automáticas)
            ads_a = fetch_insights_daily(
                act_id=act_id,
                token=token,
                api_version="v23.0",
                since_str=since_a,
                until_str=until_a,
                level="campaign",
                product_name=produto_nome
            )

            ads_b = fetch_insights_daily(
                act_id=act_id,
                token=token,
                api_version="v23.0",
                since_str=since_b,
                until_str=until_b,
                level="campaign",
                product_name=produto_nome
            )

            # Soma total de investimento de cada período
            invest_total_a = ads_a["spend"].sum() if not ads_a.empty else 0
            invest_total_b = ads_b["spend"].sum() if not ads_b.empty else 0

            # Função para distribuir investimento proporcional às vendas
            def distribuir_investimento(df, invest_total, qtd_col):
                total_qtd = df[qtd_col].sum()
                if total_qtd == 0:
                    df["Invest. (R$)"] = 0
                else:
                    df["Invest. (R$)"] = (df[qtd_col] / total_qtd) * invest_total
                return df

            # Aplica para os dois períodos
            df_a = distribuir_investimento(df_a, invest_total_a, "Qtd A")
            df_b = distribuir_investimento(df_b, invest_total_b, "Qtd B")

            # Feedback visual
            st.success(
                f"✅ Investimentos distribuídos automaticamente com base no gasto de anúncios Meta Ads ({produto_nome})"
            )

        except Exception as e:
            st.warning(f"⚠️ Não foi possível calcular investimento automático via Meta Ads. Detalhe: {e}")
            df_a["Invest. (R$)"] = 0
            df_b["Invest. (R$)"] = 0
        
        # -------------------------------------------------
        # 💲 Função auxiliar para formatar valores monetários
        # -------------------------------------------------
        def fmt_moeda(valor):
            """Formata número como moeda brasileira."""
            try:
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return valor

        # -------------------------------------------------
        # 💰 Exibir tabelas lado a lado (com formatação monetária)
        # -------------------------------------------------
        def calcular_roi_roas(df, periodo):
            """Calcula ROI, ROAS, Lucro Bruto e Lucro Líquido e adiciona linha TOTAL (produto ou variantes)."""
            df = df.copy()

            # -------------------------------------------------
            # 💸 ROI e ROAS individuais
            # -------------------------------------------------
            if "Invest. (R$)" in df.columns:
                df[f"ROI {periodo}"] = np.where(
                    df["Invest. (R$)"] > 0,
                    df[f"Lucro Bruto {periodo}"] / df["Invest. (R$)"],
                    np.nan
                )
                df[f"ROAS {periodo}"] = np.where(
                    df["Invest. (R$)"] > 0,
                    df[f"Receita {periodo}"] / df["Invest. (R$)"],
                    np.nan
                )

            # -------------------------------------------------
            # ✅ Calcula Lucro Líquido (abatendo investimento)
            # -------------------------------------------------
            df[f"Lucro Líquido {periodo}"] = df[f"Lucro Bruto {periodo}"] - df["Invest. (R$)"]

            # -------------------------------------------------
            # 🧾 Linha TOTAL consolidada (sempre mostrada)
            # -------------------------------------------------
            total_qtd = df[f"Qtd {periodo}"].sum()
            total_custo = df[f"Custo {periodo}"].sum()
            total_receita = df[f"Receita {periodo}"].sum()
            total_lucro = df[f"Lucro Bruto {periodo}"].sum()
            total_lucro_liq = df[f"Lucro Líquido {periodo}"].sum()
            total_invest = df["Invest. (R$)"].sum()

            if total_invest > 0:
                roi_total = total_lucro_liq / total_invest
                roas_total = total_receita / total_invest
            else:
                roi_total, roas_total = np.nan, np.nan

            total_row = pd.DataFrame([{
                label_nivel: "🧾 TOTAL",
                f"Qtd {periodo}": total_qtd,
                f"Custo {periodo}": total_custo,
                f"Receita {periodo}": total_receita,
                f"Lucro Bruto {periodo}": total_lucro,
                f"Lucro Líquido {periodo}": total_lucro_liq,
                "Invest. (R$)": total_invest,
                f"ROI {periodo}": roi_total,
                f"ROAS {periodo}": roas_total,
                f"Part.{periodo} (%)": 100.0
            }])

            # Junta o total no final
            df = pd.concat([df, total_row], ignore_index=True)
            return df

        # ✅ Aplica aos dois períodos
        df_a = calcular_roi_roas(df_a, "A")
        df_b = calcular_roi_roas(df_b, "B")


        # -------------------------------------------------
        # 🎨 Destaque visual da linha TOTAL
        # -------------------------------------------------
        def highlight_total(row):
            """Aplica o mesmo fundo do cabeçalho para a linha TOTAL."""
            if str(row[label_nivel]).strip().upper() == "🧾 TOTAL":
                return [
                    "background-color: #262730; font-weight: bold; color: white;"
                ] * len(row)
            return [""] * len(row)


        # -------------------------------------------------
        # 📆 Tabela Período A
        # -------------------------------------------------
        with col1:
            st.markdown("### 📆 Período A")
            styled_a = (
                df_a[[
                    label_nivel, "Qtd A", "Receita A", "Custo A",
                    "Lucro Bruto A", "Invest. (R$)", "Lucro Líquido A",
                    "ROI A", "ROAS A", "Part.A (%)"
                ]]
                .style
                .format({
                    "Qtd A": "{:.0f}",
                    "Custo A": fmt_moeda,
                    "Receita A": fmt_moeda,
                    "Lucro Bruto A": fmt_moeda,
                    "Lucro Líquido A": fmt_moeda,
                    "Invest. (R$)": fmt_moeda,
                    "ROI A": "{:.2f}x",
                    "ROAS A": "{:.2f}x",
                    "Part.A (%)": "{:.1f}%"
                })
                .set_properties(**{"text-align": "right"})
                .apply(highlight_total, axis=1)
            )
            st.dataframe(styled_a, use_container_width=True)

        # -------------------------------------------------
        # 📆 Tabela Período B
        # -------------------------------------------------
        with col2:
            st.markdown("### 📆 Período B")
            styled_b = (
                df_b[[
                    label_nivel, "Qtd B", "Receita B", "Custo B",
                    "Lucro Bruto B", "Invest. (R$)", "Lucro Líquido B",
                    "ROI B", "ROAS B", "Part.B (%)"
                ]]
                .style
                .format({
                    "Qtd B": "{:.0f}",
                    "Custo B": fmt_moeda,
                    "Receita B": fmt_moeda,
                    "Lucro Bruto B": fmt_moeda,
                    "Lucro Líquido B": fmt_moeda,
                    "Invest. (R$)": fmt_moeda,
                    "ROI B": "{:.2f}x",
                    "ROAS B": "{:.2f}x",
                    "Part.B (%)": "{:.1f}%"
                })
                .set_properties(**{"text-align": "right"})
                .apply(highlight_total, axis=1)
            )
            st.dataframe(styled_b, use_container_width=True)

        # -------------------------------------------------
        # 📈 Comparativo geral entre períodos
        # -------------------------------------------------
        st.subheader(f"📈 Comparativo entre Períodos (por {label_nivel.lower()})")

        import re

        # =====================================================
        # 🔍 Pareamento inteligente baseado em quantidade aproximada
        # =====================================================
        import re

        def extrair_qtd_pecas(nome):
            """Extrai a quantidade numérica (20, 30, 40, 60, 120...) do nome da variante."""
            if not isinstance(nome, str):
                return None
            match = re.search(r"(\d+)\s*(peças|unid|uni|pçs?)", nome.lower())
            return int(match.group(1)) if match else None

        # Cria colunas de quantidade
        df_a["qtd_variante"] = df_a[label_nivel].apply(extrair_qtd_pecas)
        df_b["qtd_variante"] = df_b[label_nivel].apply(extrair_qtd_pecas)

        # ----------------------------------------------
        # 🔁 Pareia o número mais próximo entre A e B (sem depender de 'variant_title')
        # ----------------------------------------------
        matches = []
        usadas_b = set()

        for _, row_a in df_a.iterrows():
            qtd_a = row_a.get("qtd_variante", None)
            if pd.isna(qtd_a):
                continue

            # define qual coluna usar como nome-base (Produto ou Variante)
            col_b_nome = label_nivel if label_nivel in df_b.columns else df_b.columns[0]

            # encontra as linhas B com qtd válida e ainda não pareadas
            df_b_valid = df_b[
                (~df_b["qtd_variante"].isna()) &
                (~df_b[col_b_nome].isin(usadas_b))
            ]

            if df_b_valid.empty:
                matches.append((row_a[label_nivel], None))
                continue

            # escolhe o número mais próximo
            idx_min = (df_b_valid["qtd_variante"] - qtd_a).abs().idxmin()
            match_b = df_b_valid.loc[idx_min]

            matches.append((row_a[label_nivel], match_b[col_b_nome]))
            usadas_b.add(match_b[col_b_nome])

        # adiciona variantes B que ficaram sem par
        col_b_nome = label_nivel if label_nivel in df_b.columns else df_b.columns[0]
        for _, row_b in df_b.iterrows():
            if row_b[col_b_nome] not in usadas_b:
                matches.append((None, row_b[col_b_nome]))

        # cria tabela final de correspondência
        corresp = pd.DataFrame(matches, columns=[f"{label_nivel} A", f"{label_nivel} B"])

        # --- Renomeia colunas para evitar conflito no merge
        df_a_pref = df_a.add_suffix("_A")
        df_b_pref = df_b.add_suffix("_B")

        # =====================================================
        # 🧩 Cria comparativo adaptativo — Produto vs Variante
        # =====================================================
        if produto_escolhido == "(Todos)":
            # -------------------------------------------------
            # 🔁 Modo "Todos" → compara produtos consolidados
            # -------------------------------------------------
            label_nivel = "Produto"

            comp = pd.merge(
                df_a, df_b,
                on="Produto", how="outer",
                suffixes=("_A", "_B")
            ).fillna(0)

            comp["Δ Qtd."] = comp["Qtd A"] - comp["Qtd B"]
            comp["Δ Custo"] = comp["Custo A"] - comp["Custo B"]
            comp["Δ Receita"] = comp["Receita A"] - comp["Receita B"]
            comp["Δ Lucro B."] = comp["Lucro Bruto A"] - comp["Lucro Bruto B"]
            comp["Δ Lucro Líq."] = comp["Lucro Líquido A"] - comp["Lucro Líquido B"]
            comp["Δ Invest."] = comp["Invest. (R$)_A"] - comp["Invest. (R$)_B"]
            comp["Δ ROI"] = comp["ROI A"] - comp["ROI B"]
            comp["Δ ROAS"] = comp["ROAS A"] - comp["ROAS B"]
            comp["Δ Part.(p.p)"] = comp["Part.A (%)"] - comp["Part.B (%)"]

            comp["Δ Qtd.(%)"] = np.where(
                comp["Qtd B"] > 0,
                (comp["Qtd A"] - comp["Qtd B"]) / comp["Qtd B"] * 100,
                np.nan
            )
            comp["Δ Receita(%)"] = np.where(
                comp["Receita B"] > 0,
                (comp["Receita A"] - comp["Receita B"]) / comp["Receita B"] * 100,
                np.nan
            )
            comp["Δ Lucro Líq.(%)"] = np.where(
                comp["Lucro Líquido B"] > 0,
                (comp["Lucro Líquido A"] - comp["Lucro Líquido B"]) / comp["Lucro Líquido B"] * 100,
                np.nan
            )

        else:
            # -------------------------------------------------
            # 🔁 Produto específico → compara variantes
            # -------------------------------------------------
            comp = (
                corresp
                .merge(df_a_pref, left_on=f"{label_nivel} A", right_on=f"{label_nivel}_A", how="left")
                .merge(df_b_pref, left_on=f"{label_nivel} B", right_on=f"{label_nivel}_B", how="left")
            )

        # =====================================================
        # 💡 Preenche corretamente nomes de variantes ausentes
        # =====================================================
        col_a = f"{label_nivel} A" if f"{label_nivel} A" in comp.columns else label_nivel
        col_b = f"{label_nivel} B" if f"{label_nivel} B" in comp.columns else label_nivel

        comp[col_a] = np.where(
            comp[col_a].isna() | (comp[col_a] == 0),
            comp[col_b],
            comp[col_a]
        )
        comp[col_b] = np.where(
            comp[col_b].isna() | (comp[col_b] == 0),
            comp[col_a],
            comp[col_b]
        )

        comp = comp.fillna(0)

        # =====================================================
        # 📊 Cálculos de diferenças e variações
        # =====================================================
        if produto_escolhido != "(Todos)":
            # 🔒 Garante que comp já exista antes dos cálculos
            if "comp" not in locals():
                st.error("❌ ERRO: A variável 'comp' não foi criada antes deste ponto.")
                st.stop()

            # -------------------------------------------------
            # 📈 Diferenças diretas (valores absolutos)
            # -------------------------------------------------
            comp["Δ Qtd."] = comp.get("Qtd A_A", 0) - comp.get("Qtd B_B", 0)
            comp["Δ Custo"] = comp.get("Custo A_A", 0) - comp.get("Custo B_B", 0)
            comp["Δ Lucro B."] = comp.get("Lucro Bruto A_A", 0) - comp.get("Lucro Bruto B_B", 0)
            comp["Δ Lucro Líq."] = comp.get("Lucro Líquido A_A", 0) - comp.get("Lucro Líquido B_B", 0)
            comp["Δ Receita"] = comp.get("Receita A_A", 0) - comp.get("Receita B_B", 0)
            comp["Δ Invest."] = comp.get("Invest. (R$)_A", 0) - comp.get("Invest. (R$)_B", 0)
            comp["Δ ROI"] = comp.get("ROI A_A", 0) - comp.get("ROI B_B", 0)
            comp["Δ ROAS"] = comp.get("ROAS A_A", 0) - comp.get("ROAS B_B", 0)
            comp["Δ Part.(p.p)"] = comp.get("Part.A (%)_A", 0) - comp.get("Part.B (%)_B", 0)

            # -------------------------------------------------
            # 📊 Diferenças percentuais (%)
            # -------------------------------------------------
            def safe_div(a, b):
                """Evita divisões por zero e retorna variação percentual"""
                return np.where(b != 0, (a - b) / b * 100, np.nan)

            comp["Δ Qtd.(%)"] = safe_div(comp.get("Qtd A_A", 0), comp.get("Qtd B_B", 0))
            comp["Δ Custo(%)"] = safe_div(comp.get("Custo A_A", 0), comp.get("Custo B_B", 0))
            comp["Δ Lucro B.(%)"] = safe_div(comp.get("Lucro Bruto A_A", 0), comp.get("Lucro Bruto B_B", 0))
            comp["Δ Lucro Líq.(%)"] = safe_div(comp.get("Lucro Líquido A_A", 0), comp.get("Lucro Líquido B_B", 0))
            comp["Δ Receita(%)"] = safe_div(comp.get("Receita A_A", 0), comp.get("Receita B_B", 0))
            comp["Δ Invest.(%)"] = safe_div(comp.get("Invest. (R$)_A", 0), comp.get("Invest. (R$)_B", 0))

            # -------------------------------------------------
            # 📊 Pontos percentuais (variações absolutas)
            # -------------------------------------------------
            comp["Δ ROI(p.p)"] = comp.get("ROI A_A", 0) - comp.get("ROI B_B", 0)
            comp["Δ ROAS(p.p)"] = comp.get("ROAS A_A", 0) - comp.get("ROAS B_B", 0)

        # =====================================================
        # 🧾 Ajuste de nome da coluna e remoção de TOTAL duplicado
        # =====================================================
        if produto_escolhido == "(Todos)":
            label_nivel = "Produto"
        else:
            label_nivel = "Variante"

        # Garante que a linha TOTAL apareça só uma vez
        col_ref = f"{label_nivel} A" if f"{label_nivel} A" in comp.columns else label_nivel
        comp = comp[~comp[col_ref].astype(str).str.contains("TOTAL", case=False, na=False)]

        # =====================================================
        # 📋 Garante que a linha TOTAL fique no final (modo seguro)
        # =====================================================
        col_ref = None
        for col in [f"{label_nivel} A", f"{label_nivel} B", label_nivel]:
            if col in comp.columns:
                col_ref = col
                break

        if col_ref is not None:
            mask_total = comp[col_ref].astype(str).str.contains("TOTAL", case=False, na=False)
            comp = pd.concat([comp[~mask_total], comp[mask_total]], ignore_index=True)
        else:
            # Fallback: se não encontrar nenhuma coluna válida
            mask_total = pd.Series([False] * len(comp))

        # =====================================================
        # 🎨 Estilo visual da tabela comparativa (com cor na linha TOTAL)
        # =====================================================
        def highlight_total(row):
            """Mantém o fundo escuro tanto na linha TOTAL quanto na Δ TOTAL."""
            col_check = f"{label_nivel} A" if f"{label_nivel} A" in comp.columns else label_nivel
            nome = str(row.get(col_check, "")).upper()

            if "TOTAL" in nome:  # cobre 🧾 TOTAL e 🧾 Δ TOTAL
                styles = []
                for col, val in row.items():
                    base = 'background-color: #262730; font-weight: bold;'
                    # Mantém verde/vermelho mesmo na linha TOTAL ou Δ TOTAL
                    if isinstance(val, (int, float)):
                        if val > 0:
                            styles.append(base + 'color: #00bf63;')
                        elif val < 0:
                            styles.append(base + 'color: #ff4d4d;')
                        else:
                            styles.append(base + 'color: white;')
                    else:
                        styles.append(base + 'color: white;')
                return styles

            return [''] * len(row)

        def highlight_variations(val):
            """Aplica verde para melhora e vermelho para piora."""
            try:
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #00bf63; font-weight: 600;'
                    elif val < 0:
                        return 'color: #ff4d4d; font-weight: 600;'
                elif isinstance(val, str) and any(ch in val for ch in ['+', '-', '%', 'x']):
                    num = float(val.replace('%', '').replace('x', '').replace('+', '').replace(',', '.').strip())
                    if num > 0:
                        return 'color: #00bf63; font-weight: 600;'
                    elif num < 0:
                        return 'color: #ff4d4d; font-weight: 600;'
            except Exception:
                pass
            return 'color: white;'

        # =====================================================
        # 📋 Define colunas dinamicamente (produto vs variante)
        # =====================================================
        if f"{label_nivel} A" in comp.columns and f"{label_nivel} B" in comp.columns:
            cols_base = [f"{label_nivel} A", f"{label_nivel} B"]
        else:
            cols_base = [label_nivel]

        cols_deltas = [
            "Δ Qtd.", "Δ Qtd.(%)",
            "Δ Custo", "Δ Custo(%)",
            "Δ Lucro B.", "Δ Lucro B.(%)",
            "Δ Lucro Líq.", "Δ Lucro Líq.(%)",
            "Δ Receita", "Δ Receita(%)",
            "Δ Invest.", "Δ Invest.(%)",
            "Δ ROI", "Δ ROAS", "Δ Part.(p.p)"
        ]

        # Filtra apenas colunas que realmente existem
        cols_existentes = [c for c in cols_base + cols_deltas if c in comp.columns]

        # =====================================================
        # 🧾 Adiciona linha Δ TOTAL consolidada (comparativo A vs B)
        # =====================================================
        if not comp.empty:
            total_cols_sum = [
                "Δ Qtd.", "Δ Custo", "Δ Lucro B.", "Δ Lucro Líq.",
                "Δ Receita", "Δ Invest."
            ]
            total_cols_media = [
                "Δ Qtd.(%)", "Δ Custo(%)", "Δ Lucro B.(%)",
                "Δ Lucro Líq.(%)", "Δ Receita(%)", "Δ Invest.(%)",
                "Δ ROI", "Δ ROAS", "Δ Part.(p.p)"
            ]

            # Soma direta dos valores absolutos
            total_data = {
                col: comp[col].sum() if col in comp.columns else np.nan
                for col in total_cols_sum
            }

            # Média ponderada ou simples (dependendo da disponibilidade)
            for col in total_cols_media:
                if col in comp.columns:
                    if "Receita A_A" in comp.columns and comp["Receita A_A"].sum() > 0:
                        pesos = comp["Receita A_A"] / comp["Receita A_A"].sum()
                        total_data[col] = np.average(comp[col], weights=pesos)
                    else:
                        total_data[col] = np.nanmean(comp[col])

            # Corrige rótulo (preenche tanto col_A quanto col_B)
            col_a = f"{label_nivel} A" if f"{label_nivel} A" in comp.columns else label_nivel
            col_b = f"{label_nivel} B" if f"{label_nivel} B" in comp.columns else label_nivel

            total_data[col_a] = "🧾 Δ TOTAL"
            total_data[col_b] = "🧾 Δ TOTAL"

            # Adiciona linha Δ TOTAL no final
            comp = pd.concat([comp, pd.DataFrame([total_data])], ignore_index=True)
        

        styled_comp = (
            comp[cols_existentes]
            .style
            .format({
                "Δ Qtd.": "{:.0f}",
                "Δ Qtd.(%)": "{:+.1f}%",
                "Δ Custo": fmt_moeda,
                "Δ Custo(%)": "{:+.1f}%",
                "Δ Lucro B.": fmt_moeda,
                "Δ Lucro B.(%)": "{:+.1f}%",
                "Δ Lucro Líq.": fmt_moeda,
                "Δ Lucro Líq.(%)": "{:+.1f}%",
                "Δ Receita": fmt_moeda,
                "Δ Receita(%)": "{:+.1f}%",
                "Δ Invest.": fmt_moeda,
                "Δ Invest.(%)": "{:+.1f}%",
                "Δ ROI": "{:+.2f}x",
                "Δ ROAS": "{:+.2f}x",
                "Δ Part.(p.p)": "{:+.1f}"
            })
            .applymap(highlight_variations, subset=[c for c in cols_deltas if c in comp.columns])
            .apply(highlight_total, axis=1)
            .set_properties(**{
                "text-align": "right",
                "font-size": "14px",
                "border-color": "rgba(255,255,255,0.1)"
            })
        )

        st.data_editor(styled_comp, use_container_width=True, disabled=True)


        # =====================================================
        # 🧾 Cria versão formatada da planilha para edição
        # =====================================================
        df_display = df_custos.copy()
        for col in ["Custo AliExpress (R$)", "Custo Estoque (R$)"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    if pd.notna(x) else ""
                )

        # -------------------------------------------------
        # 🔄 Função para atualizar planilha de custos
        # -------------------------------------------------
        def atualizar_planilha_custos(df):
            """Atualiza dados na planilha de custos no Google Sheets"""
            import gspread
            from google.oauth2.service_account import Credentials

            try:
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
                gcp_info = dict(st.secrets["gcp_service_account"])
                if isinstance(gcp_info.get("private_key"), str):
                    gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")

                creds = Credentials.from_service_account_info(gcp_info, scopes=scopes)
                client = gspread.authorize(creds)
                sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"]).sheet1

                df_safe = (
                    df.copy()
                    .fillna("")
                    .astype(str)
                    .replace("nan", "", regex=False)
                )

                body = [df_safe.columns.values.tolist()] + df_safe.values.tolist()
                sheet.batch_clear(["A:Z"])
                sheet.update(body)

                st.success("✅ Planilha atualizada com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao atualizar planilha: {e}")

        # =====================================================
        # 📝 Edição direta da planilha no app
        # =====================================================
        st.subheader("📝 Custos por Variante")

        edit_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("💾 Salvar alterações na planilha"):
            # ⚙️ Converte R$ 25,00 → 25.00 antes de enviar
            for col in ["Custo AliExpress (R$)", "Custo Estoque (R$)"]:
                if col in edit_df.columns:
                    edit_df[col] = (
                        edit_df[col]
                        .astype(str)
                        .str.replace("R$", "", regex=False)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                        .replace("", np.nan)
                        .astype(float)
                    )

            atualizar_planilha_custos(edit_df)
            st.cache_data.clear()
            st.rerun()
    
    # -------------------- ABA 2: FUNIL --------------------
    with tab_funnel:
        # === Moeda detectada e override opcional ===
        currency_detected = (df_daily["currency"].dropna().iloc[0]
                             if "currency" in df_daily.columns and not df_daily["currency"].dropna().empty else "BRL")
        col_curA, col_curB = st.columns([1, 2])
        with col_curA:
            use_brl_display = st.checkbox("Fixar exibição em BRL (símbolo R$)", value=True)

        currency_label = "BRL" if use_brl_display else currency_detected

        with col_curB:
            if use_brl_display and currency_detected != "BRL":
                st.caption("⚠️ Exibindo com símbolo **R$** apenas para **formatação visual**. "
                           "Os valores permanecem na moeda da conta.")

        st.caption(f"Moeda da conta detectada: **{currency_detected}** — Exibindo como: **{currency_label}**")

        # ---- Filtro por produto na Visão diária (AGORA DENTRO DA ABA) ----
        produto_sel_daily = st.selectbox(
            "Filtrar por produto (opcional)",
            ["(Todos)"] + PRODUTOS,
            key="daily_produto"
        )

        df_daily_view = _filter_by_product(df_daily, produto_sel_daily)

        # 🔹 FILTRO CORRETO DE PERÍODO (importante!)
        mask_period = (df_daily_view["date"] >= pd.Timestamp(since)) & (df_daily_view["date"] <= pd.Timestamp(until))
        df_daily_view = df_daily_view.loc[mask_period].copy()
        
        if df_daily_view.empty:
            st.info("Sem dados para o produto selecionado nesse período.")
            st.stop()

        if produto_sel_daily != "(Todos)":
            st.caption(f"🔎 Filtrando por produto: **{produto_sel_daily}**")

        # ========= KPIs do período =========
        tot_spend = float(df_daily_view["spend"].sum())
        tot_purch = float(df_daily_view["purchases"].sum())
        tot_rev   = float(df_daily_view["revenue"].sum())
        roas_g    = (tot_rev / tot_spend) if tot_spend > 0 else np.nan

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
            roas_txt = _fmt_ratio_br(roas_g) if pd.notnull(roas_g) else "—"
            st.markdown('<div class="kpi-card"><div class="small-muted">ROAS</div>'
                        f'<div class="big-number">{roas_txt}</div></div>',
                        unsafe_allow_html=True)

        st.divider()

        # ========= Série diária =========
        st.subheader("Série diária — Investimento e Conversão")
        daily = df_daily_view.groupby("date", as_index=False)[["spend", "revenue", "purchases"]].sum()
        daily_pt = daily.rename(columns={"spend": "Gasto", "revenue": "Faturamento"})
        st.line_chart(daily_pt.set_index("date")[["Faturamento", "Gasto"]])
        st.caption("Linhas diárias de Receita e Gasto. Vendas na tabela abaixo.")

        # ========= FUNIL (Período) — FUNIL VISUAL =========
        st.subheader("Funil do período (Total) — Cliques → LPV → Checkout → Add Pagamento → Compra")

        f_clicks = float(df_daily_view["link_clicks"].sum())
        f_lpv    = float(df_daily_view["lpv"].sum())
        f_ic     = float(df_daily_view["init_checkout"].sum())
        f_api    = float(df_daily_view["add_payment"].sum())
        f_pur    = float(df_daily_view["purchases"].sum())

        labels_total = ["Cliques", "LPV", "Checkout", "Add Pagamento", "Compra"]
        values_total = [int(round(f_clicks)), int(round(f_lpv)), int(round(f_ic)), int(round(f_api)), int(round(f_pur))]

        force_shape = st.checkbox("Forçar formato de funil (sempre decrescente)", value=True)
        values_plot = enforce_monotonic(values_total) if force_shape else values_total

        st.plotly_chart(
            funnel_fig(labels_total, values_plot, title="Funil do período"),
            use_container_width=True
        )

        core_rows = [
            ("LPV / Cliques",     _rate(values_total[1], values_total[0])),
            ("Checkout / LPV",    _rate(values_total[2], values_total[1])),
            ("Compra / Checkout", _rate(values_total[4], values_total[2])),
        ]
        extras_def = {
            "Add Pagto / Checkout": _rate(values_total[3], values_total[2]),
            "Compra / Add Pagto":   _rate(values_total[4], values_total[3]),
            "Compra / LPV":         _rate(values_total[4], values_total[1]),
            "Compra / Cliques":     _rate(values_total[4], values_total[0]),
            "Checkout / Cliques":   _rate(values_total[2], values_total[0]),
            "Add Pagto / LPV":      _rate(values_total[3], values_total[1]),
        }

        with st.expander("Comparar outras taxas (opcional)"):
            extras_selected = st.multiselect(
                "Escolha métricas adicionais para visualizar:",
                options=list(extras_def.keys()),
                default=[],
            )
        rows = core_rows + [(name, extras_def[name]) for name in extras_selected]
        sr = pd.DataFrame(rows, columns=["Taxa", "Valor"])
        sr["Valor"] = sr["Valor"].map(lambda x: f"{x*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                                      if pd.notnull(x) else "")
        base_h, row_h = 160, 36
        height = base_h + row_h * len(extras_selected)
        st.dataframe(sr, use_container_width=True, height=height)

        # ========= TAXAS POR DIA (variação diária — com banda saudável e resumo) =========
        st.markdown("### Taxas por dia — evolução e leitura guiada")

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
        daily_conv = (
            df_daily_view.groupby("date", as_index=False)[
                ["link_clicks","lpv","init_checkout","add_payment","purchases"]
            ].sum()
            .rename(columns={"link_clicks":"clicks","init_checkout":"checkout","add_payment":"addpay"})
        )

        # evita ruído (dias com pouquíssimos eventos)
        daily_conv = daily_conv[daily_conv["clicks"] >= min_clicks_day].copy()
        if daily_conv.empty:
            st.info("Sem dias suficientes após o filtro de cliques mínimos.")
        else:
            # taxas (frações 0–1)
            daily_conv["LPV/Cliques"]     = daily_conv.apply(lambda r: _safe_div(r["lpv"],       r["clicks"]),   axis=1)
            daily_conv["Checkout/LPV"]    = daily_conv.apply(lambda r: _safe_div(r["checkout"],  r["lpv"]),      axis=1)
            daily_conv["Compra/Checkout"] = daily_conv.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)

            ### ADD: calcula período anterior com mesma duração
            period_len = (until - since).days + 1
            prev_since = since - timedelta(days=period_len)
            prev_until = since - timedelta(days=1)

            df_prev = fetch_insights_daily(
                act_id=act_id,
                token=token,
                api_version=api_version,
                since_str=str(prev_since),
                until_str=str(prev_until),
                level=level,
                product_name=st.session_state.get("daily_produto")
            )

            daily_prev = pd.DataFrame()
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

            def _fmt_pct_series(s):  # 0–1 -> 0–100
                return (s*100).round(2)

            # helper geral do gráfico
            def _line_pct_banded(df, col, lo_pct, hi_pct, title):
                import plotly.graph_objects as go

                x = df["date"]
                y = (df[col] * 100).round(2)

                def _status(v):
                    if not pd.notnull(v):
                        return "sem"
                    v_pct = float(v) * 100.0  # fração -> %
                    if v_pct < lo_pct:
                        return "abaixo"
                    if v_pct > hi_pct:
                        return "acima"
                    return "dentro"

                status = df[col].map(_status).tolist()
                colors = [{"abaixo": "#dc2626", "dentro": "#16a34a", "acima": "#0ea5e9", "sem": "#9ca3af"}[s] for s in status]
                hover = [f"{title}<br>{d:%Y-%m-%d}<br>Taxa: {val:.2f}%" for d, val in zip(x, y.fillna(0))]

                fig = go.Figure()

                # banda saudável
                if show_band:
                    fig.add_shape(
                        type="rect",
                        xref="x", yref="y",
                        x0=x.min(), x1=x.max(),
                        y0=lo_pct, y1=hi_pct,
                        fillcolor="rgba(34,197,94,0.08)", line=dict(width=0),
                        layer="below"
                    )

                # fins de semana
                if mark_weekends:
                    for d in x:
                        if d.weekday() >= 5:
                            fig.add_shape(
                                type="rect", xref="x", yref="paper",
                                x0=d, x1=d + pd.Timedelta(days=1),
                                y0=0, y1=1,
                                line=dict(width=0),
                                fillcolor="rgba(2,132,199,0.06)"
                            )

                # série
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines+markers",
                    name="Diário",
                    marker=dict(size=7, color=colors),
                    line=dict(width=1.5, color="#1f77b4"),
                    hovertext=hover, hoverinfo="text"
                ))

                # período anterior (se houver)
                if not daily_prev.empty and col in daily_prev.columns:
                    x_aligned = df["date"].values[:len(daily_prev)]
                    y_prev = (daily_prev[col] * 100).round(2)
                    fig.add_trace(go.Scatter(
                        x=x_aligned, y=y_prev,
                        mode="lines",
                        name="Período anterior (sobreposto)",
                        line=dict(width=2.2, color="#ef4444", dash="dot")
                    ))

                fig.update_layout(
                    title=title,
                    yaxis_title="%",
                    xaxis_title="Data",
                    height=340,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=48, b=10),
                    separators=",.",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )

                # limites Y
                y_min = max(0, min(y.min(), lo_pct) - 5)
                y_max = min(100, max(y.max(), hi_pct) + 5)
                fig.update_yaxes(range=[y_min, y_max])

                # 🔧 hover e formato — DENTRO DA FUNÇÃO
                fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Taxa: %{y:.2f}%<extra></extra>")
                fig.update_layout(hovermode="x unified")
                fig.update_xaxes(hoverformat="%Y-%m-%d", showspikes=False)
                fig.update_yaxes(hoverformat=".2f%", ticksuffix="%", showspikes=False)

                return fig

            # ======== RESUMO DAS TAXAS (dinâmico por período) ========
            def _trend_vs_previous_period(series_vals: pd.Series,
                                          since_dt: date,
                                          until_dt: date,
                                          rate_name: str,
                                          produto_nome: str | None) -> tuple[float | None, float | None]:
                cur_mean = float(series_vals.mean() * 100.0) if series_vals.size else None
                period_len = (until_dt - since_dt).days + 1
                prev_since = since_dt - timedelta(days=period_len)
                prev_until = since_dt - timedelta(days=1)

                try:
                    df_prev = fetch_insights_daily(
                        act_id=act_id,
                        token=token,
                        api_version=api_version,
                        since_str=str(prev_since),
                        until_str=str(prev_until),
                        level=level,
                        product_name=produto_nome
                    )
                    if df_prev is not None and not df_prev.empty:
                        prev = (
                            df_prev.groupby("date", as_index=False)[
                                ["link_clicks", "lpv", "init_checkout", "purchases"]
                            ]
                            .sum()
                            .rename(columns={"link_clicks": "clicks", "init_checkout": "checkout"})
                        )
                        if rate_name == "LPV/Cliques":
                            prev["rate"] = prev.apply(lambda r: _safe_div(r["lpv"], r["clicks"]), axis=1)
                        elif rate_name == "Checkout/LPV":
                            prev["rate"] = prev.apply(lambda r: _safe_div(r["checkout"], r["lpv"]), axis=1)
                        else:  # "Compra/Checkout"
                            prev["rate"] = prev.apply(lambda r: _safe_div(r["purchases"], r["checkout"]), axis=1)

                        prev_mean = float(prev["rate"].mean() * 100.0) if not prev.empty else None
                    else:
                        prev_mean = None
                except Exception:
                    prev_mean = None

                if prev_mean is not None and cur_mean is not None:
                    return cur_mean, (cur_mean - prev_mean)

                # fallback: metade inicial vs. final do período atual
                n = series_vals.size
                if n == 0:
                    return cur_mean, None
                mid = max(1, n // 2)
                first_mean = float(series_vals.iloc[:mid].mean() * 100.0)
                last_mean = float(series_vals.iloc[mid:].mean() * 100.0) if n - mid > 0 else first_mean
                return cur_mean, (last_mean - first_mean)

            def _resume_box(df_rates: pd.DataFrame,
                            col: str,
                            lo_pct: int,
                            hi_pct: int,
                            label: str) -> None:
                """Mostra: média do período, % de dias dentro da banda e tendência vs período anterior."""
                vals = df_rates[col].dropna()
                if vals.empty:
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric(label, "—")
                    mcol2.metric("% dias dentro", "—")
                    mcol3.metric("Tendência (período)", "—")
                    return

                mean_pct = float(vals.mean() * 100.0)
                inside = float(((vals * 100.0 >= lo_pct) & (vals * 100.0 <= hi_pct)).mean() * 100.0)

                cur_mean, delta_pp = _trend_vs_previous_period(
                    series_vals=vals,
                    since_dt=since,
                    until_dt=until,
                    rate_name=label.split(" (")[0],  # tira o “(média)” do fim
                    produto_nome=st.session_state.get("daily_produto")
                )

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric(label, f"{mean_pct:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                mcol2.metric("% dias dentro", f"{inside:,.0f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                if delta_pp is None:
                    mcol3.metric("Tendência (período)", "—")
                else:
                    delta_txt = f"{delta_pp:,.2f} pp".replace(",", "X").replace(".", ",").replace("X", ".")
                    mcol3.metric("Tendência (período)", ("+" if delta_pp >= 0 else "") + delta_txt)

            # === Chame as três caixinhas de resumo ===
            st.markdown("**Resumo das taxas (período filtrado)**")
            _resume_box(daily_conv, "LPV/Cliques",     lpv_cli_low, lpv_cli_high, "LPV/Cliques (média)")
            _resume_box(daily_conv, "Checkout/LPV",    co_lpv_low,  co_lpv_high,  "Checkout/LPV (média)")
            _resume_box(daily_conv, "Compra/Checkout", buy_co_low,  buy_co_high,  "Compra/Checkout (média)")

            st.markdown("---")

            # === Gráficos lado a lado (mantém como estava) ===
            left, mid, right = st.columns(3)
            with left:
                st.plotly_chart(
                    _line_pct_banded(daily_conv, "LPV/Cliques", lpv_cli_low, lpv_cli_high, "LPV/Cliques"),
                    use_container_width=True
                )
            with mid:
                st.plotly_chart(
                    _line_pct_banded(daily_conv, "Checkout/LPV", co_lpv_low, co_lpv_high, "Checkout/LPV"),
                    use_container_width=True
                )
            with right:
                st.plotly_chart(
                    _line_pct_banded(daily_conv, "Compra/Checkout", buy_co_low, buy_co_high, "Compra/Checkout"),
                    use_container_width=True
                )

            st.caption(
                "Leitura: pontos **verdes** estão dentro da banda saudável, **vermelhos** abaixo e **azuis** acima. "
                "A área verde mostra o alvo; a linha vermelha tracejada mostra o **período anterior** para comparação. "
                "Fins de semana ganham fundo azul claro (opcional)."
            )

            # === NOTIFICAÇÃO DIDÁTICA DE ALOCAÇÃO DE VERBA =================================
            st.subheader("🔔 Para onde vai a verba? (recomendação automática)")

            # usa as MESMAS faixas saudáveis definidas no bloco "Taxas por dia"
            # lpv_cli_low/high, co_lpv_low/high, buy_co_low/high
            # adiciona apenas o volume mínimo para liberar "Escala"
            min_purchases_to_scale = st.number_input(
                "Compras mínimas para sugerir Escala (volume)",
                min_value=0, value=50, step=1
            )

            # taxas do período (a partir do funil total já calculado)
            r1 = _safe_div(values_total[1], values_total[0])   # LPV/Cliques
            r2 = _safe_div(values_total[2], values_total[1])   # Checkout/LPV
            r3 = _safe_div(values_total[4], values_total[2])   # Compra/Checkout

            # quedas absolutas por etapa (onde as pessoas “somem”)
            drop1 = max(0, values_total[0] - values_total[1])  # Cliques -> LPV (Criativo/LP)
            drop2 = max(0, values_total[1] - values_total[2])  # LPV -> Checkout (Interesse/Oferta)
            drop3 = max(0, values_total[2] - values_total[4])  # Checkout -> Compra (RMK/Pagamento)

            # helpers de status e forma de exibir “chips”
            def _band_status(val, lo, hi):
                if not pd.notnull(val):
                    return "sem_dado"
                v = val * 100
                if v < lo:
                    return "abaixo"
                if v > hi:
                    return "acima"
                return "dentro"

            def _chip(label, val, lo, hi):
                status = _band_status(val, lo, hi)
                if status == "abaixo":
                    return f"❌ **{label}** — {_fmt_pct_br(val)} (alvo {lo}–{hi}%)"
                if status == "dentro":
                    return f"✅ **{label}** — {_fmt_pct_br(val)} (dentro de {lo}–{hi}%)"
                if status == "acima":
                    return f"🟢 **{label}** — {_fmt_pct_br(val)} (acima de {hi}%)"
                return f"⛔ **{label}** — sem dados suficientes"

            # mapa didático das etapas (reaproveitando faixas)
            stages = {
                "Teste de criativo": {
                    "rate": r1, "lo": lpv_cli_low, "hi": lpv_cli_high, "drop": drop1,
                    "explain": "Perda entre Cliques → LPV (qualidade do clique, criativo, velocidade e UX da landing).",
                    "todo": [
                        "Testar variações de criativo (ângulo, thumb, 3s iniciais, CTA).",
                        "Melhorar tempo de carregamento e primeira dobra da LP.",
                        "Revisar promessa/título para alinhar com o anúncio."
                    ]
                },
                "Teste de interesse": {
                    "rate": r2, "lo": co_lpv_low, "hi": co_lpv_high, "drop": drop2,
                    "explain": "Perda entre LPV → Checkout (público/segmentação e proposta de valor).",
                    "todo": [
                        "Refinar públicos/lookalikes e excluir desinteressados.",
                        "Evidenciar prova social e benefícios acima do CTA.",
                        "Harmonizar oferta (preço/parcelas/bundle) com o público certo."
                    ]
                },
                "Remarketing": {
                    "rate": r3, "lo": buy_co_low, "hi": buy_co_high, "drop": drop3,
                    "explain": "Perda entre Checkout → Compra (confiança, meios de pagamento, follow-up).",
                    "todo": [
                        "RMK dinâmico com objeções, frete e garantia claros.",
                        "Oferecer alternativas de pagamento (pix/boleto/parcelas).",
                        "Recuperar carrinhos (e-mail/SMS/Whats) em até 24h."
                    ]
                }
            }

            # decide foco principal
            abaixos = {k: v for k, v in stages.items() if _band_status(v["rate"], v["lo"], v["hi"]) == "abaixo"}

            if abaixos:
                # se >1 abaixo, escolhe onde há maior queda absoluta de pessoas
                foco, foco_dat = max(abaixos.items(), key=lambda kv: kv[1]["drop"])
            else:
                total_purch = values_total[4]
                todas_ok = all(_band_status(v["rate"], v["lo"], v["hi"]) in ["dentro", "acima"] for v in stages.values())
                if todas_ok and total_purch >= min_purchases_to_scale:
                    foco, foco_dat = "Escala", {
                        "rate": None, "lo": None, "hi": None, "drop": 0,
                        "explain": "Taxas saudáveis e volume suficiente. Hora de aumentar alcance nas melhores campanhas."
                    }
                else:
                    # sem crítica clara: sugerir ganho de volume onde a queda é maior
                    foco, foco_dat = max(stages.items(), key=lambda kv: kv[1]["drop"])

            # intensidade (ajuda a sugerir % de verba)
            total_drop = max(1, drop1 + drop2 + drop3)  # evita divisão por zero
            share = foco_dat["drop"] / total_drop
            if share > 0.60:
                intensidade = "Alta"; faixa_verba = "↑ realocar **20–30%** do budget"
            elif share >= 0.30:
                intensidade = "Média"; faixa_verba = "↑ realocar **10–15%** do budget"
            else:
                intensidade = "Baixa"; faixa_verba = "↑ realocar **5–10%** do budget"

            # cartão-resumo
            st.markdown("---")
            colA, colB = st.columns([1, 2])

            with colA:
                st.markdown("**Taxas do período**")
                st.markdown(_chip("LPV/Cliques", r1, lpv_cli_low, lpv_cli_high))
                st.markdown(_chip("Checkout/LPV", r2, co_lpv_low,  co_lpv_high))
                st.markdown(_chip("Compra/Checkout", r3, buy_co_low,  buy_co_high))

            with colB:
                if foco == "Escala":
                    st.success(
                        f"**✅ Recomendação: Escala**\n\n"
                        f"- Motivo: {foco_dat['explain']}\n"
                        f"- Compras no período: **{_fmt_int_br(values_total[4])}** "
                        f"(mín. para escalar: **{_fmt_int_br(min_purchases_to_scale)}**)\n"
                        f"- Ação: aumentar orçamento nas campanhas com melhor ROAS; manter horários e públicos vencedores."
                    )
                else:
                    st.warning(
                        f"**⚠️ Recomendação: {foco}**\n\n"
                        f"- Motivo: {foco_dat['explain']}\n"
                        f"- Queda concentrada nessa etapa: **{_fmt_int_br(foco_dat['drop'])}** pessoas "
                        f"(intensidade **{intensidade}** → {faixa_verba})."
                    )
                    st.markdown("**O que fazer agora**")
                    for tip in foco_dat["todo"]:
                        st.markdown(f"- {tip}")

            # ajuda didática
            with st.expander("ℹ️ Como interpretar"):
                st.markdown(
                    """
- **LPV/Cliques** baixo → **Criativo/LP** (as pessoas clicam mas não engajam na página).
- **Checkout/LPV** baixo → **Interesse/Oferta** (as pessoas veem, mas não avançam).
- **Compra/Checkout** baixo → **Remarketing/Pagamento** (travou na finalização).
- Se tudo está saudável **e** há volume de compras → **Escala**.
                    """
                )

            # ========= COMPARATIVOS (Período A vs Período B) =========
            with st.expander("Comparativos — Período A vs Período B (opcional)", expanded=False):
                st.subheader("Comparativos — descubra o que mudou e por quê")

                period_len = (until - since).days + 1
                default_sinceA = since - timedelta(days=period_len)
                default_untilA = since - timedelta(days=1)

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Período A**")
                    sinceA = st.date_input("Desde (A)", value=default_sinceA, key="sinceA")
                    untilA = st.date_input("Até (A)", value=default_untilA, key="untilA")
                with colB:
                    st.markdown("**Período B**")
                    sinceB = st.date_input("Desde (B)", value=since, key="sinceB")
                    untilB = st.date_input("Até (B)", value=until, key="untilB")


                if sinceA > untilA or sinceB > untilB:
                    st.warning("Confira as datas: 'Desde' não pode ser maior que 'Até'.")
                else:
                    with st.spinner("Comparando períodos…"):
                        dfA = fetch_insights_daily(
                            act_id, token, api_version, str(sinceA), str(untilA), level,
                            product_name=produto_sel_daily
                        )
                        dfB = fetch_insights_daily(
                            act_id, token, api_version, str(sinceB), str(untilB), level,
                            product_name=produto_sel_daily
                        )

                    if dfA.empty or dfB.empty:
                        st.info("Sem dados em um dos períodos selecionados.")
                    else:
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

                        A = _agg(dfA)
                        B = _agg(dfB)

                        roasA = _safe_div(A["revenue"], A["spend"])
                        roasB = _safe_div(B["revenue"], B["spend"])
                        cpaA = _safe_div(A["spend"], A["purchases"])
                        cpaB = _safe_div(B["spend"], B["purchases"])
                        cpcA = _safe_div(A["spend"], A["clicks"])
                        cpcB = _safe_div(B["spend"], B["clicks"])

                        dir_map = {
                            "Valor usado": "neutral",
                            "Faturamento": "higher",
                            "Vendas": "higher",
                            "ROAS": "higher",
                            "CPC": "lower",
                            "CPA": "lower",
                        }
                        delta_map = {
                            "Valor usado": B["spend"] - A["spend"],
                            "Faturamento": B["revenue"] - A["revenue"],
                            "Vendas": B["purchases"] - A["purchases"],
                            "ROAS": (roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else np.nan,
                            "CPC": (cpcB - cpcA) if pd.notnull(cpcA) and pd.notnull(cpcB) else np.nan,
                            "CPA": (cpaB - cpaA) if pd.notnull(cpaA) and pd.notnull(cpaB) else np.nan,
                        }

                        kpi_rows = [
                            ("Valor usado", _fmt_money_br(A["spend"]), _fmt_money_br(B["spend"]), _fmt_money_br(B["spend"] - A["spend"])),
                            ("Faturamento", _fmt_money_br(A["revenue"]), _fmt_money_br(B["revenue"]), _fmt_money_br(B["revenue"] - A["revenue"])),
                            ("Vendas", _fmt_int_br(A["purchases"]), _fmt_int_br(B["purchases"]), _fmt_int_br(B["purchases"] - A["purchases"])),
                            ("ROAS", _fmt_ratio_br(roasA), _fmt_ratio_br(roasB), (_fmt_ratio_br(roasB - roasA) if pd.notnull(roasA) and pd.notnull(roasB) else "")),
                            ("CPC", _fmt_money_br(cpcA) if pd.notnull(cpcA) else "", _fmt_money_br(cpcB) if pd.notnull(cpcB) else "", _fmt_money_br(cpcB - cpcA) if pd.notnull(cpcA) and pd.notnull(cpcB) else ""),
                            ("CPA", _fmt_money_br(cpaA) if pd.notnull(cpaA) else "", _fmt_money_br(cpaB) if pd.notnull(cpaB) else "", _fmt_money_br(cpaB - cpaA) if pd.notnull(cpaA) and pd.notnull(cpaB) else ""),
                        ]
                        kpi_df_disp = pd.DataFrame(kpi_rows, columns=["Métrica", "Período A", "Período B", "Δ (B - A)"])

                        def _style_kpi(row):
                            metric = row["Métrica"]
                            d = delta_map.get(metric, np.nan)
                            rule = dir_map.get(metric, "neutral")
                            styles = [""] * len(row)
                            try:
                                idxB = list(row.index).index("Período B")
                                idxD = list(row.index).index("Δ (B - A)")
                            except Exception:
                                return styles
                            if pd.isna(d) or rule == "neutral" or d == 0:
                                return styles
                            better = (d > 0) if rule == "higher" else (d < 0)
                            color = "#16a34a" if better else "#dc2626"
                            weight = "700"
                            styles[idxB] = f"color:{color}; font-weight:{weight};"
                            styles[idxD] = f"color:{color}; font-weight:{weight};"
                            return styles

                        st.markdown("**KPIs do período (A vs B)**")
                        st.dataframe(kpi_df_disp.style.apply(_style_kpi, axis=1), use_container_width=True, height=260)

                        st.markdown("---")

                        # Taxas do funil
                        rates_num = pd.DataFrame({
                            "Taxa": ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout"],
                            "Período A": [
                                _safe_div(A["lpv"], A["clicks"]),
                                _safe_div(A["checkout"], A["lpv"]),
                                _safe_div(A["purchases"], A["checkout"]),
                            ],
                            "Período B": [
                                _safe_div(B["lpv"], B["clicks"]),
                                _safe_div(B["checkout"], B["lpv"]),
                                _safe_div(B["purchases"], B["checkout"]),
                            ],
                        })
                        rates_num["Δ"] = rates_num["Período B"] - rates_num["Período A"]

                        rates_disp = rates_num.copy()
                        for col in ["Período A", "Período B", "Δ"]:
                            rates_disp[col] = rates_disp[col].map(_fmt_pct_br)

                        st.markdown("**Taxas do funil (A vs B)**")
                        st.dataframe(rates_disp, use_container_width=True, height=180)

                        st.markdown("---")

        # ========= 📦 FUNIL POR CAMPANHA =========
        st.divider()
        st.header("📦 Funil por campanha")

        if level == "campaign":
            # 🔹 Filtra o período selecionado (inclui hoje se estiver dentro do range)
            today = pd.Timestamp.today().normalize()
            since_ts = pd.Timestamp(since)
            until_ts = pd.Timestamp(until)

            if until_ts >= today:
                mask_period = (df_daily_view["date"] >= since_ts) & (df_daily_view["date"] <= today)
                realtime_mode = True
            else:
                mask_period = (df_daily_view["date"] >= since_ts) & (df_daily_view["date"] <= until_ts)
                realtime_mode = False

            df_filtered = df_daily_view.loc[mask_period].copy()

            # 🔹 Atualiza dados de hoje se o período inclui a data atual
            if realtime_mode:
                with st.spinner("⏱️ Atualizando dados de hoje em tempo real (Meta Ads)..."):
                    try:
                        df_today_live = fetch_insights_daily(
                            act_id=act_id,
                            token=token,
                            api_version=api_version,
                            since_str=str(today),
                            until_str=str(today),
                            level=level,
                            product_name=None
                        )
                        if df_today_live is not None and not df_today_live.empty:
                            df_filtered = pd.concat([df_filtered, df_today_live], ignore_index=True)
                            df_filtered.drop_duplicates(subset=["date", "campaign_id"], inplace=True)
                            st.success("✅ Dados de hoje atualizados com sucesso!")
                        else:
                            st.info("Nenhum dado adicional encontrado para hoje (Meta ainda sincronizando).")
                    except Exception as e:
                        st.warning(f"⚠️ Falha ao atualizar dados de hoje: {e}")

            if df_filtered.empty:
                st.info("Sem dados de campanha no período selecionado.")
                st.stop()

            # 🔹 Detecta campanhas ativas (teve gasto > 0 no período)
            active_campaigns = (
                df_filtered.groupby("campaign_id")["spend"]
                .sum()
                .loc[lambda s: s > 0]
                .index
                .tolist()
            )

            df_active = df_filtered[df_filtered["campaign_id"].isin(active_campaigns)].copy()

            if df_active.empty:
                st.info("Nenhuma campanha ativa no período selecionado.")
                st.stop()

            # 🔹 Agrega e calcula KPIs
            agg_cols = ["spend", "link_clicks", "lpv", "init_checkout", "add_payment", "purchases", "revenue"]
            camp = df_active.groupby(["campaign_id", "campaign_name"], as_index=False)[agg_cols].sum()

            # ===== Divisões seguras linha a linha =====
            def safe_div(num, den):
                try:
                    return num / den if (pd.notnull(den) and den != 0) else np.nan
                except Exception:
                    return np.nan

            camp["ROAS"] = camp.apply(lambda r: safe_div(r["revenue"], r["spend"]), axis=1)
            camp["CPA"] = camp.apply(lambda r: safe_div(r["spend"], r["purchases"]), axis=1)
            camp["LPV/Cliques"] = camp.apply(lambda r: safe_div(r["lpv"], r["link_clicks"]), axis=1)
            camp["Checkout/LPV"] = camp.apply(lambda r: safe_div(r["init_checkout"], r["lpv"]), axis=1)
            camp["Compra/Checkout"] = camp.apply(lambda r: safe_div(r["purchases"], r["init_checkout"]), axis=1)

            disp = camp[[
                "campaign_name", "spend", "revenue", "purchases", "ROAS", "CPA",
                "LPV/Cliques", "Checkout/LPV", "Compra/Checkout"
            ]].rename(columns={
                "campaign_name": "Campanha",
                "spend": "Gasto",
                "revenue": "Faturamento",
                "purchases": "Vendas"
            })

            # 🔹 Formatação visual
            disp["Gasto"] = disp["Gasto"].map(_fmt_money_br)
            disp["Faturamento"] = disp["Faturamento"].map(_fmt_money_br)
            disp["Vendas"] = disp["Vendas"].map(_fmt_int_br)
            disp["ROAS"] = disp["ROAS"].map(_fmt_ratio_br)
            disp["CPA"] = disp["CPA"].map(_fmt_money_br)
            disp["LPV/Cliques"] = disp["LPV/Cliques"].map(_fmt_pct_br)
            disp["Checkout/LPV"] = disp["Checkout/LPV"].map(_fmt_pct_br)
            disp["Compra/Checkout"] = disp["Compra/Checkout"].map(_fmt_pct_br)

            disp = disp.sort_values(by="Faturamento", ascending=False)

            if realtime_mode:
                st.caption("⏱️ Modo tempo real ativado — exibindo dados parciais do dia atual.")

            st.dataframe(disp, use_container_width=True, height=420)

        else:
            st.info("Troque o nível para 'campaign' para visualizar o detalhamento por campanha.")


    # -------------------- ABA 2: HORÁRIOS (PRINCIPAL) --------------------
    with tab_daypart:

        # =================== CONTROLES INICIAIS ===================
        min_spend = st.number_input(
            "💰 Gasto mínimo por hora (R$)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="hr_min_spend"
        )

        safe_div = _safe_div

        if "df_hourly" in st.session_state and not st.session_state["df_hourly"].empty:
            d = st.session_state["df_hourly"].copy()
        else:
            with st.spinner("Carregando dados horários..."):
                d = fetch_insights_hourly(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(since),
                    until_str=str(until),
                    level=level,
                )
                st.session_state["df_hourly"] = d

        if d is None or d.empty:
            st.warning("Sem dados horários disponíveis para o período selecionado.")
            st.stop()

        # =================== FILTRO DE PRODUTO/CAMPANHA ===================
        if "campaign_name" in d.columns:
            prod_opts = ["(Todos)"] + sorted(d["campaign_name"].dropna().unique().tolist())
        else:
            prod_opts = ["(Todos)"]

        produto_sel_hr = st.selectbox(
            "🎯 Filtrar por produto/campanha (opcional)",
            options=prod_opts,
            index=0,
            key="sel_produto_hr"
        )

        if produto_sel_hr != "(Todos)":
            d = d[d["campaign_name"].str.contains(produto_sel_hr, case=False, na=False)].copy()
        else:
            with st.spinner("Carregando dados horários..."):
                d = fetch_insights_hourly(
                    act_id=act_id,
                    token=token,
                    api_version=api_version,
                    since_str=str(since),
                    until_str=str(until),
                    level=level,
                )
                st.session_state["df_hourly"] = d

        if d is None or d.empty:
            st.warning("Sem dados horários disponíveis para o período selecionado.")
            st.stop()

        # ============== 1) HEATMAP HORA × DIA (TOPO) ==============
        st.subheader("📆 Heatmap — Hora × Dia")
        cube_hm = d.groupby(["dow_label","hour"], as_index=False)[
            ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
        ].sum()
        cube_hm["roas"] = cube_hm.apply(lambda r: safe_div(r["revenue"], r["spend"]), axis=1)

        if min_spend > 0:
            cube_hm = cube_hm[cube_hm["spend"] >= min_spend]

        metric_hm = st.selectbox(
            "Métrica para o heatmap",
            ["Compras","Faturamento","Gasto","ROAS"],
            index=0,
            key="hm_metric_top"
        )

        mcol_hm = {
            "Compras": "purchases",
            "Faturamento": "revenue",
            "Gasto": "spend",
            "ROAS": "roas"
        }[metric_hm]

        if mcol_hm == "roas":
            pvt = cube_hm.groupby(["dow_label","hour"], as_index=False)[mcol_hm].mean()
        else:
            pvt = cube_hm.groupby(["dow_label","hour"], as_index=False)[mcol_hm].sum()

        order = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        pvt["dow_label"] = pd.Categorical(pvt["dow_label"], categories=order, ordered=True)
        pvt = pvt.sort_values(["dow_label","hour"])
        heat = pvt.pivot(index="dow_label", columns="hour", values=mcol_hm).fillna(0)

        hours_full = list(range(24))
        heat = heat.reindex(columns=hours_full, fill_value=0)
        heat.columns = list(range(24))

        fig_hm = go.Figure(data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorbar=dict(title=metric_hm),
            hovertemplate="Dia: %{y}<br>Hora: %{x}h<br>"+metric_hm+": %{z}<extra></extra>"
        ))
        fig_hm.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            separators=",."
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")

        # ============== 3) APANHADO GERAL POR HORA (período) ==============
        st.subheader("📦 Apanhado geral por hora (período selecionado)")
        cube_hr = d.groupby("hour", as_index=False)[
            ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
        ].sum()
        cube_hr["ROAS"] = cube_hr.apply(lambda r: safe_div(r["revenue"], r["spend"]), axis=1)

        if min_spend > 0:
            cube_hr = cube_hr[cube_hr["spend"] >= min_spend]

        top_hr = cube_hr.sort_values(["purchases","ROAS"], ascending=[False,False]).copy()
        show_cols = ["hour","ROAS","spend","revenue","link_clicks","lpv","init_checkout","add_payment","purchases"]

        disp_top = top_hr[show_cols].rename(columns={
            "hour":"Hora",
            "purchases":"Compras",
            "spend":"Valor usado",
            "revenue":"Valor de conversão"
        })
        disp_top["Valor usado"] = disp_top["Valor usado"].apply(_fmt_money_br)
        disp_top["Valor de conversão"] = disp_top["Valor de conversão"].apply(_fmt_money_br)
        disp_top["ROAS"] = disp_top["ROAS"].map(_fmt_ratio_br)

        st.dataframe(disp_top, use_container_width=True, height=360)

        fig_bar = go.Figure(go.Bar(
            x=cube_hr.sort_values("hour")["hour"],
            y=cube_hr.sort_values("hour")["purchases"]
        ))
        fig_bar.update_layout(
            title="Compras por hora (total do período)",
            xaxis_title="Hora do dia",
            yaxis_title="Compras",
            height=380,
            template="plotly_white",
            margin=dict(l=10, r=10, t=48, b=10),
            separators=",."
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info("Dica: use o 'Gasto mínimo' para filtrar horas com investimento muito baixo e evitar falsos positivos.")

        # ============== 2) TAXAS POR HORA (gráficos em cima + tabela abaixo) ==============
        st.subheader("🎯 Taxas por hora — médias diárias (sinais puros, com cap de funil)")

        cube_hr_all = d.groupby("hour", as_index=False)[
            ["link_clicks", "lpv", "init_checkout", "add_payment", "purchases"]
        ].sum()

        _hours_full = list(range(24))
        cube_hr_all = (
            cube_hr_all.set_index("hour")
                       .reindex(_hours_full, fill_value=0.0)
                       .rename_axis("hour")
                       .reset_index()
        )

        cube_hr_all["LPV_cap"] = np.minimum(cube_hr_all["lpv"], cube_hr_all["link_clicks"])
        cube_hr_all["Checkout_cap"] = np.minimum(cube_hr_all["init_checkout"], cube_hr_all["LPV_cap"])

        cube_hr_all["tx_lpv_clicks"]      = cube_hr_all.apply(lambda r: safe_div(r["LPV_cap"],      r["link_clicks"]),  axis=1)
        cube_hr_all["tx_checkout_lpv"]    = cube_hr_all.apply(lambda r: safe_div(r["Checkout_cap"], r["LPV_cap"]),      axis=1)
        cube_hr_all["tx_compra_checkout"] = cube_hr_all.apply(lambda r: safe_div(r["purchases"],    r["Checkout_cap"]), axis=1)

        show_cum = st.checkbox("Mostrar linha cumulativa (até a hora)", value=True, key="hr_show_cum")
        cum = cube_hr_all.sort_values("hour").copy()
        cum["cum_clicks"] = cum["link_clicks"].cumsum()
        cum["cum_lpv"]    = cum["lpv"].cumsum()
        cum["cum_ic"]     = cum["init_checkout"].cumsum()
        cum["cum_purch"]  = cum["purchases"].cumsum()

        cum["LPV_cap_cum"]      = np.minimum(cum["cum_lpv"], cum["cum_clicks"])
        cum["Checkout_cap_cum"] = np.minimum(cum["cum_ic"],  cum["LPV_cap_cum"])

        tx_lpv_clicks_cum      = np.divide(cum["LPV_cap_cum"],      cum["cum_clicks"],      out=np.full(len(cum), np.nan), where=cum["cum_clicks"]>0)
        tx_checkout_lpv_cum    = np.divide(cum["Checkout_cap_cum"], cum["LPV_cap_cum"],     out=np.full(len(cum), np.nan), where=cum["LPV_cap_cum"]>0)
        tx_compra_checkout_cum = np.divide(cum["cum_purch"],        cum["Checkout_cap_cum"],out=np.full(len(cum), np.nan), where=cum["Checkout_cap_cum"]>0)

        def _get_band_from_state(key, default_pair):
            v = st.session_state.get(key)
            return v if (isinstance(v, tuple) and len(v) == 2) else default_pair

        _lpv_lo_def, _lpv_hi_def = _get_band_from_state("tx_lpv_cli_band", (70, 85))
        _co_lo_def,  _co_hi_def  = _get_band_from_state("tx_co_lpv_band",  (10, 20))
        _buy_lo_def, _buy_hi_def = _get_band_from_state("tx_buy_co_band",  (25, 40))

        with st.expander("Ajustes de exibição das bandas (opcional)", expanded=True):
            show_band_hour = st.checkbox("Mostrar banda saudável (faixa alvo)", value=True, key="hr_show_band")
            b1, b2, b3 = st.columns(3)
            with b1:
                lpv_cli_low, lpv_cli_high = st.slider("LPV/Cliques alvo (%)", 0, 100, (_lpv_lo_def, _lpv_hi_def), 1, key="hr_band_lpv_clicks")
            with b2:
                co_lpv_low,  co_lpv_high  = st.slider("Checkout/LPV alvo (%)", 0, 100, (_co_lo_def,  _co_hi_def),  1, key="hr_band_checkout_lpv")
            with b3:
                buy_co_low,  buy_co_high  = st.slider("Compra/Checkout alvo (%)", 0, 100, (_buy_lo_def, _buy_hi_def), 1, key="hr_band_buy_checkout")

        st.markdown("### Resumo das taxas (período filtrado)")

        _last = cum.iloc[-1]
        _clicks_tot   = float(_last["cum_clicks"])
        _lpv_cap_tot  = float(_last["LPV_cap_cum"])
        _chk_cap_tot  = float(_last["Checkout_cap_cum"])
        _purch_tot    = float(_last["cum_purch"])

        avg_lpv_clicks   = safe_div(_lpv_cap_tot, _clicks_tot)
        avg_chk_lpv      = safe_div(_chk_cap_tot, _lpv_cap_tot)
        avg_buy_checkout = safe_div(_purch_tot,   _chk_cap_tot)

        def _pct(x): 
            return "–" if (x is None or np.isnan(x)) else f"{x*100:,.2f}%"

        card_css = """
        <style>
        .kpi-wrap {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; margin: 6px 0 12px;}
        .kpi {
          background: #0f172a;
          border: 1px solid #1f2937;
          border-radius: 12px; padding: 14px 16px;
        }
        .kpi h4 {margin: 0 0 6px; font-size: 0.92rem; color: #cbd5e1; font-weight: 600;}
        .kpi .val {font-size: 2rem; font-weight: 700; color: #ffffff; letter-spacing: .2px;}
        @media (max-width: 900px){ .kpi-wrap{grid-template-columns:1fr;}}
        </style>
        """
        st.markdown(card_css, unsafe_allow_html=True)

        cards_html = f"""
        <div class="kpi-wrap">
          <div class="kpi">
            <h4>LPV/Cliques (média)</h4>
            <div class="val">{_pct(avg_lpv_clicks)}</div>
          </div>
          <div class="kpi">
            <h4>Checkout/LPV (média)</h4>
            <div class="val">{_pct(avg_chk_lpv)}</div>
          </div>
          <div class="kpi">
            <h4>Compra/Checkout (média)</h4>
            <div class="val">{_pct(avg_buy_checkout)}</div>
          </div>
        </div>
        """
        st.markdown(cards_html, unsafe_allow_html=True)

        def _line_hour_pct(x, y, title, band_range=None, show_band=False, y_aux=None, aux_label="Cumulativa"):
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="lines+markers", name=title,
                hovertemplate=f"<b>{title}</b><br>Hora: %{{x}}h<br>Taxa: %{{y:.2f}}%<extra></extra>"
            ))
            if show_band and band_range and len(band_range) == 2:
                lo, hi = band_range
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=-0.5, x1=23.5, y0=lo, y1=hi,
                    fillcolor="rgba(34,197,94,0.10)", line=dict(width=0), layer="below"
                )
            if show_cum and y_aux is not None:
                fig.add_trace(go.Scatter(
                    x=x, y=y_aux, mode="lines", name=f"{aux_label}",
                    line=dict(width=3, color="#f59e0b")
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Hora do dia",
                yaxis_title="Taxa (%)",
                height=340,
                template="plotly_white",
                margin=dict(l=10, r=10, t=48, b=10),
                separators=",.",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5])
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            return fig

        x_hours = cube_hr_all["hour"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(
                _line_hour_pct(
                    x_hours, cube_hr_all["tx_lpv_clicks"]*100,
                    "LPV/Cliques (%)",
                    band_range=(lpv_cli_low, lpv_cli_high),
                    show_band=show_band_hour,
                    y_aux=tx_lpv_clicks_cum*100, aux_label="Cumulativa (até a hora)"
                ),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                _line_hour_pct(
                    x_hours, cube_hr_all["tx_checkout_lpv"]*100,
                    "Checkout/LPV (%)",
                    band_range=(co_lpv_low, co_lpv_high),
                    show_band=show_band_hour,
                    y_aux=tx_checkout_lpv_cum*100, aux_label="Cumulativa (até a hora)"
                ),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                _line_hour_pct(
                    x_hours, cube_hr_all["tx_compra_checkout"]*100,
                    "Compra/Checkout (%)",
                    band_range=(buy_co_low, buy_co_high),
                    show_band=show_band_hour,
                    y_aux=tx_compra_checkout_cum*100, aux_label="Cumulativa (até a hora)"
                ),
                use_container_width=True
            )

        st.markdown("---")

        # ============== 4) COMPARAR DOIS PERÍODOS (A vs B) — HORA A HORA ==============
        st.subheader("🆚 Comparar dois períodos (A vs B) — hora a hora")

        base_len = (until - since).days + 1
        default_sinceA = (since - timedelta(days=base_len))
        default_untilA = (since - timedelta(days=1))

        colA1, colA2, colB1, colB2 = st.columns(4)

        def _fmt_date(d):
            return d.strftime("%d/%m/%Y")
        
        with colA1:
            period_sinceA = st.date_input("Desde (A)", value=default_sinceA, key="cmp_sinceA", format="DD/MM/YYYY")
        with colA2:
            period_untilA = st.date_input("Até (A)", value=default_untilA, key="cmp_untilA", format="DD/MM/YYYY")
        with colB1:
            period_sinceB = st.date_input("Desde (B)", value=since, key="cmp_sinceB", format="DD/MM/YYYY")
        with colB2:
            period_untilB = st.date_input("Até (B)", value=until, key="cmp_untilB", format="DD/MM/YYYY")

        if period_sinceA > period_untilA or period_sinceB > period_untilB:
            st.warning("Confira as datas: em cada período, 'Desde' não pode ser maior que 'Até'.")
        else:
            union_since = min(period_sinceA, period_sinceB)
            union_until = max(period_untilA, period_untilB)
            level_union = "campaign"

            with st.spinner("Carregando dados por hora dos períodos selecionados…"):
                df_hourly_union = fetch_insights_hourly(
                    act_id=act_id, token=token, api_version=api_version,
                    since_str=str(union_since), until_str=str(union_until), level=level_union
                )

            if df_hourly_union is not None and not df_hourly_union.empty and produto_sel_hr != "(Todos)":
                mask_union = df_hourly_union["campaign_name"].str.contains(produto_sel_hr, case=False, na=False)
                df_hourly_union = df_hourly_union[mask_union].copy()

            if df_hourly_union is None or df_hourly_union.empty:
                st.info("Sem dados no intervalo combinado dos períodos selecionados.")
            else:
                d_cmp = df_hourly_union.dropna(subset=["hour"]).copy()
                d_cmp["hour"] = d_cmp["hour"].astype(int).clip(0, 23)
                d_cmp["date_only"] = d_cmp["date"].dt.date

                A_mask = (d_cmp["date_only"] >= period_sinceA) & (d_cmp["date_only"] <= period_untilA)
                B_mask = (d_cmp["date_only"] >= period_sinceB) & (d_cmp["date_only"] <= period_untilB)
                datA, datB = d_cmp[A_mask], d_cmp[B_mask]

                if datA.empty or datB.empty:
                    st.info("Sem dados em um dos períodos selecionados.")
                else:
                    agg_cols = ["spend","revenue","purchases","link_clicks","lpv","init_checkout","add_payment"]
                    gA = datA.groupby("hour", as_index=False)[agg_cols].sum()
                    gB = datB.groupby("hour", as_index=False)[agg_cols].sum()

                    merged = pd.merge(gA, gB, on="hour", how="outer", suffixes=(" (A)", " (B)")).fillna(0.0)
                    if min_spend > 0:
                        keep = (merged["spend (A)"] >= min_spend) | (merged["spend (B)"] >= min_spend)
                        merged = merged[keep]

                    if merged.empty:
                        st.info("Após o filtro de gasto mínimo, não sobraram horas para comparar.")
                    else:
                        hours_full = list(range(24))
                        merged = (
                            merged.set_index("hour")
                                  .reindex(hours_full, fill_value=0)
                                  .rename_axis("hour")
                                  .reset_index()
                        )

                        x = merged["hour"].astype(int)

                        barsA_max = (merged["spend (A)"] + merged["revenue (A)"]).max()
                        barsB_max = (merged["spend (B)"] + merged["revenue (B)"]).max()
                        bars_max = max(barsA_max, barsB_max)
                        bars_max = 1.05 * (bars_max if np.isfinite(bars_max) and bars_max > 0 else 1.0)

                        lineA_max = merged["purchases (A)"].max()
                        lineB_max = merged["purchases (B)"].max()
                        line_max = 1.05 * (max(lineA_max, lineB_max) if np.isfinite(max(lineA_max, lineB_max)) else 1.0)

                        COLOR_SPEND, COLOR_REVENUE, COLOR_LINE = "#E74C3C", "#3498DB", "#2ECC71"

                        def _build_barline(title_prefix, suffix, spend_col, rev_col, purch_col, period_label):
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            fig.add_trace(go.Bar(name=f"Gasto {suffix}", x=x, y=merged[spend_col],
                                                 marker_color=COLOR_SPEND, offsetgroup=suffix))
                            fig.add_trace(go.Bar(name=f"Faturamento {suffix}", x=x, y=merged[rev_col],
                                                 marker_color=COLOR_REVENUE, offsetgroup=suffix))
                            fig.add_trace(go.Scatter(name=f"Compras {suffix} — {period_label}", x=x,
                                                     y=merged[purch_col], mode="lines+markers",
                                                     line=dict(color=COLOR_LINE, width=2)), secondary_y=True)
                            fig.update_layout(
                                title=f"{title_prefix} {period_label} (Gasto + Faturamento + Compras)",
                                barmode="stack", bargap=0.15, template="plotly_white",
                                height=460, margin=dict(l=10, r=10, t=48, b=10), separators=",."
                            )
                            fig.update_xaxes(title_text="Hora do dia", tickmode="linear", tick0=0, dtick=1)
                            fig.update_yaxes(title_text="Valores (R$)", secondary_y=False, range=[0, bars_max])
                            fig.update_yaxes(title_text="Compras (unid.)", secondary_y=True, range=[0, line_max])
                            return fig

                        st.plotly_chart(_build_barline("Período A —", "(A)", "spend (A)", "revenue (A)", "purchases (A)",
                                                       f"{period_sinceA} a {period_untilA}"), use_container_width=True)
                        st.plotly_chart(_build_barline("Período B —", "(B)", "spend (B)", "revenue (B)", "purchases (B)",
                                                       f"{period_sinceB} a {period_untilB}"), use_container_width=True)

                        st.markdown("### 🔎 Insights — Período A")
                        a = merged.sort_values("hour").copy()
                        a_spend, a_rev, a_purch = a["spend (A)"], a["revenue (A)"], a["purchases (A)"]
                        a_roas_ser = np.where(a_spend > 0, a_rev / a_spend, np.nan)

                        a_tot_spend = float(a_spend.sum())
                        a_tot_rev = float(a_rev.sum())
                        a_tot_purch = int(round(float(a_purch.sum())))
                        a_roas = (a_tot_rev / a_tot_spend) if a_tot_spend > 0 else np.nan

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Valor usado (A)", _fmt_money_br(a_tot_spend))
                        c2.metric("Faturamento (A)", _fmt_money_br(a_tot_rev))
                        c3.metric("Vendas (A)", f"{a_tot_purch:,}".replace(",", "."))
                        c4.metric("ROAS (A)", _fmt_ratio_br(a_roas) if pd.notnull(a_roas) else "—")

                        h_best_purch = int(a.loc[a["purchases (A)"].idxmax(), "hour"]) if len(a_purch) and a_purch.max() > 0 else None
                        best_purch_val = int(a_purch.max()) if len(a_purch) else 0
                        mask_roasA = (a_spend >= float(min_spend)) & (a_spend > 0)
                        if mask_roasA.any():
                            roasA_vals = a_roas_ser.copy()
                            roasA_vals[~mask_roasA] = np.nan
                            h_best_roasA = int(a.loc[np.nanargmax(roasA_vals), "hour"])
                            best_roasA_val = float(np.nanmax(roasA_vals))
                        else:
                            h_best_roasA, best_roasA_val = None, np.nan

                        rollA = a_purch.rolling(3, min_periods=1).sum()
                        iA = int(rollA.idxmax()) if len(rollA) else 0
                        def _bA(ix): return int(a.loc[min(max(ix, 0), len(a)-1), "hour"])
                        winA_start, winA_mid, winA_end = _bA(iA-1), _bA(iA), _bA(iA+1)
                        winA_sum = int(rollA.max()) if len(rollA) else 0

                        wastedA = a[(a_spend > 0) & (a_purch == 0)]
                        wastedA_hours = ", ".join(f"{int(h)}h" for h in wastedA["hour"].tolist()) if not wastedA.empty else "—"

                        st.markdown(f"""
**Pontos-chave (A)**  
- 🕐 **Pico de compras:** **{str(h_best_purch)+'h' if h_best_purch is not None else '—'}** ({best_purch_val} compras).  
- 💹 **Melhor ROAS** (gasto ≥ R$ {min_spend:,.0f}): **{(str(h_best_roasA)+'h') if h_best_roasA is not None else '—'}** ({_fmt_ratio_br(best_roasA_val) if pd.notnull(best_roasA_val) else '—'}).  
- ⏱️ **Janela forte (3h):** **{winA_start}–{winA_end}h** (centro {winA_mid}h) somando **{winA_sum}** compras.  
- 🧯 **Horas com gasto e 0 compras:** {wastedA_hours}.
""".replace(",", "X").replace(".", ",").replace("X", "."))

                        st.markdown("### 🔎 Insights — Período B")
                        b = merged.sort_values("hour").copy()
                        b_spend, b_rev, b_purch = b["spend (B)"], b["revenue (B)"], b["purchases (B)"]
                        b_roas_ser = np.where(b_spend > 0, b_rev / b_spend, np.nan)

                        b_tot_spend = float(b_spend.sum())
                        b_tot_rev = float(b_rev.sum())
                        b_tot_purch = int(round(float(b_purch.sum())))
                        b_roas = (b_tot_rev / b_tot_spend) if b_tot_spend > 0 else np.nan

                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Valor usado (B)", _fmt_money_br(b_tot_spend))
                        d2.metric("Faturamento (B)", _fmt_money_br(b_tot_rev))
                        d3.metric("Vendas (B)", f"{b_tot_purch:,}".replace(",", "."))
                        d4.metric("ROAS (B)", _fmt_ratio_br(b_roas) if pd.notnull(b_roas) else "—")

                        h_best_purchB = int(b.loc[b["purchases (B)"].idxmax(), "hour"]) if len(b_purch) and b_purch.max() > 0 else None
                        best_purch_valB = int(b_purch.max()) if len(b_purch) else 0
                        mask_roasB = (b_spend >= float(min_spend)) & (b_spend > 0)
                        if mask_roasB.any():
                            roasB_vals = b_roas_ser.copy()
                            roasB_vals[~mask_roasB] = np.nan
                            h_best_roasB = int(b.loc[np.nanargmax(roasB_vals), "hour"])
                            best_roasB_val = float(np.nanmax(roasB_vals))
                        else:
                            h_best_roasB, best_roasB_val = None, np.nan

                        rollB = b_purch.rolling(3, min_periods=1).sum()
                        iB = int(rollB.idxmax()) if len(rollB) else 0
                        def _bB(ix): return int(b.loc[min(max(ix, 0), len(b)-1), "hour"])
                        winB_start, winB_mid, winB_end = _bB(iB-1), _bB(iB), _bB(iB+1)
                        winB_sum = int(rollB.max()) if len(rollB) else 0

                        wastedB = b[(b_spend > 0) & (b_purch == 0)]
                        wastedB_hours = ", ".join(f"{int(h)}h" for h in wastedB["hour"].tolist()) if not wastedB.empty else "—"

                        st.markdown(f"""
**Pontos-chave (B)**  
- 🕐 **Pico de compras:** **{(str(h_best_purchB)+'h') if h_best_purchB is not None else '—'}** ({best_purch_valB} compras).  
- 💹 **Melhor ROAS** (gasto ≥ R$ {min_spend:,.0f}): **{(str(h_best_roasB)+'h') if h_best_roasB is not None else '—'}** ({_fmt_ratio_br(best_roasB_val) if pd.notnull(best_roasB_val) else '—'}).  
- ⏱️ **Janela forte (3h):** **{winB_start}–{winB_end}h** (centro {winB_mid}h) somando **{winB_sum}** compras.  
- 🧯 **Horas com gasto e 0 compras:** {wastedB_hours}.
""".replace(",", "X").replace(".", ",").replace("X", "."))

                        st.info("💡 Sugestão: direcione orçamento para as horas com melhor ROAS e pause/teste criativos nas horas com gasto e 0 compras.")

        # -------------------- ABA 3: 📊 DETALHAMENTO --------------------
        with tab_detail:
            st.caption(
                "Explore por dimensão: Idade, Gênero, Idade+Gênero, País, Plataforma, "
                "Posicionamento, Dia e Hora. Há um modo 'Populares' com os TOP 5."
            )

            # ===== Filtros =====
            colf1, colf2 = st.columns([2, 1])
            with colf1:
                produto_sel_det = st.selectbox(
                    "Filtrar por produto (opcional)",
                    ["(Todos)"] + PRODUTOS,
                    key="det_produto",
                )
            with colf2:
                min_spend_det = st.slider(
                    "Gasto mínimo para considerar (R$)",
                    0.0, 2000.0, 0.0, 10.0,
                    key="det_min_spend",
                )

            dimensao = st.radio(
                "Dimensão",
                [
                    "Populares", "Idade", "Gênero", "Idade + Gênero",
                    "Região", "País", "Plataforma", "Posicionamento", "Dia da Semana",
                ],
                index=0,
                horizontal=True,
            )

            # ========= Helpers =========
            def _apply_prod_filter(df_base: pd.DataFrame) -> pd.DataFrame:
                if produto_sel_det and produto_sel_det != "(Todos)":
                    mask = df_base["campaign_name"].str.contains(produto_sel_det, case=False, na=False)
                    return df_base[mask].copy()
                return df_base

            def _ensure_cols_exist(df: pd.DataFrame) -> pd.DataFrame:
                for col in ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]:
                    if col not in df.columns:
                        df[col] = 0.0
                return df

            def _agg_and_format(df: pd.DataFrame, group_cols: list[str]):
                if df is None or df.empty:
                    return pd.DataFrame(), pd.DataFrame()
                df2 = _apply_prod_filter(df)
                if df2.empty:
                    return pd.DataFrame(), pd.DataFrame()
                df2 = _ensure_cols_exist(df2)

                agg_cols = ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
                g = df2.groupby(group_cols, dropna=False, as_index=False)[agg_cols].sum()
                g["ROAS"] = np.divide(
                    g["revenue"].astype(float),
                    g["spend"].replace(0, np.nan).astype(float)
                )

                if min_spend_det and float(min_spend_det) > 0:
                    g = g[g["spend"] >= float(min_spend_det)]
                if not g.empty:
                    g = g.sort_values(["purchases", "ROAS"], ascending=[False, False])
                return g, g.copy()

            def _bar_chart(x_labels, y_values, title, x_title, y_title):
                fig = go.Figure(go.Bar(x=x_labels, y=y_values, text=y_values, textposition="outside"))
                fig.update_layout(
                    title=title,
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    height=420,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=48, b=10),
                    separators=".,",
                )
                st.plotly_chart(fig, use_container_width=True)

            # === Cores e formatação de período ===
            COLOR_A = "#636EFA"   # azul
            COLOR_B = "#EF553B"   # laranja

            def _fmt_range_br(d1, d2) -> str:
                d1 = pd.to_datetime(str(d1)).date()
                d2 = pd.to_datetime(str(d2)).date()
                return f"{d1.strftime('%d/%m/%Y')} → {d2.strftime('%d/%m/%Y')}"

            # ========= Helpers p/ taxas =========
            def _rates_from_raw(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
                df = df.copy()
                df["LPV/Cliques"]        = df.apply(lambda r: _safe_div(r["lpv"], r["link_clicks"]), axis=1)
                df["Checkout/LPV"]       = df.apply(lambda r: _safe_div(r["init_checkout"], r["lpv"]), axis=1)
                df["Compra/Checkout"]    = df.apply(lambda r: _safe_div(r["purchases"], r["init_checkout"]), axis=1)
                df["Add Pagto/Checkout"] = df.apply(lambda r: _safe_div(r["add_payment"], r["init_checkout"]), axis=1)
                df["Compra/Add Pagto"]   = df.apply(lambda r: _safe_div(r["purchases"], r["add_payment"]), axis=1)
                return df

            # ========= POPULARES =========
            if dimensao == "Populares":
                base = df_daily.copy()
                base = _apply_prod_filter(base)
                base = _ensure_cols_exist(base)

                g = (
                    base.groupby(["campaign_id", "campaign_name"], as_index=False)[
                        ["spend", "revenue", "purchases", "link_clicks", "lpv", "init_checkout", "add_payment"]
                    ].sum()
                )
                g["ROAS"] = np.divide(
                    g["revenue"].astype(float),
                    g["spend"].replace(0, np.nan).astype(float)
                )

                if min_spend_det and float(min_spend_det) > 0:
                    g = g[g["spend"] >= float(min_spend_det)]

                top_comp = g.sort_values(["purchases", "ROAS"], ascending=[False, False]).head(5).copy()
                top_roas = g[g["spend"] > 0].sort_values("ROAS", ascending=False).head(5).copy()

                def _fmt_disp(df_):
                    out = df_.copy()
                    out["Valor usado"]        = out["spend"].apply(_fmt_money_br)
                    out["Valor de conversão"] = out["revenue"].apply(_fmt_money_br)
                    out["ROAS"]               = out["ROAS"].map(_fmt_ratio_br)
                    out.rename(
                        columns={
                            "campaign_name": "Campanha",
                            "purchases": "Compras",
                            "link_clicks": "Cliques",
                            "lpv": "LPV",
                            "init_checkout": "Checkout",
                            "add_payment": "Add Pagto",
                        },
                        inplace=True,
                    )
                    return out

                disp_comp = _fmt_disp(top_comp)[
                    ["Campanha", "Compras", "Valor usado", "Valor de conversão", "ROAS", "Cliques", "LPV", "Checkout", "Add Pagto"]
                ]
                disp_roas = _fmt_disp(top_roas)[
                    ["Campanha", "ROAS", "Compras", "Valor usado", "Valor de conversão", "Cliques", "LPV", "Checkout", "Add Pagto"]
                ]

                st.subheader("TOP 5 — Campanhas")
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(disp_comp, use_container_width=True, height=260)
                with c2:
                    st.dataframe(disp_roas, use_container_width=True, height=260)
                st.stop()

            # ========= DIA DA SEMANA =========
            if dimensao == "Dia da Semana":
                base = df_daily.copy()
                base = _apply_prod_filter(base)
                base = _ensure_cols_exist(base)

                # Detecta coluna de data automaticamente
                date_col = None
                for c in base.columns:
                    if "date" in c.lower() or "data" in c.lower():
                        date_col = c
                        break
                if not date_col:
                    st.error("Nenhuma coluna de data encontrada em df_daily.")
                    st.stop()

                # Converter datas e gerar nome do dia (sem depender de locale)
                base["date_start"] = pd.to_datetime(base[date_col], errors="coerce")
                base["Dia da Semana"] = base["date_start"].dt.day_name()
                traducao_dias = {
                    "Monday": "segunda-feira",
                    "Tuesday": "terça-feira",
                    "Wednesday": "quarta-feira",
                    "Thursday": "quinta-feira",
                    "Friday": "sexta-feira",
                    "Saturday": "sábado",
                    "Sunday": "domingo",
                }
                base["Dia da Semana"] = base["Dia da Semana"].map(traducao_dias)

                # Ordenar dias na sequência natural
                ordem_dias = [
                    "segunda-feira", "terça-feira", "quarta-feira",
                    "quinta-feira", "sexta-feira", "sábado", "domingo"
                ]
                base["Dia da Semana"] = pd.Categorical(base["Dia da Semana"], categories=ordem_dias, ordered=True)

                # Agregar dados principais
                agg_cols = ["spend", "revenue", "purchases"]
                g = base.groupby("Dia da Semana", dropna=False, as_index=False)[agg_cols].sum()
                g["ROAS"] = np.divide(
                    g["revenue"].astype(float),
                    g["spend"].replace(0, np.nan).astype(float)
                )
                g["Custo por Compra"] = np.divide(
                    g["spend"].astype(float),
                    g["purchases"].replace(0, np.nan).astype(float)
                )

                if min_spend_det and float(min_spend_det) > 0:
                    g = g[g["spend"] >= float(min_spend_det)]

                # Preenche dias faltantes com 0 (caso algum dia não tenha vendas)
                g = g.set_index("Dia da Semana").reindex(ordem_dias, fill_value=0).reset_index()

                # ====== VISUAL ======
                st.subheader("📊 Investimento × Vendas por Dia da Semana")

                def fmt_real(v):
                    return f"R${v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                # Barras = Compras
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=g["Dia da Semana"],
                    y=g["purchases"],
                    name="Compras",
                    marker_color="#1f77b4",
                    text=g["purchases"],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Compras: %{y}",
                    yaxis="y1",
                ))

                # Linha = Investimento
                custom_hover = []
                for _, row in g.iterrows():
                    hover_text = (
                        f"<b>{row['Dia da Semana']}</b><br>"
                        f"Investimento: {fmt_real(row['spend'])}<br>"
                        f"ROAS: {row['ROAS']:.2f}<br>"
                        f"Custo por compra: {fmt_real(row['Custo por Compra'])}"
                    )
                    custom_hover.append(hover_text)

                fig.add_trace(go.Scatter(
                    x=g["Dia da Semana"],
                    y=g["spend"],
                    name="Invest. (R$)",
                    mode="lines+markers+text",
                    text=[fmt_real(v) for v in g["spend"]],
                    textposition="top center",
                    marker_color="#ff7f0e",
                    line=dict(width=3),
                    hovertext=custom_hover,
                    hoverinfo="text",
                    yaxis="y2",
                ))

                fig.update_layout(
                    title=dict(
                        text="Relação entre Investimento e Vendas por Dia da Semana",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16)
                    ),
                    xaxis=dict(title="Dia da Semana"),
                    yaxis=dict(title="Compras", side="left", showgrid=False, zeroline=False),
                    yaxis2=dict(title="Invest. (R$)", overlaying="y", side="right", showgrid=False, zeroline=False),
                    legend=dict(
                        orientation="h",
                        x=0.5, y=-0.2,
                        xanchor="center", yanchor="top",
                        bgcolor="rgba(255,255,255,0.8)",
                        font=dict(color="black", size=12)
                    ),
                    height=480,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=60, b=80),
                    separators=".,",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)


                if not g.empty and g["spend"].sum() > 0:
                    media_roas = g["ROAS"].mean()
                    media_cpa = g["Custo por Compra"].mean()
                    best_roas = g.loc[g["ROAS"].idxmax()]
                    best_cpa = g.loc[g["Custo por Compra"].idxmin()]
                    best_pur = g.loc[g["purchases"].idxmax()]

                    st.markdown("### 🧠 Insights Automáticos (Período Selecionado)")
                    # ==== Melhores dias ====
                    best_roas = g.loc[g["ROAS"].idxmax()]
                    best_cpa = g.loc[g["Custo por Compra"].idxmin()]
                    best_pur = g.loc[g["purchases"].idxmax()]
                    # ==== Piores dias ====
                    worst_roas = g.loc[g["ROAS"].idxmin()]
                    worst_cpa = g.loc[g["Custo por Compra"].idxmax()]
                    worst_pur = g.loc[g["purchases"].idxmin()]

                    st.markdown("#### 🟢 Melhores Desempenhos")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div style="background-color:#E8F9E9;padding:18px;border-radius:12px;
                                    border:1px solid #6EE7B7;color:#1B4332;">
                            <h5 style="color:#1B4332;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                💰 Melhor Eficiência (ROAS)
                            </h5>
                            <b>{best_roas['Dia da Semana'].capitalize()}</b><br>
                            ROAS: <b>{best_roas['ROAS']:.2f}</b><br>
                            ↑ {(best_roas['ROAS']/media_roas - 1)*100:.1f}% acima da média<br>
                            Investimento: {fmt_real(best_roas['spend'])}<br>
                            CPA: {fmt_real(best_roas['Custo por Compra'])}<br>
                            Compras: {int(best_roas['purchases'])}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="background-color:#E8F9E9;padding:18px;border-radius:12px;
                                    border:1px solid #86EFAC;color:#1B4332;">
                            <h5 style="color:#1B4332;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                ⚡ Maior Volume de Vendas
                            </h5>
                            <b>{best_pur['Dia da Semana'].capitalize()}</b><br>
                            Compras: <b>{int(best_pur['purchases'])}</b><br>
                            ROAS: {best_pur['ROAS']:.2f}<br>
                            CPA: {fmt_real(best_pur['Custo por Compra'])}<br>
                            Inv.: {fmt_real(best_pur['spend'])}
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="background-color:#E8F9E9;padding:18px;border-radius:12px;
                                    border:1px solid #A7F3D0;color:#1B4332;">
                            <h5 style="color:#1B4332;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                💸 Melhor Rentabilidade
                            </h5>
                            <b>{best_cpa['Dia da Semana'].capitalize()}</b><br>
                            CPA: <b>{fmt_real(best_cpa['Custo por Compra'])}</b><br>
                            ↓ {(1 - best_cpa['Custo por Compra']/media_cpa)*100:.1f}% abaixo da média<br>
                            ROAS: {best_cpa['ROAS']:.2f}<br>
                            Compras: {int(best_cpa['purchases'])}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 🔴 Piores Desempenhos")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div style="background-color:#FDECEC;padding:18px;border-radius:12px;
                                    border:1px solid #FCA5A5;color:#7F1D1D;">
                            <h5 style="color:#7F1D1D;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                📉 Pior Eficiência (ROAS)
                            </h5>
                            <b>{worst_roas['Dia da Semana'].capitalize()}</b><br>
                            ROAS: <b>{worst_roas['ROAS']:.2f}</b><br>
                            ↓ {(1 - worst_roas['ROAS']/media_roas)*100:.1f}% abaixo da média<br>
                            Inv.: {fmt_real(worst_roas['spend'])}<br>
                            CPA: {fmt_real(worst_roas['Custo por Compra'])}<br>
                            Compras: {int(worst_roas['purchases'])}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="background-color:#FDECEC;padding:18px;border-radius:12px;
                                    border:1px solid #F87171;color:#7F1D1D;">
                            <h5 style="color:#7F1D1D;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                🐢 Menor Volume de Vendas
                            </h5>
                            <b>{worst_pur['Dia da Semana'].capitalize()}</b><br>
                            Compras: <b>{int(worst_pur['purchases'])}</b><br>
                            ROAS: {worst_pur['ROAS']:.2f}<br>
                            CPA: {fmt_real(worst_pur['Custo por Compra'])}<br>
                            Inv.: {fmt_real(worst_pur['spend'])}
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="background-color:#FDECEC;padding:18px;border-radius:12px;
                                    border:1px solid #FCA5A5;color:#7F1D1D;">
                            <h5 style="color:#7F1D1D;font-weight:700;font-size:15px;
                                       margin:0 0 6px 0;">
                                🚨 Pior Rentabilidade (Maior CPA)
                            </h5>
                            <b>{worst_cpa['Dia da Semana'].capitalize()}</b><br>
                            CPA: <b>{fmt_real(worst_cpa['Custo por Compra'])}</b><br>
                            ↑ {(worst_cpa['Custo por Compra']/media_cpa - 1)*100:.1f}% acima da média<br>
                            ROAS: {worst_cpa['ROAS']:.2f}<br>
                            Compras: {int(worst_cpa['purchases'])}
                        </div>
                        """, unsafe_allow_html=True)

                    st.caption("Essas métricas consideram apenas o período e filtros aplicados.")

                    # ====== RANKING GERAL POR DESEMPENHO ======
                    st.markdown("### 🏆 Ranking Geral — Desempenho Consolidado")

                    df_rank = g.copy()
                    df_rank["score_vendas"] = (df_rank["purchases"] - df_rank["purchases"].min()) / (df_rank["purchases"].max() - df_rank["purchases"].min() + 1e-9)
                    df_rank["score_roas"] = (df_rank["ROAS"] - df_rank["ROAS"].min()) / (df_rank["ROAS"].max() - df_rank["ROAS"].min() + 1e-9)
                    df_rank["score_invest"] = (df_rank["spend"].max() - df_rank["spend"]) / (df_rank["spend"].max() - df_rank["spend"].min() + 1e-9)

                    PESO_VENDAS = 0.35
                    PESO_ROAS = 0.50
                    PESO_INVEST = 0.15

                    df_rank["score_final"] = (
                        df_rank["score_vendas"] * PESO_VENDAS
                        + df_rank["score_roas"] * PESO_ROAS
                        + df_rank["score_invest"] * PESO_INVEST
                    )

                    df_rank = df_rank.sort_values("score_final", ascending=False).reset_index(drop=True)
                    df_rank["Posição"] = df_rank.index + 1

                    df_rank["Investimento"] = df_rank["spend"].apply(fmt_real)
                    df_rank["Faturamento"] = df_rank["revenue"].apply(fmt_real)
                    df_rank["Custo por Compra"] = df_rank["Custo por Compra"].apply(fmt_real)
                    df_rank["ROAS"] = df_rank["ROAS"].round(2)
                    df_rank["Compras"] = df_rank["purchases"].astype(int)
                    df_rank["Score (%)"] = (df_rank["score_final"] * 100).round(1)

                    disp_rank = df_rank[
                        ["Posição", "Dia da Semana", "Compras", "Investimento", "Faturamento", "ROAS", "Custo por Compra", "Score (%)"]
                    ]

                    def _highlight_rank(row):
                        if row["Posição"] == 1:
                            return ['background-color: #d1fae5; font-weight: bold; color: #065f46;'] * len(row)
                        elif row["Posição"] == 2:
                            return ['background-color: #fef3c7; font-weight: bold; color: #92400e;'] * len(row)
                        elif row["Posição"] == 3:
                            return ['background-color: #dbeafe; font-weight: bold; color: #1e3a8a;'] * len(row)
                        elif row["Posição"] == len(df_rank):
                            return ['background-color: #fee2e2; color: #991b1b;'] * len(row)
                        else:
                            return [''] * len(row)

                    st.dataframe(
                        disp_rank.style.apply(_highlight_rank, axis=1),
                        use_container_width=True,
                        height=380,
                    )

                    st.caption("Ranking pondera Compras (35%), ROAS (50%) e Investimento (15%) — equilíbrio entre eficiência e volume.")

                # ====== TABELA ======
                disp = g.copy()
                disp["Investimento"] = disp["spend"].apply(_fmt_money_br)
                disp["Faturamento"] = disp["revenue"].apply(_fmt_money_br)
                disp["ROAS"] = disp["ROAS"].map(_fmt_ratio_br)
                disp["Custo por Compra"] = disp["Custo por Compra"].apply(_fmt_money_br)
                disp["Compras"] = disp["purchases"].astype(int)

                st.markdown("### 🧾 Detalhamento por Dia da Semana")
                st.dataframe(
                    disp[["Dia da Semana", "Compras", "Investimento", "Faturamento", "ROAS", "Custo por Compra"]],
                    use_container_width=True,
                    height=380,
                )
                st.stop()

            # ========= DEMAIS DIMENSÕES =========
            dim_to_breakdowns = {
                "Idade": ["age"],
                "Gênero": ["gender"],
                "Idade + Gênero": ["age", "gender"],
                "Região": ["region"],
                "País": ["country"],
                "Plataforma": ["publisher_platform"],
                "Posicionamento": ["publisher_platform", "platform_position"],
            }

            level_bd = level
            if dimensao == "Posicionamento" and level_bd not in ["adset", "ad"]:
                level_bd = "adset"

            if dimensao in dim_to_breakdowns:
                bks = dim_to_breakdowns[dimensao]
                df_bd = fetch_insights_breakdown(
                    act_id, token, api_version, str(since), str(until),
                    bks, level_bd, product_name=st.session_state.get("det_produto"),
                )

                if df_bd.empty:
                    st.info(f"Sem dados para {dimensao} no período/filtro.")
                    st.stop()

                rename_map = {
                    "age": "Idade",
                    "gender": "Gênero",
                    "region": "Região",
                    "country": "País",
                    "publisher_platform": "Plataforma",
                    "platform_position": "Posicionamento",
                }
                group_cols = [rename_map.get(c, c) for c in bks]

                raw, disp = _agg_and_format(df_bd.rename(columns=rename_map), group_cols)
                if disp.empty:
                    st.info(f"Sem dados para {dimensao} após aplicar filtros.")
                    st.stop()

                st.subheader(f"Desempenho por {dimensao}")

                # ----- gráfico -----
                if len(group_cols) == 1:
                    xlab = group_cols[0]
                    _bar_chart(raw[xlab], raw["purchases"], f"Compras por {xlab}", xlab, "Compras")
                else:
                    idx, col = group_cols
                    pvt = raw.pivot_table(index=idx, columns=col, values="purchases", aggfunc="sum").fillna(0)
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=pvt.values,
                            x=pvt.columns.astype(str),
                            y=pvt.index.astype(str),
                            colorbar=dict(title="Compras"),
                            hovertemplate=(f"{idx}: " + "%{y}<br>" + f"{col}: " + "%{x}<br>" + "Compras: %{z}<extra></extra>"),
                        )
                    )
                    fig.update_layout(
                        title=f"Heatmap — Compras por {idx} × {col}",
                        height=460,
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=48, b=10),
                        separators=".,",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ====== COMPARAR PERÍODOS ======
                st.markdown("### Comparar períodos A vs B (Taxas em p.p.)")
                from datetime import timedelta
                since_dt = pd.to_datetime(str(since)).date()
                until_dt = pd.to_datetime(str(until)).date()
                delta = until_dt - since_dt

                colp1, colp2 = st.columns(2)
                with colp1:
                    perA = st.date_input("Período A", (since_dt, until_dt), key="perA_det_tbl", format="DD/MM/YYYY")
                with colp2:
                    default_b_end = since_dt - timedelta(days=1)
                    default_b_start = default_b_end - delta
                    perB = st.date_input("Período B", (default_b_start, default_b_end), key="perB_det_tbl", format="DD/MM/YYYY")

                since_A, until_A = perA
                since_B, until_B = perB

                def _load_rates_for_range(d1, d2):
                    dfR = fetch_insights_breakdown(
                        act_id, token, api_version, str(d1), str(d2),
                        bks, level_bd, product_name=st.session_state.get("det_produto"),
                    )
                    rawR, _ = _agg_and_format(dfR.rename(columns=rename_map), group_cols)
                    return _rates_from_raw(rawR, group_cols)

                raw_A = _load_rates_for_range(since_A, until_A)
                raw_B = _load_rates_for_range(since_B, until_B)

                comp_num = pd.merge(raw_A, raw_B, on=group_cols, how="outer", suffixes=(" A", " B")).fillna(0)
                rate_specs = ["LPV/Cliques", "Checkout/LPV", "Compra/Checkout", "Add Pagto/Checkout", "Compra/Add Pagto"]

                deltas = comp_num[group_cols].copy()
                for label in rate_specs:
                    a_col, b_col = f"{label} A", f"{label} B"
                    if a_col in comp_num.columns and b_col in comp_num.columns:
                        deltas[f"Δ {label} (p.p.)"] = (comp_num[a_col] - comp_num[b_col]) * 100

                POS_BG = "rgba(22, 163, 74, 0.45)"
                NEG_BG = "rgba(239, 68, 68, 0.45)"

                def _bg_sign(val):
                    try:
                        v = float(val)
                    except Exception:
                        return ""
                    if v > 0:
                        return f"background-color: {POS_BG}; font-weight: 700;"
                    if v < 0:
                        return f"background-color: {NEG_BG}; font-weight: 700;"
                    return ""

                def _fmt_pp(v):
                    if pd.isna(v) or np.isinf(v):
                        return "—"
                    sign = "+" if v >= 0 else ""
                    s = f"{sign}{v:.1f}".replace(".", ",")
                    return f"{s} p.p."

                pp_cols = [c for c in deltas.columns if c.endswith("(p.p.)")]

                styled = (
                    deltas.style
                    .applymap(_bg_sign, subset=pp_cols)
                    .format({c: _fmt_pp for c in pp_cols})
                    .set_properties(subset=pp_cols, **{"text-align": "center", "padding": "6px"})
                )

                st.table(styled)
                st.caption(
                    f"Período A: **{_fmt_range_br(since_A, until_A)}** | "
                    f"Período B: **{_fmt_range_br(since_B, until_B)}**"
                )


# =====================================================
# 📦 DASHBOARD – LOGÍSTICA
# =====================================================
if menu == "📦 Dashboard – Logística":

    # =====================================================
    # 🧭 Cabeçalho fixo principal
    # =====================================================
    st.title("📦 DASHBOARD — LOGÍSTICA")
    st.caption("Visualização completa de pedidos, estoque, entregas e indicadores.")

    # =====================================================
    # 🗂️ Abas principais da Logística
    # =====================================================
    aba1, aba2, aba3 = st.tabs([
        "📋 Controle Operacional",
        "📦 Estoque",
        "🚚 Entregas"
    ])

    # =====================================================
    # 📋 ABA 1 — CONTROLE OPERACIONAL
    # =====================================================
    with aba1:
        # -------------------------------------------------
        # 🧭 SIDEBAR — Filtro lateral de período
        # -------------------------------------------------
        st.sidebar.header("📅 Período rápido")

        hoje = datetime.now(APP_TZ).date()  

        opcoes_periodo = [
            "Hoje", "Ontem", "Últimos 7 dias", "Últimos 14 dias",
            "Últimos 30 dias", "Últimos 90 dias", "Esta semana",
            "Este mês", "Máximo", "Personalizado"
        ]

        escolha_periodo = st.sidebar.radio("Selecione:", opcoes_periodo, index=0)

        if escolha_periodo == "Hoje":
            start_date, end_date = hoje, hoje
        elif escolha_periodo == "Ontem":
            start_date, end_date = hoje - timedelta(days=1), hoje - timedelta(days=1)
        elif escolha_periodo == "Últimos 7 dias":
            start_date, end_date = hoje - timedelta(days=7), hoje - timedelta(days=1)
        elif escolha_periodo == "Últimos 14 dias":
            start_date, end_date = hoje - timedelta(days=14), hoje - timedelta(days=1)
        elif escolha_periodo == "Últimos 30 dias":
            start_date, end_date = hoje - timedelta(days=30), hoje - timedelta(days=1)
        elif escolha_periodo == "Últimos 90 dias":
            start_date, end_date = hoje - timedelta(days=90), hoje - timedelta(days=1)
        elif escolha_periodo == "Esta semana":
            start_date, end_date = hoje - timedelta(days=hoje.weekday()), hoje
        elif escolha_periodo == "Este mês":
            start_date = hoje.replace(day=1)
            end_date = hoje
        elif escolha_periodo == "Máximo":
            start_date = date(2020, 1, 1)
            end_date = hoje
        else:
            periodo = st.sidebar.date_input("📆 Selecione o intervalo:", (hoje, hoje), format="DD/MM/YYYY")
            if isinstance(periodo, tuple) and len(periodo) == 2:
                start_date, end_date = periodo
            else:
                st.sidebar.warning("🟡 Selecione o fim do período.")
                st.stop()

        st.sidebar.markdown(f"**Desde:** {start_date}  \n**Até:** {end_date}")

        # -------------------------------------------------
        # 🔍 Busca rápida (no topo)
        # -------------------------------------------------
        st.subheader("🔍 Busca rápida")
        busca = st.text_input("Digite parte do nome do cliente, email ou número do pedido:")

        # -------------------------------------------------
        # 🔄 Carregamento de dados
        # -------------------------------------------------
        periodo_atual = st.session_state.get("periodo_atual")

        if busca.strip():
            with st.spinner(f"🔍 Buscando '{busca}' diretamente na Shopify..."):
                produtos = get_products_with_variants()
                pedidos = search_orders_shopify(busca)
            st.success(f"✅ {len(pedidos)} pedido(s) encontrados para '{busca}'. (busca direta na Shopify)")
            st.session_state["produtos"] = produtos
            st.session_state["pedidos"] = pedidos

        elif periodo_atual != (start_date, end_date):
            with st.spinner("🔄 Carregando dados da Shopify..."):
                produtos = get_products_with_variants()
                pedidos = get_orders(start_date=start_date, end_date=end_date)
                st.session_state["produtos"] = produtos
                st.session_state["pedidos"] = pedidos
                st.session_state["periodo_atual"] = (start_date, end_date)
            st.success(f"✅ Dados carregados de {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')}")

        # -------------------------------------------------
        # 🧩 Garantir que 'pedidos' existe mesmo se ainda não foi carregado
        # -------------------------------------------------
        if "pedidos" not in st.session_state or st.session_state["pedidos"].empty:
            pedidos = pd.DataFrame()
        else:
            pedidos = st.session_state["pedidos"]

        # -------------------------------------------------
        # 🧩 Garantir que 'produtos' existe mesmo se ainda não foi carregado
        # -------------------------------------------------
        if "produtos" not in st.session_state or st.session_state["produtos"].empty:
            try:
                with st.spinner("🔄 Carregando lista de produtos da Shopify..."):
                    produtos = get_products_with_variants()
                    st.session_state["produtos"] = produtos
            except Exception as e:
                st.error(f"❌ Erro ao carregar produtos da Shopify: {e}")
                produtos = pd.DataFrame()
        else:
            produtos = st.session_state["produtos"]

        # -------------------------------------------------
        # 🧩 Preparação dos dados
        # -------------------------------------------------
        for col in ["order_id", "order_number", "financial_status", "fulfillment_status"]:
            if col not in pedidos.columns:
                pedidos[col] = None

        merge_cols = [c for c in ["variant_id", "sku", "product_title", "variant_title"] if c in produtos.columns]
        if "variant_id" in pedidos.columns and "variant_id" in produtos.columns:
            base = pedidos.merge(produtos[merge_cols], on="variant_id", how="left", suffixes=("", "_produto"))
        else:
            base = pedidos.copy()

        for c in ["product_title", "variant_title"]:
            if f"{c}_produto" in base.columns and c not in base.columns:
                base[c] = base[f"{c}_produto"]
            elif c not in base.columns:
                base[c] = np.nan
            if c in base.columns:
                base[c] = base[c].fillna(f"({c} desconhecido)")

        # -------------------------------------------------
        # 🔧 Conversão segura da coluna "created_at"
        # -------------------------------------------------
        if "created_at" in base.columns and not base["created_at"].dropna().empty:
            try:
                base["created_at"] = (
                    pd.to_datetime(base["created_at"], errors="coerce")
                    .dt.tz_localize(None)
                )
            except Exception as e:
                st.warning(f"⚠️ Erro ao converter datas: {e}")
                base["created_at"] = pd.NaT
        else:
            base["created_at"] = pd.NaT


        if "price" in base.columns:
            base["price"] = pd.to_numeric(base["price"], errors="coerce").fillna(0)
        else:
            base["price"] = 0

        if "quantity" in base.columns:
            base["quantity"] = pd.to_numeric(base["quantity"], errors="coerce").fillna(0)
        else:
            base["quantity"] = 0

        base["line_revenue"] = base["price"] * base["quantity"]


        # -------------------------------------------------
        # 🧠 Aplicação de filtros
        # -------------------------------------------------
        if busca.strip():
            df = base.copy()
        else:
            df = base.dropna(subset=["created_at"])
            df = df[(df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)].copy()

        # -------------------------------------------------
        # 📊 Métricas de resumo
        # -------------------------------------------------
        order_col = "order_number" if df["order_number"].notna().any() else "order_id"
        total_pedidos = df[order_col].nunique()
        total_unidades = df["quantity"].sum()
        total_receita = df["line_revenue"].sum()
        ticket_medio = total_receita / total_pedidos if total_pedidos > 0 else 0

        colA, colB, colC, colD = st.columns(4)
        colA.metric("🧾 Pedidos", total_pedidos)
        colB.metric("📦 Unidades vendidas", int(total_unidades))
        def formatar_moeda(valor):
            try:
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return f"R$ {valor:.2f}"

        colC.metric("💰 Receita total", formatar_moeda(total_receita))
        colD.metric("💸 Ticket médio", formatar_moeda(ticket_medio))

        # -------------------------------------------------
        # 📋 Tabela de pedidos
        # -------------------------------------------------
        
        st.markdown("""
            <style>
            thead tr th:first-child, tbody tr td:first-child {
                text-align: right !important;
            }
            input[type="text"] {
                border-radius: 10px;
                border: 1px solid #444;
                padding: 8px 12px;
            }
            </style>
        """, unsafe_allow_html=True)

        colunas = [order_col, "fulfillment_status", "customer_name", "product_title", "variant_title", "quantity", "created_at", 
                   "forma_entrega", "customer_email", "customer_phone", "customer_cpf", "endereco", "bairro", "cep", "estado", "cidade"]
        colunas = [c for c in colunas if c in df.columns]
        tabela = df[colunas].sort_values("created_at", ascending=False).copy()

        tabela.rename(columns={
            order_col: "Pedido", "created_at": "Data do pedido", "customer_name": "Cliente", "customer_email": "E-mail", "customer_phone": "Telefone", "customer_cpf": "CPF",
            "endereco": "Endereço", "bairro": "Bairro", "cep": "CEP", "quantity": "Qtd", "product_title": "Produto", "variant_title": "Variante", 
            "price": "Preço", "fulfillment_status": "Status de processamento",
            "forma_entrega": "Frete", "estado": "Estado"
        }, inplace=True)

        if "Pedido" in tabela.columns:
            tabela["Pedido"] = tabela["Pedido"].astype(str).str.replace(",", "").str.replace(".0", "", regex=False)

        tabela["Status de processamento"] = df["fulfillment_status"].apply(
            lambda x: "✅ Processado" if str(x).lower() in ["fulfilled", "shipped", "complete"] else "🟡 Não processado"
        )

        # 🔝 Identificação de duplicados
        def identificar_duplicado(row, df_ref):
            nome = str(row.get("Cliente", "")).strip().lower()
            email = str(row.get("E-mail", "")).strip().lower()
            cpf = str(row.get("CPF", "")).strip()
            tel = str(row.get("Telefone", "")).strip()
            end = str(row.get("Endereço", "")).strip().lower()

            ignorar = ["(sem cpf)", "(sem email)", "(sem telefone)", "(sem endereço)", "(sem bairro)"]

            if cpf and cpf not in ignorar and df_ref["CPF"].eq(cpf).sum() > 1:
                return True
            if email and email not in ignorar and df_ref["E-mail"].str.lower().eq(email).sum() > 1:
                return True
            if nome and df_ref["Cliente"].str.lower().eq(nome).sum() > 1:
                return True
            if tel and tel not in ignorar and df_ref["Telefone"].eq(tel).sum() > 1:
                return True
            if end and end not in ignorar and df_ref["Endereço"].str.lower().eq(end).sum() > 1:
                return True
            return False

        tabela["duplicado"] = tabela.apply(lambda row: identificar_duplicado(row, tabela), axis=1)
        # -------------------------------------------------
        # 🚚 Identificação de SEDEX e ordenação
        # -------------------------------------------------
        if "Frete" in tabela.columns:
            tabela["is_sedex"] = tabela["Frete"].astype(str).str.contains("SEDEX", case=False, na=False)
        else:
            tabela["is_sedex"] = False  # cria coluna padrão

        # -------------------------------------------------
        # 🟩 Agrupamento lógico de duplicados (CPF, E-mail, Telefone, Nome, Endereço)
        # -------------------------------------------------
        def chave_grupo(row):
            partes = [
                str(row.get("CPF", "")).strip().lower(),
                str(row.get("E-mail", "")).strip().lower(),
                str(row.get("Telefone", "")).strip(),
                str(row.get("Cliente", "")).strip().lower(),
                str(row.get("Endereço", "")).strip().lower()
            ]
            return "|".join([p for p in partes if p and "(sem" not in p])

        tabela["grupo_id"] = tabela.apply(chave_grupo, axis=1)

        # Se o grupo tiver mais de um item e pelo menos um SEDEX, marca todos como grupo_verde
        grupo_sedex = (
            tabela.groupby("grupo_id")["is_sedex"]
            .transform(lambda x: x.any())
        )
        grupo_duplicado = (
            tabela.groupby("grupo_id")["duplicado"]
            .transform(lambda x: x.any())
        )
        tabela["grupo_verde"] = grupo_duplicado & grupo_sedex

        # ✅ Garante que colunas de ordenação existem
        colunas_ordem = [c for c in ["duplicado", "is_sedex", "Data do pedido"] if c in tabela.columns]
        if colunas_ordem:
            tabela = tabela.sort_values(by=colunas_ordem, ascending=[False, True, False][:len(colunas_ordem)])

        def highlight_prioridades(row):
            # 🟢 Grupo duplicado com SEDEX → Verde translúcido
            if row["grupo_verde"]:
                return ['background-color: rgba(0, 255, 128, 0.15)'] * len(row)
            # 🔵 Duplicado → Azul translúcido
            elif row["duplicado"]:
                return ['background-color: rgba(0, 123, 255, 0.15)'] * len(row)
            # 🟡 SEDEX → Amarelo translúcido
            elif row["is_sedex"]:
                return ['background-color: rgba(255, 215, 0, 0.15)'] * len(row)
            else:
                return [''] * len(row)

        # 🔢 Ajusta o índice antes de aplicar o estilo
        tabela.index = range(1, len(tabela) + 1)

        # Cria uma cópia apenas com as colunas visíveis + técnicas
        colunas_visiveis = [c for c in tabela.columns if c not in ["duplicado", "is_sedex", "grupo_verde", "grupo_id"]]
        tabela_exibir = tabela[colunas_visiveis + ["duplicado", "is_sedex", "grupo_verde", "grupo_id"]].copy()

        # Aplica estilo condicional
        tabela_estilizada = tabela_exibir.style.apply(highlight_prioridades, axis=1)

        # ✅ Remove colunas técnicas antes de exibir (só da visualização)
        colunas_visiveis = [
            c for c in tabela_exibir.columns 
            if c not in ["duplicado", "is_sedex", "grupo_verde", "grupo_id"]
        ]

        # ✅ Converte valores para string (evita erro React no front-end)
        tabela_exibir[colunas_visiveis] = tabela_exibir[colunas_visiveis].fillna("").astype(str)

        # ✅ Exibe tabela com estilo (mantém cores sem quebrar)
        st.write(
            tabela_estilizada.hide(axis="columns", subset=["duplicado", "is_sedex", "grupo_verde", "grupo_id"]),
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # 🎛️ Filtros adicionais
        # -------------------------------------------------
        st.subheader("🎛️ Filtros adicionais")
        col1, col2 = st.columns(2)
        with col1:
            escolha_prod = st.selectbox("Produto", ["(Todos)"] + sorted(base["product_title"].dropna().unique().tolist()))
        with col2:
            escolha_var = st.selectbox("Variante", ["(Todas)"] + sorted(base["variant_title"].dropna().unique().tolist()))

        if escolha_prod != "(Todos)":
            df = df[df["product_title"] == escolha_prod]
        if escolha_var != "(Todas)":
            df = df[df["variant_title"] == escolha_var]

        if df.empty:
            st.warning("⚠️ Nenhum pedido encontrado com os filtros selecionados.")
        else:
            st.success(f"✅ {len(df)} registros após aplicação dos filtros.")
            
        # -------------------------------------------------
        # 🚚 Processamento de pedidos
        # -------------------------------------------------
        pendentes = df[df["fulfillment_status"].isin(["unfulfilled", None, "null"])]
        if not pendentes.empty:
            if st.button("🚀 Processar TODOS os pedidos pendentes"):
                progress = st.progress(0)
                total = len(pendentes)
                for i, row in enumerate(pendentes.itertuples(), start=1):
                    try:
                        create_fulfillment(row.order_id)
                    except Exception as e:
                        st.error(f"❌ Erro ao processar pedido {row.order_id}: {str(e)}")
                        continue
                    progress.progress(i / total)
                st.success("✅ Todos os pedidos pendentes foram processados com sucesso!")
        else:
            st.info("✅ Nenhum pedido pendente para processar.")

        # -------------------------------------------------
        # 📦 Processar individualmente
        # -------------------------------------------------
        st.markdown("### 📦 Processar individualmente")
        for idx, row in df.iterrows():
            if str(row["fulfillment_status"]).lower() in ["fulfilled", "shipped", "complete"]:
                continue
            order_display = int(float(row[order_col])) if pd.notna(row[order_col]) else row["order_id"]
            status_key = f"status_{row.order_id}_{idx}"
            form_key = f"form_{row.order_id}_{idx}"
            track_key = f"track_{row.order_id}_{idx}"
            if status_key not in st.session_state:
                st.session_state[status_key] = ""

            with st.container(border=True):
                st.markdown(f"#### Pedido #{order_display} — {row['customer_name']}")
                st.caption(f"Produto: {row['product_title']} — Variante: {row['variant_title']}")
                with st.form(key=form_key, clear_on_submit=True):
                    tracking_number = st.text_input("📦 Código de rastreio (opcional)", key=track_key)
                    submitted = st.form_submit_button("✅ Processar pedido")
                    if submitted:
                        try:
                            with st.spinner(f"Processando pedido #{order_display}..."):
                                result = create_fulfillment(
                                    row.order_id,
                                    tracking_number=tracking_number or None,
                                    tracking_company="Correios"
                                )
                                log_fulfillment(row.order_id)
                            st.session_state[status_key] = f"✅ Pedido #{order_display} processado com sucesso!"
                            if tracking_number:
                                st.session_state[status_key] += f"\n📬 Código de rastreio: `{tracking_number}`"
                        except Exception as e:
                            st.session_state[status_key] = f"❌ Erro ao processar pedido #{order_display}: {e}"

                msg = st.session_state[status_key]
                if msg:
                    if msg.startswith("✅"):
                        st.success(msg)
                    elif msg.startswith("❌"):
                        st.error(msg)
                    else:
                        st.info(msg)
    
    # =====================================================
    # 📦 ABA 2 — ESTOQUE
    # =====================================================
    with aba2:
        st.subheader("Comparativo de Saídas e Custos por Variante:")

        # =====================================================
        # 🧾 Cria versão formatada da planilha para edição
        # =====================================================
        df_display = df_custos.copy()
        for col in ["Custo AliExpress (R$)", "Custo Estoque (R$)"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    if pd.notna(x) else ""
                )

        # -------------------------------------------------
        # 🔄 Função para atualizar planilha de custos
        # -------------------------------------------------
        def atualizar_planilha_custos(df):
            """Atualiza dados na planilha de custos no Google Sheets"""
            import gspread
            from google.oauth2.service_account import Credentials

            try:
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
                gcp_info = dict(st.secrets["gcp_service_account"])
                if isinstance(gcp_info.get("private_key"), str):
                    gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")

                creds = Credentials.from_service_account_info(gcp_info, scopes=scopes)
                client = gspread.authorize(creds)
                sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"]).sheet1

                df_safe = (
                    df.copy()
                    .fillna("")
                    .astype(str)
                    .replace("nan", "", regex=False)
                )

                body = [df_safe.columns.values.tolist()] + df_safe.values.tolist()
                sheet.batch_clear(["A:Z"])
                sheet.update(body)

                st.success("✅ Planilha atualizada com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao atualizar planilha: {e}")

        # =====================================================
        # 📝 Edição direta da planilha no app
        # =====================================================
        st.subheader("📝 Custos por Variante")

        edit_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("💾 Salvar alterações na planilha"):
            # ⚙️ Converte R$ 25,00 → 25.00 antes de enviar
            for col in ["Custo AliExpress (R$)", "Custo Estoque (R$)"]:
                if col in edit_df.columns:
                    edit_df[col] = (
                        edit_df[col]
                        .astype(str)
                        .str.replace("R$", "", regex=False)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                        .replace("", np.nan)
                        .astype(float)
                    )

            atualizar_planilha_custos(edit_df)
            st.cache_data.clear()
            st.rerun()

    # =====================================================
    # 🚚 ABA 3 — ENTREGAS
    # =====================================================
    with aba3:
        st.info("📍 Em breve: status de fretes, prazos e devoluções.")
