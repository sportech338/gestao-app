import requests
import pandas as pd
import streamlit as st

SHOP_NAME = st.secrets["shopify"]["shop_name"]
ACCESS_TOKEN = st.secrets["shopify"]["access_token"]
API_VERSION = "2024-10"

BASE_URL = f"https://{SHOP_NAME}/admin/api/{API_VERSION}"
HEADERS = {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}

@st.cache_data(ttl=600)
def get_products_with_variants(limit=250):
    url = f"{BASE_URL}/products.json?limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json().get("products", [])
    rows = []
    for p in data:
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
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def get_orders(limit=100):
    url = f"{BASE_URL}/orders.json?limit={limit}&status=any"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json().get("orders", [])
    rows = []
    for o in data:
        for it in o.get("line_items", []):
            rows.append({
                "order_id": o["id"],
                "created_at": o["created_at"],
                "variant_id": it.get("variant_id"),
                "title": it.get("title"),
                "variant_title": it.get("variant_title"),
                "quantity": it.get("quantity", 0),
                "price": float(it.get("price") or 0),
            })
    return pd.DataFrame(rows)
