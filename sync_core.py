import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date
from zoneinfo import ZoneInfo
import json
import os

APP_TZ = ZoneInfo("America/Sao_Paulo")

# ===============================================================
# CARREGA SECRETS DO GITHUB ACTIONS
# ===============================================================
SHOP_NAME = os.getenv("SHOP_NAME")
SHOP_ACCESS_TOKEN = os.getenv("SHOP_ACCESS_TOKEN")
GSHEET_ID = os.getenv("GSHEET_ID")
GCP_JSON = os.getenv("GCP_SERVICE_ACCOUNT")

if GCP_JSON:
    GCP_INFO = json.loads(GCP_JSON)
else:
    raise Exception("❌ GCP_SERVICE_ACCOUNT não encontrado nas secrets.")

# ===============================================================
# FUNÇÃO DE BUSCAR PEDIDOS PAGOS — limpa e independente
# ===============================================================
def get_paid_orders_today():
    hoje = datetime.now(APP_TZ).date()

    start_str = hoje.strftime("%Y-%m-%dT00:00:00-03:00")
    end_str   = hoje.strftime("%Y-%m-%dT23:59:59-03:00")

    url = (
        f"https://{SHOP_NAME}/admin/api/2024-10/orders.json"
        f"?status=any&created_at_min={start_str}&created_at_max={end_str}"
    )

    headers = {"X-Shopify-Access-Token": SHOP_ACCESS_TOKEN}

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    orders = r.json().get("orders", [])
    rows = []

    for o in orders:
        if o.get("financial_status") != "paid":
            continue
        
        for it in o.get("line_items", []):
            rows.append([
                o.get("created_at"),
                o.get("customer", {}).get("first_name", "") + " " + o.get("customer", {}).get("last_name", ""),
                o.get("financial_status"),
                it.get("title"),
                it.get("quantity"),
                o.get("email") or o.get("contact_email"),
                str(o.get("order_number")),
                "",
                "",
                ""
            ])

    df = pd.DataFrame(rows, columns=[
        "DATA", "CLIENTE", "STATUS", "PRODUTO", "QUANTIDADE",
        "EMAIL", "PEDIDO", "RASTREIO", "LINK", "OBSERVAÇÕES"
    ])

    return df


# ===============================================================
# SINCRONIZAÇÃO REAL
# ===============================================================
def sync_shopify_to_sheet():

    df_new = get_paid_orders_today()
    if df_new.empty:
        return "Nenhum pedido pago encontrado hoje."

    creds = Credentials.from_service_account_info(
        GCP_INFO,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)

    sheet = client.open_by_key(GSHEET_ID).worksheet("Logística")
    df_sheet = pd.DataFrame(sheet.get_all_records())

    df_sheet["PEDIDO"] = df_sheet["PEDIDO"].astype(str).str.replace(",", "")
    pedidos_existentes = set(df_sheet["PEDIDO"])

    novos = df_new[~df_new["PEDIDO"].isin(pedidos_existentes)]

    if novos.empty:
        return "Nenhum pedido novo para adicionar."

    sheet.append_rows(novos.astype(str).values.tolist(), value_input_option="USER_ENTERED")

    return f"{len(novos)} pedido(s) adicionados!"
