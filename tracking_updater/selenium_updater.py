import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------------------------------------
# 🔐 AUTENTICAÇÃO GOOGLE SHEETS
# ---------------------------------------------------------
def get_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    # PLANILHA LOGÍSTICA
    sheet = client.open_by_key("1WTEiRnm1OFxzn6ag1MfI8VnlQCbL8xwxY3LeanCsdxk").worksheet("Logística")
    return sheet


# ---------------------------------------------------------
# 🔍 EXTRAÇÃO COM SELENIUM
# ---------------------------------------------------------
def extrair_eventos_selenium(link):

    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")

        # 👇 AQUI ESTÁ A CORREÇÃO IMPORTANTE  
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        driver.get(link)

        # Aguarda eventos carregarem
        WebDriverWait(driver, 25).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "rptn-order-tracking-event"))
        )

        eventos_html = driver.find_elements(By.CLASS_NAME, "rptn-order-tracking-event")

        eventos = []
        for ev in eventos_html:

            txt = ev.find_element(By.CLASS_NAME, "rptn-order-tracking-text")

            def get(class_name):
                try:
                    return txt.find_element(By.CLASS_NAME, class_name).text.strip()
                except:
                    return ""

            data = get("rptn-order-tracking-date")
            label = get("rptn-order-tracking-label")
            local = get("rptn-order-tracking-location")
            desc = get("rptn-order-tracking-description")

            linha = f"{data} — {label}"
            if local:
                linha += f" — {local}"
            if desc:
                linha += f" — {desc}"

            eventos.append(linha)

        driver.quit()

        return "\n".join(eventos) if eventos else "Sem eventos"

    except Exception as e:
        return f"Erro Selenium: {e}"


# ---------------------------------------------------------
# 🔁 ATUALIZAÇÃO DA PLANILHA
# ---------------------------------------------------------
def atualizar_planilha():

    sheet = get_sheet()
    data = sheet.get_all_records()

    for i, row in enumerate(data):
        linha = i + 2   # linha real na planilha (1 = cabeçalho)

        link = str(row.get("LINK", "")).strip()

        if not link.startswith("http"):
            sheet.update_cell(linha, 10, "Sem link")
            print(f"⚠️ Linha {linha}: Sem link")
            continue

        status = extrair_eventos_selenium(link)
        sheet.update_cell(linha, 10, status)

        print(f"➡️ Linha {linha}: {status}")

        time.sleep(1.2)  # Evita limite da API


# ---------------------------------------------------------
# ▶️ EXECUÇÃO
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Iniciando atualização de rastreios...")
    atualizar_planilha()
    print("✅ Finalizado!")
