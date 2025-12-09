from sync_core import sync_shopify_to_sheet

def main():
    print("🚀 GitHub Actions executando atualização Shopify → Planilha...")
    try:
        result = sync_shopify_to_sheet()
        print("RESULTADO:", result)
    except Exception as e:
        print("❌ ERRO:", e)

if __name__ == "__main__":
    main()
