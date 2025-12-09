from streamlit_app import sync_shopify_to_sheet

def main():
    print("🚀 GitHub Actions: Atualizando pedidos Shopify → Planilha...")

    try:
        resultado = sync_shopify_to_sheet()
        print(f"✅ {resultado}")
    except Exception as e:
        print(f"❌ Erro ao atualizar: {e}")

if __name__ == "__main__":
    main()
