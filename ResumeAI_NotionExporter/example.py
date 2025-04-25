"""
Script de exemplo para demonstrar o uso do módulo NotionExporter
Parte do projeto ResumeAI - JANUS
"""

import os
from notion_exporter import NotionExporter

def main():
    print("Demonstração do Exportador para o Notion")
    print("----------------------------------------")

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        print("\nERRO: Token do Notion não encontrado.")
        print("Para usar este script, você precisa definir a variável de ambiente NOTION_TOKEN.")
        print("\nExemplo:")
        print("  export NOTION_TOKEN='seu_token_secreto_aqui'")
        print("\nVocê pode obter um token de integração em https://www.notion.so/my-integrations")
        return

    exporter = NotionExporter(token)

    title = "Resumo da Reunião de Planejamento - ResumeAI"
    content = """
    Na reunião de hoje, discutimos os seguintes pontos sobre o projeto ResumeAI:
    
    1. Cronograma do projeto para o próximo trimestre
    2. Alocação de recursos para desenvolvimento das funcionalidades de IA
    3. Estratégias de marketing para o lançamento do MVP
    
    Decidimos priorizar o desenvolvimento da funcionalidade de categorização inteligente,
    seguida pela integração com o Notion. A equipe de marketing começará a preparar
    materiais promocionais para o lançamento previsto para o final do próximo mês.
    
    Próximos passos:
    - Finalizar a implementação da API de resumo automático
    - Iniciar testes com usuários reais
    - Preparar documentação técnica
    """
    
    categories = ["Planejamento", "Desenvolvimento", "Marketing"]
    source_type = "Reunião"
    source_name = "Equipe de Produto"
    
    # Solicitar ID da página pai ou banco de dados
    print("\nPara exportar o resumo, você precisa fornecer um ID de página ou banco de dados do Notion.")
    print("Você pode encontrar o ID na URL da página ou banco de dados.")
    print("Exemplo: https://www.notion.so/workspace/1234abcd5678efgh9012ijkl3456mnop")
    print("                                      └─────────── ID ───────────┘")
    
    parent_id = input("\nDigite o ID da página pai ou banco de dados: ")
    
    if not parent_id:
        print("ID não fornecido. Encerrando demonstração.")
        return
    
    # Formatar o ID se precisar
    parent_id = parent_id.replace("-", "")
    if len(parent_id) == 32:
        # Formatar no padrão 8-4-4-4-12
        parent_id = f"{parent_id[0:8]}-{parent_id[8:12]}-{parent_id[12:16]}-{parent_id[16:20]}-{parent_id[20:32]}"
    
    # Perguntar se é uma página ou banco de dados
    parent_type = input("\nO ID fornecido é de uma página ou banco de dados? (p/b): ").lower()
    
    try:
        if parent_type == "p":
            print("\nExportando resumo para a página...")
            response = exporter.create_page(
                parent_page_id=parent_id,
                title=title,
                content=content,
                categories=categories,
                source_type=source_type,
                source_name=source_name
            )
            print(f"\nResumo exportado com sucesso!")
            print(f"URL da página criada: {response['url']}")
            
        elif parent_type == "b":
            print("\nExportando resumo para o banco de dados...")
            response = exporter.create_database_item(
                database_id=parent_id,
                title=title,
                content=content,
                categories=categories,
                source_type=source_type,
                source_name=source_name
            )
            print(f"\nResumo exportado com sucesso!")
            print(f"URL do item criado: {response['url']}")
            
        else:
            print("\nTipo inválido. Use 'p' para página ou 'b' para banco de dados.")
            return
            
    except Exception as e:
        print(f"\nErro ao exportar resumo: {str(e)}")
        print("\nVerifique se:")
        print("1. O token de integração é válido")
        print("2. O ID fornecido é válido")
        print("3. A integração tem permissão para acessar a página ou banco de dados")
        print("   (Você precisa compartilhar a página com a integração no Notion)")

if __name__ == "__main__":
    main()
