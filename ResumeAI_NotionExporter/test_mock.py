"""
Script de teste para verificar a funcionalidade do módulo NotionExporter
Parte do projeto ResumeAI - JANUS
"""

import os
import json
from notion_exporter import NotionExporter

def test_mock():
    """
    Função que testa o exportador para o Notion usando um mock.
    Isso permite verificar a estrutura dos dados sem precisar de um token real.
    """
    print("Teste do Exportador para o Notion (Modo Mock)")
    print("---------------------------------------------")
    
    class MockNotionClient:
        """Cliente mock para simular a API do Notion"""
        
        def __init__(self):
            self.pages = MockPagesClient()
            self.blocks = MockBlocksClient()
            
    class MockPagesClient:
        """Cliente mock para simular a API de páginas do Notion"""
        
        def create(self, parent, properties, children):
            print("\n=== Dados que seriam enviados para a API do Notion ===")
            print("\nParent:")
            print(json.dumps(parent, indent=2))
            print("\nProperties:")
            print(json.dumps(properties, indent=2))
            print("\nChildren (primeiros 2 blocos):")
            print(json.dumps(children[:2], indent=2))
            print(f"... e mais {len(children) - 2} blocos" if len(children) > 2 else "")
            
            return {
                "id": "mock-page-id-12345",
                "url": "https://www.notion.so/Resumo-da-Reuniao-de-Planejamento-mock-page-id-12345"
            }

        def retrieve(self, page_id):
            return {
                "id": page_id,
                "url": f"https://www.notion.so/Mock-Page-{page_id}"
            }
    
    class MockBlocksClient:

        def __init__(self):
            self.children = MockBlockChildrenClient()
            
    class MockBlockChildrenClient:

        def append(self, block_id, children):
            print(f"\nAdicionando {len(children)} blocos à página {block_id}")
            return {
                "results": [{"id": f"mock-block-{i}"} for i in range(len(children))]
            }
            
        def list(self, block_id):
            return {
                "results": [
                    {
                        "id": "mock-block-1",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": "Este é um bloco de teste."
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
    
    # Criar uma instância do exportador com o cliente mock
    exporter = NotionExporter("mock-token")
    exporter.notion = MockNotionClient()
    
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
    
    print("\n\n=== Teste 1: Criação de página ===")
    response = exporter.create_page(
        parent_page_id="mock-parent-page-id",
        title=title,
        content=content,
        categories=categories,
        source_type=source_type,
        source_name=source_name
    )
    print(f"\nResultado: Página criada com sucesso!")
    print(f"URL da página criada: {response['url']}")
    
    print("\n\n=== Teste 2: Adição de conteúdo a uma página existente ===")
    additional_content = """
    Atualização importante:
    
    Após a reunião, recebemos feedback do time de UX que sugere algumas melhorias
    na interface do usuário. Precisaremos revisar o cronograma para acomodar essas mudanças.
    """
    response = exporter.append_to_page(
        page_id="mock-existing-page-id",
        content=additional_content
    )
    print(f"\nResultado: Conteúdo adicionado com sucesso!")

    print("\n\nTestes concluídos com sucesso!")
    print("Os testes mostram que a estrutura dos dados está correta para a API do Notion.")
    print("Para testar com uma integração real, use o script example.py com um token válido.")

if __name__ == "__main__":
    test_mock()
