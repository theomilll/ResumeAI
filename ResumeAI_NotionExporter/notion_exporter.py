"""
Módulo para exportação de texto/resumo para o Notion
Parte do projeto ResumeAI - JANUS

Este módulo implementa a funcionalidade de exportação de resumos gerados
automaticamente para o Notion, permitindo que os usuários armazenem e
organizem os resumos de reuniões, aulas e palestras.

Todos os códigos nessa página seguem padrões de codificação pragmáticos para melhor leitura.
"""
from typing import Dict, List
from datetime import datetime
from notion_client import Client

class NotionExporter:
    """
    Classe responsável por exportar resumos para o Notion.
    
    Esta classe implementa a funcionalidade de exportação de texto/resumo
    para o Notion, permitindo que os resumos gerados pelo ResumeAI sejam
    armazenados e organizados na plataforma Notion.
    """
    
    def __init__(self, token: str):
        """
        Inicializa o exportador do Notion com o token de integração.
        
        Args:
            token: Token de integração do Notion (Internal Integration Token)
        """
        self.notion = Client(auth=token)
        
    def create_page(self, 
                   parent_page_id: str, 
                   title: str, 
                   content: str, 
                   categories: List[str] = None,
                   source_type: str = None,
                   source_name: str = None,
                   language: str = "pt-br") -> Dict:
        """
        Cria uma nova página no Notion com o resumo.
        
        Args:
            parent_page_id: ID da página pai no Notion onde o resumo será criado
            title: Título do resumo
            content: Texto do resumo
            categories: Lista de categorias/tags para classificar o resumo
            source_type: Tipo da fonte (reunião, aula, palestra)
            source_name: Nome da fonte
            language: Idioma do resumo
            
        Returns:
            Resposta da API do Notion com os dados da página criada
        """
        properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
        }
        
        # Adicionar data da criação
        properties["Data"] = {
            "date": {
                "start": datetime.now().strftime("%Y-%m-%d")
            }
        }
        
        # Adicionar tipo de fonte se houver
        if source_type:
            properties["Tipo"] = {
                "select": {
                    "name": source_type
                }
            }
            
        # Adicionar nome da fonte se houver
        if source_name:
            properties["Fonte"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": source_name
                        }
                    }
                ]
            }
            
        # Adicionar idioma
        properties["Idioma"] = {
            "select": {
                "name": language
            }
        }
        
        children = []
        
        # Adicionar categorias como tags
        if categories and len(categories) > 0:
            categories_text = ", ".join(categories)
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Categorias: {categories_text}"
                            }
                        }
                    ],
                    "icon": {
                        "emoji": "🏷️"
                    },
                    "color": "blue_background"
                }
            })
            
            # Adicionar linha divisória
            children.append({
                "object": "block",
                "type": "divider",
                "divider": {}
            })
        
        # Dividir o conteúdo em parágrafos
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": paragraph.strip()
                                }
                            }
                        ]
                    }
                })
        
        # Criar a página no Notion
        response = self.notion.pages.create(
            parent={
                "type": "page_id",
                "page_id": parent_page_id
            },
            properties=properties,
            children=children
        )
        
        return response
    
    def create_database_item(self, 
                           database_id: str, 
                           title: str, 
                           content: str, 
                           categories: List[str] = None,
                           source_type: str = None,
                           source_name: str = None,
                           language: str = "pt-br") -> Dict:
        """
        Cria um novo item em um banco de dados do Notion com o resumo.
        
        Args:
            database_id: ID do banco de dados no Notion
            title: Título do resumo
            content: Texto do resumo
            categories: Lista de categorias/tags para classificar o resumo
            source_type: Tipo da fonte (reunião, aula, palestra)
            source_name: Nome da fonte
            language: Idioma do resumo
            
        Returns:
            Resposta da API do Notion com os dados do item criado
        """
        database = self.notion.databases.retrieve(database_id=database_id)
        
        # Preparar propriedades do item
        properties = {}
        
        # Identificar a propriedade de título
        title_property = None
        for prop_name, prop_details in database["properties"].items():
            if prop_details["type"] == "title":
                title_property = prop_name
                break
        
        if not title_property:
            raise ValueError("O banco de dados não possui uma propriedade de título")
        
        # Adicionar título
        properties[title_property] = {
            "title": [
                {
                    "text": {
                        "content": title
                    }
                }
            ]
        }

        for prop_name, prop_details in database["properties"].items():
            # Pular a propriedade de título que já foi adicionada
            if prop_name == title_property:
                continue
                
            # Adicionar data se houver uma propriedade de data
            if prop_details["type"] == "date" and prop_name.lower() in ["data", "date", "created"]:
                properties[prop_name] = {
                    "date": {
                        "start": datetime.now().strftime("%Y-%m-%d")
                    }
                }
                
            # Adicionar tipo de fonte se houver uma propriedade de seleção apropriada
            if prop_details["type"] == "select" and prop_name.lower() in ["tipo", "type"] and source_type:
                properties[prop_name] = {
                    "select": {
                        "name": source_type
                    }
                }
                
            # Adicionar nome da fonte se houver uma propriedade de texto apropriada
            if prop_details["type"] == "rich_text" and prop_name.lower() in ["fonte", "source"] and source_name:
                properties[prop_name] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": source_name
                            }
                        }
                    ]
                }
                
            # Adicionar idioma se houver uma propriedade de seleção apropriada
            if prop_details["type"] == "select" and prop_name.lower() in ["idioma", "language"]:
                properties[prop_name] = {
                    "select": {
                        "name": language
                    }
                }
                
            # Adicionar categorias se houver uma propriedade de multi_select apropriada
            if prop_details["type"] == "multi_select" and prop_name.lower() in ["categorias", "categories", "tags"] and categories:
                multi_select_values = []
                for category in categories:
                    multi_select_values.append({"name": category})
                    
                properties[prop_name] = {
                    "multi_select": multi_select_values
                }
        
        # Preparar conteúdo do item
        children = []
        
        # Adicionar categorias como texto se não houver uma propriedade multi_select
        if categories and len(categories) > 0 and not any(prop_details["type"] == "multi_select" for prop_name, prop_details in database["properties"].items()):
            categories_text = ", ".join(categories)
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Categorias: {categories_text}"
                            }
                        }
                    ],
                    "icon": {
                        "emoji": "🏷️"
                    },
                    "color": "blue_background"
                }
            })
            
            # Adicionar linha divisória
            children.append({
                "object": "block",
                "type": "divider",
                "divider": {}
            })
        
        # Dividir o conteúdo em parágrafos
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": paragraph.strip()
                                }
                            }
                        ]
                    }
                })
        
        # Criar o item no banco de dados do Notion
        response = self.notion.pages.create(
            parent={
                "type": "database_id",
                "database_id": database_id
            },
            properties=properties,
            children=children
        )
        
        return response
    
    def append_to_page(self, page_id: str, content: str) -> Dict:
        """
        Adiciona conteúdo a uma página existente no Notion.
        
        Args:
            page_id: ID da página no Notion
            content: Texto a ser adicionado à página
            
        Returns:
            Resposta da API do Notion
        """
        # Dividir o conteúdo em parágrafos
        paragraphs = content.split('\n\n')
        children = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": paragraph.strip()
                                }
                            }
                        ]
                    }
                })
        
        # Adicionar conteúdo à página
        response = self.notion.blocks.children.append(
            block_id=page_id,
            children=children
        )
        
        return response
    
    def search_pages(self, query: str = "", filter_by_type: str = "page") -> List[Dict]:
        """
        Pesquisa páginas no Notion.
        
        Args:
            query: Termo de pesquisa
            filter_by_type: Tipo de objeto a ser pesquisado (page, database)
            
        Returns:
            Lista de páginas encontradas
        """
        response = self.notion.search(
            query=query,
            filter={
                "property": "object",
                "value": filter_by_type
            }
        )
        
        return response["results"]
    
    def get_page_content(self, page_id: str) -> List[Dict]:
        """
        Obtém o conteúdo de uma página no Notion.
        
        Args:
            page_id: ID da página no Notion
            
        Returns:
            Lista de blocos de conteúdo da página
        """
        response = self.notion.blocks.children.list(block_id=page_id)
        return response["results"]