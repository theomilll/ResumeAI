# Documentação do Módulo de Exportação para o Notion

## Visão Geral

Este módulo implementa a funcionalidade de "Exportação de um texto/resumo para o Notion" para o projeto ResumeAI da JANUS. Ele permite que resumos gerados automaticamente de reuniões, aulas e palestras sejam exportados e organizados no Notion, facilitando o acesso e a categorização das informações.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
notion_export/
├── notion_exporter.py     # Módulo principal com a classe NotionExporter
├── example.py             # Script de exemplo para uso real
└── test_mock.py           # Script de teste com cliente mock
```

## Requisitos

- Python 3.6 ou superior
- Biblioteca `notion-client`
- Token de integração do Notion (para uso real)

## Instalação

1. Instale a biblioteca necessária:
   ```
   pip install notion-client
   ```

2. Configure o token de integração do Notion como variável de ambiente:
   ```
   export NOTION_TOKEN='seu_token_secreto_aqui'
   ```

## Funcionalidades Implementadas

A classe `NotionExporter` oferece as seguintes funcionalidades:

1. **Criação de páginas no Notion**
   - Cria uma nova página com título, conteúdo e metadados
   - Suporta categorização através de tags
   - Inclui informações sobre a fonte (tipo e nome)
   - Formata o conteúdo em blocos de texto estruturados

2. **Criação de itens em bancos de dados do Notion**
   - Adiciona um novo item a um banco de dados existente
   - Adapta-se automaticamente à estrutura do banco de dados
   - Preenche propriedades relevantes com base nos metadados

3. **Adição de conteúdo a páginas existentes**
   - Permite anexar novo conteúdo a uma página já criada
   - Mantém a formatação consistente

4. **Pesquisa de páginas no Notion**
   - Busca páginas existentes com base em termos de pesquisa
   - Filtra por tipo de objeto (página, banco de dados)

5. **Recuperação de conteúdo de páginas**
   - Obtém o conteúdo de uma página existente
   - Retorna a lista de blocos para processamento

## Como Usar

### Exemplo Básico

```python
from notion_exporter import NotionExporter

# Inicializar o exportador com o token
token = "seu_token_secreto_aqui"  # Ou use os.environ.get("NOTION_TOKEN")
exporter = NotionExporter(token)

# Criar uma página com um resumo
response = exporter.create_page(
    parent_page_id="ID_DA_PAGINA_PAI",
    title="Título do Resumo",
    content="Conteúdo do resumo...",
    categories=["Categoria1", "Categoria2"],
    source_type="Reunião",
    source_name="Nome da Fonte"
)

# Obter a URL da página criada
page_url = response["url"]
print(f"Resumo exportado para: {page_url}")
```

### Exportar para um Banco de Dados

```python
response = exporter.create_database_item(
    database_id="ID_DO_BANCO_DE_DADOS",
    title="Título do Resumo",
    content="Conteúdo do resumo...",
    categories=["Categoria1", "Categoria2"],
    source_type="Aula",
    source_name="Nome do Professor"
)
```

### Adicionar Conteúdo a uma Página Existente

```python
exporter.append_to_page(
    page_id="ID_DA_PAGINA_EXISTENTE",
    content="Novo conteúdo a ser adicionado..."
)
```

## Integração com o ResumeAI

Para integrar este módulo ao sistema ResumeAI:

1. Importe a classe `NotionExporter` no módulo principal do ResumeAI
2. Inicialize o exportador com o token de integração do Notion
3. Após gerar um resumo, chame o método apropriado para exportá-lo para o Notion
4. Armazene a URL da página criada para referência futura

Exemplo de integração:

```python
def process_meeting_recording(audio_file, meeting_name):
    # Gerar resumo usando o ResumeAI
    summary = generate_summary(audio_file)
    categories = categorize_content(summary)
    
    # Exportar para o Notion
    exporter = NotionExporter(os.environ.get("NOTION_TOKEN"))
    response = exporter.create_page(
        parent_page_id=NOTION_WORKSPACE_PAGE_ID,
        title=f"Resumo: {meeting_name}",
        content=summary,
        categories=categories,
        source_type="Reunião",
        source_name=meeting_name
    )
    
    return response["url"]
```

## Configuração da Integração no Notion

Para usar este módulo, você precisa:

1. Criar uma integração interna no [Notion Developers](https://www.notion.so/my-integrations)
2. Obter o token de integração
3. Compartilhar as páginas ou bancos de dados desejados com a integração no Notion
4. Configurar o token como variável de ambiente ou passá-lo diretamente ao inicializar o exportador

## Limitações e Considerações

- A integração requer permissões explícitas para acessar páginas e bancos de dados no Notion
- O formato dos blocos de conteúdo segue a estrutura da API do Notion
- Algumas formatações avançadas podem não ser suportadas
- É necessário obter os IDs das páginas ou bancos de dados do Notion para uso

## Testes

O módulo inclui um script de teste (`test_mock.py`) que verifica a estrutura dos dados sem fazer chamadas reais à API do Notion. Para executar os testes:

```
python test_mock.py
```

Para testar com uma integração real, use o script de exemplo:

```
python example.py
```

## Próximos Passos e Melhorias Futuras

- Implementar suporte para mais tipos de blocos (listas, tabelas, etc.)
- Adicionar suporte para upload de imagens e arquivos
- Implementar sincronização bidirecional (ler modificações feitas no Notion)
- Adicionar cache para melhorar o desempenho em operações repetitivas
- Implementar tratamento de erros mais robusto para casos específicos da API
