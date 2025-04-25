# Guia de Uso - Exportação para o Notion

Este guia explica como utilizar o módulo de exportação para o Notion desenvolvido para o projeto ResumeAI.

## Configuração Inicial

1. **Instale as dependências necessárias**:
   ```bash
   pip install notion-client
   ```

2. **Crie uma integração no Notion**:
   - Acesse [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations)
   - Clique em "New integration"
   - Dê um nome à sua integração (ex: "ResumeAI")
   - Selecione o workspace onde deseja usar a integração
   - Clique em "Submit"
   - Copie o "Internal Integration Token" que será exibido

3. **Compartilhe páginas com sua integração**:
   - Abra a página do Notion onde deseja exportar os resumos
   - Clique em "..." no canto superior direito
   - Selecione "Add connections"
   - Busque e selecione sua integração na lista

4. **Configure o token como variável de ambiente**:
   ```bash
   export NOTION_TOKEN='seu_token_aqui'
   ```

## Como Usar o Módulo

### Exportar um Resumo para uma Página

```python
from notion_exporter import NotionExporter
import os

# Inicializar o exportador
token = os.environ.get("NOTION_TOKEN")
exporter = NotionExporter(token)

# Dados do resumo
titulo = "Resumo da Reunião de Planejamento"
conteudo = """
Na reunião de hoje, discutimos os seguintes pontos:

1. Cronograma do projeto ResumeAI para o próximo trimestre
2. Alocação de recursos para desenvolvimento das funcionalidades de IA
3. Estratégias de marketing para o lançamento do MVP

Decidimos priorizar o desenvolvimento da funcionalidade de categorização inteligente.
"""
categorias = ["Planejamento", "Desenvolvimento", "Marketing"]
tipo_fonte = "Reunião"
nome_fonte = "Equipe de Produto"

# ID da página pai no Notion (onde o resumo será criado)
pagina_pai_id = "sua_pagina_id_aqui"  # Substitua pelo ID real

# Exportar para o Notion
resposta = exporter.create_page(
    parent_page_id=pagina_pai_id,
    title=titulo,
    content=conteudo,
    categories=categorias,
    source_type=tipo_fonte,
    source_name=nome_fonte
)

# Exibir URL da página criada
print(f"Resumo exportado com sucesso: {resposta['url']}")
```

### Exportar para um Banco de Dados

```python
# ID do banco de dados no Notion
banco_dados_id = "seu_banco_dados_id_aqui"  # Substitua pelo ID real

# Exportar para o banco de dados
resposta = exporter.create_database_item(
    database_id=banco_dados_id,
    title=titulo,
    content=conteudo,
    categories=categorias,
    source_type=tipo_fonte,
    source_name=nome_fonte
)

print(f"Item criado com sucesso: {resposta['url']}")
```

### Adicionar Conteúdo a uma Página Existente

```python
# ID da página existente
pagina_id = "sua_pagina_id_aqui"  # Substitua pelo ID real

# Conteúdo adicional
conteudo_adicional = """
Atualização importante:

Após a reunião, recebemos feedback do time de UX que sugere algumas melhorias
na interface do usuário. Precisaremos revisar o cronograma.
"""

# Adicionar à página existente
exporter.append_to_page(
    page_id=pagina_id,
    content=conteudo_adicional
)

print("Conteúdo adicionado com sucesso!")
```

## Como Encontrar IDs no Notion

Para encontrar o ID de uma página ou banco de dados:

1. Abra a página/banco de dados no navegador
2. Copie a URL da página
3. O ID é a parte final da URL, antes de qualquer parâmetro de consulta
   - Exemplo: `https://www.notion.so/workspace/1234abcd5678efgh9012ijkl3456mnop`
   - O ID é: `1234abcd5678efgh9012ijkl3456mnop`

## Executando o Script de Exemplo

O módulo inclui um script de exemplo que demonstra o uso do exportador:

```bash
# Configure o token primeiro
export NOTION_TOKEN='seu_token_aqui'

# Execute o script de exemplo
python example.py
```

O script solicitará o ID de uma página ou banco de dados e criará um exemplo de resumo.

## Solução de Problemas

Se encontrar erros ao usar o módulo, verifique:

1. **Token inválido**: Certifique-se de que o token está correto e configurado como variável de ambiente
2. **Permissões**: Verifique se a integração tem acesso à página ou banco de dados
3. **IDs incorretos**: Confirme se os IDs de página/banco de dados estão no formato correto
4. **Estrutura do banco de dados**: Para exportar para um banco de dados, ele deve ter as propriedades adequadas

Para mais detalhes, consulte a documentação completa no arquivo README.md.
