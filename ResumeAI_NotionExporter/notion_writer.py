from notion_client import Client
from speech_to_text.resumo_generator import gerar_resumo

def dividir_texto_em_blocos(texto, tamanho_maximo=2000):
    palavras = texto.split()
    blocos = []
    bloco = ''

    for palavra in palavras:
        if len(bloco) + len(palavra) + 1 <= tamanho_maximo:
            bloco += palavra + ' '
        else:
            blocos.append(bloco.strip())
            bloco = palavra + ' '

    if bloco:
        blocos.append(bloco.strip())

    return blocos

def salvar_resumo_notion(token: str, page_id: str):
    if not page_id or not token:
        raise ValueError("Página ou token não fornecidos.")

    titulo, resumo = gerar_resumo()
    client = Client(auth=token)

    blocos_de_texto = dividir_texto_em_blocos(resumo, tamanho_maximo=1999)

    children_blocks = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": bloco
                        }
                    }
                ]
            }
        }
        for bloco in blocos_de_texto
    ]

    resposta = client.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": titulo
                            }
                        }
                    ]
                }
            }
        ] + children_blocks
    )

    return resposta


"""def salvar_resumo_notion(token: str, page_id: str):
    titulo = "Resumo teste manual"
    resumo = "Esse é um exemplo de conteúdo de resumo gerado manualmente."

    client = Client(auth=token)
    resposta = client.pages.create(
        parent={"type": "page_id", "page_id": page_id},
        properties={
            "title": [{"type": "text", "text": {"content": titulo}}]
        },
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": resumo}}]
                }
            }
        ]
    )

    return resposta
"""