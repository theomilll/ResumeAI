# notion_auth.py

import requests
import base64
import os
from notion_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv('NOTION_CLIENT_ID', '')
CLIENT_SECRET = os.getenv('NOTION_CLIENT_SECRET', '')
REDIRECT_URI = os.getenv('NOTION_REDIRECT_URI', 'http://localhost:5000/oauth/callback')

AUTHORIZATION_URL = 'https://api.notion.com/v1/oauth/authorize'
TOKEN_URL = "https://api.notion.com/v1/oauth/token"


def gerar_url_autenticacao():
    return (
        f"{AUTHORIZATION_URL}"
        f"?owner=user"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
    )


def trocar_codigo_por_token(code):
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {b64_auth_str}",
            "Content-Type": "application/json",
        },
        json={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
    )

    if response.status_code != 200:
        raise Exception(f"Erro ao obter token: {response.text}")

    return response.json().get("access_token")


def buscar_paginas(token):
    notion = Client(auth=token)
    search_results = notion.search(filter={"property": "object", "value": "page"})

    paginas = []
    for result in search_results["results"]:
        titulo = "Sem título"
        try:
            titulo_info = result.get("properties", {}).get("title", {}).get("title", [])
            if titulo_info and titulo_info[0].get("text"):
                titulo = titulo_info[0]["text"]["content"]
        except:
            pass

        paginas.append({
            "id": result["id"],
            "title": titulo
        })

    return paginas


if __name__ == '__main__':
    print("Este arquivo serve como biblioteca. Use as funções em outro app.")
