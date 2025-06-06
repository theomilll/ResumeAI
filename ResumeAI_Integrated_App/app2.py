import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, render_template_string, redirect, session
from ResumeAI_NotionExporter.notion_auth import gerar_url_autenticacao, trocar_codigo_por_token, buscar_paginas
from notion_client import Client
from speech_to_text.recorder import (
    listar_dispositivos_entrada,
    sugerir_dispositivo_padrao,
    iniciar_gravacao,
    parar_gravacao
)
from speech_to_text.resumo_generator import gerar_resumo_do_arquivo
import threading

app = Flask(__name__)
app.secret_key = 'teste123'

@app.route('/')
def index():
    token = session.get('notion_token')
    if not token:
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="pt-br">
            <head>
            <meta charset="UTF-8">
            <title>Conectar</title>
            <style>
                body {
                background-color: #f9f5ee;
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                text-align: center;
                }

                .logo {
                font-weight: bold;
                font-size: 2em;
                margin-bottom: 40px;
                color: #1e1e1e;
                animation: fadeSlideIn 1.2s ease forwards;
                opacity: 0;
                }

                a.button {
                background-color: #d5a65d;
                color: #1e1e1e;
                padding: 15px 30px;
                border-radius: 30px;
                font-size: 1.1em;
                font-weight: bold;
                text-decoration: none;
                display: inline-block;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                animation: pulse 2.5s infinite;
                }

                a.button:hover {
                transform: scale(1.05);
                background-color: #c2954e;
                }

                @keyframes fadeSlideIn {
                0% { opacity: 0; transform: translateY(30px); }
                100% { opacity: 1; transform: translateY(0); }
                }

                @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
                }
            </style>
            </head>
            <body>
            <div class="logo">
            <strong>ResumeAI</strong><br>
            Uma plataforma open-source que transcreve, resume e envia conteúdos direto para seu Notion.<br><br>
            Clique no botão abaixo para se conectar e vivenciar a experiência do ResumeAI.
            </div>
            <a href="/login" class="button">Conectar-se ao <strong>Notion<strong></a>
            </body>
            </html>
            """)

    try:
        paginas = buscar_paginas(token)
        options = ''.join(f"<option value='{p['id']}'>{p['title']}</option>" for p in paginas)
        return render_template_string(f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <title>Selecionar Página</title>
  <style>
    body {{
      background-color: #f9f5ee;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      height: 100vh;
      text-align: center;
      margin: 0;
    }}

    h2 {{
      color: #1e1e1e;
      margin-bottom: 20px;
    }}

    form {{
      background: white;
      padding: 30px;
      border-radius: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    select {{
      padding: 12px;
      font-size: 1em;
      border-radius: 10px;
      border: 1px solid #ccc;
      margin-bottom: 20px;
      width: 250px;
    }}

    button {{
      background-color: #d5a65d;
      color: #1e1e1e;
      padding: 12px 28px;
      border: none;
      border-radius: 30px;
      font-weight: bold;
      cursor: pointer;
      transition: 0.3s;
    }}

    button:hover {{
      background-color: #c2954e;
    }}
  </style>
</head>
<body>
  <h2>Escolha onde salvar o resumo:</h2>
  <form action="/iniciar" method="post">
    <select name="page_id">{options}</select><br>
    <button type="submit">🎤 Iniciar Gravação</button>
  </form>
</body>
</html>
""")
    except Exception as e:
        return f"Erro ao carregar páginas: {str(e)}"

@app.route('/login')
def login():
    return redirect(gerar_url_autenticacao())

@app.route('/oauth/callback')
def oauth_callback():
    code = request.args.get('code')
    if not code:
        return "Erro: código não fornecido."

    try:
        token = trocar_codigo_por_token(code)
        session['notion_token'] = token
        return redirect('/')
    except Exception as e:
        return f"Erro ao autenticar: {str(e)}"

@app.route('/iniciar', methods=['POST'])
def iniciar():
    token = session.get('notion_token')
    page_id = request.form.get('page_id')
    if not token or not page_id:
        return "❌ Token de autenticação ou ID da página ausente."

    session['page_id'] = page_id

    entrada_ids = listar_dispositivos_entrada()
    if not entrada_ids:
        return "❌ Nenhum dispositivo de entrada encontrado."

    padrao = sugerir_dispositivo_padrao(entrada_ids)

    # Inicia gravação em uma thread separada
    thread = threading.Thread(target=iniciar_gravacao, args=(padrao,))
    thread.start()

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
        <meta charset="UTF-8">
        <title>Gravando...</title>
        <style>
            body {
            background-color: #f9f5ee;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
            }

            h2 {
            font-size: 2em;
            color: #1e1e1e;
            margin-bottom: 30px;
            }

            button {
            background-color: #c0392b;
            color: white;
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: 0.3s;
            }

            button:hover {
            background-color: #a93226;
            }

            button:disabled {
            background-color: #aaa;
            cursor: not-allowed;
            }
        </style>
        <script>
            function desativarBotao() {
            const botao = document.getElementById('pararBtn');
            botao.disabled = true;
            botao.innerText = 'Processando...';
            }
        </script>
        </head>
        <body>
        <h2>🎙️ Gravação em andamento...</h2>
        <form action="/parar" method="post" onsubmit="desativarBotao()">
            <button id="pararBtn" type="submit">Parar e Gerar Resumo</button>
        </form>
        </body>
        </html>
        """)

@app.route('/parar', methods=['POST'])
def parar():
    try:
        
        caminho_audio = parar_gravacao()
        titulo, resumo = gerar_resumo_do_arquivo(caminho_audio)
        session['titulo'] = titulo
        session['resumo'] = resumo

        return render_template_string(f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <title>Resumo Gerado</title>
  <style>
    body {{
      background-color: #f9f5ee;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      height: 100vh;
      text-align: center;
      padding: 40px;
    }}

    h2 {{
      font-size: 2em;
      margin-bottom: 20px;
      color: #1e1e1e;
    }}

    p {{
      font-size: 1.1em;
      color: #333;
      max-width: 800px;
      margin-bottom: 30px;
    }}

    form button {{
      background-color: #5e4b2a;
      color: white;
      padding: 12px 28px;
      border-radius: 30px;
      font-weight: bold;
      border: none;
      cursor: pointer;
      transition: 0.3s;
    }}

    form button:hover {{
      background-color: #493b21;
    }}
  </style>
</head>
<body>
  <h2>Resumo gerado com sucesso!</h2>
  <p><strong>{titulo}</strong></p>
  <p>{resumo}</p>
  <form action="/enviar" method="post">
    <button type="submit">Enviar para o Notion</button>
  </form>
</body>
</html>
""")
    except Exception as e:
        return f"Erro ao finalizar gravação: {str(e)}"

@app.route('/enviar', methods=['POST'])
def enviar():
    token = session.get('notion_token')
    page_id = session.get('page_id')
    titulo = session.get('titulo')
    resumo = session.get('resumo')

    if not token or not page_id or not titulo or not resumo:
        return "❌ Informações incompletas para envio ao Notion."

    try:
        client = Client(auth=token)
        resposta = client.pages.create(
            parent={"type": "page_id", "page_id": page_id},
            properties={
                "title": [{"type": "text", "text": {"content": titulo}}]
            },
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": resumo}}]
                }
            }]
        )
        url = resposta['url']
        return render_template_string(f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <title>Resumo Enviado</title>
  <style>
    body {{
      background-color: #f9f5ee;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      height: 100vh;
      text-align: center;
      padding: 40px;
    }}

    h2 {{
      font-size: 2em;
      margin-bottom: 30px;
      color: #1e1e1e;
    }}

    a.button {{
      background-color: #5e4b2a;
      color: white;
      padding: 12px 28px;
      font-size: 1em;
      text-decoration: none;
      border-radius: 30px;
      margin: 10px;
      font-weight: bold;
      transition: 0.3s;
    }}

    a.button:hover {{
      background-color: #493b21;
    }}
  </style>
</head>
<body>
  <h2>✅ Resumo enviado com sucesso para o Notion!</h2>
  <a href="{url}" target="_blank" class="button">📄 Ver no Notion</a>
  <a href="/" class="button">🔙 Voltar</a>
</body>
</html>
""")
    except Exception as e:
        return f"<h2>Erro:</h2><pre>{str(e)}</pre><br><a href='/'>Voltar</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
