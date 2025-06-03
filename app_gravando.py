
from flask import Flask, request, render_template_string, redirect, session
from ResumeAI_NotionExporter.notion_auth import gerar_url_autenticacao, trocar_codigo_por_token, buscar_paginas
from notion_client import Client
from speech_to_text.resumo_generator import gerar_resumo

app = Flask(__name__)
app.secret_key = 'teste123'

@app.route('/')
def index():
    token = session.get('notion_token')
    if not token:
        return '<a href="/login">🔐 Conectar ao Notion</a>'

    try:
        paginas = buscar_paginas(token)
        options = ''.join(f"<option value='{p['id']}'>{p['title']}</option>" for p in paginas)
        return render_template_string(f'''
            <h2>Escolha onde salvar o resumo:</h2>
            <form action="/gravar" method="post">
                <select name="page_id">{options}</select><br><br>
                <button type="submit">🎤 Iniciar Gravação</button>
            </form>
        ''')
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

@app.route('/gravar', methods=['POST'])
def gravar():
    token = session.get('notion_token')
    page_id = request.form.get('page_id')
    if not token or not page_id:
        return "❌ Token de autenticação ou ID da página ausente."

    session['page_id'] = page_id
    try:
        titulo, resumo = gerar_resumo()
        session['titulo'] = titulo
        session['resumo'] = resumo

        return render_template_string(f'''
            <h2>✅ Resumo gerado com sucesso!</h2>
            <p><strong>{titulo}</strong></p>
            <p>{resumo}</p>
            <form action="/enviar" method="post">
                <button type="submit">📤 Enviar para Notion</button>
            </form>
        ''')
    except Exception as e:
        return f"Erro ao gravar/transcrever: {str(e)}"

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
        url = resposta['url']
        return render_template_string(f'''
            <h2>✅ Resumo enviado para o Notion!</h2>
            <p><a href="{url}" target="_blank">👉 Ver no Notion</a></p>
            <a href="/">Voltar</a>
        ''')
    except Exception as e:
        return f"<h2>Erro:</h2><pre>{str(e)}</pre><br><a href='/'>Voltar</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
