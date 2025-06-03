# app_main.py

from flask import Flask, request, render_template_string, redirect, session
from ResumeAI_NotionExporter.notion_writer import salvar_resumo_notion
from ResumeAI_NotionExporter.notion_auth import gerar_url_autenticacao, trocar_codigo_por_token, buscar_paginas

app = Flask(__name__)
app.secret_key = 'teste123'

@app.route('/')
def index():
    token = session.get('notion_token')
    if not token:
        return '<a href="/login">Conectar ao Notion</a>'
    
    try:
        paginas = buscar_paginas(token)
        options = ''.join(
            f"<option value='{p['id']}'>{p['title']}</option>" for p in paginas
        )
        return render_template_string(f"""
            <h2>Escolha onde salvar o resumo:</h2>
            <form action="/executar" method="post">
                <select name="page_id">{options}</select><br><br>
                <button type="submit">🎤 Gravar e Salvar Resumo</button>
            </form>
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
        paginas = buscar_paginas(token)

        options = ''.join(
            f"<option value='{p['id']}'>{p['title']}</option>" for p in paginas
        )

        return render_template_string(f"""
            <h2>Escolha onde salvar o resumo:</h2>
            <form action="/executar" method="post">
                <select name="page_id">{options}</select><br><br>
                <button type="submit">🎤 Gravar e Salvar Resumo</button>
            </form>
        """)

    except Exception as e:
        return f"Erro ao autenticar: {str(e)}"

@app.route('/executar', methods=['POST'])
def executar():
    token = session.get('notion_token')
    page_id = request.form.get('page_id')

    if not token or not page_id:
        return "Token de autenticação ou ID da página ausente."

    try:
        resposta = salvar_resumo_notion(token, page_id)
        url = resposta['url']
        return render_template_string(f"""
            <h2>Resumo criado com sucesso!</h2>
            <p><a href="{url}" target="_blank">Ver no Notion</a></p>
            <a href="/">Voltar</a>
        """)
    except Exception as e:
        return f"<h2>Erro:</h2><pre>{str(e)}</pre><br><a href='/'>Voltar</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
