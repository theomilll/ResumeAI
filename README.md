# 📝 ResumeAI

**ResumeAI** é uma plataforma open‑source que transcreve, resume, traduz e organiza automaticamente conteúdos de aulas, palestras e reuniões, entregando tudo no seu workspace do Notion. Nosso foco é tornar o aprendizado mais inclusivo, eficiente e acessível para todos os estudantes.

---

## 🌟 Introdução

Muitas informações se perdem em aulas longas ou reuniões densas. Pessoas com deficiência ou alunos de intercâmbio ainda lidam com barreiras de acessibilidade e idioma. O ResumeAI resolve esses problemas ao transformar áudio ou texto bruto em resumos claros, categorizados e prontos para consulta.

---

## 🎯 Propósito & Público‑Alvo

- **Estudantes** que desejam revisar conteúdos rapidamente.
- **Pessoas com deficiência** que precisam de transcrições acessíveis e leitura adaptada.
- **Intercambistas** que enfrentam barreiras linguísticas e necessitam de traduções.

---

## 🚀 Funcionalidades Principais

1. **Transcrição Automática Inteligente** – converte áudio em texto com alta precisão.
2. **Resumos Automatizados** – destaca os pontos‑chave para revisão rápida.
3. **Integração com o Notion** – organiza o material em páginas ou bancos de dados.
4. **Tradução Multilíngue** – facilita o entendimento para estudantes estrangeiros.
5. **Acessibilidade Aprimorada** – leitura em voz alta e formatação inclusiva.
---

## 🧠 Categorização (modelo treinado por nosso grupo)

- **TF‑IDF** → transforma cada resumo em um vetor esparso de uni‑ e bigramas ponderados.
- **Logistic Regression** → classificador linear (`solver="liblinear"`, `class_weight='balanced'`).
- **Treino estratificado** → pipeline ajusta vetorização + modelo e avalia por *accuracy* e F1.

## 📂 Estrutura do Repositório

```
.
├── categorizacao/                # Modelos e scripts de NLP (Python)
├── ResumeAI_NotionExporter/      # Integração com a API do Notion (Python)
├── speech-to-text/               # Web‑app de transcrição (HTML/JS/CSS)
└── README.md                     # ← este arquivo
```

Cada subpasta possui um **README** próprio com instruções completas de instalação, uso e testes.

---

## 🌱 Roadmap

- Suporte a mais idiomas de transcrição
- Dashboard web full‑stack para gerenciamento de resumos
- Sincronização bidirecional com o Notion
- Upload de arquivos de mídia (áudio/vídeo) gravados

Contribuições são bem‑vindas! Abra um *issue* ou *pull request* para participar.

---

## 🛠️ Requisitos de Sistema

Além das dependências Python listadas em `requirements.txt`, este projeto requer os seguintes programas instalados no sistema operacional:

---

### ✅ Java (necessário para o funcionamento do `language_tool_python`)

- O LanguageTool agora exige **Java 17 ou superior**.
- Baixe o **JDK 17 para Windows (Instalador .exe)** diretamente da Oracle:
  
  🔗 [https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.exe](https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.exe)

- Após a instalação:
  1. Reinicie o terminal ou VS Code.
  2. Verifique com:
     ```bash
     java -version
     ```

---

### ✅ FFmpeg (necessário para o Whisper)

- O Whisper usa `ffmpeg` para processar arquivos de áudio (.wav, .mp3, etc.).
- Baixe uma build confiável no site:

  🔗 [https://www.gyan.dev/ffmpeg/builds/#](https://www.gyan.dev/ffmpeg/builds/#)

#### 📦 Instalação no Windows:
1. Baixe o arquivo `ffmpeg-release-full.zip`.
2. Extraia o conteúdo em uma pasta fixa, como `C:\ffmpeg`.
3. Adicione `C:\ffmpeg\bin` à variável de ambiente `Path`:
   - Menu Iniciar → “Variáveis de Ambiente” → Edite `Path`.

4. Reinicie o terminal e teste com:

```bash
ffmpeg -version

## 📜 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para detalhes.