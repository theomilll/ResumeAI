
# 🎙️ Speech to Text

Uma aplicação simples de reconhecimento de voz que converte fala em texto diretamente no navegador. Desenvolvida com JavaScript puro e a API Web Speech, ela permite que usuários gravem sua voz, visualizem a transcrição em tempo real, salvem múltiplas gravações e copiem o texto com um clique.

---

## 📁 Estrutura do Projeto

- `index.html` — HTML principal da aplicação.
- `speech-to-text.js` — Script JavaScript com toda a lógica de reconhecimento de voz.
- `stylebutton.css` — Estilos visuais da aplicação.

---

## ✅ Funcionalidades

- **Reconhecimento de voz contínuo (em português)** com transcrição em tempo real.
- **Botão "Falar/Parar"** para controle do reconhecimento.
- **Lista de gravações anteriores** clicáveis para reexibir textos.
- **Botão de cópia rápida** para salvar a transcrição na área de transferência.
- **Design leve e responsivo**.

---

## 🚀 Como Usar

### 1. Pré-requisitos
- Um navegador moderno que suporte a [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API), como o Google Chrome.
- Permitir o uso do microfone quando solicitado.

### 2. Execução
1. Faça o download dos arquivos `index.html`, `speech-to-text.js`, e `stylebutton.css` em uma mesma pasta.
2. Abra o arquivo `index.html` em seu navegador.

### 3. Interação
- Clique no botão **“Falar”** para começar a gravar.
- O botão muda para **“Parar”** enquanto o áudio está sendo capturado.
- O texto transcrito aparece em tempo real.
- Ao parar, a transcrição é salva em uma lista de gravações abaixo.
- Clique em uma gravação da lista para ver novamente o conteúdo.
- Use o botão **📋 Copiar** para copiar o texto atual para a área de transferência.

---

## 🛠️ Detalhes Técnicos

- A API utilizada é `SpeechRecognition` (`window.SpeechRecognition || window.webkitSpeechRecognition`).
- Suporte ao idioma **pt-BR**.
- Reconhecimento contínuo com múltiplas entradas.
- Uso de `navigator.clipboard.writeText()` para copiar texto.
- Interface minimalista com `CSS` customizado.

---

## ⚠️ Observações

- Alguns navegadores não suportam a API de reconhecimento de voz.
- Certifique-se de testar no **Google Chrome** para melhor compatibilidade.
- A aplicação não salva dados permanentemente — tudo é mantido apenas enquanto a aba está aberta.
