const toggleButton = document.querySelector("#toggle")
const copyButton = document.querySelector("#copy")
const textElement = document.querySelector("#text")
const listaGravacoes = document.querySelector("lista-gravacoes");

let recognition= null;
let isRecording = false;
let gravacoes = [];
let textoAtual = "";

toggleButton.addEventListener("click", toggleRecognition)

function toggleRecognition(){
    if(!isRecording) {
        startRecognition();
        toggleButton.textContent = "Parar";
        toggleButton.classList.add("Stop");
        toggleButton.classList.remove("talk");
    } else {
        stopRecognition();
        toggleButton.textContent = "Falar";
        toggleButton.classList.add("talk");
        toggleButton.classList.remove("stop");
    }
    isRecording = !isRecording;
}

function startRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Seu navegador não suporta reconhecimento de voz.");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "pt-br";
    recognition.interimResults = false;
    recognition.continuous = true;

    textoAtual = "";
    textElement.textContent = "Ouvindo...\n";


    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript + " ";
        }
        textoAtual += transcript.trim() + "\n";
        textElement.textContent = textoAtual;
    };

    recognition.onerror = (event) => {
        console.error("Erro no reconhecimento:", event.error);
        textElement.textContent += "\nErro: " + event.error;
    };

    recognition.onend = () => {
        if (isRecording) {
            recognition.start();
        }
    };

    recognition.start();
}

function stopRecognition() {
    if (recognition) {
        recognition.onend = null;
        recognition.stop();
    }

    if (textoAtual.trim() !=="") {
        gravacoes.push(textoAtual.trim());
        rederizarLista();
    }

    textoAtual = "";
}

function renderizarLista() {
    listaGravacoes.innerHTML = "";

    gravacoes.forEach((texto, index) => {
        const item = document.createElement("li");
        item.textContent = `Gravação ${index + 1}`;
        item.style.cursor = "pointer";

        item.addEventListener("click", () => {
            textElement.textContent = texto;
        });

        listaGravacoes.appendChild(item);
    });
}


copyButton.addEventListener("click", () => {
    navigator.clipboard.writeText(textElement.textContent);
    setTimeout(() => {
        alert("Texto copiado!");
    }, 300);
});