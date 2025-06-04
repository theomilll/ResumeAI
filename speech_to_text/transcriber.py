import whisper 
import language_tool_python
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

tool = language_tool_python.LanguageTool('pt-BR')

TEXTGEARS_API_KEY = os.getenv("TEXTGEARS_API_KEY", "")

#filtro 1
def corrigir_com_languagetool(texto):
    matches = tool.check(texto)
    return language_tool_python.utils.correct(texto, matches)

#filtro 2
def corrigir_com_textgears(texto):
    if not TEXTGEARS_API_KEY:
        print("[Aviso] TEXTGEARS_API_KEY não configurada. Pulando correção do TextGears.")
        return texto
    
    try:
        response = requests.get(
            "https://api.textgears.com/grammar",
            params={
                "text": texto,
                "language": "pt-BR",
                "key": TEXTGEARS_API_KEY
            }
        )
        data = response.json()
        for error in data.get("response", {}).get("errors", []):
            if error["better"]:
                texto = texto.replace(error["bad"], error["better"][0])
    except Exception as e:
        print(f"[Aviso] Falha ao usar TextGears: {e}")
    return texto


def transcribe_audio(audio_path): 

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    texto = result["text"]

    texto_corrigido = corrigir_com_languagetool(texto)
    texto_final = corrigir_com_textgears(texto_corrigido)

    return texto_final