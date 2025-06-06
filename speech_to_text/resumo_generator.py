from .recorder import listar_dispositivos_entrada, sugerir_dispositivo_padrao, parar_gravacao
from speech_to_text.transcriber import transcribe_audio
from speech_to_text.summarizer import resumir_com_gemini
from resumo.ML_Summarizer import gerar_resumo_completo
from datetime import datetime

def gerar_resumo_do_arquivo(caminho_audio):
    texto = transcribe_audio(caminho_audio)
    resumo_intermediario = gerar_resumo_completo(texto)
    resumo = resumir_com_gemini(resumo_intermediario)
    titulo = "Resumo de Reunião - " + datetime.now().strftime("%d/%m/%Y %H:%M")
    return titulo, resumo