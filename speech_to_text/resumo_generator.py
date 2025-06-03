from speech_to_text.recorder import listar_dispositivos_entrada, sugerir_dispositivo_padrao, gravar_sem_limite
from speech_to_text.transcriber import transcribe_audio
from speech_to_text.summarizer import resumir_com_gemini
from resumo.ML_Summarizer import gerar_resumo_completo
from datetime import datetime
import sounddevice as sd

def gerar_resumo():
    print("🎤 Captura de reunião iniciada")

    entrada_ids = listar_dispositivos_entrada()
    if not entrada_ids:
        raise RuntimeError("Nenhum dispositivo de entrada detectado.")

    padrao = sugerir_dispositivo_padrao(entrada_ids)
    print(f"\nDispositivo sugerido: [{padrao}] {sd.query_devices(padrao)['name']}")
    usar = input("Deseja usar este dispositivo? (S/n): ").strip().lower()

    if usar == "n":
        while True:
            try:
                escolha = int(input("Digite o número do dispositivo desejado: "))
                if escolha in entrada_ids:
                    padrao = escolha
                    break
                else:
                    print("Número inválido. Tente novamente.")
            except ValueError:
                print("Entrada inválida. Use apenas números.")
    else:
        escolha = padrao

    caminho_audio = gravar_sem_limite(escolha)

    print("\n📝 Transcrevendo o áudio...")
    texto = transcribe_audio(caminho_audio)

    print("\n🗒️ Transcrição:")
    print(texto)

    resumo_intermediario = gerar_resumo_completo(texto)
    resumo = resumir_com_gemini(resumo_intermediario)
    titulo = "Resumo de Reunião - " + datetime.now().strftime("%d/%m/%Y %H:%M")

    return titulo, resumo
