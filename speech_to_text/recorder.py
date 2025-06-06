import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from datetime import datetime
import threading

taxa_amostragem = 44100
frames = []
gravando = threading.Event()
stream = None
arquivo_saida = None

def callback(indata, frames_count, time_info, status):
    if gravando.is_set():
        frames.append(indata.copy())

def listar_dispositivos_entrada():
    dispositivos = sd.query_devices()
    entrada_ids = []
    for i, d in enumerate(dispositivos):
        if d['max_input_channels'] > 0:
            entrada_ids.append(i)
    return entrada_ids

def sugerir_dispositivo_padrao(entrada_ids):
    for i in entrada_ids:
        nome = sd.query_devices(i)['name'].lower()
        if "vb-cable" in nome or "stereo mix" in nome:
            return i
    return entrada_ids[0] if entrada_ids else None

def iniciar_gravacao(dispositivo):
    global frames, stream, arquivo_saida
    frames = []

    info = sd.query_devices(dispositivo)
    canais = min(info['max_input_channels'], 2)

    gravando.set()

    stream = sd.InputStream(samplerate=taxa_amostragem, channels=canais, callback=callback, device=dispositivo)
    stream.start()
    print(f"🎙️ Iniciando gravação com '{info['name']}'...")

def parar_gravacao():
    global arquivo_saida, stream
    gravando.clear()

    if stream:
        stream.stop()
        stream.close()

    audio = np.concatenate(frames, axis=0)
    os.makedirs("gravacoes", exist_ok=True)

    arquivo_saida = datetime.now().strftime("gravacoes/reuniao_%Y%m%d_%H%M%S.wav")
    write(arquivo_saida, taxa_amostragem, audio)
    print(f"💾 Gravação salva: {arquivo_saida}")
    return arquivo_saida
