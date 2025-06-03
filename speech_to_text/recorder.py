import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from datetime import datetime

taxa_amostragem = 44100
frames = []

def callback(indata, frames_count, time_info, status):
    frames.append(indata.copy())

def listar_dispositivos_entrada():
    dispositivos = sd.query_devices()
    entrada_ids = []

    print("\nDispositivos de entrada disponíveis:")
    for i, d in enumerate(dispositivos):
        if d['max_input_channels'] > 0:
            print(f"[{i}] {d['name']}")
            entrada_ids.append(i)

    return entrada_ids

def sugerir_dispositivo_padrao(entrada_ids):
    for i in entrada_ids:
        nome = sd.query_devices(i)['name'].lower()
        if "vb-cable" in nome or "stereo mix" in nome:
            return i
    return entrada_ids[0] if entrada_ids else None

def gravar_sem_limite(dispositivo):
    global frames
    frames = [] 

    info = sd.query_devices(dispositivo)
    canais = min(info['max_input_channels'], 2)

    print(f"\nGravando com dispositivo '{info['name']}' usando {canais} canal(is)... Pressione ENTER para parar.")

    stream = sd.InputStream(samplerate=taxa_amostragem, channels=canais, callback=callback, device=dispositivo)
    with stream:
        input()

    audio = np.concatenate(frames, axis=0)

    os.makedirs("gravacoes", exist_ok=True)

    nome_arquivo = datetime.now().strftime("gravacoes/reuniao_%Y%m%d_%H%M%S.wav")
    write(nome_arquivo, taxa_amostragem, audio)

    print(f"\nGravação salva: {nome_arquivo}")
    return nome_arquivo
