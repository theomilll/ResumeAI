from transformers import pipeline
import os
from bert_score import score

resumidor = pipeline("summarization", model="facebook/bart-large-cnn")

input_dir = "texts"
output_dir = "resumos"
os.makedirs(output_dir, exist_ok=True)

# Taxas de conteúdo pra manter
taxas_conteudo = {
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75
}

for nome_arquivo in os.listdir(input_dir):
    if nome_arquivo.endswith(".txt"):
        caminho_arquivo = os.path.join(input_dir, nome_arquivo)
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            texto = f.read()

        total_palavras = len(texto.split())

        for rotulo, taxa in taxas_conteudo.items():
            num_palavras_desejadas = int(total_palavras * taxa)
            max_tokens = int(num_palavras_desejadas * 1.5)  # tokens ~= palavras * 1.5
            min_tokens = max(20, int(max_tokens * 0.5))

            # Geração do resumo
            try:
                resumo_bruto = resumidor(
                    texto,
                    max_length=max_tokens,
                    min_length=min_tokens,
                    do_sample=False
                )[0]['summary_text']

                palavras_resumo = resumo_bruto.split()
                resumo_final = ' '.join(palavras_resumo[:num_palavras_desejadas])

            except Exception as e:
                print(f"Erro ao resumir {nome_arquivo} com taxa {rotulo}: {e}")
                continue

            # Avaliação com BERTScore
            _, _, bert_scores = score([resumo_final], [texto], lang="pt", verbose=False)
            bert_score = bert_scores.mean().item()

            nome_base = os.path.splitext(nome_arquivo)[0]
            nome_resumo = f"{nome_base}_resumo_{rotulo}.txt"
            caminho_resumo = os.path.join(output_dir, nome_resumo)

            # Salvando resultado
            with open(caminho_resumo, "w", encoding="utf-8") as out:
                out.write(f"Resumo ({rotulo} do conteúdo original - {num_palavras_desejadas} palavras):\n")
                out.write(resumo_final + "\n\n")
                out.write(f"Qualidade do Resumo (BERTScore): {bert_score}\n")

            print(f"[✓] Resumo {rotulo} de '{nome_arquivo}' salvo em '{nome_resumo}' com {num_palavras_desejadas} palavras")
