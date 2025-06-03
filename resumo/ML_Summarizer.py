from transformers import pipeline
from bert_score import score

# Inicializa o pipeline uma vez
resumidor = pipeline("summarization", model="facebook/bart-large-cnn")

def gerar_resumo_completo(texto):
    total_palavras = len(texto.split())
    num_palavras_desejadas = total_palavras
    max_tokens = int(num_palavras_desejadas * 1.5)
    min_tokens = max(50, int(max_tokens * 0.8))

    try:
        resumo_bruto = resumidor(
            texto,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False
        )[0]['summary_text']

        resumo_final = resumo_bruto.strip()

        # Avaliação opcional com BERTScore
        _, _, bert_scores = score([resumo_final], [texto], lang="pt", verbose=False)
        bert_score_val = bert_scores.mean().item()

        print(f"[✓] Resumo com ML gerado. BERTScore: {bert_score_val:.4f}")
        return resumo_final

    except Exception as e:
        print(f"[Erro] Falha ao gerar resumo com ML: {e}")
        return texto  # fallback: retorna o texto completo
