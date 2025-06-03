# Resumidor de Textos com Avaliação por BERTScore

Este projeto realiza a geração de resumos automáticos para textos `.txt` utilizando o modelo pré-treinado `facebook/bart-large-cnn`, da Hugging Face. Além de gerar os resumos,
ele também calcula a qualidade de cada resumo com base na métrica BERTScore.

Para cada arquivo de texto, são geradas três versões de resumo, mantendo 25%, 50% e 75% do número original de palavras.

---

## Pré-requisitos

- Python 3.8 ou superior
- Instale as dependências com:

```
pip install -r requirements.txt
```

## Estrutura do Projeto

├── main.py                  # Script principal que gera e avalia os resumos
├── texts/                   # Pasta onde devem ser colocados os arquivos .txt de entrada
├── resumos/                 # Pasta onde os resumos gerados serão salvos
└── requirements.txt         # Dependências do projeto

## Como Usar
Coloque seus arquivos .txt dentro da pasta texts/.

Execute o script principal:

```python3 main.py ```

Os resumos serão gerados automaticamente dentro da pasta resumos/. Para cada texto, serão criados três arquivos correspondentes às taxas de resumo 25%, 50% e 75%.

## Detalhes do Funcionamento

Modelo Utilizado: facebook/bart-large-cnn
Pipeline: Hugging Face Transformers (pipeline("summarization"))
Tamanhos de resumo: calculados proporcionalmente ao número de palavras do texto original.
Avaliação: utiliza o bert_score para medir a similaridade semântica entre o resumo e o texto original.

## Exemplo de Saída

Dado um arquivo relatorio.txt, o programa irá gerar:

resumos/
├── relatorio_resumo_25%.txt
├── relatorio_resumo_50%.txt
└── relatorio_resumo_75%.txt

Cada arquivo conterá:

O resumo gerado com a taxa indicada
A quantidade de palavras alvo
O valor da métrica BERTScore para o resumo

## Observações
O script suporta apenas textos em inglês.

O modelo BART pode demorar para processar textos longos dependendo do hardware.

Precisa de melhorias na parte de semântica, visto que ele suporta apenas textos em inglês, a parte de traduzir ele, e depois traduzir de volta faz com que 
aconteçam erros gramaticáis.

O modelo BART não obedece completamente a função de permanecer fiel a ''25,50 e 75%'' do texto original, por mais que tenha aumentado o resumo, ainda tende
a deixar ele mais resumido, independentemente de como você força ele a aumentar o resumo. Além disso, por mais que não seja sempre, ainda tende a ''comer'' 
algumas palavras por achar que seja necessário a frase inteira.

Modelo sem Fine-Tunning. Em construção. Trazer uma database maior para treinar o modelo e evitar que ele corte palavras pela metade ou para ajudar ele a especificar mais o resumo.
