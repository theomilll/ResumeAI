# Ferramenta de Categorização de Resumos de Reuniões

Esta ferramenta permite treinar e usar modelos de Machine Learning para categorizar resumos de reuniões em cinco categorias:
- Atualizações de Projeto
- Achados de Pesquisa
- Gestão de Equipe
- Reuniões com Clientes
- Outras

## Pré-requisitos

- Python 3.8 ou superior
- Instale as dependências:

  ```bash
  pip install -r requirements.txt
  ```

## Estrutura do Projeto

```
├── data/
│   └── resumos.csv         # Seu dataset de treinamento (colunas: text, category)
├── models/                 # Modelos e encoders salvos
├── src/                    # Código-fonte
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train_tfidf.py      # Pipeline TF-IDF + LogisticRegression
│   ├── train_bert.py       # Pipeline de fine-tuning BERT em português
│   └── predict_tfidf.py    # CLI para modelo TF-IDF + LogisticRegression
├── tests/                  # Testes automatizados
└── requirements.txt
```

## Como Treinar

### 1) TF-IDF + LogisticRegression (maior percentual de sucesso nos treinos realizados)

```bash
python3 src/train_tfidf.py \
  --data_path data/resumos.csv \
  --label_column category \
  --output_dir models/tfidf
```

### 2) BERT

```bash
python3 src/train_bert.py \
  --data_path data/resumos.csv \
  --label_column category \
  --epochs 5 \
  --batch_size 8 \
  --output_dir models/bert_finetuned
```

### Explicação das Estratégias de Treino

#### TF-IDF + LogisticRegression (maior percentual de sucesso nos treinos realizados)
- Vetorização: TF-IDF com uni- e bigramas (max_features=10000).
- Classificador: LogisticRegression com solver liblinear e class_weight='balanced'.
- Vantagens: rápido, leve, dependências mínimas, serve como baseline para comparar.
- Caso de uso: validações e testes iniciais, datasets menores ou desempenho em tempo real.

#### BERT
- Pipeline: fine-tuning de `neuralmind/bert-base-portuguese-cased` usando HuggingFace Trainer.
- Configurações chave: epochs=5, batch_size=8, max_len=128, seed fixo, validação estratificada.
- Vantagens: melhor compreensão semântica e contexto, maior precisão em linguagem natural.
- Caso de uso: aplicações que necessitam de alta acurácia e toleram tempo de treino e inferência maiores.

## Como Prever

### TF-IDF + LogisticRegression

```bash
python3 src/predict_tfidf.py "Seu texto de reunião aqui"
```

#### Dica para facilitar a execução

Adicione um alias no seu shell (~/.bashrc ou ~/.zshrc):

```bash
alias predict="python3 src/predict_tfidf.py"
```

A partir de então, basta executar:

```bash
predict "Resumo da reunião"
```
```
