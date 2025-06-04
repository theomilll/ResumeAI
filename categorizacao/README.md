# Ferramenta de Categorização de Resumos de Reuniões

Esta ferramenta permite treinar e usar modelos de Machine Learning para categorizar resumos de reuniões em cinco categorias:
- Atualizações de Projeto
- Achados de Pesquisa
- Gestão de Equipe
- Reuniões com Clientes
- Outras

## 🚀 Novidades - BERT Enhanced

**Novo modelo com 78% de acurácia!** Principais melhorias implementadas:

- ✅ **+53% de melhoria** comparado ao modelo anterior
- ✅ **Aumento de dados automático** com text augmentation
- ✅ **Balanceamento inteligente** de classes desbalanceadas  
- ✅ **Análise de confiança** para cada predição
- ✅ **Contexto expandido** (256 vs 128 tokens)
- ✅ **Early stopping** e regularização avançada
- ✅ **Integração completa** na interface web

**Performance por categoria:**
- 🎯 Achados de Pesquisa: **100%**
- 🎯 Reuniões com Clientes: **100%** 
- 🎯 Atualizações de Projeto: **90%**
- 🎯 Gestão de Equipe: **87.5%**

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
│   ├── train_bert.py       # Pipeline BERT legacy
│   ├── train_bert_enhanced.py  # Pipeline BERT Enhanced ⭐ (NOVO)
│   ├── predict_tfidf.py    # CLI para modelo TF-IDF
│   └── predict_bert.py     # CLI para modelo BERT Enhanced ⭐ (NOVO)
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

### 2) BERT Enhanced (RECOMENDADO - 78% de Acurácia) ⭐

```bash
python3 src/train_bert_enhanced.py \
  --data_path data/resumos.csv \
  --use_augmentation \
  --epochs 10 \
  --batch_size 16
```

### 3) BERT Legacy

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

#### BERT Enhanced ⭐ (RECOMENDADO)
- **Acurácia**: 78% no conjunto de validação (significativa melhoria)
- **Modelo base**: Fine-tuning de `neuralmind/bert-base-portuguese-cased`
- **Melhorias implementadas**:
  - ✅ **Aumento de dados**: Técnicas de text augmentation (sinonímia, troca de palavras, exclusão aleatória)
  - ✅ **Balanceamento de classes**: Pesos automáticos para lidar com desbalanceamento
  - ✅ **Hiperparâmetros otimizados**: Learning rate com warmup, weight decay, gradient accumulation
  - ✅ **Contexto expandido**: Sequências de até 256 tokens (vs. 128 anterior)
  - ✅ **Early stopping**: Evita overfitting com paciência de 3 épocas
  - ✅ **Regularização**: Dropout aumentado para melhor generalização
- **Performance por categoria**:
  - Achados de Pesquisa: 100% acurácia
  - Reuniões com Clientes: 100% acurácia  
  - Atualizações de Projeto: 90% acurácia
  - Gestão de Equipe: 87.5% acurácia
  - Outras: Categoria mais desafiadora (necessita mais dados)
- **Vantagens**: Melhor precisão, análise de confiança, tratamento robusto de classes desbalanceadas
- **Caso de uso**: Aplicações de produção que exigem alta acurácia

#### BERT Legacy
- Pipeline: fine-tuning básico de `neuralmind/bert-base-portuguese-cased`
- Configurações: epochs=5, batch_size=8, max_len=128
- Vantagens: Mais rápido para treinar, menor uso de memória
- Caso de uso: Testes rápidos e desenvolvimento inicial

## Como Prever

### BERT Enhanced (RECOMENDADO) ⭐

```bash
# Predição simples
python3 src/predict_bert.py "Seu texto de reunião aqui"

# Com análise de confiança
python3 src/predict_bert.py "Seu texto de reunião aqui" --show_confidence
```

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
