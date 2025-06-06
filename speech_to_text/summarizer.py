import google.generativeai as genai
import os
from dotenv import load_dotenv
from categorizacao.src.predict_bert import BERTSummaryClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_MODEL_PATH = os.path.join(BASE_DIR, "categorizacao", "models", "bert_enhanced")

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Caminho completo do modelo BERT
bert_classifier = BERTSummaryClassifier(BERT_MODEL_PATH)

def classificar_tipo_reuniao(texto_resumo: str) -> str:
    try:
        categoria = bert_classifier.classify(texto_resumo)
        return categoria
    except Exception as e:
        print(f"[Erro] Falha ao classificar com BERT aprimorado: {e}")
        return "Erro na Classificação"


def resumir_com_gemini(texto):
    if not GOOGLE_GEMINI_API_KEY:
        print("[Aviso] GOOGLE_GEMINI_API_KEY não configurada. Retornando texto original.")
        return texto
    
    model = genai.GenerativeModel("gemini-1.5-flash-002")

    categoria = classificar_tipo_reuniao(texto)

    prompt_atualizacoes_de_projeto = """
                Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de reuniões, aulas, palestras ou eventos, e gerar um resumo analítico e completo.

                **⚠️ IMPORTANTE:** A resposta final deve conter no máximo **2000 caracteres**, incluindo espaços.

                Siga as diretrizes abaixo:

                1. Apresente um **texto introdutório corrido**, resumindo com fluidez os assuntos discutidos, de forma coerente e bem redigida.
                2. Em seguida, destaque seções específicas:
                - **Tópicos e Discussões Relevantes**: Liste os principais assuntos abordados com um parágrafo explicativo para cada.
                - **Decisões Tomadas e Acordos**: Detalhe claramente qualquer conclusão, plano de ação ou consenso.
                - **Datas, Horários e Prazos**: Identifique qualquer menção a datas e horas.
                - **Encaminhamentos e Responsabilidades**: Aponte as tarefas atribuídas, quem as executará, e até quando.
                - **Observações Importantes**: Inclua avisos, recomendações, preocupações ou ideias relevantes levantadas durante o encontro.

                **Estilo:**
                - Use linguagem formal e clara.
                - Evite listas secas; desenvolva minimamente cada ponto.
                - Mantenha a fidelidade ao conteúdo, sem interpretações subjetivas.
                - Se o conteúdo estiver confuso, organize logicamente, mas sinalize isso.

                **⚠️ Lembrete:** O resumo inteiro deve caber em até **2000 caracteres**.
                """
    

    prompt_achados_de_pesquisa = """Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de reuniões, workshops ou apresentações de achados de pesquisa, e gerar um resumo analítico e completo.

                ⚠️ IMPORTANTE: A resposta final deve conter no máximo 2000 caracteres, incluindo espaços.

                Siga as diretrizes abaixo:

                1. Apresente um texto introdutório corrido, resumindo com fluidez os principais achados, metodologias utilizadas e discussões realizadas, de forma coerente e bem redigida.

                2. Em seguida, destaque seções específicas:
                - **Principais Achados e Insights**: Explique os resultados mais relevantes encontrados na pesquisa, com um parágrafo explicativo para cada ponto.
                - **Discussões e Interpretações**: Aponte como os dados foram interpretados e quais hipóteses ou hipóteses alternativas surgiram.
                - **Implicações e Possíveis Aplicações**: Detalhe os impactos práticos, recomendações ou caminhos futuros decorrentes dos achados.
                - **Limitações e Considerações Metodológicas**: Comente sobre restrições da pesquisa ou alertas importantes discutidos.
                - **Encaminhamentos e Próximos Passos**: Indique as ações futuras acordadas, responsáveis e prazos mencionados.

                Estilo:
                - Use linguagem formal e clara.
                - Evite listas secas; desenvolva minimamente cada ponto.
                - Mantenha a fidelidade ao conteúdo, sem interpretações subjetivas.
                - Se o conteúdo estiver confuso, organize logicamente, mas sinalize isso.

                ⚠️ Lembrete: O resumo inteiro deve caber em até 2000 caracteres."""

    prompt_gestao_de_equipe ="""
                Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de reuniões, aulas, palestras ou eventos, e gerar um resumo analítico e completo.

                **⚠️ IMPORTANTE:** A resposta final deve conter no máximo **2000 caracteres**, incluindo espaços.

                Siga as diretrizes abaixo:

                1. Apresente um **texto introdutório corrido**, resumindo com fluidez os assuntos discutidos, de forma coerente e bem redigida.
                2. Em seguida, destaque seções específicas:
                - **Tópicos e Discussões Relevantes**: Liste os principais assuntos abordados com um parágrafo explicativo para cada.
                - **Decisões Tomadas e Acordos**: Detalhe claramente qualquer conclusão, plano de ação ou consenso.
                - **Datas, Horários e Prazos**: Identifique qualquer menção a datas e horas.
                - **Encaminhamentos e Responsabilidades**: Aponte as tarefas atribuídas, quem as executará, e até quando.
                - **Observações Importantes**: Inclua avisos, recomendações, preocupações ou ideias relevantes levantadas durante o encontro.

                **Estilo:**
                - Use linguagem formal e clara.
                - Evite listas secas; desenvolva minimamente cada ponto.
                - Mantenha a fidelidade ao conteúdo, sem interpretações subjetivas.
                - Se o conteúdo estiver confuso, organize logicamente, mas sinalize isso.

                **⚠️ Lembrete:** O resumo inteiro deve caber em até **2000 caracteres**.
                """
    prompt_reunioes_com_clientes = """Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de reuniões com clientes e gerar um resumo analítico e completo, preservando a clareza, os objetivos comerciais e os compromissos firmados.

                ⚠️ IMPORTANTE: A resposta final deve conter no máximo 2000 caracteres, incluindo espaços.

                Siga as diretrizes abaixo:

                1. Apresente um texto introdutório corrido, resumindo de forma clara o propósito da reunião, os temas abordados e o contexto do cliente.

                2. Em seguida, destaque seções específicas:
                - **Necessidades e Expectativas do Cliente**: Aponte os principais pedidos, dores ou metas apresentadas pelo cliente.
                - **Soluções Apresentadas e Discussões Técnicas**: Descreva as propostas ou alternativas discutidas, com um parágrafo para cada solução relevante.
                - **Decisões Tomadas e Acordos Comerciais**: Indique claramente as decisões consensuais, serviços contratados ou etapas aprovadas.
                - **Prazos, Entregas e Condições**: Registre datas combinadas, próximos marcos de entrega ou condições acordadas.
                - **Encaminhamentos e Responsáveis**: Liste as tarefas definidas, indicando responsáveis e prazos associados.
                - **Observações Relevantes**: Inclua preocupações levantadas, feedbacks do cliente ou pontos que requerem acompanhamento.

                Estilo:
                - Use linguagem formal e clara.
                - Evite listas secas; desenvolva minimamente cada ponto.
                - Mantenha a fidelidade ao conteúdo, sem interpretações subjetivas.
                - Se o conteúdo estiver confuso, organize logicamente, mas sinalize isso.

                ⚠️ Lembrete: O resumo inteiro deve caber em até 2000 caracteres."""
    prompt_outras = """Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de encontros, reuniões, conversas ou eventos diversos, e gerar um resumo analítico e completo, mesmo quando a natureza da reunião não estiver claramente definida.

                ⚠️ IMPORTANTE: A resposta final deve conter no máximo 2000 caracteres, incluindo espaços.

                Siga as diretrizes abaixo:

                1. Apresente um texto introdutório corrido, contextualizando a reunião de forma clara, identificando os assuntos abordados, mesmo que não tenha sido possível classificar a reunião em uma categoria específica.

                2. Em seguida, destaque seções específicas:
                - **Tópicos Identificados**: Liste os temas tratados, agrupando por afinidade ou relevância, com um parágrafo explicativo para cada tópico.
                - **Discussões e Pontos Relevantes**: Descreva os argumentos, dúvidas ou ideias trocadas, com base nos trechos falados.
                - **Possíveis Decisões ou Sinais de Alinhamento**: Registre qualquer indício de decisão ou concordância, mesmo que informal.
                - **Prazos, Datas ou Menções Temporais**: Anote qualquer citação a datas, prazos ou períodos relevantes.
                - **Encaminhamentos e Responsáveis (se houver)**: Caso tenham sido definidos, aponte tarefas, responsáveis e prazos.
                - **Outros Elementos Importantes**: Inclua observações, preocupações ou sugestões dignas de registro, mesmo que isoladas.

                Estilo:
                - Use linguagem formal e clara.
                - Evite listas secas; desenvolva minimamente cada ponto.
                - Mantenha a fidelidade ao conteúdo, sem interpretações subjetivas.
                - Se o conteúdo estiver confuso, organize logicamente, mas sinalize isso.

                ⚠️ Lembrete: O resumo inteiro deve caber em até 2000 caracteres."""

    prompt_map = {
        "Atualizações de Projeto": prompt_atualizacoes_de_projeto,
        "Achados de Pesquisa": prompt_achados_de_pesquisa,
        "Gestão de Equipe": prompt_gestao_de_equipe,
        "Reuniões com Clientes": prompt_reunioes_com_clientes,
        "Outras": prompt_outras,
        "Categoria Desconhecida": prompt_outras,
        "Erro na Classificação": prompt_outras
    }

    prompt_escolhido = prompt_map.get(categoria, prompt_outras)

    prompt_final = f"{prompt_escolhido}\n\nTexto Original:\n{texto}"
    resposta = model.generate_content(prompt_final)
    return resposta.text.strip()
