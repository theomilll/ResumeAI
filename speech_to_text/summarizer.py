import google.generativeai as genai

genai.configure(api_key="")

def resumir_com_gemini(texto):
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    
    prompt = f"""
Você é um assistente especializado em registrar, resumir e relatar conteúdos falados de forma estruturada e clara. Sua tarefa é analisar transcrições ou descrições de reuniões, aulas, palestras ou eventos, e gerar um resumo analítico e completo.

**⚠️ IMPORTANTE:** A resposta final deve conter no máximo **2000 caracteres**, incluindo espaços.

**Texto Original:** 
{texto}

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

    resposta = model.generate_content(prompt)
    return resposta.text.strip()
