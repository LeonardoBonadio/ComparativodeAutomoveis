import os
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field
from typing import Optional

openai.api_key = os.getenv("OPENAI_API_KEY")

class CarSpecsModel(BaseModel):
    car_name: str = Field(..., description="Nome do modelo do carro")
    fuel_type: str = Field(..., description="Tipo de combustível (ex: Gasolina, Elétrico, Híbrido)")
    engine_size: Optional[float] = Field(None, description="Cilindrada em litros (ex: 1.0, 2.0)")
    range_ev: Optional[float] = Field(None, description="Autonomia elétrica em km")
    msrp_price: float = Field(..., description="Preço sugerido ou valor aproximado em Reais (BRL)")
    summary: str = Field(..., description="Resumo de 2 frases sobre os destaques do carro")

def ExtrairDadosdosPdfs(text):
    system_prompt = """
    Você é um engenheiro de dados especialista em mercado automotivo.
    Extraia as especificações do carro do texto fornecido.
    Se o carro for elétrico, forneça a autonomia em 'range_ev' e null para 'engine_size'. 
    Se for a combustão, faça o oposto.
    Retorne estritamente um JSON.
    """
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    import json
    specs_json = json.loads(response.choices[0].message.content)
    return CarSpecsModel(**specs_json)





def get_chroma_client():
    """Inicializa o banco vetorial localmente."""
    return chromadb.Client()

def process_rag(question, user_chunks, fixed_chunks=[]):
    """
    Recebe a pergunta, os pedaços de texto dos PDFs do usuário E do PDF fixo.
    Vetoriza tudo, acha a resposta e gera o texto final.
    """
    chroma_client = get_chroma_client()
    embedding_func = OpenAIEmbeddingFunction(api_key=openai.api_key)
    
    # 1. Limpa a coleção anterior para não misturar conversas diferentes no MVP
    try:
        chroma_client.delete_collection(name="car_specs")
    except:
        pass # Se não existir, segue o jogo
        
    collection = chroma_client.create_collection(
        name="car_specs", 
        embedding_function=embedding_func
    )

    # 2. Junta o conhecimento fixo com o que o usuário subiu agora
    all_chunks = fixed_chunks + user_chunks
    ids = [f"doc_{i}" for i in range(len(all_chunks))]
    
    if not all_chunks:
        return "Nenhum documento encontrado para analisar."

    # Adiciona no banco (isso consome a API de embeddings da OpenAI)
    collection.add(documents=all_chunks, ids=ids)

    # 3. Busca no banco as partes que importam para a pergunta
    results = collection.query(
        query_texts=[question], 
        n_results=4 # Pega os 4 pedaços mais relevantes de texto
    )
    
    context_text = "\n\n---\n\n".join(results['documents'][0])

    # 4. Gera a resposta final com a IA
    rag_prompt = f"""
    Você é um assistente comparativo automotivo premium.
    Responda à dúvida do usuário usando APENAS as informações do CONTEXTO abaixo.
    Se a resposta não estiver no contexto, diga que não tem essa informação nos manuais.
    
    CONTEXTO DOS MANUAIS:
    {context_text}
    
    PERGUNTA DO USUÁRIO:
    {question}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente útil e direto."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.3, # Um pouco de criatividade na escrita, mas focado nos fatos
    )
    
    return response.choices[0].message.content