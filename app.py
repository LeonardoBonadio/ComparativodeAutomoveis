import streamlit as st
import pandas as pd
import os
import time

# Importando os nossos módulos personalizados
from ExtrairPdf import extract_text_from_pdf, chunk_text
from ia_services import extract_structured_specs, process_rag

# Configuração inicial da página (Layer 3 - Interface)
st.set_page_config(
    page_title="Comparador Automotivo IA", 
    page_icon="🚙", 
    layout="wide"
)

# Função para carregar a base de conhecimento fixa e deixá-la em cache
@st.cache_data
def load_fixed_knowledge():
    """Lê automaticamente todos os PDFs da pasta 'base_conhecimento'."""
    fixed_text = ""
    fixed_dir = "base_conhecimento"
    
    # Verifica se a pasta existe
    if os.path.exists(fixed_dir):
        for filename in os.listdir(fixed_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(fixed_dir, filename)
                # Abre o arquivo local no modo binário para leitura
                with open(file_path, "rb") as f:
                    fixed_text += extract_text_from_pdf(f)
                    
    # Retorna o texto já fatiado (chunked) para o RAG
    return chunk_text(fixed_text) if fixed_text else []


def main():
    st.title("🚙 Assistente Comparativo Automotivo Inteligente")
    st.markdown("""
        Bem-vindo! Este sistema analisa manuais e fichas técnicas de veículos.
        Ele utiliza um documento base fixo de mercado e permite que você adicione PDFs de modelos específicos para comparação.
    """)

    # Carrega a base fixa logo ao iniciar o app
    fixed_chunks = load_fixed_knowledge()
    if fixed_chunks:
        st.success(f"📚 Base de conhecimento fixa carregada com sucesso ({len(fixed_chunks)} blocos de texto).")
    else:
        st.warning("Nenhum PDF fixo encontrado na pasta 'base_conhecimento'. O sistema funcionará apenas com os arquivos enviados.")

    # --- MENU LATERAL (Layer 3) ---
    with st.sidebar:
        st.header("1️⃣ Envie Novos Manuais")
        uploaded_files = st.file_uploader(
            "Adicione PDFs de carros para comparar:", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        st.header("2️⃣ Busca Inteligente (RAG)")
        user_question = st.text_input(
            "O que deseja saber?", 
            placeholder="Qual a diferença de porta-malas entre os modelos?"
        )
        
        process_button = st.button("Analisar Dados e Responder", use_container_width=True)

    # --- PROCESSAMENTO PRINCIPAL ---
    if process_button:
        all_text_chunks = []
        extracted_data_raw = []
        
        if uploaded_files:
            with st.spinner("Lendo os arquivos enviados..."):
                for uploaded_file in uploaded_files:
                    # Extrai texto do PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    # Prepara os chunks para o RAG
                    all_text_chunks.extend(chunk_text(text))
                    
                    # Usa a IA para extrair os dados estruturados (Layer 2 - JSON)
                    try:
                        # Mandamos os primeiros 4000 caracteres para não estourar o limite de tokens rapidamente
                        specs = extract_structured_specs(text[:4000])
                        extracted_data_raw.append(specs.model_dump())
                    except Exception as e:
                        st.error(f"Erro ao analisar o arquivo {uploaded_file.name}: {e}")
        
        # --- RESPOSTA DO RAG (Layer 2) ---
        if user_question:
            st.markdown("---")
            st.header("💡 Resposta do Assistente")
            with st.spinner("Consultando manuais e base de conhecimento..."):
                # Junta os chunks dos PDFs enviados com os chunks do PDF fixo
                rag_answer = process_rag(user_question, all_text_chunks, fixed_chunks)
                st.info(rag_answer)

        # --- PROCESSAMENTO DE DADOS COM PANDAS (Layer 1) ---
        if extracted_data_raw:
            st.markdown("---")
            st.header("📊 Dashboard Comparativo")
            
            # 1. Transformar em DataFrame
            df = pd.DataFrame(extracted_data_raw)
            
            # 2. Operação de Limpeza / Tipagem
            # Trata valores nulos dependendo do tipo de motorização
            df['engine_size'] = df['engine_size'].fillna(0.0)
            df['range_ev'] = df['range_ev'].fillna(0.0)
            
            # 3. Operação de Agregação e Filtro (Groupby)
            # Agrupa os carros pelo tipo de combustível e calcula a média de preço
            agg_df = df.groupby('fuel_type', as_index=False).agg(
                Preço_Medio=('msrp_price', 'mean'),
                Total_Modelos=('car_name', 'count')
            )
            
            # Layout em colunas para o Streamlit
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Especificações Extraídas")
                st.dataframe(
                    df[['car_name', 'fuel_type', 'msrp_price', 'engine_size', 'range_ev']], 
                    use_container_width=True,
                    hide_index=True
                )
                
            with col2:
                st.subheader("Resumo por Combustível")
                st.dataframe(agg_df, use_container_width=True, hide_index=True)
                
            # Gráfico de comparação de preços
            st.subheader("Comparativo de Preços (BRL)")
            # Setamos o index para o nome do carro para o gráfico ficar com as legendas corretas
            chart_data = df.set_index("car_name")[["msrp_price"]]
            st.bar_chart(chart_data)
            
            # Mostra os resumos gerados pela IA
            st.subheader("Resumo dos Modelos")
            for index, row in df.iterrows():
                with st.expander(f"Destaques: {row['car_name']}"):
                    st.write(row['summary'])

if __name__ == "__main__":
    main()