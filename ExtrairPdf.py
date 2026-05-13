import PyPDF2

def ExtrairPdf(PDF):
    text = ""
    leitorPdf = PyPDF2.PdfReader(PDF)
    
    for pagina in leitorPdf.pages:
        text += pagina.ExtrairPdf() or ""
        
    return text

def chunk_text(text, chunk_size=1000):
    """
    Divide o texto longo em pedaços menores (chunks).
    Isso é essencial para o RAG, pois o banco vetorial e o LLM lidam melhor com fragmentos.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]