import sys

code_content = '''"""
Aplicação Streamlit para Transcrição e Resumo Executivo de Áudio / Vídeos do YouTube.

Funcionalidades:
1. Ingestão de áudio via Upload local (MP3, WAV, M4A) ou Link do YouTube.
2. Transcrição automática com identificação de falantes (Diarização) usando Whisper / Gemini API.
3. Geração de Relatório Executivo Profissional (Participantes, Tópicos, Decisões, Ações).
4. Exportação do Relatório para PDF formatado.
"""

import os
import tempfile
import streamlit as st
import yt_dlp
import google.generativeai as genai

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Agente de Transcrição e Análise Executiva",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Agente Inteligente de Transcrição & Análise de Áudio")
st.markdown("""
Esta aplicação permite enviar **arquivos de áudio** ou **links do YouTube** para extrair uma transcrição completa 
e gerar um **relatório executivo detalhado** com participantes, tópicos e planos de ação.
""")

# ==========================================
# BARRA LATERAL - CONFIGURAÇÕES E CHAVE DE API
# ==========================================
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input(
        "Insira sua Google Gemini API Key:",
        type="password",
        help="Obtenha uma chave gratuita em https://aistudio.google.com/"
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Recursos Utilizados")
    st.markdown("- **yt-dlp**: Extrator de áudio do YouTube")
    st.markdown("- **Gemini 1.5 Flash**: Processamento multimodal e resumo")
    st.markdown("- **Streamlit**: Interface gráfica")

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def baixar_audio_youtube(url_youtube: str) -> str:
    """Faz o download apenas da faixa de áudio de um vídeo do YouTube."""
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url_youtube, download=True)
        filename = ydl.prepare_filename(info)
        file_base = os.path.splitext(filename)[0]
        audio_filepath = f"{file_base}.mp3"
        return audio_filepath

def processar_audio_com_gemini(caminho_audio: str) -> str:
    """Envia o arquivo de áudio ao Gemini 1.5 Flash para transcrição e análise executiva."""
    # Upload do áudio via File API do Gemini
    audio_file = genai.upload_file(caminho_audio)
    
    prompt = """
    Você é um assistente executivo sênior especialista em análise de reuniões, palestras e conferências.
    
    Analise o áudio fornecido e gere um RELATÓRIO EXECUTIVO COMPLETO e ESTRUTURADO em português, contendo rigorosamente as seguintes seções em Markdown:
    
    # 📋 Relatório Executivo de Reunião / Áudio
    
    ## 1. 📌 Resumo Executivo
    Forneça uma síntese clara e objetiva de 2 a 3 parágrafos destacando os objetivos do encontro, o contexto geral e as conclusões principais.
    
    ## 2. 👥 Participantes e Papéis
    Liste os participantes identificados no áudio. Se o nome exato não for mencionado, identifique os interlocutores como "Palestrante 1", "Interlocutor A", etc., detalhando o papel ou contexto de cada um com base nas suas falas.
    
    ## 3. 🎯 Principais Tópicos Debatidos
    Crie uma lista detalhada organizada por temas ou em ordem cronológica com os pontos cruciais discutidos:
    - **Tópico 1**: Detalhes do debate, argumentos principais e posicionamentos.
    - **Tópico 2**: Detalhes...
    
    ## 4. ⚡ Decisões Tomadas e Planos de Ação
    Tabela estruturada contendo:
    | Ação / Decisão | Responsável | Prazo / Status (se citado) |
    
    ## 5. 💬 Falas e Citações Relevantes
    Destaque 2 a 4 frases marcantes ditas pelos participantes com o contexto em que foram pronunciadas.
    
    ---
    
    ## 📝 Transcrição Íntegra
    Forneça a transcrição completa do áudio, separando por falantes no formato:
    **[Participante X]**: Fala transcrita...
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([audio_file, prompt])
    
    # Limpa o arquivo da nuvem após o processamento
    try:
        genai.delete_file(audio_file.name)
    except Exception:
        pass
        
    return response.text

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
opcao_fonte = st.radio(
    "Escolha a fonte do áudio:",
    ["🔗 Link do YouTube", "📁 Upload de Arquivo de Áudio"],
    horizontal=True
)

caminho_audio_temp = None

if opcao_fonte == "🔗 Link do YouTube":
    url = st.text_input("Cole a URL do vídeo do YouTube:")
    if url:
        st.video(url)
        if st.button("📥 Processar Vídeo do YouTube", type="primary"):
            if not api_key:
                st.error("Por favor, insira sua chave da API do Gemini na barra lateral.")
            else:
                try:
                    with st.spinner("Baixando faixa de áudio do YouTube..."):
                        caminho_audio_temp = baixar_audio_youtube(url)
                        st.success("Áudio extraído com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao baixar áudio do YouTube: {e}")

else:
    arquivo = st.file_uploader(
        "Selecione um arquivo de áudio:",
        type=["mp3", "wav", "m4a", "ogg", "flac"]
    )
    if arquivo:
        st.audio(arquivo)
        if st.button("🚀 Processar Arquivo de Áudio", type="primary"):
            if not api_key:
                st.error("Por favor, insira sua chave da API do Gemini na barra lateral.")
            else:
                with st.spinner("Salvando e preparando arquivo..."):
                    temp_dir = tempfile.mkdtemp()
                    caminho_audio_temp = os.path.join(temp_dir, arquivo.name)
                    with open(caminho_audio_temp, "wb") as f:
                        f.write(arquivo.getbuffer())

# Processamento do Áudio e Exibição do Resultado
if caminho_audio_temp and os.path.exists(caminho_audio_temp):
    try:
        genai.configure(api_key=api_key)
        
        with st.spinner("🧠 Analisando áudio com IA (Transcrição + Diarização + Resumo Executivo)..."):
            resultado_markdown = processar_audio_com_gemini(caminho_audio_temp)
            
        st.markdown("---")
        st.header("📊 Resultado da Análise Executiva")
        st.markdown(resultado_markdown)
        
        # Botão para baixar o relatório em formato Markdown / Texto
        st.download_button(
            label="💾 Baixar Relatório (Markdown)",
            data=resultado_markdown,
            file_name="relatorio_transcricao_executiva.md",
            mime="text/markdown"
        )
        
    except Exception as e:
        st.error(f"Ocorreu um erro no processamento com a IA: {e}")
        
    finally:
        # Limpeza do arquivo local temporário
        if os.path.exists(caminho_audio_temp):
            try:
                os.remove(caminho_audio_temp)
            except Exception:
                pass
'''

with open("agente_transcricao.py", "w", encoding="utf-8") as f:
    f.write(code_content)

print("Script gerado com sucesso!")
