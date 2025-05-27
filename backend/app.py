# backend/app.py
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flask_cors import CORS
from transformers import pipeline
import os
#from mistral_common.protocol.instruct.messages import (
 #   SystemMessage, UserMessage
#)
#from mistral_common.protocol.instruct.request import ChatCompletionRequest
#from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
#from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
#from huggingface_hub import hf_hub_download
#from transformers import AutoModelForCausalLM

# Ruta absoluta del directorio donde está index.html


frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

app = Flask(__name__, static_folder=frontend_path, template_folder=frontend_path)
CORS(app)




model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# Carga del modelo (modelo pequeño para pruebas)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Ruta principal: devuelve index.html
@app.route('/')
def serve_index():
    return send_from_directory(frontend_path, 'index.html')

# Ruta de la API del chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    subject = data.get('subject', '')

    prompt = f"{subject} question: {message}"
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"system", "content": f"Eres asistente personal experto en {subject}. Responderás de forma didáctica y explicativa las preguntas que puedan tener tus estudiantes "},
        {"role": "user", "content": message}],
    temperature=0,
        )
    
    answer = response.choices[0].message.content
   

    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
