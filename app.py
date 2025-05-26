from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# API Configuration
API_CONFIG = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key": OPENAI_API_KEY,
        "model": "gpt-4o"
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key": ANTHROPIC_API_KEY,
        "model": "claude-3-5-sonnet-20241022"
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": GROQ_API_KEY,
        "model": "llama3-70b-8192"
    }
}

# In-memory conversation state (use a database in production)
chat_sessions = {}

def call_ai_api(provider, prompt, context):
    config = API_CONFIG.get(provider)
    if not config or not config["key"]:
        raise ValueError(f"Provider {provider} not configured or missing API key")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['key']}"
    }
    body = {
        "model": config["model"],
        "messages": context + [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }

    if provider == "anthropic":
        headers["x-api-key"] = config["key"]
        del headers["Authorization"]
        body["max_tokens"] = 500

    response = requests.post(config["url"], headers=headers, json=body)
    response.raise_for_status()

    if provider == "anthropic":
        return response.json()["content"][0]["text"]
    return response.json()["choices"][0]["message"]["content"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_id = data.get("user_id", "default")
    message = data.get("message", "").strip()
    chat_state = data.get("chat_state", "initial")
    symptoms = data.get("symptoms", [])

    if not message and chat_state != "initial":
        return jsonify({"response": "Please enter a message.", "chat_state": chat_state, "symptoms": symptoms})

    # Initialize session if not exists
    if user_id not in chat_sessions:
        chat_sessions[user_id] = {"history": [], "chat_state": "initial", "symptoms": []}

    # Update session
    if message:
        chat_sessions[user_id]["history"].append({"role": "user", "content": message})
    chat_sessions[user_id]["chat_state"] = chat_state
    chat_sessions[user_id]["symptoms"] = symptoms

    try:
        if chat_state == "initial":
            response = """
                Please select an option:<br>
                <button onclick="handleChatbotOption('symptom')" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Check Symptoms</button>
                <button onclick="handleChatbotOption('doctor')" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Talk to a Doctor</button>
                <button onclick="handleChatbotOption('patient')" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Join Patient Community</button>
            """
            model_used = None
        elif chat_state == "symptom":
            response, model_used = handle_symptom_check(message, chat_sessions[user_id]["symptoms"], chat_sessions[user_id]["history"])
            chat_sessions[user_id]["symptoms"] = [] if response else chat_sessions[user_id]["symptoms"]
            chat_sessions[user_id]["chat_state"] = "initial" if response else "symptom"
        elif chat_state == "doctor":
            response, model_used = handle_doctor_chat(message, chat_sessions[user_id]["history"])
            chat_sessions[user_id]["chat_state"] = "initial"
        elif chat_state == "patient":
            response, model_used = handle_patient_chat(message, chat_sessions[user_id]["history"])
            chat_sessions[user_id]["chat_state"] = "initial"
        else:
            response = "Invalid state. Please start over."
            model_used = None

        if response:
            chat_sessions[user_id]["history"].append({"role": "assistant", "content": response})
        return jsonify({
            "response": response,
            "chat_state": chat_sessions[user_id]["chat_state"],
            "symptoms": chat_sessions[user_id]["symptoms"],
            "model_used": model_used
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "response": "Sorry, an error occurred. Please try again.",
            "chat_state": "initial",
            "symptoms": [],
            "model_used": None
        })

def handle_symptom_check(message, symptoms, history):
    if not message:
        return "Please describe your symptoms (e.g., fever, cough, headache).", None
    symptoms.append(message.lower())
    if len(symptoms) < 3 and message.lower() != "done":
        return "Please share another symptom or type 'done'.", None

    prompt = f"""
        You are a medical assistant analyzing symptoms for a patient in rural Kenya. The patient has reported: {', '.join(symptoms)}. 
        Provide a concise possible diagnosis (e.g., cold, flu, malaria) and recommend consulting a doctor. 
        Avoid definitive diagnoses; emphasize professional medical advice.
    """
    response = call_ai_api("groq", prompt, history)
    return f"{response}<br><button onclick=\"handleChatbotOption('doctor')\" class=\"bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600\">Connect with a Doctor</button>", "Llama 3.1 (Groq)"

def handle_doctor_chat(message, history):
    prompt = f"""
        You are a medical assistant simulating a doctor consultation. The user has asked: "{message}". 
        Provide general advice, avoid definitive diagnoses, and emphasize consulting a licensed doctor. 
        Keep the response concise and professional.
    """
    response = call_ai_api("anthropic", prompt, history)
    return f"{response}<br><button onclick=\"handleChatbotOption('initial')\" class=\"bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600\">Back to Menu</button>", "Claude 3.5 Sonnet"

def handle_patient_chat(message, history):
    prompt = f"""
        You are a friendly assistant in a patient community chat. The user shared: "{message}". 
        Respond empathetically, encouraging sharing or advice from others, and keep the tone supportive.
    """
    response = call_ai_api("openai", prompt, history)
    return f"{response}<br><button onclick=\"handleChatbotOption('initial')\" class=\"bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600\">Back to Menu</button>", "ChatGPT (GPT-4o)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)