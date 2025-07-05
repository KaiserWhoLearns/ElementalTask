from dotenv import load_dotenv
from openai import OpenAI
from together import Together
import os
import re



# === Determine backend ===
openai_models = [
    "gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14", "o1-mini-2024-09-12",
    "o3-mini-2025-01-31", "o4-mini-2025-04-16"
]

together_models = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
]


# === Load API keys from .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


def build_model(model_name):
    if model_name in openai_models:
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif model_name in together_models:
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
        client = Together()
    else:
        raise ValueError("❌ WRONG MODEL NAME!")
    return client


def ask_llm(client, model, msgs, temperature=1.0, top_p=1.0, max_tokens=2048):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR calling model]: {e}"


def extract_last_json(s):
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, s)
    if matches:
        return matches[-1]
    return s