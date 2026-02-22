import os

# ✅ MUST be set before importing transformers or torch
# Leapcell's /app directory is read-only — only /tmp is writable
# This tells HuggingFace to cache model files in /tmp instead
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch

app = FastAPI(title="LFM2-350M Chat API")

MODEL_ID = "LiquidAI/LFM2-350M"
tokenizer = None
model = None


class ChatRequest(BaseModel):
    message: str


def load_model():
    """Load model only once, on first use (lazy loading)."""
    global tokenizer, model

    if model is not None:
        return  # Already loaded, skip

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"⏳ Loading model: {MODEL_ID} into /tmp/hf_cache ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()
    print("✅ Model loaded!")


@app.get("/health")
def health():
    """Instant health check — Leapcell pings this on startup."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:sans-serif; padding:40px; background:#f0f4ff">
        <h1>🌊 Liquid AI LFM2-350M</h1>
        <p>Send a POST to <code>/chat</code> with JSON:</p>
        <pre style="background:#fff;padding:12px;border-radius:8px">{"message": "Hello!"}</pre>
        <p><b>⚠️ First request:</b> slow (~60–120s) — model is downloading + loading into RAM.</p>
        <p><a href="/docs">📖 Swagger API Docs →</a></p>
    </body>
    </html>
    """


@app.post("/chat")
def chat(body: ChatRequest):
    """
    Send a message to LFM2-350M and get a response.
    Body: { "message": "your question here" }
    """
    user_message = body.message.strip()
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)

    try:
        load_model()
    except Exception as e:
        return JSONResponse({"error": f"Model failed to load: {str(e)}"}, status_code=500)

    try:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.05,
                max_new_tokens=256,
            )

        new_tokens = output[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"response": response_text}

    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)
