import os

# Must be set BEFORE any huggingface/transformers imports
# Leapcell's /app is read-only — only /tmp is writable
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch

app = FastAPI(title="LFM2-350M API")

MODEL_ID = "LiquidAI/LFM2-350M"
tokenizer = None
model = None


class ChatRequest(BaseModel):
    message: str


def load_model():
    global tokenizer, model
    if model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"⏳ Downloading + loading {MODEL_ID} into /tmp/hf_cache ...")

    # Exact same as official model card — only change is cpu + float32 (no GPU on Leapcell)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",       # Leapcell has no GPU
        torch_dtype=torch.float32,  # bfloat16 is NOT safe on all CPUs — float32 always works
    )
    model.eval()
    print("✅ Model ready!")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "torch_version": torch.__version__,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:sans-serif;padding:40px;background:#f0f4ff">
        <h1>🌊 Liquid AI LFM2-350M</h1>
        <p>POST to <code>/chat</code>:</p>
        <pre style="background:#fff;padding:12px;border-radius:8px">{"message": "Hello!"}</pre>
        <p>⚠️ <b>First request is slow (~60–120s)</b> — model is downloading 700MB into /tmp</p>
        <p><a href="/health">/health</a> | <a href="/docs">/docs</a></p>
    </body>
    </html>
    """


@app.post("/chat")
def chat(body: ChatRequest):
    user_message = body.message.strip()
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)

    try:
        load_model()
    except Exception as e:
        return JSONResponse({"error": f"Model failed to load: {str(e)}"}, status_code=500)

    try:
        # Exact generation code from official LFM2 model card
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model.device)  # sends input to same device as model (cpu)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.3,
                min_p=0.15,              # official recommended param
                repetition_penalty=1.05, # official recommended param
                max_new_tokens=512,
            )

        # Decode only new tokens (skip the input prompt)
        new_tokens = output[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"response": response_text}

    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)
