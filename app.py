from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import torch

app = FastAPI(title="LFM2-350M Chat API")

# ✅ Model is NOT loaded at startup — it loads on the FIRST /chat request
# This lets the server start instantly and pass Leapcell's 9800ms health check
MODEL_ID = "LiquidAI/LFM2-350M"

tokenizer = None
model = None


def load_model():
    """Load model only once, on first use."""
    global tokenizer, model

    if model is not None:
        return  # Already loaded, skip

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"⏳ Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()
    print("✅ Model loaded!")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Leapcell hits this to check if the server is alive.
    Responds instantly — model does NOT need to be loaded.
    """
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:sans-serif; padding:40px; background:#f0f4ff">
        <h1>🌊 Liquid AI LFM2-350M</h1>
        <p>Server is running! Send a POST to <code>/chat</code>:</p>
        <pre>{"message": "Hello, who are you?"}</pre>
        <p><b>Note:</b> First request will be slow (~30–60s) — model is loading into RAM.</p>
        <p><a href="/docs">API Docs →</a></p>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(request: Request):
    # Load model on first request (lazy loading)
    load_model()

    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return JSONResponse(
            {"error": "Please provide a 'message' field."},
            status_code=400
        )

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
