from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="LFM2-350M Chat API")

# ✅ Model is loaded ONCE when the server starts (not on every request)
MODEL_ID = "LiquidAI/LFM2-350M"
print(f"Loading model: {MODEL_ID} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",        # Leapcell has no GPU, so we use CPU
    torch_dtype=torch.float32,  # bfloat16 may not work on all CPUs; float32 is safe
)
model.eval()
print("✅ Model loaded successfully!")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:sans-serif; padding:40px; background:#f0f4ff">
        <h1>🌊 Liquid AI LFM2-350M is running!</h1>
        <p>Send a POST request to <code>/chat</code> with JSON body:</p>
        <pre>{"message": "Hello, who are you?"}</pre>
        <p>Or use the <a href="/docs">API docs</a></p>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return JSONResponse({"error": "Please provide a 'message' field."}, status_code=400)

    # Apply the chat template — this wraps message in the correct format LFM2 expects
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
            max_new_tokens=256,   # Keep responses short for faster CPU inference
        )

    # Decode only the NEW tokens (not the input prompt)
    new_tokens = output[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {"response": response_text}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
