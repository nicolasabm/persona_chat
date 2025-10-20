# api_model.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "Nicolasabm/llama3_2_3b_finetuned_complete")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Persona Model API")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 500
    temperature: float = 0.6
    do_sample: bool = True

@app.on_event("startup")
def load_model():
    global tokenizer, model, device
    print(f"Starting up. Device: {DEVICE}")
    device = torch.device(DEVICE)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a while)...")
    # Use float16 and device_map to leverage GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    print("Model loaded to device successfully.")

@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=req.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )

        # decode and remove the prompt portion if needed
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Optionally strip the prompt prefix if your front-end sends full prompt
        # Here we return whole decoding; frontend can strip if wanted.
        return {"response": decoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Recommended: run with uvicorn from command line. This is for quick local test.
    uvicorn.run("api_model:app", host="0.0.0.0", port=8000, reload=False)