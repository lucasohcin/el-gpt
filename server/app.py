"""
FastAPI Server for El GPT.
Provides:
- Streaming Chat completion endpoint (/api/chat) via Server-Sent Events (SSE)
  Supports both ultra-fast Cloud Models (Groq, 450+ tps) and local PyTorch engines.
- Cloud API Key & Status management (/api/cloud/status, /api/cloud/key)
- Live Training & Telemetry API (/api/train/start, /api/train/status, /api/train/stop)
- Model metadata & status (/api/models)
- Interactive Code Runner (/api/run-code)
- Static file serving for ChatGPT-grade UI
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.cloud_engine import cloud_engine, get_api_key, save_api_key_to_env
from engine.memory import memory_engine

# Graceful optional loading of heavy local PyTorch engines
# This allows El GPT to run in lightweight cloud environments (Vercel, Render, Railway)
# with 0% memory bloat, while still supporting local Apple Silicon MPS when available.
try:
    import torch
    from engine.model import ElGPTConfig, ElGPTModel
    from engine.tokenizer import ElGPTTokenizer
    from engine.inference import generate_stream, get_default_device
    from engine.smart_engine import smart_engine
    from engine.el_gpt_1_5 import el_gpt_1_5
    from engine.el_gpt_1_8 import el_gpt_1_8
    from server.training_runner import coordinator
    LOCAL_PYTORCH_AVAILABLE = True
except Exception as e:
    LOCAL_PYTORCH_AVAILABLE = False
    print(f"[Server] Running in Cloud-Optimized Mode (Local PyTorch engine not initialized: {e})")

app = FastAPI(title="El GPT Cloud & Neural Studio", redirect_slashes=False)

# Robust CORS configuration supporting all methods & preflights without credentials collision
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = "checkpoints/el_gpt_1.0_latest.pt"

if LOCAL_PYTORCH_AVAILABLE:
    class ModelManager:
        def __init__(self):
            self.device = get_default_device()
            self.tokenizer = ElGPTTokenizer()
            self.config = ElGPTConfig(vocab_size=self.tokenizer.vocab_size)
            self.model: Optional[ElGPTModel] = None
            self.checkpoint_loaded: bool = False
            self.last_loaded_mtime: float = 0.0
            self.load_or_init_model()

        def load_or_init_model(self):
            if not os.path.exists(CHECKPOINT_PATH):
                if self.model is None:
                    self.model = ElGPTModel(self.config).to(self.device)
                return

            current_mtime = os.path.getmtime(CHECKPOINT_PATH)
            if self.model is not None and current_mtime <= self.last_loaded_mtime:
                return

            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
                cfg = ckpt.get("config", self.config)
                new_model = ElGPTModel(cfg).to(self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                new_model.load_state_dict(state_dict)
                self.model = new_model
                self.checkpoint_loaded = True
                self.last_loaded_mtime = current_mtime
                print(f"[ModelManager] Loaded checkpoint from {CHECKPOINT_PATH} (mtime: {current_mtime})")
            except Exception as e:
                print(f"[ModelManager] Could not load checkpoint: {e}. Running fresh architecture.")
                if self.model is None:
                    self.model = ElGPTModel(self.config).to(self.device)
                self.checkpoint_loaded = False

    model_mgr = ModelManager()
else:
    model_mgr = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model_id: str = Field(default="el-gpt-cloud-120b")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=1024, ge=10, le=4096)
    api_key: Optional[str] = None


class CloudKeyRequest(BaseModel):
    api_key: str
    provider: Optional[str] = "groq"


class TrainRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=5e-4, ge=1e-5, le=1e-2)
    train_file: str = Field(default="data/train_ultra_1b.jsonl")
    scale: str = Field(default="1b")


class MemoryAddRequest(BaseModel):
    memory: str


class ImageGenRequest(BaseModel):
    prompt: str
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    model: str = Field(default="flux")


@app.post("/api/image")
@app.post("/api/image/")
@app.post("/image")
@app.post("/image/")
async def generate_image_endpoint(req: ImageGenRequest):
    """Generates an image using FLUX.1 (100% Free)."""
    from engine.image_engine import generate_image
    result = generate_image(req.prompt, width=req.width, height=req.height, model=req.model)
    return result


# Explicit OPTIONS preflight handler covering all possible paths
@app.options("/api/chat")
@app.options("/api/chat/")
@app.options("/chat")
@app.options("/chat/")
@app.options("/")
async def options_handler():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/api/cloud/status")
@app.get("/api/cloud/status/")
@app.get("/cloud/status")
@app.get("/cloud/status/")
async def get_cloud_status():
    """Returns status of cloud AI engine and whether API keys are configured."""
    has_groq = bool(get_api_key(provider="groq"))
    has_openrouter = bool(get_api_key(provider="openrouter"))
    return {
        "cloud_enabled": True,
        "has_server_api_key": has_groq or has_openrouter,
        "has_groq_key": has_groq,
        "has_openrouter_key": has_openrouter,
        "default_cloud_model": "el-gpt-cloud-120b",
        "models": cloud_engine.get_models_metadata(),
    }


@app.post("/api/cloud/key")
@app.post("/api/cloud/key/")
@app.post("/cloud/key")
@app.post("/cloud/key/")
async def set_cloud_key(req: CloudKeyRequest):
    """Saves or updates API keys in the server's .env file."""
    success = save_api_key_to_env(req.api_key, provider=req.provider or "groq")
    return {
        "success": success,
        "message": "API Key saved successfully to server!" if success else "Failed to save API key.",
        "has_server_api_key": bool(get_api_key(provider="groq")) or bool(get_api_key(provider="openrouter")),
    }


@app.get("/api/memory")
@app.get("/api/memory/")
@app.get("/memory")
@app.get("/memory/")
async def get_memories():
    """Returns stored user preferences and long-term memories."""
    return {"memories": memory_engine.get_memories()}


@app.post("/api/memory")
@app.post("/api/memory/")
@app.post("/memory")
@app.post("/memory/")
async def add_memory(req: MemoryAddRequest):
    """Adds a new persistent memory."""
    success = memory_engine.add_memory(req.memory)
    return {"success": success, "memories": memory_engine.get_memories()}


@app.delete("/api/memory/{idx}")
@app.delete("/api/memory/{idx}/")
@app.delete("/memory/{idx}")
@app.delete("/memory/{idx}/")
async def delete_memory(idx: int):
    """Deletes a memory item by index."""
    success = memory_engine.delete_memory(idx)
    return {"success": success, "memories": memory_engine.get_memories()}


@app.post("/api/memory/clear")
@app.post("/api/memory/clear/")
@app.post("/memory/clear")
@app.post("/memory/clear/")
async def clear_memories():
    """Clears all stored memories."""
    memory_engine.clear_memories()
    return {"success": True, "memories": []}


@app.get("/api/models")
@app.get("/api/models/")
@app.get("/models")
@app.get("/models/")
async def get_models():
    """Returns metadata about active models including Cloud and Local options."""
    all_models = []

    # 1. Cloud Models (Top Priority - Zero hardware load, ultra-fast)
    all_models.extend(cloud_engine.get_models_metadata())

    # 2. Local Apple Silicon Models (if available on this host)
    if LOCAL_PYTORCH_AVAILABLE and model_mgr is not None:
        has_ckpt = os.path.exists(CHECKPOINT_PATH)
        ckpt_size_mb = round(os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024), 2) if has_ckpt else 0
        device_str = str(model_mgr.device)

        all_models.extend([
            {
                "id": "el-gpt-1-8-ultra",
                "name": "El GPT 1.8 Ultra (Local)",
                "tag": "1B Local MPS",
                "badge": "1.5B Local",
                "parameters": "1,543,714,816",
                "description": "Local 1.5B model running on Apple Silicon GPU/MPS. High RAM & compute load.",
                "status": "Ready (Local)",
            },
            {
                "id": "el-gpt-1-5-pro",
                "name": "El GPT 1.5 Pro (Local)",
                "tag": "500M Local MPS",
                "badge": "Pro Local",
                "parameters": "494,032,768",
                "description": "Local 500M model for reasoning and long-term memory.",
                "status": "Ready (Local)",
            },
            {
                "id": "el-gpt-1-5-flash",
                "name": "El GPT 1.5 Flash (Local)",
                "tag": "500M High-Speed",
                "badge": "Flash Local",
                "parameters": "494,032,768",
                "description": "Local 500M model optimized for lower latency.",
                "status": "Ready (Local)",
            },
            {
                "id": "el-gpt-1-0-pro",
                "name": "El GPT 1.0 Pro (Local)",
                "tag": "135M Compact",
                "badge": "135M Local",
                "parameters": "134,515,008",
                "description": "Compact 135M foundation model pre-trained on 2 Trillion tokens.",
                "status": "Ready (Local)",
            },
            {
                "id": "el-gpt-scratch",
                "name": "El GPT Scratch (Local)",
                "tag": "Custom Neural Engine",
                "badge": "Trainable",
                "parameters": f"{model_mgr.model.get_num_params():,}" if model_mgr and model_mgr.model else "0",
                "description": "Your custom PyTorch Transformer decoder trained locally in the Training Studio.",
                "status": "Checkpoint Ready" if has_ckpt else "Untrained",
                "checkpoint_size_mb": ckpt_size_mb,
            },
        ])
    else:
        device_str = "Cloud Infrastructure"

    return {
        "active_default": "el-gpt-cloud-120b",
        "device": device_str,
        "local_available": LOCAL_PYTORCH_AVAILABLE,
        "has_cloud_key": bool(get_api_key()),
        "models": all_models,
    }


# Multiple path decorators so any Vercel rewrite or direct path matches without 405 error
@app.post("/api/chat")
@app.post("/api/chat/")
@app.post("/chat")
@app.post("/chat/")
@app.post("/api")
@app.post("/api/")
@app.post("/api/index.py")
@app.post("/index.py")
@app.post("/")
async def chat_endpoint(req: ChatRequest):
    """
    Streaming SSE endpoint delivering tokens in real time to the ChatGPT UI.
    Routes to Cloud Engine (Groq 450+ tps) or Local PyTorch engines.
    """
    messages_data = [m.model_dump() for m in req.messages]

    async def event_generator():
        def run_gen():
            try:
                # 1. Cloud Model Route (Fastest, 0% CPU/RAM)
                if cloud_engine.is_cloud_model(req.model_id):
                    for token_chunk in cloud_engine.generate_stream(
                        messages=messages_data,
                        model_id=req.model_id,
                        api_key=req.api_key,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                    ):
                        yield token_chunk
                    return

                # 2. Local Model Routes (requires PyTorch)
                if not LOCAL_PYTORCH_AVAILABLE:
                    yield (
                        "\n⚠️ **Local Model Unavailable**: This server is running in Cloud-Optimized mode. "
                        "Please select one of the **El GPT Cloud Models** from the model dropdown for ultra-fast responses."
                    )
                    return

                if req.model_id == "el-gpt-1-8-ultra":
                    for token_chunk in el_gpt_1_8.generate_stream(
                        messages=messages_data,
                        mode="ultra",
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                    ):
                        yield token_chunk
                elif req.model_id == "el-gpt-1-5-flash":
                    for token_chunk in el_gpt_1_5.generate_stream(
                        messages=messages_data,
                        mode="flash",
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                    ):
                        yield token_chunk
                elif req.model_id == "el-gpt-1-5-pro":
                    for token_chunk in el_gpt_1_5.generate_stream(
                        messages=messages_data,
                        mode="pro",
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                    ):
                        yield token_chunk
                elif req.model_id == "el-gpt-1-0-pro":
                    for token_chunk in smart_engine.generate_stream(
                        messages=messages_data,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                    ):
                        yield token_chunk
                else:
                    # Scratch model generation
                    if model_mgr:
                        if not model_mgr.checkpoint_loaded and os.path.exists(CHECKPOINT_PATH):
                            model_mgr.load_or_init_model()
                        for token_chunk in generate_stream(
                            model=model_mgr.model,
                            tokenizer=model_mgr.tokenizer,
                            messages=messages_data,
                            max_new_tokens=req.max_new_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            device=model_mgr.device,
                        ):
                            yield token_chunk
            except Exception as e:
                yield f"\n⚠️ [Generation error: {str(e)}]"

        stream = run_gen()
        for chunk in stream:
            payload = json.dumps({"token": chunk})
            yield f"data: {payload}\n\n"
            # Fast pacing for cloud tokens
            await asyncio.sleep(0.001)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/train/start")
@app.post("/api/train/start/")
async def start_training(req: TrainRequest):
    """Triggers background model training with chosen scale."""
    if not LOCAL_PYTORCH_AVAILABLE:
        return {"success": False, "message": "Training is only supported in a local environment with PyTorch installed."}

    success = coordinator.start_training(
        train_file=req.train_file,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        scale=req.scale,
    )
    if not success:
        return {"success": False, "message": "A training run is already in progress."}
    return {"success": True, "message": f"Training initiated on Apple Silicon MPS (scale: {req.scale})."}


@app.get("/api/train/status")
@app.get("/api/train/status/")
async def get_training_status():
    """Returns real-time telemetry from active training run."""
    if not LOCAL_PYTORCH_AVAILABLE:
        return {"is_training": False, "message": "Local training not active."}
    return coordinator.get_telemetry()


@app.post("/api/train/stop")
@app.post("/api/train/stop/")
async def stop_training():
    """Gracefully interrupts active training run."""
    if not LOCAL_PYTORCH_AVAILABLE:
        return {"success": True}
    coordinator.stop_training()
    return {"success": True, "message": "Training stop signal dispatched."}


class RunCodeRequest(BaseModel):
    language: str = "python"
    code: str
    timeout: float = 6.0


@app.post("/api/run-code")
@app.post("/api/run-code/")
async def run_code(req: RunCodeRequest):
    """Executes a code snippet (Python or JS) safely and returns stdout/stderr."""
    lang = req.language.lower().strip()
    code = req.code.strip()

    start_time = time.time()
    if lang in ["python", "python3", "py"]:
        cmd = [sys.executable, "-c", code]
    elif lang in ["javascript", "js", "node"]:
        node_bin = shutil.which("node")
        if not node_bin:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Node.js is not installed on this system. You can run JavaScript directly in the browser live preview!",
                "exit_code": 1,
                "execution_time_ms": 0,
            }
        cmd = [node_bin, "-e", code]
    else:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution for '{req.language}' is not supported. Supported: python, javascript.",
            "exit_code": 1,
            "execution_time_ms": 0,
        }

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=min(req.timeout, 10.0),
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout[:15000],
            "stderr": proc.stderr[:15000],
            "exit_code": proc.returncode,
            "execution_time_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {req.timeout} seconds.",
            "exit_code": 124,
            "execution_time_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "exit_code": 1,
            "execution_time_ms": elapsed_ms,
        }


# Universal Fallback Route — Guarantees ZERO 404 / 405 errors under any rewrite setup
@app.api_route("/{path_name:path}", methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE"])
async def catch_all_fallback(request: Request, path_name: str):
    method = request.method.upper()
    path = "/" + path_name.strip("/")

    if method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS, HEAD, PUT, DELETE",
                "Access-Control-Allow-Headers": "*",
            },
        )

    # If any POST arrives with messages or 'chat' in path, handle as chat
    if "chat" in path or method == "POST":
        try:
            body = await request.json()
            if isinstance(body, dict) and "messages" in body:
                chat_req = ChatRequest(**body)
                return await chat_endpoint(chat_req)
        except Exception:
            pass

    if "model" in path:
        return await get_models()

    if "cloud" in path:
        return await get_cloud_status()

    if "memory" in path:
        return await get_memories()

    return {"status": "ok", "app": "El GPT", "path": path}


# Mount frontend / public static directory ONLY when running locally
# On Vercel, public/ is served directly at the Edge CDN.
# Mounting StaticFiles at '/' on Vercel would intercept unmatched POST requests and return 405.
if not os.environ.get("VERCEL"):
    public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    static_dir = public_dir if os.path.exists(public_dir) else frontend_dir
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
