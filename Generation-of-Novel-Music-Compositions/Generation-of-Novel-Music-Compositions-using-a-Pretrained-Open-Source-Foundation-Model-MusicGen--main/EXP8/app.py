import base64
import io
import json
import os
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

PORT = int(os.getenv("PORT", "8000"))
MODEL_ID = os.getenv("MUSICGEN_MODEL_ID", "facebook/musicgen-small")
_model = None
_processor = None
_lock = threading.Lock()


def load_model():
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                _processor = AutoProcessor.from_pretrained(MODEL_ID)
                _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
                _model.eval()
    return _processor, _model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"success": True, "model": MODEL_ID, "loaded": _model is not None})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/generate":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(request.get("prompt", "")).strip()
            duration = max(1, min(int(request.get("duration", 5)), 20))
            if not prompt:
                self.send_json(400, {"error": "prompt is required"})
                return
            processor, model = load_model()
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            with torch.no_grad():
                audio = model.generate(**inputs, max_new_tokens=duration * 256)
            samples = audio[0].cpu().numpy()
            if samples.ndim == 2:
                samples = samples.T
            samples = np.clip(samples, -1.0, 1.0)
            pcm = (samples * 32767).astype(np.int16)
            if pcm.ndim == 1:
                pcm = pcm[:, None]
            output = io.BytesIO()
            with wave.open(output, "wb") as wav_file:
                wav_file.setnchannels(pcm.shape[1])
                wav_file.setsampwidth(2)
                wav_file.setframerate(model.config.audio_encoder.sampling_rate)
                wav_file.writeframes(pcm.tobytes())
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            self.send_json(200, {"success": True, "audio_base64": encoded, "format": "wav"})
        except Exception as error:
            self.send_json(500, {"success": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"MusicGen service listening on http://127.0.0.1:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
