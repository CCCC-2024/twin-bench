#!/usr/bin/env python3
"""Check isolated native Ollama CLI/API output and text/vision inference after renaming."""
import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib


def request(port, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


@contextlib.contextmanager
def server(runtime, store, port, log):
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=str(store),
               OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_CONTEXT_LENGTH="2048")
    with log.open("w") as out:
        process = subprocess.Popen([str(runtime / "bin/ollama"), "serve"], env=env,
                                   stdout=out, stderr=out, start_new_session=True)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {log}")
                try:
                    request(port, "/api/version")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"Server not ready: {log}")
            yield env
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def red_image():
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
    data += chunk(b"IEND", b"")
    return base64.b64encode(data).decode()


def check(runtime, store, port, log, custom):
    with server(runtime, store, port, log) as env:
        show = request(port, "/api/show", {"model": "laqmwe"})
        if custom:
            assert show["details"]["family"] == "laqmwe", show["details"]
            assert show["model_info"]["general.architecture"] == "laqmwe"
            assert "qwen" not in json.dumps(show).lower(), "Original name in default /api/show"
        cli = {}
        for args in (["list"], ["show", "laqmwe"], ["show", "laqmwe", "--modelfile"],
                     ["show", "laqmwe", "--parameters"], ["show", "laqmwe", "--template"],
                     ["show", "laqmwe", "--license"]):
            result = subprocess.run([str(runtime / "bin/ollama"), *args], env=env,
                                    capture_output=True, text=True, check=True)
            cli[" ".join(args)] = result.stdout
            if custom:
                assert "qwen" not in (result.stdout + result.stderr).lower(), args
        responses = {}
        for label, content, images in (
                ("text", "Compute 17 + 25. Reply only with the integer.", None),
                ("vision", "What is the dominant color in this image? Reply with one word.", [red_image()])):
            message = {"role": "user", "content": content}
            if images:
                message["images"] = images
            result = request(port, "/api/chat", {"model": "laqmwe", "messages": [message],
                             "stream": False, "think": False,
                             "options": {"temperature": 0, "seed": 42, "num_ctx": 2048,
                                         "num_predict": 32, "num_thread": 2}})
            responses[label] = result["message"]["content"].strip()
            assert ("42" if label == "text" else "red") in responses[label].lower(), result
        result = request(port, "/v1/chat/completions", {"model": "laqmwe", "messages": [
                         {"role": "user", "content": 'Reply only with the JSON object {"ok":true}.'}],
                         "temperature": 0, "max_tokens": 64, "reasoning_effort": "none"})
        responses["openai"] = result["choices"][0]["message"]["content"].strip()
        assert json.loads(responses["openai"]) == {"ok": True}, result
        ps = subprocess.run([str(runtime / "bin/ollama"), "ps"], env=env,
                            capture_output=True, text=True, check=True).stdout
        processes = []
        if custom:
            assert "laqmwe" in ps and "qwen" not in ps.lower(), ps
            assert "qwen" not in log.read_text().lower(), "Original name in loader log"
            assert request(port, "/api/show", {"model": "gpt-oss:20b"})["details"]["family"] == "gpt-oss"
            listing = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True,
                                     text=True, check=True).stdout
            processes = [line.strip() for line in listing.splitlines() if line.split(None, 1)
                         and line.split(None, 1)[0] in {"ollama", "llama-server"} and str(runtime) in line]
            assert processes and all("qwen" not in line.lower() for line in processes), processes
            result = request(port, "/api/chat", {"model": "gpt-oss:20b", "messages": [
                             {"role": "user", "content": "Compute 17 + 25. Reply only with the integer."}],
                             "stream": False, "think": "low", "options": {"temperature": 0,
                             "num_ctx": 2048, "num_predict": 256, "num_thread": 2}})
            assert "42" in result["message"]["content"], result
            assert "qwen" not in log.read_text().lower(), "Original name in final loader log"
        return {"responses": responses, "cli": cli, "ps": ps, "processes": processes,
                "family": show["details"]["family"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock-runtime", type=Path, required=True)
    p.add_argument("--stock-store", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    stock = check(args.stock_runtime, args.stock_store, 12671,
                  args.report.with_suffix(".stock.log"), False)
    print("Stock text, vision and OpenAI API checks passed", flush=True)
    custom = check(args.runtime, args.store, 12672, args.report.with_suffix(".custom.log"), True)
    assert stock["responses"] == custom["responses"], (stock["responses"], custom["responses"])
    args.report.write_text(json.dumps({"stock": stock, "custom": custom,
                                      "responses_identical": True}, indent=2))
    print("Custom native CLI/API/log checks passed; text and vision responses match stock", flush=True)


if __name__ == "__main__":
    main()
