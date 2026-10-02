"""The stdio MCP bridge against a real server process: initialize, tools/list, tools/call."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    port = free_port()
    env = {**os.environ, "PYGMALION_DATA_DIR": str(tmp_path / "data"), "PYGMALION_PORT": str(port), "PORT_STRICT": "1", "PYTHONUNBUFFERED": "1",
           "PYGMALION_SCHEDULER": "0", "PYGMALION_OFFLINE": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "pygmalion_hoard"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{url}/api/health", timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        yield url, tmp_path / "data", env
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def rpc(proc, payload):
    proc.stdin.write((json.dumps(payload) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_bridge_roundtrip(server):
    url, data, env = server
    benv = {**env, "PYGMALION_URL": url, "PYGMALION_TOKEN_FILE": str(data / "mcp-token"), "PYGMALION_BRIDGE_AUTOSTART": "0"}
    proc = subprocess.Popen([sys.executable, str(ROOT / "mcp_server.py")], cwd=ROOT, env=benv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    try:
        init = rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                                                                       "clientInfo": {"name": "t", "version": "0"}}})
        assert init["result"]["serverInfo"]["name"] == "pygmalion-hoard"
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
        proc.stdin.flush()
        tools = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        names = {t["name"] for t in tools}
        assert {"pygmalion_overview", "train_plan", "train_start", "quantize_start", "evaluate_start", "publish_ollama", "lineage_graph"} <= names
        assert len(names) == 44
        ov = rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "pygmalion_overview", "arguments": {}}})
        body = json.loads(ov["result"]["content"][0]["text"])
        assert body["allowed_gpus"] == [2, 3] and "summary" in body
        made = rpc(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "dataset_create", "arguments": {
            "name": "Puente", "wait_s": 20, "sources": [{"type": "jsonl", "text": "\n".join(json.dumps({"prompt": f"Pregunta {i} larga", "response": f"Respuesta {i} larga"}) for i in range(25))}]}}})
        built = json.loads(made["result"]["content"][0]["text"])
        assert built["state"] == "done" and built["result"]["version"]["records"] == 25
        listed = rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "datasets_list", "arguments": {}}})
        assert json.loads(listed["result"]["content"][0]["text"])["datasets"][0]["name"] == "Puente"
        err = rpc(proc, {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "artifact_get", "arguments": {"artifact": "a_none"}}})
        detail = json.loads(err["result"]["content"][0]["text"])
        assert detail["code"] == "not_found" and detail["hint"]
    finally:
        proc.terminate()
        proc.wait(10)


def test_bridge_refuses_a_non_local_server(server):
    _url, data, env = server
    out = subprocess.run([sys.executable, str(ROOT / "mcp_server.py")], cwd=ROOT, input=b"", capture_output=True, timeout=30,
                         env={**env, "PYGMALION_URL": "http://example.org:5202", "PYGMALION_TOKEN_FILE": str(data / "mcp-token"), "PYGMALION_BRIDGE_AUTOSTART": "0"})
    assert out.returncode != 0 and b"local" in (out.stderr + out.stdout).lower()
