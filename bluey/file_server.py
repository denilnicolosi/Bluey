"""
File server to serve exported files.
"""

import os
import threading
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.getenv("FILE_SERVER_PORT", "8765"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data/output")).resolve()


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def _start_server():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handler = partial(CORSHandler, directory=str(OUTPUT_DIR))
    
    try:
        server = ThreadingHTTPServer(("localhost", PORT), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"📁 File server: http://localhost:{PORT}")
    except OSError as e:
        pass # Server already running


def get_download_url(filename: str) -> str:
    _start_server()
    return f"http://localhost:{PORT}/{urllib.parse.quote(filename)}"


def get_output_dir() -> Path:
    _start_server()
    return OUTPUT_DIR