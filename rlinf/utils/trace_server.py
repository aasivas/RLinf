# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("rlinf.trace_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TraceHTTPRequestHandler(BaseHTTPRequestHandler):
    # Class-level lock to ensure thread-safe appends to the JSONL trace file
    file_lock = threading.Lock()
    trace_file_path = "trace_events.jsonl"

    def log_message(self, format, *args):
        # Prevent default http.server logging to stdout to keep logs clean
        pass

    def do_GET(self):
        if self.path == "/sync":
            # Returns server time in microseconds (UTC+0 Unix epoch time)
            server_time_us = time.time_ns() // 1000
            response = {"server_time_us": server_time_us}
            self._send_json(response)
        elif self.path == "/status" or self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_error_response(404, "Not Found")

    def do_POST(self):
        if self.path == "/trace":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_error_response(400, "Empty Body")
                return

            try:
                post_data = self.rfile.read(content_length)
                events = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self._send_error_response(400, f"Invalid JSON: {str(e)}")
                return

            if not isinstance(events, list):
                self._send_error_response(
                    400, "Expected a JSON list of trace events"
                )
                return

            # Append to file thread-safely
            try:
                with self.file_lock:
                    # Ensure directory exists
                    dir_name = os.path.dirname(self.trace_file_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    
                    with open(self.trace_file_path, "a") as f:
                        for event in events:
                            f.write(json.dumps(event) + "\n")
            except Exception as e:
                logger.error(f"Failed to write to file: {e}")
                self._send_error_response(500, "Internal Server Error")
                return

            self._send_json({"status": "success", "count": len(events)})
        else:
            self._send_error_response(404, "Not Found")

    def _send_json(self, data, status_code=200):
        try:
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    def _send_error_response(self, status_code, message):
        self._send_json({"error": message}, status_code=status_code)


def start_server(host: str, port: int, output_file: str):
    TraceHTTPRequestHandler.trace_file_path = os.path.abspath(output_file)
    server_address = (host, port)
    
    # Initialize the ThreadingHTTPServer to handle concurrency safely
    httpd = ThreadingHTTPServer(server_address, TraceHTTPRequestHandler)
    logger.info(f"Starting http trace server on {host}:{port}")
    logger.info(f"Writing trace events to: {TraceHTTPRequestHandler.trace_file_path}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLinf HTTP Trace Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument(
        "--file", default="trace_events.jsonl", help="Output JSONL trace filepath"
    )
    args = parser.parse_args()

    start_server(args.host, args.port, args.file)
