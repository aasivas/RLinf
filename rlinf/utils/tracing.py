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

import atexit
import functools
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager

logger = logging.getLogger("rlinf.tracing")

# 1. Hydra-safe Interception of Command Line Options at Import-Time
TRACE_SERVER_IP = None
TRACE_SERVER_PORT = 8888

if "--trace-server-ip" in sys.argv:
    try:
        idx = sys.argv.index("--trace-server-ip")
        if idx + 1 < len(sys.argv):
            TRACE_SERVER_IP = sys.argv[idx + 1]
            del sys.argv[idx:idx + 2]
    except ValueError:
        pass

if "--trace-server-port" in sys.argv:
    try:
        idx = sys.argv.index("--trace-server-port")
        if idx + 1 < len(sys.argv):
            try:
                TRACE_SERVER_PORT = int(sys.argv[idx + 1])
            except ValueError:
                TRACE_SERVER_PORT = 8888
            del sys.argv[idx:idx + 2]
    except ValueError:
        pass


class DistTracer:
    """Distributed tracing client that sends trace events to a central HTTP server."""

    def __init__(self, server_ip: str, port: int = 8888, process_name: str = None, thread_name: str = None):
        self.server_ip = server_ip
        self.port = port
        self.server_url = f"http://{server_ip}:{port}"
        
        # Identity labels for Chrome Trace representation
        self.pid = process_name if process_name is not None else str(os.getpid())
        self.tid = thread_name if thread_name is not None else str(threading.get_ident())

        # Synchronization state
        self.offset = 0
        self.last_sync_time = 0.0
        self.sync_lock = threading.Lock()

        # Buffering state
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_limit = 1000

        # Run time synchronization
        self.sync_clock()

        # Background thread control
        self.running = True
        self.bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.bg_thread.start()

        # Emit initial metadata events to label process/thread in the trace viewer
        self.emit_metadata("process_name", {"name": self.pid})
        self.emit_metadata("thread_name", {"name": self.tid})

        # Register exit handler for clean final flush
        atexit.register(self.shutdown)

    def sync_clock(self):
        """Synchronize time with the server using Cristian's algorithm over HTTP GET."""
        logger.info(f"Synchronizing clock with trace server at {self.server_url}")
        best_offset = 0
        min_rtt = float("inf")
        successful_rounds = 0

        # Perform 5 round-trip measurements
        for i in range(5):
            try:
                t0 = time.time_ns() // 1000
                req = urllib.request.Request(f"{self.server_url}/sync", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    t_server = data["server_time_us"]
                t1 = time.time_ns() // 1000

                rtt = t1 - t0
                offset = t_server - (t0 + rtt // 2)

                if rtt < min_rtt:
                    min_rtt = rtt
                    best_offset = offset
                successful_rounds += 1
            except Exception as e:
                # Log warning but don't fail completely to keep system robust
                logger.warning(f"Clock sync round {i} failed: {e}")
                time.sleep(0.05)

        if successful_rounds > 0:
            with self.sync_lock:
                self.offset = best_offset
                self.last_sync_time = time.time()
            logger.info(
                f"Clock synchronized. Offset: {best_offset} us, Min RTT: {min_rtt} us"
            )
        else:
            logger.error("Could not sync clock with trace server. Defaulting to 0 offset.")

    def log_event(self, name: str, cat: str = "default", ph: str = "X", ts: int = None, dur: int = None, args: dict = None):
        """Append a trace event to the buffer thread-safely."""
        if ts is None:
            # Adjust local time to UTC+0 synchronized server time
            with self.sync_lock:
                current_offset = self.offset
            ts = (time.time_ns() // 1000) + current_offset

        event = {
            "name": name,
            "cat": cat,
            "ph": ph,
            "ts": ts,
            "pid": self.pid,
            "tid": self.tid,
        }
        if dur is not None:
            event["dur"] = dur
        if args is not None:
            event["args"] = args

        with self.buffer_lock:
            self.buffer.append(event)
            buffer_len = len(self.buffer)

        # Trigger immediate flush if buffer limit is reached
        if buffer_len >= self.buffer_limit:
            threading.Thread(target=self.flush, daemon=True).start()

    def emit_metadata(self, name: str, args: dict):
        """Log a Chrome Trace metadata event (ph: M) to label processes/threads."""
        with self.sync_lock:
            current_offset = self.offset
        ts = (time.time_ns() // 1000) + current_offset
        event = {
            "name": name,
            "ph": "M",
            "ts": ts,
            "pid": self.pid,
            "tid": self.tid,
            "args": args,
        }
        with self.buffer_lock:
            self.buffer.append(event)

    def flush(self):
        """Flush buffered events to the HTTP trace server."""
        with self.buffer_lock:
            if not self.buffer:
                return
            events_to_send = self.buffer
            self.buffer = []

        try:
            data = json.dumps(events_to_send).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/trace",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
        except Exception as e:
            # Log failure but avoid crashing client execution
            logger.warning(f"Failed to flush {len(events_to_send)} trace events: {e}")
            # Restore events to buffer so we don't lose them
            with self.buffer_lock:
                self.buffer = events_to_send + self.buffer

    def _background_loop(self):
        """Loop running every 2 seconds in a background thread to flush buffers and sync clock daily."""
        while self.running:
            try:
                time.sleep(2.0)
                # 1. Periodically flush buffer
                self.flush()

                # 2. Daily Re-synchronization (every 86400 seconds)
                time_since_sync = time.time() - self.last_sync_time
                if time_since_sync >= 86400.0:
                    self.sync_clock()
            except Exception as e:
                logger.error(f"Error in tracer background loop: {e}")

    def shutdown(self):
        """Gracefully shut down the background thread and perform a final synchronous flush."""
        if self.running:
            self.running = False
            # Final synchronous flush of any remaining events
            self.flush()


# Global tracer client instance
_tracer = None
_tracer_lock = threading.Lock()


def init_tracer(server_ip: str, port: int = 8888, process_name: str = None, thread_name: str = None):
    """Initialize the global distributed tracer client."""
    global _tracer
    with _tracer_lock:
        if server_ip:
            try:
                _tracer = DistTracer(
                    server_ip=server_ip,
                    port=port,
                    process_name=process_name,
                    thread_name=thread_name
                )
            except Exception as e:
                logger.error(f"Failed to initialize tracer client: {e}")
                _tracer = None
        else:
            _tracer = None


def get_tracer():
    """Retrieve the global tracer instance, or None if disabled."""
    global _tracer
    with _tracer_lock:
        return _tracer


@contextmanager
def trace_span(name: str, cat: str = "default", args: dict = None):
    """Context manager to trace execution of a code block."""
    tracer = get_tracer()
    if tracer is None:
        yield
        return

    # Measure start time aligned to server epoch
    with tracer.sync_lock:
        offset = tracer.offset
    start_ts = (time.time_ns() // 1000) + offset

    try:
        yield
    finally:
        end_ts = (time.time_ns() // 1000) + offset
        dur = end_ts - start_ts
        tracer.log_event(name=name, cat=cat, ph="X", ts=start_ts, dur=dur, args=args)


def trace_func(cat: str = "default"):
    """Decorator to trace functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_span(func.__name__, cat=cat):
                return func(*args, **kwargs)
        return wrapper
    return decorator
