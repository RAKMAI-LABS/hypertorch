"""HyperTorch vs Python serving benchmark — the Problem 1 numbers.

Uses a raw-socket HTTP/1.1 client (identical for both servers) instead of
urllib: on Windows, urllib was found to add ~15ms of client-side overhead
unevenly between servers, corrupting latency comparisons. The raw client
opens a new connection per request (like real short-lived clients), sets
TCP_NODELAY, sends the request in one write, and reads a Content-Length
response.

Measures per server:
  1. cold start   - process launch until /health responds
  2. memory (RSS) - after warmup (needs psutil; skipped if absent)
  3. latency      - p50/p99, sequential, new connection per request
  4. keep-alive   - p50 on one persistent connection (server compute view)
  5. throughput   - req/s with 8 concurrent clients

Usage:
  python python/bench.py --rust "target/release/examples/serve model.rtw 7878" \
                         --python "python python/serve_baseline.py model.rtw 7879"
Or one already-running server:
  python python/bench.py --url http://127.0.0.1:7878
"""

import argparse
import concurrent.futures
import os
import shlex
import socket
import statistics
import struct
import subprocess
import time

IN_FEATURES = 784

try:
    import psutil
except ImportError:
    psutil = None


def parse_hostport(url):
    hp = url.split("//", 1)[-1].split("/", 1)[0]
    host, _, port = hp.partition(":")
    return host, int(port or 80)


def build_request(method, path, host, body=b"", keep_alive=False):
    conn = "keep-alive" if keep_alive else "close"
    head = (f"{method} {path} HTTP/1.1\r\nHost: {host}\r\n"
            f"Connection: {conn}\r\nContent-Length: {len(body)}\r\n\r\n")
    return head.encode() + body


def read_response(s):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(65536)
        if not chunk:
            raise ConnectionError("server closed before headers")
        data += chunk
    head, rest = data.split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    clen = 0
    for line in head.split(b"\r\n")[1:]:
        k, _, v = line.partition(b":")
        if k.strip().lower() == b"content-length":
            clen = int(v)
    while len(rest) < clen:
        chunk = s.recv(65536)
        if not chunk:
            break
        rest += chunk
    return status, rest[:clen]


def request_once(host, port, req):
    s = socket.create_connection((host, port), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        s.sendall(req)
        status, body = read_response(s)
    finally:
        s.close()
    return status, body


def wait_health(host, port, proc, timeout=90.0):
    req = build_request("GET", "/health", host)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            status, _ = request_once(host, port, req)
            if status == 200:
                return time.perf_counter() - t0
        except OSError:
            time.sleep(0.005)
    raise TimeoutError("server never became healthy")


def one_image_body():
    vals = [((i * 2654435761) % 1000) / 1000.0 for i in range(IN_FEATURES)]
    return struct.pack(f"<{IN_FEATURES}f", *vals)


def bench_server(name, cmd, url):
    print(f"\n=== {name} ===")
    host, port = parse_hostport(url)
    proc = None
    cold = None
    if cmd:
        args_list = cmd if os.name == "nt" else shlex.split(cmd)
        proc = subprocess.Popen(args_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cold = wait_health(host, port, proc)
        print(f"cold start:      {cold * 1000:.1f} ms  (launch -> first /health OK)")
    else:
        wait_health(host, port, None, timeout=5)
        print("cold start:      (server was already running; not measured)")

    body = one_image_body()
    post = build_request("POST", "/predict", host, body)

    for _ in range(20):
        request_once(host, port, post)

    if psutil and proc:
        p = psutil.Process(proc.pid)
        rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
        print(f"memory (RSS):    {rss / 1e6:.1f} MB")
    elif proc:
        print("memory (RSS):    (pip install psutil to measure)")

    # sequential latency, fresh connection per request
    lat = []
    for _ in range(300):
        t0 = time.perf_counter()
        request_once(host, port, post)
        lat.append(time.perf_counter() - t0)
    lat.sort()
    p50 = statistics.median(lat) * 1e6
    p99 = lat[int(len(lat) * 0.99) - 1] * 1e6
    print(f"latency p50:     {p50:.0f} us   (new connection per request)")
    print(f"latency p99:     {p99:.0f} us")

    # keep-alive latency (pure request/response, connection reused)
    ka_req = build_request("POST", "/predict", host, body, keep_alive=True)
    ka = []
    try:
        s = socket.create_connection((host, port), timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for _ in range(300):
            t0 = time.perf_counter()
            s.sendall(ka_req)
            read_response(s)
            ka.append(time.perf_counter() - t0)
        s.close()
        ka_p50 = statistics.median(ka) * 1e6
        print(f"keep-alive p50:  {ka_p50:.0f} us   (persistent connection)")
    except (OSError, ConnectionError):
        ka_p50 = None
        print("keep-alive p50:  (server closed persistent connection; skipped)")

    # concurrent throughput
    n_req, workers = 1000, 8
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda _: request_once(host, port, post), range(n_req)))
    dt = time.perf_counter() - t0
    print(f"throughput:      {n_req / dt:.0f} req/s  ({workers} concurrent clients)")

    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"cold_ms": cold * 1000 if cold else None, "p50_us": p50,
            "ka_p50_us": ka_p50, "rps": n_req / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", help="command to launch the Rust server")
    ap.add_argument("--python", dest="pybase", help="command to launch the Python baseline")
    ap.add_argument("--url", help="benchmark one already-running server")
    args = ap.parse_args()

    results = {}
    if args.url:
        bench_server("server", None, args.url)
        return
    if args.rust:
        results["rust"] = bench_server("HyperTorch (Rust)", args.rust, "http://127.0.0.1:7878")
    if args.pybase:
        results["python"] = bench_server("FastAPI + PyTorch", args.pybase, "http://127.0.0.1:7879")

    if "rust" in results and "python" in results:
        r, p = results["rust"], results["python"]
        print("\n=== head to head ===")
        if r["cold_ms"] and p["cold_ms"]:
            print(f"cold start:  {p['cold_ms'] / r['cold_ms']:.1f}x faster in Rust")
        print(f"p50 latency: {p['p50_us'] / r['p50_us']:.1f}x lower in Rust")
        if r["ka_p50_us"] and p["ka_p50_us"]:
            print(f"keep-alive:  {p['ka_p50_us'] / r['ka_p50_us']:.1f}x lower in Rust")
        print(f"throughput:  {r['rps'] / p['rps']:.1f}x higher in Rust")


if __name__ == "__main__":
    main()
