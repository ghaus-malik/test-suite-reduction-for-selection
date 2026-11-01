"""System-wide safety net. Every 0.5 s: if available memory falls below LOW, kill the largest process that
belongs to these experiments (venv Python or a sandboxed candidate), never anything else. Logs every action."""
import os, sys, time, psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW = 3.5 * 2**30
LOG = os.path.join(ROOT, "memguard.log")


def ours(p):
    """Only Python processes started for these experiments: run from this folder or from a candidate temp dir."""
    try:
        if "python" not in p.name().lower() or p.pid == os.getpid():
            return False
        cmd = " ".join(p.cmdline())
        cwd = p.cwd()
    except psutil.Error:
        return False
    return ROOT in cmd or cwd.startswith(ROOT) or "evalrun_" in cwd


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


log(f"memguard started, threshold {LOW / 2**30:.1f} GB available")
last_beat = 0
while True:
    vm = psutil.virtual_memory()
    if vm.available < LOW:
        cands = []
        for p in psutil.process_iter(["pid"]):
            if ours(p):
                try:
                    cands.append((p.memory_info().rss, p))
                except psutil.Error:
                    pass
        if cands:
            rss, p = max(cands, key=lambda x: x[0])
            try:
                cmd = " ".join(p.cmdline())[:160]
                p.kill()
                log(f"LOW MEMORY {vm.available / 2**30:.1f} GB available: killed pid {p.pid} rss {rss / 2**30:.1f} GB: {cmd}")
            except psutil.Error:
                pass
        time.sleep(1.0)
    if time.time() - last_beat > 600:
        log(f"ok, available {vm.available / 2**30:.1f} GB, swap used {psutil.swap_memory().used / 2**30:.1f} GB")
        last_beat = time.time()
    time.sleep(0.5)
