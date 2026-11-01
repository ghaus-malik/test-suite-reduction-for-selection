"""Runs inside sandbox-exec. Reads one job as JSON on stdin, prints one @@RESULT@@ line.
A candidate that kills the process (os._exit, segfault) produces no result line, which the caller counts as a failure."""
import sys, json, signal, io, contextlib, os

job = json.loads(sys.stdin.read())
RESULT = dict(bits=None, vis=None, status="ok")
OUT = sys.__stdout__


def emit():
    OUT.write("\n@@RESULT@@" + json.dumps(RESULT) + "\n")
    OUT.flush()


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)
PER = job["per_test"]


def guarded(fn, t):
    try:
        signal.setitimer(signal.ITIMER_REAL, t)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
        signal.setitimer(signal.ITIMER_REAL, 0)
        return 1, None
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return 0, type(e).__name__


g = {"__name__": "__candidate__"}
ok, err = guarded(lambda: exec(compile(job["code"], "<candidate>", "exec"), g), 10.0)
if not ok:
    RESULT["status"] = "load_error:" + str(err)
    emit()
    os._exit(0)

# visible asserts (MBPP only), each in a fresh copy of the candidate namespace
vis = []
for a in job["visible"]:
    ns = dict(g)
    ok, _ = guarded(lambda: exec(job["test_imports"] + "\n" + a, ns), PER)
    vis.append(ok)
RESULT["vis"] = vis

bits = []
state = {"timeouts": 0}


def _RUN_ONE_(fn):
    if state["timeouts"] >= 3:
        bits.append(0)
        return
    ok, err = guarded(fn, PER)
    if err == "_Timeout":
        state["timeouts"] += 1
    bits.append(ok)


tg = dict(g)
tg["_RUN_ONE_"] = _RUN_ONE_
ok, err = guarded(lambda: exec(job["test_imports"] + "\n" + job["test"], tg), 600.0)
if job["src"] == "he" and ok:
    ok, err = guarded(lambda: tg["check"](g[job["entry"]]), 600.0)
if not ok and not bits:
    RESULT["status"] = "test_error:" + str(err)
RESULT["bits"] = bits
emit()
os._exit(0)
