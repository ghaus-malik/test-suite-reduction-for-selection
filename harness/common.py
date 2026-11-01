"""Shared harness: problem loading, sandboxed execution and LLM judges.

Data: EvalPlus MBPP+ and HumanEval+ (many hidden tests per problem).
Execution: every candidate runs in a macOS sandbox (no network, no writes under /Users)
with a per-test timeout, and we record one pass/fail bit per test.
Judges: local MLX models score P(Yes) for "is this correct" from one forward pass.
"""
import gzip, json, os, re, subprocess, sys, tempfile, hashlib, math, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
PY = sys.executable
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner_exec.py")
SANDBOX = '(version 1)(allow default)(deny network*)(deny file-write* (subpath "/Users"))'
EXEC_TMP = tempfile.mkdtemp(prefix="evalrun_")

POLICY = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
JUDGE_A = "mlx-community/Qwen2.5-7B-Instruct-4bit"
JUDGE_B = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

LOOP_RE = re.compile(
    r"^([ \t]*)for i, \(inp, exp\) in enumerate\(zip\(inputs, results\)\):\n([ \t]+)(assertion\(.*\))[ \t]*$",
    re.M,
)


def load_problems():
    """Return a list of dicts with a uniform schema. 14 of the 542 EvalPlus problems are dropped:
    10 whose test file has no per-input loop, and 4 MBPP problems whose function name cannot be
    parsed from the first visible assert."""
    from datasets import load_dataset
    probs = []
    for ex in load_dataset("evalplus/mbppplus")["test"]:
        if not LOOP_RE.search(ex["test"]):
            continue
        m = re.search(r"assert\s+(?:set\()?\s*(?:math\.isclose\()?\s*(\w+)\(", ex["test_list"][0])
        entry = m.group(1) if m else None
        if entry is None:
            continue
        probs.append(dict(
            pid=f"MBPP/{ex['task_id']}", src="mbpp", entry=entry,
            task=f"{ex['prompt'].strip()}\nYour code should pass this test:\n{ex['test_list'][0]}",
            test=ex["test"], test_imports="\n".join(ex["test_imports"]),
            visible=list(ex["test_list"]), reference=ex["code"],
        ))
    for ex in load_dataset("evalplus/humanevalplus")["test"]:
        if not LOOP_RE.search(ex["test"]):
            continue
        probs.append(dict(
            pid=ex["task_id"], src="he", entry=ex["entry_point"],
            task=ex["prompt"], test=ex["test"], test_imports="",
            visible=[], reference=ex["prompt"] + ex["canonical_solution"], he_prompt=ex["prompt"],
        ))
    return probs


def policy_messages(p):
    if p["src"] == "mbpp":
        user = (p["task"] + "\nWrite the Python function in a single ```python code block. "
                "Do not include test code or explanations.")
    else:
        user = ("Complete the following Python function. Return the full function, with any imports, "
                "in a single ```python code block, without test code or explanations.\n```python\n"
                + p["task"] + "```")
    return [{"role": "system", "content": "You are a helpful Python programmer."},
            {"role": "user", "content": user}]


def extract_code(text, p):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    if blocks:
        code = max(blocks, key=len)
    else:
        code = text.split("```")[0] if text.count("```") == 1 else text
        code = re.sub(r"^```(?:python)?\n?", "", code)
    code = code.strip("\n") + "\n"
    if p["src"] == "he":
        imports = "\n".join(l for l in p["he_prompt"].splitlines() if l.startswith(("import ", "from ")))
        if f"def {p['entry']}" not in code:
            code = p["he_prompt"] + code
        elif imports:
            code = imports + "\n" + code
    return code


def instrument_test(test):
    """Rewrite the EvalPlus loop so each test is recorded instead of stopping at the first failure."""
    def rep(m):
        ind0, ind1, line = m.group(1), m.group(2), m.group(3)
        return (f"{ind0}for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                f"{ind1}_RUN_ONE_(lambda: {line})")
    return LOOP_RE.sub(rep, test, count=1)


MEM_CAP = 800 * 2**20        # kill a candidate process above 800 MB resident
MIN_FREE = 5 * 2**30          # and kill it if the whole machine drops below 5 GB available


def run_guarded(cmd, stdin_text, total, cwd=None):
    """Popen with a memory watchdog. Model-written code can allocate memory very fast, and a few such
    programs at once can exhaust a 24 GB machine, so every child is polled every 20 ms and killed
    when it passes MEM_CAP or the machine runs low. Returns (returncode, stdout, stderr, status)."""
    import psutil
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=fo, stderr=fe, text=True, cwd=cwd)
        try:
            pr.stdin.write(stdin_text)
            pr.stdin.close()
        except BrokenPipeError:
            pass
        ps = psutil.Process(pr.pid)
        t0, status = time.time(), "ok"
        while pr.poll() is None:
            try:
                rss = sum(c.memory_info().rss for c in [ps] + ps.children(recursive=True))
            except psutil.Error:
                rss = 0
            if rss > MEM_CAP:
                status = "mem_limit"
            elif psutil.virtual_memory().available < MIN_FREE:
                status = "low_system_memory"
            elif time.time() - t0 > total:
                status = "total_timeout"
            if status != "ok":
                try:
                    for c in ps.children(recursive=True):
                        c.kill()
                    pr.kill()
                except psutil.Error:
                    pass
                pr.wait()
                break
            time.sleep(0.02)
        fo.seek(0); fe.seek(0)
        return pr.returncode, fo.read(), fe.read(), status


def run_candidate(code, p, per_test=2.0, total=90, mode="plus"):
    """Run one candidate. mode='plus' gives a bit per hidden test plus visible asserts (MBPP).
    Returns dict(bits=[...], vis=[...], status=str)."""
    job = dict(code=code, test=instrument_test(p["test"]), test_imports=p["test_imports"],
               entry=p["entry"], src=p["src"], visible=p["visible"], per_test=per_test, mode=mode)
    rc, out, err, status = run_guarded(["sandbox-exec", "-p", SANDBOX, PY, "-I", RUNNER],
                                       json.dumps(job), total, cwd=EXEC_TMP)
    if status != "ok":
        return dict(bits=None, vis=None, status=status)
    last = [l for l in out.splitlines() if l.startswith("@@RESULT@@")]
    if not last:
        return dict(bits=None, vis=None, status=f"no_result rc={rc} {err[-300:]}")
    return json.loads(last[-1][len("@@RESULT@@"):])


def run_many(jobs, workers=4):
    """jobs: list of (code, problem). Returns results in order."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda cp: run_candidate(cp[0], cp[1]), jobs))


def h(code):
    return hashlib.sha1(code.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- judges

def limit_mlx_memory(gb=11, cache_gb=1):
    """Keep MLX well under 24 GB of unified memory so the rest of the system never starves."""
    import mlx.core as mx
    mx.set_memory_limit(gb * 2**30)
    mx.set_cache_limit(cache_gb * 2**30)


class LogitJudge:
    """Scores P(Yes) / (P(Yes)+P(No)) for the next token after a chat prompt.
    Right padding is exact under a causal mask because we read each row at its own last token."""

    def __init__(self, repo):
        import mlx.core as mx
        from mlx_lm import load
        limit_mlx_memory()
        self.mx = mx
        self.repo = repo
        self.model, self.tok = load(repo)
        self.yes = self._first_id("Yes")
        self.no = self._first_id("No")

    def _first_id(self, w):
        ids = self.tok.encode(w, add_special_tokens=False)
        return ids[0]

    def encode(self, messages):
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok.encode(s, add_special_tokens=False)

    def _head(self, hdn):
        m = self.model
        if getattr(m, "lm_head", None) is not None and not getattr(getattr(m, "args", None), "tie_word_embeddings", False):
            return m.lm_head(hdn)
        return m.model.embed_tokens.as_linear(hdn)

    def score_tokens(self, toks_list, batch_tokens=6000):
        mx = self.mx
        order = sorted(range(len(toks_list)), key=lambda i: len(toks_list[i]))
        out = [None] * len(toks_list)
        i = 0
        while i < len(order):
            L = len(toks_list[order[i]])
            j = i
            while j < len(order) and (j - i + 1) * len(toks_list[order[j]]) <= max(batch_tokens, len(toks_list[order[j]])):
                j += 1
            idx = order[i:j]
            maxlen = max(len(toks_list[k]) for k in idx)
            arr = [toks_list[k] + [0] * (maxlen - len(toks_list[k])) for k in idx]
            x = mx.array(arr)
            hdn = self.model.model(x)
            last = mx.array([len(toks_list[k]) - 1 for k in idx])
            hl = hdn[mx.arange(len(idx)), last]
            logits = self._head(hl).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            ly, ln = lp[:, self.yes], lp[:, self.no]
            mx.eval(ly, ln)
            for r, k in enumerate(idx):
                a, b = float(ly[r]), float(ln[r])
                py = 1.0 / (1.0 + math.exp(b - a))
                out[k] = dict(p_yes=py, mass=math.exp(a) + math.exp(b), logit_margin=a - b)
            i = j
        return out

    def score(self, messages_list, **kw):
        return self.score_tokens([self.encode(m) for m in messages_list], **kw)


def judge_messages(task, code):
    return [
        {"role": "system", "content": "You are an expert Python code reviewer."},
        {"role": "user", "content": (
            f"Task:\n{task}\n\nCandidate solution:\n```python\n{code.rstrip()}\n```\n\n"
            "Does this solution correctly solve the task for all valid inputs? "
            "Answer with a single word: Yes or No.")},
    ]


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_json(path):
    """Reads path, or path + '.gz' if only the compressed copy exists."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)
