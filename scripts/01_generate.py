"""Sample N candidate solutions per problem from the small policy model (Qwen2.5-Coder-1.5B-Instruct, 4-bit)."""
import sys, time, random, json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
TEMP = 1.0
out_path = os.path.join(DATA, f"generations_n{N}.json")
bad = set(load_json(os.path.join(DATA, "harness_check.json"))["bad_reference"])
probs = [p for p in load_problems() if p["pid"] not in bad]
if LIMIT:
    probs = probs[:LIMIT]
mx.random.seed(1)  # the released samples: problems 1-288 used seed 0, the rest seed 1 (the run was resumed once)
limit_mlx_memory()
model, tok = load(POLICY)
sampler = make_sampler(temp=TEMP, top_p=0.95)
done = load_json(out_path) if os.path.exists(out_path) and not LIMIT else {}
t0 = time.time(); ntok = 0
CHUNK = 16  # problems per batch call -> 8*N sequences
todo = [p for p in probs if p["pid"] not in done]
for c in range(0, len(todo), CHUNK):
    chunk = todo[c:c + CHUNK]
    prompts = []
    for p in chunk:
        ids = tok.apply_chat_template(policy_messages(p), add_generation_prompt=True)
        prompts += [ids] * N
    r = batch_generate(model, tok, prompts, max_tokens=512, sampler=sampler, completion_batch_size=64, prefill_batch_size=8)
    for k, p in enumerate(chunk):
        texts = r.texts[k * N:(k + 1) * N]
        ntok += sum(len(tok.encode(t)) for t in texts)
        done[p["pid"]] = [dict(raw=t, code=extract_code(t, p)) for t in texts]
    if not LIMIT:
        save_json(done, out_path)
    el = time.time() - t0
    print(f"{len(done)}/{len(probs)} problems, {ntok} tokens, {ntok/el:.0f} tok/s, {el/60:.1f} min", flush=True)
if LIMIT:
    save_json(done, os.path.join(DATA, "gen_test.json"))
