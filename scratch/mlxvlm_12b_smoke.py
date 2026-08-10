"""Smoke test: can mlx-vlm load + text-generate gemma-4-12B-it (gemma4_unified)?
Downloads the model on first run (~12GB for 8bit). Fail-fast before wiring the probe."""
import sys, time
sys.path.insert(0, "src")
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-12B-it-8bit"
t0 = time.time()
model, processor = load(REPO)
print(f"[load] {REPO} in {time.time()-t0:.0f}s", flush=True)
cfg = model.config if hasattr(model, "config") else load.__self__  # config for chat template
# apply_chat_template needs the model config object
try:
    from mlx_vlm.utils import load_config
    config = load_config(REPO)
except Exception:
    config = getattr(model, "config", {})

prompt_text = "You are playing Slay the Spire. In ONE sentence: what happens when you play a Skill against a Gremlin Nob?"
for think in (False, True):
    try:
        formatted = apply_chat_template(processor, config, prompt_text,
                                        add_generation_prompt=True, num_images=0,
                                        enable_thinking=think)
    except TypeError:
        formatted = apply_chat_template(processor, config, prompt_text,
                                        add_generation_prompt=True, num_images=0)
    t1 = time.time()
    res = generate(model, processor, formatted, image=None, max_tokens=200,
                   temperature=0.0, verbose=False)
    dt = time.time() - t1
    text = res.text if hasattr(res, "text") else str(res)
    ntok = getattr(res, "generation_tokens", None)
    print(f"[gen think={think}] {dt:.0f}s tokens={ntok} tok/s~{(ntok/dt if ntok else 0):.0f}", flush=True)
    print("  out:", repr(text[:300]), flush=True)
