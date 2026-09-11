# PHEF handoff — 8M smoke validated; full eval stopped as CPU-bound and non-resumable

**Handoff date:** 2026-09-10, 00:41 MSK
**Project root actually used in current session:** `~/agi`
**Python actually used:** `/home/linux/agi/.venv/bin/python` (Python 3.14.4)
**Git state last freshly observed at session start:** `HEAD = origin/main = ca3d438` (`chore: add /data snapshot backup script with 3-snapshot rotation`)

> This document is a dated handoff, not proof of current machine state. Every mutable fact—paths, branch, Git status, checkpoint, files, jobs, GPU usage, tool CLI—must be rechecked from a fresh primary artifact before use.

---

## 0. Operating discipline

### 0.1 Source of truth

1. A fresh primary artifact supplied by the user is the only source of truth for local state: full command output, file contents, JSON, checkpoint metadata, Git output, process listing, GPU output, or traceback.
2. Do not claim a file, checkpoint, result, process, GPU state, flag, configuration, commit, remote, or log exists unless it is freshly confirmed.
3. The assistant has no direct shell, repository, filesystem, GPU, Git, or process access on the user's machine. The user runs all commands.
4. Historical handoffs are context only. They never establish current state.
5. Any unexplained discrepancy is **[UNKNOWN]** and cannot support the next experiment.
6. Do not infer a JSON schema from stdout or from another tool. Confirm JSON keys in the target file.

### 0.2 Claim labels and response order

Label substantive claims:

- **[FACT]** directly present in a primary artifact
- **[INFERENCE]** logically follows from confirmed facts
- **[HYPOTHESIS]** testable explanation, not established
- **[UNKNOWN]** insufficient data; requires a minimal discriminating read-only step

Response order:

1. fact check;
2. inference;
3. decision;
4. exactly one next command/step.

Before suggesting a command, state:

- what it checks or changes;
- expected result;
- success signal.

### 0.3 Explicit consent for writes

Require explicit user approval before any action that writes or changes state, including:

- train/eval runs that create JSON, logs, checkpoints, or other artifacts;
- file creation, modification, moving, or deletion;
- Git staging, restore, commit, push, reset, cleanup;
- stopping a process;
- system changes.

Before approval, show the exact command or full diff, all affected paths, expected artifacts, and risk/irreversibility. After any write, request a read-only verification of the actual output.

### 0.4 Git discipline

- Never use `git add .` or `git add -A`.
- Never mass-delete, mass-clean, or stage untracked files without a separate inventory and approval.
- Use exact whitelisted paths for staging.
- Do not automatically add `.pt`, logs, drafts, backups, or unknown artifacts.
- Before any commit: show exact staged file list, `git diff --cached --stat`, `git diff --cached --check`, and a meaningful staged diff/summary.
- Commit and push are separate approved actions.
- Before push, inspect remote URL, branch/upstream, `origin/main..HEAD`, and commit list. After push, verify `HEAD` equals `origin/main`.

### 0.5 GPU and long jobs

- Before a CUDA job, read-only check relevant processes, GPU compute apps, free VRAM, and output-path conflicts.
- Never run parallel CUDA jobs on the single GPU.
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for CUDA Python jobs.
- Long jobs must be started only after approval and should have observable progress and resumability. Do not repeat the old evaluator's design: it emits no progress and only writes the JSON at the very end.
- Do not interpret a quiet or empty redirected Python log as a hang by itself; buffered stdout can remain empty until process exit.
- Measure VRAM externally with `nvidia-smi` sampling. `/usr/bin/time -v` reports host RSS, not VRAM.
- Any external sampler command must be structured in an explicit shell script/function: do not combine background `&` with an implicit `&&` chain that can lose variable scope or let later commands escape an earlier refusal.

---

## 1. Freshly confirmed project state from this session

### 1.1 Actual paths and environment

**[FACT]** The prior historical project root and venv paths were stale. The commands actually ran from:

```text
~/agi
/home/linux/agi/.venv/bin/python
Python 3.14.4
```

**[FACT]** The current evaluator file was present at:

```text
phef/experiments/eval_recall_checkpoint.py
```

It was 4.8K with observed mtime 2026-09-07 22:52.

**[FACT]** At the session preflight, the Git state was:

```text
## main...origin/main
ca3d438 (HEAD -> main, origin/main, origin/HEAD)
chore: add /data snapshot backup script with 3-snapshot rotation
```

**[UNKNOWN]** Git state may have changed after the session start. Recheck it first in the next chat.

### 1.2 Current evaluator contract

**[FACT]** CLI observed from `--help`:

```text
usage: eval_recall_checkpoint.py [-h] --checkpoint CHECKPOINT
  [--n-keys N_KEYS] [--n-values N_VALUES] [--n-segments N_SEGMENTS]
  [--overwrite-p OVERWRITE_P] [--key-dim KEY_DIM] [--dim DIM]
  [--eval-batch-size EVAL_BATCH_SIZE] [--eval-batches EVAL_BATCHES]
  [--seed SEED] [--cpu] --out OUT
```

**[FACT]** There is no `--conditions` CLI flag. The evaluator always runs three conditions sequentially:

```python
real = evaluate(...)
shuffled = evaluate(..., shuffle=True)
masked = evaluate(..., mask_memory=True)
```

**[FACT]** The evaluator JSON schema differs from MFR. In evaluator JSON, conditions are top-level:

```text
checkpoint
checkpoint_epoch
checkpoint_sel_acc
hard_eval
answer_chance
real
shuffled
masked
config
```

MFR conditions are under `results.real` instead; do not mix schemas.

### 1.3 Batch-size invariant discovered

**[FACT]** `eval_batch_size=1` is invalid for the current evaluator's shuffled condition. `PHEF.forward(..., shuffle=True)` runs:

```python
shift = torch.randint(1, batch_size, (), device=values.device)
```

For `batch_size=1`, this fails with:

```text
RuntimeError: random_ expects 'from' to be less than 'to', but got from=1 >= to=1
```

**[INFERENCE]** The minimum valid evaluation batch size in the current unmodified evaluator is 2.

---

## 2. Scientific status

### 2.1 Task mechanism

**[FACT]** The PHEF task is last-occurrence retrieval: select the final position of the queried key in a sequence. Codebook keys are resampled each batch and L2-normalized, so a static lookup table cannot solve the task.

**[FACT]** The address score is:

\[
s_j = \alpha\frac{\langle q,k_{\mathrm{slot}_j}\rangle}{\sqrt d} + \beta\frac{t_j}{S-1}.
\]

All duplicate occurrences of the same key have the same semantic score; only the recency term can choose the last one among duplicates.

**[INFERENCE]** Retrieval errors partition exactly into:

```text
same_key_recency_error: predicted slot is queried slot but wrong occurrence
wrong_key_semantic_error: predicted slot is not queried slot
```

### 2.2 Established Phase-B failure mechanism

**[FACT]** Prior paired MFR measurements on replay checkpoints at `K=131072`, `S=524288`, same seed and 2048 episodes yielded:

| Metric | epoch 750 | epoch 1500 | delta |
|---|---:|---:|---:|
| `answer_acc` | 0.916016 | 0.868164 | -0.047852 |
| `retrieval_acc` | 0.916016 | 0.867676 | -0.048340 |
| `wrong_key_semantic_error` | 172 | 271 | +99 |
| `same_key_recency_error` | 0 | 0 | 0 |
| `semantic_inversion` | 167 | 255 | +88 |
| `semantic_inversion_rate` | 0.081543 | 0.124512 | +0.042969 |
| `semantic_lead_rate` | 0.918457 | 0.875488 | -0.042969 |
| `margin_nonpositive_rate` | 0.083984 | 0.132324 | +0.048340 |
| `B_norm_mean` | 2.2946 | 2.2013 | -0.0933 |

**[INFERENCE]** The mechanism is **semantic collapse**, not a failure to select the latest duplicate: of 99 new errors, 88 were semantic inversions and 11 were positive-gap episodes overridden by recency (88.9% vs 11.1%).

**[INFERENCE]** Rejected CAMP targets for this measured failure interval:

- `B_norm` / recency budget;
- alpha-to-beta ratio;
- L2 anchoring to parameters;
- only the mean semantic gap.

The functional loss target must protect the lower tail of per-episode semantic gap:

\[
\Delta_{\mathrm{sem},i} = \operatorname{sem}_{\mathrm{true},i} - \max_{k\ne k_{\mathrm{true},i}}\operatorname{sem}_{i,k}.
\]

High-`n_keys` anchor episodes are necessary; a training-size `n_keys=512` anchor is structurally too weak to expose the extreme-competitor failure mode.

### 2.3 CAMP status

**[FACT]** CAMP has not been implemented.

**[INFERENCE]** Before CAMP implementation, preserve/measure a high-`K` anchor signature from the healthy mechanism and define an experiment with a matched same-seed baseline. Do not write CAMP code before a clear change plan and explicit approval.

---

## 3. Early KD32 high-capacity baseline

### 3.1 Validated 4M independent PASS

**[FACT]** Primary target JSON was read in this session:

```text
phef/results/early_s65536_seed7_rw10_keys4194304_b2_eval16384.json
```

It reported:

```text
checkpoint: results/early_s65536_seed7_rw10_e6000_best.pt
checkpoint_epoch: 4500
checkpoint_sel_acc: 1.0
hard_eval: True
n_keys: 4194304
n_segments: 65536
n_values: 64
overwrite_p: 0.3
key_dim: 32
dim: 128
eval_batch_size: 2
eval_batches: 8192
seed: 7
```

Full 16,384 episode result per condition:

| Condition | episodes | answer_acc | retrieval_acc | Criterion | Verdict |
|---|---:|---:|---:|---:|---|
| real | 16384 | 0.97845458984375 | 0.978271484375 | both >= 0.90 | PASS |
| shuffled | 16384 | 0.0140380859375 | 0.0 | answer < 0.03125 | PASS |
| masked | 16384 | 0.01629638671875 | null | answer < 0.03125 | PASS |

**[INFERENCE]** Early KD32 has an independently confirmed lower clean key-capacity bound of \(2^{22}=4,194,304\) keys under this protocol.

### 3.2 Healthy 4M MFR anchor

**[FACT]** MFR target JSON read this session:

```text
phef/results/mfr_early_k4194304_s65536_b2x1024.json
```

Its validated MFR configuration:

```text
checkpoint: phef/results/early_s65536_seed7_rw10_e6000_best.pt
checkpoint_epoch: 4500
hard_eval: true
n_keys: 4194304
n_segments: 65536
n_values: 64
batch_size: 2
batches: 1024
episodes: 2048
seed: 7
conditions: real
```

Headline MFR numbers:

| Metric | Value |
|---|---:|
| answer_acc | 0.97802734375 |
| retrieval_acc | 0.97802734375 |
| margin_nonpositive_rate | 0.02197265625 |
| same_key_recency_error | 0 |
| wrong_key_semantic_error | 45 |
| wrong_key_semantic_rate | 0.02197265625 |
| semantic_inversion | 33 |
| semantic_inversion_rate | 0.01611328125 |
| semantic_lead_rate | 0.98388671875 |
| delta_sem mean | 1.95082103647 |
| delta_sem positive mean | 1.98706601997 |
| recency_budget_norm mean | 2.94365767748 |
| true_rank mean | 1.037109375 |
| alpha/abs(beta) | 1.48140909956 |
| semantic threshold abs(beta)/alpha | 0.67503298062 |
| score_scale alpha | 2.01473617554 |
| recency_scale beta | 1.36001336575 |

**[INFERENCE]** This MFR result is the current healthy high-capacity semantic anchor. It is a mechanism diagnostic, not an operational PASS criterion.

---

## 4. 8M work performed this session

### 4.1 Provenance audit

**[FACT]** A full 4M eval log and JSON exist, plus MFR logs/JSON, confirming the 4M result was genuinely executed. Relevant historical artifacts included:

```text
phef/logs/early_s65536_seed7_rw10_keys4194304_b2_eval16384.log
phef/results/early_s65536_seed7_rw10_keys4194304_b2_eval16384.json
phef/logs/mfr_early_k4194304_s65536_b2x1024.log
phef/results/mfr_early_k4194304_s65536_b2x1024.json
```

### 4.2 Invalid 8M batch-1 attempt

**[FACT]** A first 8M `eval_batch_size=1` probe reached `real` then failed at shuffled due to the batch-size invariant above. It wrote no JSON.

**[FACT]** Its external VRAM log recorded a peak of 9728 MiB before the failure, but this is not a valid full protocol measurement because shuffled and masked did not complete.

An old log exists:

```text
phef/logs/vram_early_s65536_seed7_rw10_keys8388608_b1_probe.log
```

Do not overwrite/delete it without explicit approval.

### 4.3 Valid 8M batch-2 smoke

**[FACT]** A valid short 8M smoke completed using the current evaluator:

```text
checkpoint: phef/results/early_s65536_seed7_rw10_e6000_best.pt
n_keys: 8388608
n_segments: 65536
n_values: 64
overwrite_p: 0.3
key_dim: 32
dim: 128
eval_batch_size: 2
eval_batches: 1
seed: 7
```

Artifacts:

```text
phef/results/early_s65536_seed7_rw10_keys8388608_b2_probe_retry2.json
phef/logs/vram_early_s65536_seed7_rw10_keys8388608_b2_probe_retry2.log
```

Reported:

```text
real answer = 1.0, real retrieval = 1.0
shuffled answer = 0.0, shuffled retrieval = 0.0
masked answer = 0.0
exit status = 0
peak external VRAM = 20994 MiB
```

**[INFERENCE]** At `K=8,388,608`, `S=65,536`, batch 2, all three control paths execute and fit on this RTX 4090. Two episodes are not evidence of quality/PASS.

### 4.4 Full 8M run: stopped

**[FACT]** A full independent 8M eval was started with:

```text
n_keys=8388608
eval_batch_size=2
eval_batches=8192
seed=7
out=phef/results/early_s65536_seed7_rw10_keys8388608_b2_eval16384.json
log=phef/logs/early_s65536_seed7_rw10_keys8388608_b2_eval16384.log
PID=31830
```

**[FACT]** It ran for approximately 24 hours. Repeated observed states:

```text
CPU: 100% on one core
GPU utilization: 0% in two samples 10 seconds apart
GPU memory: about 20970–20996 MiB, stable
GPU power: about 65 W
host RSS: grew from about 3.1 GiB to 6.05 GiB
log: 0 bytes (stdout buffered under nohup)
final JSON: absent
```

**[FACT]** The process was explicitly stopped with `kill -TERM 31830` and exited. The final full-eval JSON was never created. The retained full-eval log remained 0 bytes.

**[INFERENCE]** No 8M full PASS/FAIL result exists. The old evaluator is unsuitable for a costly long 8M run as currently instrumented because it has no progress reporting, no partial-result checkpoints, and no resume facility.

**[IMPORTANT CORRECTION]** The old evaluator source proves that it does *not* retain CUDA tensors across batches: it uses scalar counters and `torch.inference_mode()`. Therefore its stable VRAM does not show a GPU tensor leak. The CPU RSS growth is observed but its exact cause is **[UNKNOWN]**; do not call it a memory leak without a targeted measurement.

---

## 5. Current source facts about evaluator performance

**[FACT]** Current `evaluate()` loop:

```python
@torch.inference_mode()
def evaluate(model, world, batch_size, device, n_batches, shuffle=False, mask_memory=False):
    model.eval()
    correct = 0
    retrieval_correct = 0
    total = 0
    score_retrieval = not mask_memory

    for _ in range(n_batches):
        (...) = move_batch(world.batch(batch_size), device)
        logits, retrieval_scores = model(..., return_retrieval=True)
        correct += logits.argmax(dim=-1).eq(answers).sum().item()
        if score_retrieval:
            retrieval_correct += retrieval_scores.argmax(dim=-1).eq(answer_positions).sum().item()
        total += answers.numel()

    return {
        "answer_acc": correct / total,
        "retrieval_acc": retrieval_correct / total if score_retrieval else None,
        "episodes": total,
    }
```

**[FACT]** `world.batch(batch_size)` is constructed before `move_batch(..., device)`, so CPU generation and CPU-to-GPU transfer are candidate bottlenecks.

**[HYPOTHESIS]** The full 8M run's slow wall time is dominated by CPU-side world generation and/or host-to-device transfer at large K. This is consistent with 100% CPU and instantaneous GPU idleness, but one must not claim this is fully established without block-level timing.

**[FACT]** The `masked` model path skips the heavy semantic route (`key_encoder(codebook)`, batched `einsum`, gather semantic scores, and `memory_values` indexing). `real` and `shuffled` have matching main shapes; `shuffled` permutes batch members.

---

## 6. Correct next research plan

### Goal A — make 8M evaluation measurable and resumable

Do **not** relaunch the unmodified full 8M evaluator.

The next implementation should be a separate tool, not a silent modification of `eval_recall_checkpoint.py`, for example:

```text
phef/experiments/eval_recall_checkpoint_resumable.py
```

This is an unimplemented proposal, not an existing file.

Required design constraints:

1. Same functional protocol as old evaluator: fixed seed, same `BranchingWorld`, same `PHEF`, `hard_eval=True`, same real/shuffled/masked semantics.
2. Progress report every fixed block, e.g. 32/64/128 batches: condition, completed batches, total batches, elapsed seconds, average batch time, current answer/retrieval counters, CPU RSS, optional current GPU memory.
3. Atomic partial-state persistence after each block, using a temporary file plus rename.
4. State must include: version, fully explicit config, condition index/name, completed batch count, scalar counters, all relevant RNG states (`random`, NumPy, CPU Torch, CUDA Torch if used), and identity/provenance of checkpoint.
5. Resume must refuse config/checkpoint mismatch.
6. Final JSON must be a clearly distinguishable format/version and must not overwrite old evaluator output by default.
7. Preserve deterministic sequence equivalence on resume: checkpoint RNG state only at exact batch boundaries after counters are updated.
8. Add a `--stop-after-batches` or equivalent test limit only if its semantics are explicit and it helps validation; avoid hidden test behavior.
9. Do not serialize GPU tensors in progress state; only scalar counters and CPU-serializable RNG state.
10. Always flush progress output (`print(..., flush=True)`) in long jobs.

### Goal B — validate the new tool before 8M

Validation must precede 8M full rerun:

1. Read existing evaluator and relevant RNG/world code carefully.
2. Propose an exact diff and artifact naming plan.
3. Get explicit approval for code creation/modification.
4. Run a small baseline equivalence test with the same checkpoint/config/seed and a small batch count, comparing old evaluator and new evaluator results exactly where deterministic equality is expected.
5. Test interrupted/resumed run against uninterrupted new-tool run with the same seed/config and prove equal final counters/metrics.
6. Only then use it for full 8M.

### Goal C — answer 8M capacity

After resumable validation, run 8M with:

```text
checkpoint       = phef/results/early_s65536_seed7_rw10_e6000_best.pt
n_keys           = 8388608
n_segments       = 65536
n_values         = 64
overwrite_p      = 0.3
key_dim / dim    = 32 / 128
eval_batch_size  = 2
eval_batches     = 8192
episodes         = 16384 per condition
seed             = 7
```

Operational PASS is only:

```text
hard_eval = true
real.answer_acc >= 0.90
real.retrieval_acc >= 0.90
shuffled.answer_acc < 0.03125
masked.answer_acc < 0.03125
episodes = 16384 in real, shuffled, masked
```

### Goal D — then return to CAMP

- If 8M passes: capacity lower bound becomes 8M; decide deliberately whether a higher K or CAMP is the next priority.
- If 8M fails: run MFR at 8M (with a separate memory-safe probe first) on the same seed to quantify semantic inversion, margin and tail changes relative to the 4M anchor.
- Only after high-K anchor/boundary is established should CAMP implementation begin.

---

## 7. First command in the new chat

**Purpose:** Freshly confirm that the terminated process is gone, GPU is free, relevant artifacts still exist, Git state is known, and no relevant files have been modified before planning code changes.

**Type:** read-only. It creates/modifies/deletes nothing.

```bash
cd ~/agi && \
printf '%s\n' '===== time ====='; \
date '+%F %T %Z'; \
printf '%s\n' '===== git ====='; \
git status -sb; \
git log -3 --oneline --decorate; \
printf '%s\n' '===== Python ====='; \
command -v python; \
python --version; \
printf '%s\n' '===== relevant processes ====='; \
pgrep -af 'train_recall.py|eval_recall_checkpoint.py|probe_memory_flight_recorder.py' || true; \
printf '%s\n' '===== GPU ====='; \
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader; \
printf '%s\n' '===== GPU compute apps ====='; \
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader; \
printf '%s\n' '===== key artifacts ====='; \
ls -lh \
  phef/results/early_s65536_seed7_rw10_keys4194304_b2_eval16384.json \
  phef/results/mfr_early_k4194304_s65536_b2x1024.json \
  phef/results/early_s65536_seed7_rw10_keys8388608_b2_probe_retry2.json \
  phef/logs/vram_early_s65536_seed7_rw10_keys8388608_b2_probe_retry2.log \
  phef/logs/early_s65536_seed7_rw10_keys8388608_b2_eval16384.log \
  2>&1; \
printf '%s\n' '===== relevant source status ====='; \
git status --short -- \
  phef/experiments/eval_recall_checkpoint.py \
  phef/phef/model.py \
  phef/world.py
```

**Expected success signal:** no active relevant process; no GPU compute app; GPU largely free; expected 4M and 8M smoke artifacts visible; exact Git state shown.

---

## 8. Do not lose these caveats

- The target 4M evaluator JSON is the primary evidence of full 4M PASS; historical handoffs merely agree with it.
- MFR and evaluator JSON schemas differ.
- A two-episode smoke cannot establish capacity or accuracy.
- The 8M full run was intentionally stopped; it produced no usable partial metrics.
- `eval_batch_size=1` is invalid without a deliberate code change to shuffled behavior.
- Do not infer current filesystem/Git/GPU state from this handoff.
- Do not patch the existing validated evaluator in place merely to make 8M convenient; build, validate, and name a separate resumable tool.
- Before any code write, enumerate target files, show full diff, output paths, testing plan, and request user approval.
