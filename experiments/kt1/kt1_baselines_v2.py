"""Baseline v2: honest delta selection on same-length validation."""
import sys, json, os, numpy as np
sys.path.insert(0, 'experiments/kt1')
from kt1_baselines import ExactHashMap, QuantizedDict, key_scale, make_episode, perturb_keys
from kt1_fix_delta import select_delta_fixed
from kt1_stats import wilson

SIG=(0.0,0.30,0.50,0.70,1.0,1.5)
N_SEG=20000; EP=64
os.makedirs('artifacts/kt1/baseline_v2', exist_ok=True)

print(f"{'sigma':>6} | {'delta':>6} | {'hash':>8} | {'qdict':>8} | {'qdict Wilson95':>18}")
for sg in SIG:
    bd,_=select_delta_fixed(sg, seed=7, n_seg=N_SEG)
    h=q=0
    for ep in range(EP):
        d=make_episode(N_SEG, rng=np.random.default_rng([7,ep]))
        sc=key_scale(d["keys"])
        qk=perturb_keys(d["query_key"], sg, sc, np.random.default_rng([7,ep,int(sg*1e6)]))
        m=ExactHashMap(); m.build(d["keys"],d["values"])
        h+=int(np.all(m.query(qk)==d["answer"]))
        m2=QuantizedDict(delta=bd,scale=sc); m2.build(d["keys"],d["values"])
        q+=int(np.all(m2.query(qk)==d["answer"]))
    lo,hi=wilson(q,EP)
    print(f"{sg:>6} | {bd:>6} | {h:>4}/{EP:<3} | {q:>4}/{EP:<3} | [{lo:.4f}, {hi:.4f}]")
    for name,cnt in (("exact_hash_map",h),("quantized_dict",q)):
        json.dump({"schema":"kt1_baseline_v2","mechanism":name,"sigma":sg,
                   "episodes":EP,"retrieval_correct":cnt,"n_segments":N_SEG,
                   "delta":bd if name=="quantized_dict" else None,
                   "delta_selection":"same_length_validation_32_queries"},
                  open(f'artifacts/kt1/baseline_v2/kt1_{name}_sigma{sg}.json','w'), indent=2)
