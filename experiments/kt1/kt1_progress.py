"""Прогресс KT-1: читает готовые JSON, считает интервалы и проверяет пороги."""
import glob, json, math, os, re, sys

THRESH = {0.0:1.0000, 0.3:0.2462, 0.5:0.1702, 0.7:0.1290, 1.0:0.1290, 1.5:0.1290}

def wilson(x,n,z=1.959963984540054):
    if n<=0: return (0.,1.)
    p=x/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,c-h),min(1.,c+h)

agg={}
for f in sorted(glob.glob('artifacts/kt1/phef/kt1_sigma*.json')):
    d=json.load(open(f))
    sg=float(d.get('query_noise_sigma', d['config'].get('query_noise_sigma',0.0)))
    r=d['real']; n=r['episodes']; c=round(r['retrieval_acc']*n)
    s=agg.setdefault(sg,{'c':0,'n':0,'cells':0,'qcos':[]})
    s['c']+=c; s['n']+=n; s['cells']+=1
    if r.get('query_state_cosine') is not None: s['qcos'].append(r['query_state_cosine'])

print(f"{'sigma':>6} {'ячеек':>6} {'счёт':>10} {'оценка':>8} {'Wilson95':>18} {'q_cos':>8} {'порог':>8} {'дом.':>5}")
for sg in sorted(agg):
    s=agg[sg]; lo,hi=wilson(s['c'],s['n'])
    thr=THRESH.get(round(sg,2))
    qc=sum(s['qcos'])/len(s['qcos']) if s['qcos'] else 1.0
    dom='—' if thr is None else ('ДА' if lo>thr else 'нет')
    print(f"{sg:>6} {s['cells']:>4}/4 {s['c']:>4}/{s['n']:<5} {s['c']/s['n']:>8.4f} "
          f"[{lo:.4f},{hi:.4f}] {qc:>8.4f} {thr if thr else 0:>8.4f} {dom:>5}")
print(f"\nвсего ячеек: {sum(a['cells'] for a in agg.values())}/24  (6 sigma x 4)")
