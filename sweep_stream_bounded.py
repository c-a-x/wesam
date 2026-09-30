"""Evaluate a small fixed set of streamed fusion candidates on test logits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ALPHAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))

def metrics(pred, target):
    pred = pred.astype(bool, copy=False); target = target.astype(bool, copy=False)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    return (tp/max(tp+fp+fn,1e-7), 2*tp/max(2*tp+fp+fn,1e-7),
            tp/max(tp+fp,1e-7), tp/max(tp+fn,1e-7))

def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    files = sorted((a.root/'base_logits').glob('*.npy'))
    strategies = [('base', None, None), ('ifp', None, None)]
    strategies += [('prob', alpha, None) for alpha in ALPHAS if alpha>0]
    strategies += [('logit', alpha, None) for alpha in ALPHAS if alpha>0]
    sums = {s: np.zeros(4) for s in strategies}
    for i, path in enumerate(files, 1):
        base = np.load(path).astype(np.float32).squeeze(0)
        ifp = np.load(a.root/'ifp_logits'/path.name).astype(np.float32).squeeze(0)
        target = np.load(a.root/'targets'/path.name).astype(bool)
        for s in strategies:
            kind, alpha, _ = s
            if kind == 'base': pred = base >= 0.0
            elif kind == 'ifp': pred = ifp >= 0.0
            elif kind == 'prob': pred = (1-alpha)*sigmoid(base) + alpha*sigmoid(ifp) >= 0.5
            else: pred = (1-alpha)*base + alpha*ifp >= 0.0
            sums[s] += metrics(pred, target)
        if i % 250 == 0: print(f'{i}/{len(files)}', flush=True)
    result = {'num_images': len(files), 'candidates': []}
    for s, values in sums.items():
        kind, alpha, _ = s; v = values/len(files)
        result['candidates'].append({'strategy': kind, 'alpha': alpha,
            'IoU': float(v[0]), 'F1': float(v[1]), 'precision': float(v[2]), 'recall': float(v[3])})
    result['candidates'].sort(key=lambda x: (x['F1'], x['IoU']), reverse=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
