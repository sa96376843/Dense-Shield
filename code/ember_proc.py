import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Preprocess EMBER 2018 into monthly NumPy archives.')
parser.add_argument('--ember-dir', type=Path, default=REPO_ROOT / 'dataset' / 'ember2018')
parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'dataset' / 'ember_proc')
args = parser.parse_args()
EMBER = args.ember_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(42)

def num(x):
    try: return float(x)
    except: return float(abs(hash(str(x)))%1000)

def feat_vec(o):
    v=[]
    v += o['histogram']                      # 256
    v += o['byteentropy']                    # 256
    g=o['general']
    v += [g['size'],g['vsize'],g['has_debug'],g['exports'],g['imports'],g['has_relocations'],
          g['has_resources'],g['has_signature'],g['has_tls'],g['symbols']]  # 10
    s=o['strings']
    v += [s['numstrings'],s['avlength'],s['printables'],s['entropy'],s['paths'],s['urls'],
          s['registry'],s['MZ']]             # 8
    v += s['printabledist']                  # 96
    h=o['header']['coff']
    v += [h['timestamp'], num(h['machine']), len(h['characteristics'])]
    v += h['characteristics'][:0]
    oh=o['header']['optional']
    v += [num(oh['subsystem']), len(oh['dll_characteristics']),
          oh['major_image_version'], oh['minor_image_version'],
          oh['major_linker_version'], oh['minor_linker_version'],
          oh['sizeof_code'], oh['sizeof_headers'], oh['sizeof_heap_commit']]
    sec=o['section']
    v += [len(sec['sections']), 1 if len(sec['entry'])>0 else 0]
    return np.asarray(v, dtype=np.float64)

train_X, train_y = [], []
month_X = {m: [] for m in range(5,11)}
month_y = {m: [] for m in range(5,11)}

for fn in sorted(EMBER.iterdir()):
    if not fn.name.startswith('train_features'): continue
    path = fn
    print('processing', fn.name, flush=True)
    with open(path) as f:
        for line in f:
            o=json.loads(line)
            lab=o.get('label',-1)
            ap=o.get('appeared','')
            try: mo=int(ap.split('-')[1])
            except: mo=0
            if lab not in (0,1): continue
            if 1<=mo<=4:
                if rng.rand()>0.10: continue
                v=feat_vec(o); train_X.append(v); train_y.append(lab)
            elif 5<=mo<=10:
                if rng.rand()>0.12: continue
                v=feat_vec(o); month_X[mo].append(v); month_y[mo].append(lab)
    # cap months at 10000
print('saving...')
np.savez_compressed(OUT / 'train_m1_4.npz',
                    X=np.asarray(train_X,dtype=np.float32), y=np.asarray(train_y,dtype=np.int64))
for m in range(5,11):
    X=np.asarray(month_X[m],dtype=np.float32); y=np.asarray(month_y[m],dtype=np.int64)
    if len(X)>10000:
        idx=rng.choice(len(X),10000,replace=False); X,y=X[idx],y[idx]
    np.savez_compressed(OUT / f'test_m{m}.npz', X=X, y=y)
    print('month',m,X.shape, 'malware ratio', y.mean(), flush=True)
print('train', len(train_X))
print('DONE')
