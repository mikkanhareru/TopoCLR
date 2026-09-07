import glob, json, random
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple
from ase.io import read
from Data_Preprocessing.ase_process_topo import MolecularGraphBuilder

def parse_conect(pdb_path: str) -> Tuple[int, int]: # Tuple[i, j]
    bonds = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("CONECT"):
                parts = line.split()
                src = int(parts[1]) - 1 # 1-based -> 0-based
                for p in parts[2:]:
                    dst = int(p) - 1
                    if src < dst:
                        bonds.add((src, dst))
    return bonds

def predict_bonds(atoms, s):
    builder = MolecularGraphBuilder(atoms=atoms, multiplier=s)
    builder.build_graph(apply_correction=False, 
                        use_aromatic_check=False,
                        detect_rings=False,
                        verbose=False)
    return set(tuple(sorted(b)) for b in builder.bonds)

pdb_files = glob.glob('data/mettalic/*.pdb')
random.shuffle(pdb_files)
pdb_files = pdb_files[:1000]

s_values = [round(0.80 + i * 0.05, 2) for i in range(11)]
stats = {s: {'tp':0,
             'fp':0,
             'fn':0} for s in s_values}

for path in pdb_files:
    try:
        atoms = read(path)
    except Exception:
        continue
    gt = parse_conect(path)
    if not gt:
        continue
    for s in s_values:
        pred = predict_bonds(atoms, s)
        stats[s]['tp'] += len(pred & gt)
        stats[s]['fp'] += len(pred - gt)
        stats[s]['fn'] += len(gt - pred)

best_s, best_f1 = 1.0, 0.0
f1_scores = []
for s in s_values:
    tp, fp, fn = stats[s]['tp'], stats[s]['fp'], stats[s]['fn']
    P = tp/(tp+fp+1e-9)
    R = tp/(tp+fn+1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    print(f"s={s:.2f} P={P:.4f} R={R:.4f} F1={F1:.4f}")
    f1_scores.append(F1)
    if F1 > best_f1:
        best_f1, best_s = F1, s
        

print(f"\nBest s = {best_s}  (F1 = {best_f1:.4f})")
os.makedirs('results/graph_calibrate', exist_ok = True)
json.dump({'bond_multiplier': best_s}, open('results/graph_calibrate/bond_config.json', 'w'), indent=2)



plt.plot(s_values, f1_scores)  # collect F1 list while looping
plt.xlabel('multiplier s')
plt.ylabel('Bond F1')
plt.savefig('results/graph_calibrate/bond_calibration.png')
    