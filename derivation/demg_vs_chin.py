#!/usr/bin/env python3
"""dEMG 有效性驗證(2026-10-04 審查補充):H3 dEMG(Fp1−Fp2,10–35 Hz)與 PSG 下頦肌電
於 REM vs NREM 之分辨力比較(肌張力消失 → REM 期 EMG 功率較低)。

每個 PSG epoch(比較視窗內、人工有判讀者):兩訊號皆取 10–35 Hz 之 Welch 功率(4 s、50% 重疊,取 log),
下頦肌電以 10–35 Hz 帶通與 dEMG 同頻帶;各受試者計算 NREM(N1+N2+N3)vs REM 之 AUC(NREM 為正類,各類 ≥ 10 epoch)。
排除:平線(SD < 0.01 µV)或峰對峰 > 2000 µV 之 epoch。
輸出 derivation/demg_vs_chin.json;彙總:兩法 AUC 平均 ± SD 與 Wilcoxon 配對檢定。
"""
import os, re, json, glob, numpy as np, pyedflib
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import wilcoxon

ROOT = os.path.expanduser('~/sandbox/收案'); OUT = os.path.dirname(os.path.abspath(__file__))
OVR = {12, 3, 24}; EXC = {18}; EP = 30

def case_dir(no): return [os.path.join(ROOT, x) for x in os.listdir(ROOT) if re.match(rf'{no}\D', x)][0]
def mapping(al):
    segs = al.get('segments')
    if segs:
        (i0, s0), (i1, s1) = [(g['offset_intercept_s'], g['drift_slope']) for g in segs[:2]]; brk = segs[0]['break_h3_t'] * (1 + s0) + i0
        return lambda t: (t - i0) / (1 + s0) if t < brk else (t - i1) / (1 + s1)
    return lambda t: (t - al['offset_intercept_s']) / (1 + al['drift_slope'])
def rml_offset(d):
    for gj in glob.glob(d + '/*.grid.json'): return json.load(open(gj)).get('rml_offset_s', 0)
    return 0

def epoch_logpow(x, fs):
    if x.std() < 0.01 or np.ptp(x) > 2000: return None
    f, p = welch(x, fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
    v = p[(f >= 10) & (f < 35)].sum() * (f[1] - f[0])
    return float(np.log10(v)) if v > 0 else None

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 10 or len(neg) < 10: return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def main():
    cases = sorted(int(re.match(r'\d+', x).group()) for x in os.listdir(ROOT) if re.match(r'\d+\D', x) and os.path.exists(os.path.join(ROOT, x, 'h3_vs_psg_metrics.json')))
    cases = [c for c in cases if c not in EXC]
    res = {'cases': {}}
    for no in cases:
        d = case_dir(no); m = json.load(open(os.path.join(d, 'nogrid_h3_vs_psg_metrics.json' if no in OVR else 'h3_vs_psg_metrics.json')))
        lo, psg = m['series']['lo_ep'], m['series']['psg']; off = rml_offset(d)
        al = json.load(open(os.path.join(d, 'alignment_case.json'))); f_h3 = mapping(al)
        pf = glob.glob(os.path.join(d, '*', al['psg_file'].replace('[', '[[]')))[0]
        sig = {}
        with pyedflib.EdfReader(pf) as f:
            L = f.getSignalLabels(); i = L.index('EMG Chin'); fs = f.getSampleFrequency(i)
            sos = butter(3, [10, 35], btype='band', fs=fs, output='sos')
            sig['chin'] = (sosfiltfilt(sos, f.readSignal(i)), fs, lambda t: t - off)
        h3f = [x for x in glob.glob(d + '/*_derived.EDF') if '_grid_' not in x][0]
        with pyedflib.EdfReader(h3f) as f:
            L = f.getSignalLabels(); i = L.index('dEMG'); fs = f.getSampleFrequency(i)
            sig['demg'] = (f.readSignal(i), fs, lambda t: f_h3(t - off))   # 已於產生時 10–35 Hz 帶通
        feat = {k: {'NREM': [], 'R': []} for k in sig}; nrej = {k: 0 for k in sig}
        for k, st in enumerate(psg):
            if st not in ('N1', 'N2', 'N3', 'R'): continue
            grp = 'R' if st == 'R' else 'NREM'; t_rml = (lo + k) * EP
            for key, (x, fs, tmap) in sig.items():
                t = tmap(t_rml); a = int(round(t * fs)); b = a + int(EP * fs)
                if a < 0 or b > len(x): continue
                v = epoch_logpow(x[a:b], fs)
                if v is None: nrej[key] += 1; continue
                feat[key][grp].append(v)
        out = {k: {'auc': auc(feat[k]['NREM'], feat[k]['R']), 'n_nrem': len(feat[k]['NREM']), 'n_rem': len(feat[k]['R']), 'n_rej': nrej[k]} for k in sig}
        res['cases'][str(no)] = out
        print(f'#{no}', {k: (round(out[k]['auc'], 2) if out[k]['auc'] is not None else None) for k in sig}, flush=True)
        json.dump(res, open(os.path.join(OUT, 'demg_vs_chin.json'), 'w'))
    pair = [(v['demg']['auc'], v['chin']['auc']) for v in res['cases'].values() if v['demg']['auc'] is not None and v['chin']['auc'] is not None]
    a = np.array(pair)
    res['summary'] = {'n': len(pair), 'demg_mean': float(a[:, 0].mean()), 'demg_sd': float(a[:, 0].std(ddof=1)),
                      'chin_mean': float(a[:, 1].mean()), 'chin_sd': float(a[:, 1].std(ddof=1)),
                      'wilcoxon_p': float(wilcoxon(a[:, 0], a[:, 1]).pvalue),
                      'diff_mean': float((a[:, 0] - a[:, 1]).mean())}
    json.dump(res, open(os.path.join(OUT, 'demg_vs_chin.json'), 'w'))
    print('SUMMARY', res['summary'])

if __name__ == '__main__':
    main()
