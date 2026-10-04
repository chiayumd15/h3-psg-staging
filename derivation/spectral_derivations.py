#!/usr/bin/env python3
"""額極 vs 標準導程之睡眠分期腦波特徵比較(以人工 PSG 判讀為各 epoch 之期別)。

導程:Fp1(H3 EEG1,Fp1−右耳前點)、Fp2(H3 EEG2,Fp2−左耳前點)、F4-A1、C4-A1、O2-A1(PSG)。
每個 PSG epoch(比較視窗內、人工有判讀者):0.3–35 Hz 帶通 → Welch(4 s、50% 重疊)→
  絕對與相對頻帶功率(相對 = 佔 0.5–30 Hz 總功率):δ 0.5–4、θ 4–8、α 8–11、σ 11–16、β 16–30 Hz。
H3 時間對映:RML 時間 − rml_offset_s = PSG 檔時間 → alignment_case.json 之 psg_to_h3(含分段)。
假影排除(各導程分別):峰對峰 > 800 µV 或 SD < 0.5 µV 之 epoch 不用。
判期特徵可分辨度(各受試者、各導程 AUC,Mann–Whitney;09-29 看過前 9 案相對功率結果後改為下列預設特徵,因相對功率受其他頻段牽動):
  W vs N1:log(α/θ)(W 較高;AASM 定義 N1 為 α 被 4–7 Hz 取代);N1 vs N2:log 絕對 σ(N2 較高);N2 vs N3:log 絕對 δ(N3 較高)。
  各對比每期需 ≥ 10 epoch。輸出 derivation/spectral_results.json(逐受試者 AUC 與各期平均頻譜)。
"""
import os, re, json, glob, numpy as np, pyedflib
from scipy.signal import butter, sosfiltfilt, welch
ROOT = os.path.expanduser('~/sandbox/收案'); OUT = os.path.dirname(os.path.abspath(__file__))
OVR = {12, 3, 24}; EXC = {18}; EP = 30
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 11), 'sigma': (11, 16), 'beta': (16, 30)}
DERS = ['Fp1', 'Fp2', 'F4', 'C4', 'O2']; PSG_CH = {'F4': 'EEG F4-A1', 'C4': 'EEG C4-A1', 'O2': 'EEG O2-A1'}
FGRID = np.arange(0.5, 30.01, 0.25)

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

def epoch_features(x, fs):
    if np.ptp(x) > 800 or x.std() < 0.5: return None
    f, p = welch(x, fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
    tot = p[(f >= 0.5) & (f < 30)].sum()
    ab = {b: p[(f >= lo) & (f < hi)].sum() * (f[1] - f[0]) for b, (lo, hi) in BANDS.items()}
    rel = {b: p[(f >= lo) & (f < hi)].sum() / tot for b, (lo, hi) in BANDS.items()}
    return ab, rel, np.interp(FGRID, f, p)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 10 or len(neg) < 10: return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def main():
    cases = sorted(int(re.match(r'\d+', x).group()) for x in os.listdir(ROOT) if re.match(r'\d+\D', x) and os.path.exists(os.path.join(ROOT, x, 'h3_vs_psg_metrics.json')))
    cases = [c for c in cases if c not in EXC]
    res = {'bands': BANDS, 'fgrid': FGRID.tolist(), 'cases': {}}
    for no in cases:
        d = case_dir(no); m = json.load(open(os.path.join(d, 'nogrid_h3_vs_psg_metrics.json' if no in OVR else 'h3_vs_psg_metrics.json')))
        lo, psg = m['series']['lo_ep'], m['series']['psg']; off = rml_offset(d)
        al = json.load(open(os.path.join(d, 'alignment_case.json'))); f_h3 = mapping(al)
        pf = glob.glob(os.path.join(d, '*', al['psg_file'].replace('[', '[[]')))[0]
        sig = {}
        with pyedflib.EdfReader(pf) as f:
            L = f.getSignalLabels()
            for der, ch in PSG_CH.items():
                i = L.index(ch); fs = f.getSampleFrequency(i); sos = butter(3, [0.3, 35], btype='band', fs=fs, output='sos')
                sig[der] = (sosfiltfilt(sos, f.readSignal(i)), fs, lambda t: t - off)
        h3f = [x for x in glob.glob(d + '/*_derived.EDF') if '_grid_' not in x][0]
        with pyedflib.EdfReader(h3f) as f:
            L = f.getSignalLabels()
            for der, ch in [('Fp1', 'EEG1'), ('Fp2', 'EEG2')]:
                i = L.index(ch); fs = f.getSampleFrequency(i); sos = butter(3, [0.3, 35], btype='band', fs=fs, output='sos')
                sig[der] = (sosfiltfilt(sos, f.readSignal(i)), fs, lambda t: f_h3(t - off))
        feat = {der: {s: [] for s in ('W', 'N1', 'N2', 'N3', 'R')} for der in DERS}; nrej = {der: 0 for der in DERS}
        for k, st in enumerate(psg):
            if not st: continue
            t_rml = (lo + k) * EP
            for der, (x, fs, tmap) in sig.items():
                t = tmap(t_rml); a = int(round(t * fs)); b = a + int(EP * fs)
                if a < 0 or b > len(x): continue
                r = epoch_features(x[a:b], fs)
                if r is None: nrej[der] += 1; continue
                feat[der][st].append(r)
        out = {'n_rejected': nrej, 'auc': {}, 'spec': {}, 'n': {der: {s: len(v) for s, v in feat[der].items()} for der in DERS}}
        for der in DERS:
            F = feat[der]; rel = lambda s, b: [e[1][b] for e in F[s]]; lab = lambda s, b: [np.log10(e[0][b]) for e in F[s]]
            at = lambda s: [np.log10(e[0]['alpha'] / e[0]['theta']) for e in F[s]]
            out['auc'][der] = {'W_N1_alpha_theta': auc(at('W'), at('N1')),
                               'N2_N1_sigma': auc(lab('N2', 'sigma'), lab('N1', 'sigma')),
                               'N3_N2_delta': auc(lab('N3', 'delta'), lab('N2', 'delta'))}
            out.setdefault('abs', {})[der] = {s: [[round(float(np.log10(e[0][b])), 4) for b in BANDS] for e in v] for s, v in F.items()}
            out['spec'][der] = {s: np.log10(np.mean([e[2] for e in v], 0)).tolist() for s, v in F.items() if len(v) >= 10}
            out.setdefault('relpow', {})[der] = {s: {b: float(np.mean([e[1][b] for e in v])) for b in BANDS} for s, v in F.items() if len(v) >= 10}
        res['cases'][str(no)] = out
        print(f'#{no}', {der: {k_[:4]: (round(v, 2) if v is not None else None) for k_, v in out['auc'][der].items()} for der in ('Fp1', 'C4', 'O2')}, flush=True)
        json.dump(res, open(os.path.join(OUT, 'spectral_results.json'), 'w'))

if __name__ == '__main__':
    main()
