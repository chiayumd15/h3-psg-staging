#!/usr/bin/env python3
"""眼動(EOG)互相關時間校正:與 ECG 法互相驗證。

PSG 的 'EOG ROC-A2' 與 H3 的 EEG4(ROC − A1)記錄同一顆右眼的動作。兩邊都做 0.3–5 Hz 帶通、
降到 20 Hz、z 分數,再在每個 15 分鐘視窗內於 ±500 s 範圍互相關取峰值 → 該視窗的時間差。
保留 r 夠高且與中位數接近的視窗,線性回歸得偏移與漂移,並與 alignment_case.json(ECG 法)比較。

用法: python3 align_eog.py <收案資料夾> [...]
輸出: <case>/alignment_eog.json;彙總印在終端。
"""
import sys, os, json
import numpy as np
from scipy import signal as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import EDF
from align_case import find_pair

FS = 20.0


def eog_series(edf, ch, t0, t1):
    x, fs = edf.get(ch, t0, t1)
    b, a = sg.butter(3, [0.3 / (fs / 2), 5.0 / (fs / 2)], 'band')
    y = sg.filtfilt(b, a, x - np.mean(x))
    n = int(round(len(y) * FS / fs))
    y = sg.resample(y, n)
    y = np.clip(y, *np.percentile(y, [0.5, 99.5]))          # 壓掉極端雜訊
    return (y - y.mean()) / (y.std() + 1e-9)


def lag_at(h3, psg, t0, nominal, h3_ch, psg_ch, dur=900, search=500):
    yh = eog_series(h3, h3_ch, t0, t0 + dur)
    ya = eog_series(psg, psg_ch, t0 + nominal - search, t0 + nominal + dur + search)
    c = sg.correlate(ya, yh, mode='valid')
    r = c.max() / np.sqrt((yh ** 2).sum() * (ya ** 2).sum()) * np.sqrt(len(ya) / len(yh)) / np.sqrt(len(ya) / len(yh))
    return np.argmax(c) / FS - search, float(c.max() / (len(yh)))


def run(folder):
    name = os.path.basename(folder.rstrip('/'))
    h3, psg = find_pair(folder)
    nominal = h3.start_seconds() - psg.start_seconds()
    if nominal < -43200: nominal += 86400
    psg_ch = next(c for c in ('EOG ROC-A2', 'EOG ROC-A1', 'EOG ROC') if c in psg.lab)
    probes = list(range(600, int(h3.length) - 1500, 1500))
    res = []
    for t in probes:
        try:
            lag, r = lag_at(h3, psg, t, nominal, 'EEG4', psg_ch)
            res.append((t, lag, r))
        except Exception:
            pass
    R = np.array([x[2] for x in res]); L = np.array([x[1] for x in res]); T = np.array([x[0] for x in res], float)
    # 穩健:先取 r 前 60% 的視窗算中位數,再留 |lag − 中位數| < 5 s 者
    thr = np.percentile(R, 40)
    cand = (R >= thr)
    med = float(np.median(L[cand]))
    ok = cand & (np.abs(L - med) < 5)
    if ok.sum() < 3:
        ok = np.abs(L - med) < 5
    slope, icpt = np.polyfit(T[ok], L[ok], 1)
    out = {'case': name, 'psg_eog_channel': psg_ch, 'h3_channel': 'EEG4 (ROC − A1)',
           'header_offset_s': round(nominal, 2), 'true_offset_s': round(nominal + med, 2), 'header_error_s': round(med, 2),
           'offset_intercept_s': round(nominal + icpt, 3), 'drift_slope': slope, 'drift_s_per_hour': round(slope * 3600, 3),
           'fit_resid_sd_s': round(float(np.std(L[ok] - (icpt + slope * T[ok]))), 2),
           'windows_used': f'{int(ok.sum())}/{len(res)}', 'mean_r': round(float(R[ok].mean()), 3),
           'windows': [{'h3_t': int(t), 'lag_correction_s': round(float(l), 2), 'r': round(float(r), 3), 'used': bool(u)} for t, l, r, u in zip(T, L, R, ok)]}
    ecg_path = os.path.join(folder, 'alignment_case.json')
    if os.path.exists(ecg_path):
        ecg = json.load(open(ecg_path))
        out['vs_ecg'] = {'ecg_true_offset_s': ecg['true_offset_s'], 'diff_offset_s': round(out['true_offset_s'] - ecg['true_offset_s'], 2),
                         'ecg_drift_s_per_hour': ecg['drift_s_per_hour'], 'diff_drift_s_per_hour': round(out['drift_s_per_hour'] - ecg['drift_s_per_hour'], 3)}
    json.dump(out, open(os.path.join(folder, 'alignment_eog.json'), 'w'), ensure_ascii=False, indent=2)
    v = out.get('vs_ecg', {})
    print(f"[{name}] EOG 法:偏移 {out['true_offset_s']} s(標頭誤差 {out['header_error_s']:+} s)漂移 {out['drift_s_per_hour']:+} s/hr 殘差 {out['fit_resid_sd_s']} s 視窗 {out['windows_used']}"
          + (f" | 與 ECG 法差 {v['diff_offset_s']:+} s、漂移差 {v['diff_drift_s_per_hour']:+} s/hr" if v else ''))
    return out


if __name__ == '__main__':
    for f in sys.argv[1:]:
        run(f)
