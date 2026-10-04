#!/usr/bin/env python3
"""H3 與 Alice 6 LDx 同步錄音的時間校準。

兩台機器各自的 EDF 標頭時間並不可靠(實測誤差可達數分鐘),
因此改用兩邊都有的心電圖做互相關,求出真實時間差與時鐘漂移。

用法:
    python3 align.py <資料根目錄>/<收案資料夾>
    python3 align.py <資料根目錄>          # 批次處理所有子資料夾
"""
import sys, json, glob, os
import numpy as np
from scipy import signal as sg

EPOCH = 30.0


class EDF:
    """只讀需要的通道,避免把整個 500 MB 檔案載入記憶體。"""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        h = self.f.read(256)
        self.startdate = h[168:176].decode('latin1')
        self.starttime = h[176:184].decode('latin1')
        self.nrec = int(h[236:244])
        self.dur = float(h[244:252])
        self.ns = int(h[252:256])
        blk = lambda n: [self.f.read(n).decode('latin1').strip() for _ in range(self.ns)]
        self.lab = blk(16)
        blk(80); blk(8)
        self.pmin = [float(x) for x in blk(8)]
        self.pmax = [float(x) for x in blk(8)]
        self.dmin = [float(x) for x in blk(8)]
        self.dmax = [float(x) for x in blk(8)]
        blk(80)
        self.nsamp = [int(x) for x in blk(8)]
        self.hdr = 256 + self.ns * 256
        self.recsz = sum(self.nsamp) * 2
        self.off = np.cumsum([0] + self.nsamp)
        self.length = self.nrec * self.dur

    def start_seconds(self):
        hh, mm, ss = (int(x) for x in self.starttime.split('.'))
        return hh * 3600 + mm * 60 + ss

    def fs(self, name):
        return self.nsamp[self.lab.index(name)] / self.dur

    def get(self, name, t0, t1):
        i = self.lab.index(name)
        r0 = max(0, int(t0 / self.dur))
        r1 = min(self.nrec, int(np.ceil(t1 / self.dur)))
        n = self.nsamp[i]
        out = np.empty((r1 - r0) * n, dtype=np.int16)
        for k, r in enumerate(range(r0, r1)):
            self.f.seek(self.hdr + r * self.recsz + self.off[i] * 2)
            out[k * n:(k + 1) * n] = np.frombuffer(self.f.read(n * 2), dtype='<i2')
        gain = (self.pmax[i] - self.pmin[i]) / (self.dmax[i] - self.dmin[i])
        return (out - self.dmin[i]) * gain + self.pmin[i], n / self.dur


def rpeaks(x, fs):
    """自適應閾值的 R 波偵測,對兩台機器不同的訊號品質都夠穩健。"""
    b, a = sg.butter(3, [8 / (fs / 2), 22 / (fs / 2)], 'band')
    env = sg.filtfilt(*sg.butter(2, 6 / (fs / 2), 'low'), np.abs(sg.filtfilt(b, a, x)))
    w = int(10 * fs)
    th = np.repeat([np.percentile(env[i:i + w], 88) for i in range(0, len(env), w)], w)[:len(env)]
    pk, _ = sg.find_peaks(env, height=th, distance=int(0.33 * fs))
    return pk / fs


def ecg_quality(t):
    """以 RR 間期的規律性評分,分數越高越像乾淨的心電圖。"""
    rr = np.diff(t)
    rr = rr[(rr > 0.4) & (rr < 1.8)]
    if len(rr) < 30:
        return 0.0, 0.0
    hr = 60 / np.median(rr)
    if not 35 < hr < 110:
        return 0.0, hr
    cv = np.std(rr) / np.mean(rr)
    return float(len(rr)) / (1 + 40 * cv ** 2), hr


def best_ecg(edf, candidates, probe=(3600, 10800)):
    """同一台機器可能有多條心電圖導程,品質差很多,挑最好的那條。"""
    best, score = None, 0.0
    for ch in candidates:
        if ch not in edf.lab:
            continue
        s = []
        for t0 in probe:
            if t0 + 300 > edf.length:
                continue
            try:
                x, fs = edf.get(ch, t0, t0 + 300)
            except Exception:
                continue
            s.append(ecg_quality(rpeaks(x, fs))[0])
        if s and np.median(s) > score:
            best, score = ch, float(np.median(s))
    return best, score


def _train(t, n, fs=20):
    v = np.zeros(n)
    i = (t * fs).astype(int)
    v[i[(i >= 0) & (i < n)]] = 1
    return sg.filtfilt(*sg.butter(2, 2.5 / (fs / 2), 'low'), v)


def lag_at(h3, psg, t0, nominal, h3_ch, psg_ch, dur=900, search=500, fs=20):
    """回傳 (lag, r):lag 是相對於 nominal 的修正量,單位秒。"""
    a0, a1 = t0 + nominal - search, t0 + nominal + dur + search
    if a0 < 0 or a1 > psg.length or t0 + dur > h3.length:   # 視窗落在任一檔範圍外(兩機開錄時間差很大時)
        return None, 0.0
    xh, fh = h3.get(h3_ch, t0, t0 + dur)
    xa, fa = psg.get(psg_ch, a0, a1)
    th, ta = rpeaks(xh, fh), rpeaks(xa, fa)
    if len(th) < 300 or len(ta) < 300:
        return None, 0.0
    vh, va = _train(th, int(dur * fs), fs), _train(ta, int((dur + 2 * search) * fs), fs)
    vh, va = vh - vh.mean(), va - va.mean()
    c = sg.correlate(va, vh, mode='valid')
    r = c.max() / np.sqrt((vh ** 2).sum() * (va ** 2).sum())
    return np.argmax(c) / fs - search, r


def classify(path):
    """判斷一個 EDF 是 H3 還是 Alice。"""
    e = EDF(path)
    if 'ECG II' in e.lab and len(e.lab) > 20:
        return 'psg', e
    if 'EEG3' in e.lab and 'X' in e.lab:
        return 'h3', e
    return None, e


def analyse(folder, verbose=True):
    h3 = psg = None
    for p in glob.glob(os.path.join(folder, '*.[eE][dD][fF]')):
        if os.path.basename(p).startswith('._'):
            continue
        kind, e = classify(p)
        if kind == 'h3':
            h3 = e
        elif kind == 'psg':
            psg = e
    name = os.path.basename(folder.rstrip('/'))
    if not (h3 and psg):
        if verbose:
            have = 'H3' if h3 else ('PSG' if psg else '無')
            print(f"[{name}] 缺配對(只有 {have}),跳過")
        return None

    nominal = h3.start_seconds() - psg.start_seconds()
    if nominal < -43200:
        nominal += 86400

    h3_ch, hq = best_ecg(h3, ['EEG3', 'EEG4', 'EEG1', 'EEG2'])
    psg_ch, pq = best_ecg(psg, ['ECG II', 'ECG I', 'ECG IIHF'])
    if not (h3_ch and psg_ch):
        if verbose:
            print(f"[{name}] 找不到可用的心電圖通道(H3={h3_ch} PSG={psg_ch})")
        return None

    probes = [t for t in range(900, int(h3.length) - 1500, 3300)][:8]
    res = [(t, *lag_at(h3, psg, t, nominal, h3_ch, psg_ch)) for t in probes]
    ok = [(t, l, r) for t, l, r in res if l is not None and r > 0.40]
    if len(ok) < 3:
        if verbose:
            print(f"[{name}] 校準失敗:只有 {len(ok)} 個可信視窗")
        return None

    T = np.array([o[0] for o in ok], float)
    L = np.array([o[1] for o in ok], float)
    slope, _ = np.polyfit(T, L, 1)
    true_off = nominal + float(np.median(L))

    # Alice 的 epoch 網格為準,找出兩邊都有訊號的範圍
    first = max(1, int(np.ceil(true_off / EPOCH)) + 1)
    last = int(min(psg.length, h3.length + true_off) // EPOCH)
    out = {
        'case': name,
        'h3_file': os.path.basename(h3.path), 'psg_file': os.path.basename(psg.path),
        'h3_start': h3.starttime, 'psg_start': psg.starttime,
        'ecg_channels': {'h3': h3_ch, 'psg': psg_ch},
        'header_offset_s': round(nominal, 2),
        'true_offset_s': round(true_off, 2),
        'header_error_s': round(true_off - nominal, 2),
        'drift_s_per_hour': round(slope * 3600, 3),
        'lag_sd_s': round(float(np.std(L)), 2),
        'windows_used': f"{len(ok)}/{len(res)}",
        'mean_r': round(float(np.mean([o[2] for o in ok])), 3),
        'overlap_psg_epochs': [first, last],
        'overlap_hours': round((last - first + 1) * EPOCH / 3600, 2),
        # PSG 第 k 個 epoch 對應的 H3 樣本索引(fs=250)
        'h3_sample_of_psg_epoch': f"round((30*(k-1) - {true_off:.2f}) * 250)",
    }
    if verbose:
        print(f"[{name}] 標頭差 {nominal:>7.1f}s → 實測 {true_off:>7.1f}s "
              f"(誤差 {true_off - nominal:+7.1f}s) | 漂移 {slope*3600:+.2f}s/hr | "
              f"r={out['mean_r']:.2f} | 可比對 PSG epoch {first}–{last} ({out['overlap_hours']}h)")
    return out


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    folders = [root] if glob.glob(os.path.join(root, '*.[eE][dD][fF]')) else \
        sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    results = [r for r in (analyse(f) for f in folders) if r]
    if results:
        dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alignment.json')
        json.dump(results, open(dest, 'w'), ensure_ascii=False, indent=2)
        print(f"\n{len(results)} 個案校準完成 → {dest}")
