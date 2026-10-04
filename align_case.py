#!/usr/bin/env python3
"""單一收案資料夾的 H3–PSG 時間校準(PSG EDF 可放在子資料夾)。

用 ECG 互相關求出「PSG 秒數 → H3 秒數」的線性關係:
    h3_t = psg_t - offset(psg_t),  offset(t) = off0 + drift * t
結果寫到 <case>/alignment_case.json,並合併進 h3-align/alignment.json。
用法: python3 align_case.py <收案資料夾>
"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import EDF, best_ecg, lag_at, classify, EPOCH


def find_pair(folder):
    h3 = psg = None
    for p in glob.glob(os.path.join(folder, '**', '*.[eE][dD][fF]'), recursive=True):
        b = os.path.basename(p)
        if b.startswith('._') or b.endswith('-T.edf') or '_derived' in b:
            continue
        kind, e = classify(p)
        if kind == 'h3':
            h3 = e
        elif kind == 'psg':
            psg = e
    return h3, psg


def align_case(folder, verbose=True):
    name = os.path.basename(folder.rstrip('/'))
    h3, psg = find_pair(folder)
    if not (h3 and psg):
        raise SystemExit(f"[{name}] 找不到 H3/PSG 配對 (H3={bool(h3)} PSG={bool(psg)})")
    nominal = h3.start_seconds() - psg.start_seconds()
    if nominal < -43200:
        nominal += 86400
    h3_ch, hq = best_ecg(h3, ['EEG3', 'EEG4', 'EEG1', 'EEG2'])
    psg_ch, pq = best_ecg(psg, ['ECG II', 'ECG I', 'ECG IIHF'])
    if not (h3_ch and psg_ch):
        raise SystemExit(f"[{name}] 找不到可用的 ECG 通道 (H3={h3_ch} PSG={psg_ch})")
    # 每 ~25 分鐘取一個 15 分鐘視窗,涵蓋整夜
    probes = list(range(600, int(h3.length) - 1500, 1500))
    res = [(t, *lag_at(h3, psg, t, nominal, h3_ch, psg_ch)) for t in probes]
    ok = [(t, l, r) for t, l, r in res if l is not None and r > 0.30]
    low_r_fallback = False
    if len(ok) < 3:
        # r 全都偏低(例如 PSG ECG 飽和/雜訊大,#12),改用「多數視窗互相一致」當可信依據:
        # 取全部視窗 lag 的中位數,彼此在 5 s 內的視窗 ≥ 6 個才接受,並在輸出標記 low_r_fallback
        allv = [(t, l, r) for t, l, r in res if l is not None]
        med0 = float(np.median([o[1] for o in allv]))
        agree = [o for o in allv if abs(o[1] - med0) <= 5]
        if len(agree) >= 6:
            ok, low_r_fallback = agree, True
            if verbose:
                print(f"[{name}] ⚠ ECG 相關係數偏低(max r={max(o[2] for o in allv):.2f}),改以 {len(agree)}/{len(allv)} 個視窗互相一致(±5 s)為依據")
    # 穩健篩選:與中位數差超過 5 秒的視窗視為互相關鎖錯,剔除後再擬合漂移
    med = float(np.median([o[1] for o in ok]))
    rejected = [(t, l, r) for t, l, r in res if l is not None and ((t, l, r) not in ok or abs(l - med) > 5)]
    ok = [(t, l, r) for t, l, r in ok if abs(l - med) <= 5]
    if len(ok) < 3:
        raise SystemExit(f"[{name}] 校準失敗:只有 {len(ok)} 個可信視窗")
    ok.sort()
    T = np.array([o[0] for o in ok], float)   # H3 時間 (s)
    L = np.array([o[1] for o in ok], float)   # 相對 nominal 的修正 (s)
    # 跳階偵測:相鄰可信視窗的修正量差 > 3 s 視為錄音不連續(例如 H3 掉資料),分兩段各自擬合
    segments, disc = [], None
    jumps = np.abs(np.diff(L))
    if len(L) >= 6 and jumps.max() > 3:
        k = int(np.argmax(jumps))
        if k + 1 >= 3 and len(L) - (k + 1) >= 3:
            disc = {'between_h3_t': [float(T[k]), float(T[k + 1])], 'jump_s': round(float(L[k + 1] - L[k]), 2),
                    'note': 'H3 相對 PSG 少了 jump_s 秒的資料(正值)或多了(負值);兩視窗之間受試者狀態見分期'}
            for Ts, Ls in ((T[:k + 1], L[:k + 1]), (T[k + 1:], L[k + 1:])):
                sl, ic = np.polyfit(Ts, Ls, 1)
                segments.append({'h3_t_from': float(Ts[0]), 'h3_t_to': float(Ts[-1]), 'offset_intercept_s': round(nominal + ic, 3), 'drift_slope': sl,
                                 'drift_s_per_hour': round(sl * 3600, 3), 'fit_resid_sd_s': round(float(np.std(Ls - (ic + sl * Ts))), 2), 'n_windows': len(Ts)})
    if segments:
        # 頂層欄位沿用第一段(向下相容);分段邊界取兩視窗中點
        slope, icpt = segments[0]['drift_slope'], segments[0]['offset_intercept_s'] - nominal
        resid = np.concatenate([L[:k + 1] - (icpt + slope * T[:k + 1]), L[k + 1:] - (segments[1]['offset_intercept_s'] - nominal + segments[1]['drift_slope'] * T[k + 1:])])
        for sgm in segments: sgm['break_h3_t'] = float((T[k] + T[k + 1]) / 2)
    else:
        slope, icpt = np.polyfit(T, L, 1)
        resid = L - (icpt + slope * T)
    # offset(psg_t): psg_t = h3_t + nominal + icpt + slope*h3_t  →  h3_t = (psg_t - nominal - icpt)/(1+slope)
    out = {
        'case': name,
        'h3_file': os.path.basename(h3.path), 'psg_file': os.path.basename(psg.path),
        'h3_start': h3.starttime, 'psg_start': psg.starttime,
        'h3_length_s': h3.length, 'psg_length_s': psg.length,
        'ecg_channels': {'h3': h3_ch, 'psg': psg_ch},
        'header_offset_s': round(nominal, 2),
        'true_offset_s': round(nominal + float(np.median(L)), 2),
        'header_error_s': round(float(np.median(L)), 2),
        'offset_intercept_s': round(nominal + icpt, 3),
        'drift_s_per_hour': round(slope * 3600, 3),
        'drift_slope': slope,
        'fit_resid_sd_s': round(float(np.std(resid)), 2),
        'lag_sd_s': round(float(np.std(L)), 2),
        'windows_used': f"{len(ok)}/{len(res)}",
        'mean_r': round(float(np.mean([o[2] for o in ok])), 3),
        'windows': [{'h3_t': t, 'lag_correction_s': round(l, 2), 'r': round(r, 3), 'used': (t, l, r) in ok} for t, l, r in res if l is not None],
        'windows_rejected': len(rejected),
        'psg_to_h3': 'h3_t = (psg_t - offset_intercept_s) / (1 + drift_slope)' + (';有 segments 時依 break_h3_t 分段套用' if segments else ''),
    }
    if segments:
        out['segments'] = segments; out['discontinuity'] = disc
    if low_r_fallback:
        out['low_r_fallback'] = True
    if verbose:
        print(f"[{name}] 標頭差 {nominal:.1f}s → 實測 {out['true_offset_s']:.1f}s (標頭誤差 {out['header_error_s']:+.1f}s) | "
              f"漂移 {out['drift_s_per_hour']:+.2f}s/hr | 殘差SD {out['fit_resid_sd_s']}s | r={out['mean_r']} | 視窗 {out['windows_used']}")
        if disc:
            print(f"  ⚠ 偵測到不連續:H3 t={disc['between_h3_t'][0]:.0f}–{disc['between_h3_t'][1]:.0f}s 之間跳 {disc['jump_s']:+.2f}s;"
                  f"分兩段擬合 漂移 {segments[0]['drift_s_per_hour']:+.2f} / {segments[1]['drift_s_per_hour']:+.2f} s/hr,殘差SD {segments[0]['fit_resid_sd_s']} / {segments[1]['fit_resid_sd_s']} s")
    json.dump(out, open(os.path.join(folder, 'alignment_case.json'), 'w'), ensure_ascii=False, indent=2)
    # 合併進總表
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alignment.json')
    allres = json.load(open(dest)) if os.path.exists(dest) else []
    allres = [r for r in allres if r['case'] != name] + [{k: v for k, v in out.items() if k not in ('windows', 'drift_slope')}]
    json.dump(allres, open(dest, 'w'), ensure_ascii=False, indent=2)
    return out


if __name__ == '__main__':
    align_case(sys.argv[1])
