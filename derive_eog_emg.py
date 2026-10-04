#!/usr/bin/env python3
"""從 H3 腦波儀的現有通道合成模擬 EOG 與 EMG。

H3 沒有獨立的下巴 EMG,也只有單一 ROC。實際貼法(2026-09-10 確認):
  EEG1 = Fp1(紅) − 右耳前(棕)     EEG2 = Fp2(黃) − 左耳前(橘)
  EEG3 = ECG 左鎖骨下(藍) − 右鎖骨下(綠)
  EEG4 = ROC(灰) − A1 左耳後(紫)   接地 = A2 右耳後(黑)

利用兩條額極導程參考電極不同,可以離線合成:

  Derived LOC  = EEG1 − EEG2 = (Fp1 − Fp2) + (左耳前 − 右耳前)
  Derived ROC  = EEG2 − EEG1 = 上式的反相

  Fp1/Fp2 在眼睛正上方,水平共軛眼動在此差值通道裡表現為明顯的
  慢波偏轉,與真實 LOC/ROC 的反相特徵等效。

  Derived EMG  = highpass(EEG1 − EEG2, fc=10 Hz)
  高通後腦電與眼動(<10 Hz)被移除,殘留的高頻成分主要來自
  耳前(顳肌 / 咀嚼肌附近)參考電極的肌電。這是下巴 EMG 的粗略代理,
  可用於觀察 REM atonia。

用法:
    python3 derive_eog_emg.py <資料根目錄>/<收案資料夾>
    python3 derive_eog_emg.py /Users/xiaoyu/sandbox/收案   # 批次
"""
import sys, os, glob, json
import numpy as np
from scipy import signal as sg
from align import EDF

MAPPING = {'EEG1': 'Fp1', 'EEG2': 'Fp2', 'EEG3': 'ECG', 'EEG4': 'ROC'}
HP_EEG, LP_EEG = 0.3, 35.0
HP_EMG = 10.0


def derive_epoch(edf, t0, t1):
    """取一段時間,回傳 derived LOC、ROC、EMG 與原始 ROC (EEG4)。"""
    x1, fs = edf.get('EEG1', t0, t1)
    x2, _ = edf.get('EEG2', t0, t1)
    x4, _ = edf.get('EEG4', t0, t1)
    n = min(len(x1), len(x2), len(x4))
    x1, x2, x4 = x1[:n], x2[:n], x4[:n]

    # Derived EOG: Fp1−Fp2 方向 = EEG1−EEG2
    diff = x1 - x2

    # 0.3-35 Hz bandpass for EOG display
    sos_eog = sg.butter(4, [HP_EEG / (fs / 2), LP_EEG / (fs / 2)], 'band', output='sos')
    loc = sg.sosfiltfilt(sos_eog, diff)
    roc = -loc  # 反相

    # Derived EMG: 10-35 Hz of the same difference
    sos_emg = sg.butter(4, [HP_EMG / (fs / 2), LP_EEG / (fs / 2)], 'band', output='sos')
    emg = sg.sosfiltfilt(sos_emg, diff)

    # 原始 ROC (EEG4) for comparison
    roc_orig = sg.sosfiltfilt(sos_eog, x4)

    # 原始 Fp1/Fp2 (filtered) for context
    fp1 = sg.sosfiltfilt(sos_eog, x1)
    fp2 = sg.sosfiltfilt(sos_eog, x2)

    return dict(fs=fs, loc=loc, roc=roc, emg=emg, roc_orig=roc_orig,
                fp1=fp1, fp2=fp2, t0=t0, t1=t1)


def emg_rms_profile(edf, win_s=3.0, step_s=1.0):
    """整夜 derived EMG 的 RMS 時間序列,用於觀察 REM atonia。"""
    x1, fs = edf.get('EEG1', 0, edf.length)
    x2, _ = edf.get('EEG2', 0, edf.length)
    n = min(len(x1), len(x2))
    diff = x1[:n] - x2[:n]
    sos = sg.butter(4, [HP_EMG / (fs / 2), LP_EEG / (fs / 2)], 'band', output='sos')
    emg = sg.sosfiltfilt(sos, diff)
    win, step = int(win_s * fs), int(step_s * fs)
    rms = np.array([np.sqrt(np.mean(emg[i:i + win] ** 2))
                    for i in range(0, len(emg) - win, step)])
    t = np.arange(len(rms)) * step_s
    return t, rms


def plot_epoch(data, save_path=None):
    """6 通道堆疊圖:Fp1, Fp2, Derived LOC, Derived ROC, 原始 ROC, Derived EMG。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fs = data['fs']
    n = len(data['loc'])
    t = np.arange(n) / fs + data['t0']

    channels = [
        ('Fp1 (EEG1)', data['fp1'], 100),
        ('Fp2 (EEG2)', data['fp2'], 100),
        ('Derived LOC\n(Fp1−Fp2)', data['loc'], 150),
        ('Derived ROC\n(Fp2−Fp1)', data['roc'], 150),
        ('Original ROC\n(EEG4)', data['roc_orig'], 150),
        ('Derived EMG\n(M1−M2, >10Hz)', data['emg'], 30),
    ]

    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'H3 Derived EOG & EMG  (t = {data["t0"]:.0f}–{data["t1"]:.0f} s)',
                 fontsize=13, fontweight='bold')

    for ax, (label, sig, scale) in zip(axes, channels):
        ax.plot(t, sig, 'k', linewidth=0.4)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center')
        ax.set_ylim(-scale, scale)
        ax.axhline(0, color='grey', linewidth=0.3)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)


def plot_emg_overview(t, rms, case_name, save_path=None):
    """整夜 EMG RMS 概觀圖。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 3))
    hours = t / 3600
    ax.plot(hours, rms, 'k', linewidth=0.3)
    median_rms = np.median(rms)
    ax.axhline(median_rms, color='blue', linewidth=0.8, linestyle='--',
               label=f'median = {median_rms:.1f} µV')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('EMG RMS (µV)')
    ax.set_title(f'{case_name} — Derived EMG (M1−M2 >10 Hz) RMS', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(hours[0], hours[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    plt.close(fig)


def analyse(folder):
    edfs = [p for p in glob.glob(os.path.join(folder, '*.[eE][dD][fF]'))
            if not os.path.basename(p).startswith('._')]
    h3 = None
    for p in edfs:
        e = EDF(p)
        if 'EEG3' in e.lab and 'X' in e.lab:
            h3 = e
            break
    name = os.path.basename(folder.rstrip('/'))
    if h3 is None:
        print(f'[{name}] 找不到 H3 檔,跳過')
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'derived_plots')
    os.makedirs(out_dir, exist_ok=True)

    # 取三個時段的 60 秒 epoch:前段/中段/後段
    length = h3.length
    probes = [
        ('early', max(600, length * 0.15)),
        ('mid', length * 0.5),
        ('late', min(length * 0.85, length - 660)),
    ]

    for tag, t0 in probes:
        try:
            data = derive_epoch(h3, t0, t0 + 60)
            path = os.path.join(out_dir, f'{name}_{tag}_eog_emg.png')
            plot_epoch(data, path)
            print(f'  [{name}] {tag} epoch → {path}')
        except Exception as ex:
            print(f'  [{name}] {tag} epoch failed: {ex}')

    # 整夜 EMG RMS
    try:
        t, rms = emg_rms_profile(h3)
        path = os.path.join(out_dir, f'{name}_emg_overview.png')
        plot_emg_overview(t, rms, name, path)
        print(f'  [{name}] EMG overview → {path}')
    except Exception as ex:
        print(f'  [{name}] EMG overview failed: {ex}')

    # 統計摘要
    stats = {}
    for tag, t0 in probes:
        try:
            data = derive_epoch(h3, t0, t0 + 120)
            stats[tag] = {
                'loc_rms': float(np.sqrt(np.mean(data['loc'] ** 2))),
                'roc_orig_rms': float(np.sqrt(np.mean(data['roc_orig'] ** 2))),
                'emg_rms': float(np.sqrt(np.mean(data['emg'] ** 2))),
            }
        except Exception:
            pass

    print(f'  [{name}] 完成 | EOG RMS ~ {np.mean([s["loc_rms"] for s in stats.values()]):.1f} µV | '
          f'EMG RMS ~ {np.mean([s["emg_rms"] for s in stats.values()]):.1f} µV')
    return stats


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    folders = [root] if glob.glob(os.path.join(root, '*.[eE][dD][fF]')) else \
        sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    for f in folders:
        analyse(f)
