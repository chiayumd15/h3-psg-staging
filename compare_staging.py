#!/usr/bin/env python3
"""H3 人工判讀 vs PSG(Alice)判讀:時間校正後、以 PSG 關燈→開燈為範圍的逐 epoch 比較。

輸入(都在收案資料夾內):
  alignment_case.json          align_case.py 的輸出(PSG 秒 → H3 秒 的線性關係)
  *.staging.json               Sleepwell H3 存的人工判讀(H3 時間軸、30 秒 epoch)
  <psg>/*.rml                  Alice 的判讀(Stage 變化點 + Arousal 事件)
  <psg>/*-report.txt           Alice 報告(取 Recording start / Lights out / Lights on)
輸出:
  <case>/h3_vs_psg_report.md, h3_vs_psg_metrics.json, h3_vs_psg_hypnogram.png, h3_vs_psg_confusion.png
用法: python3 compare_staging.py <收案資料夾>
"""
import sys, os, glob, json, re, csv, io
import numpy as np
import xml.etree.ElementTree as ET

EPOCH = 30
STAGES = ['W', 'N1', 'N2', 'N3', 'R']
MAP = {'Wake': 'W', 'NonREM1': 'N1', 'NonREM2': 'N2', 'NonREM3': 'N3', 'NonREM4': 'N3', 'REM': 'R',
       'Stage1': 'N1', 'Stage2': 'N2', 'Stage3': 'N3', 'Stage4': 'N3', 'NotScored': None}


# ---------------------------------------------------------------- 讀取
def hms_to_s(txt):
    """'PM 09:50:17' → 秒(當日)。"""
    # Alice 中文系統匯出會寫「上午/下午 10:02:20」(Big5),等同 AM/PM
    txt = txt.replace('上午', 'AM').replace('下午', 'PM')
    m = re.match(r'\s*(AM|PM)?\s*(\d+):(\d+):(\d+)', txt)
    if not m:
        return None
    ap, h, mi, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if ap == 'PM' and h != 12:
        h += 12
    if ap == 'AM' and h == 12:
        h = 0
    return h * 3600 + mi * 60 + s


def read_report_times(path):
    """從 Alice report.txt 取 Recording start / Lights out / Lights on(相對錄製開始的秒數)。"""
    raw = open(path, 'rb').read()
    try:
        txt = raw.decode('utf-8')
    except UnicodeDecodeError:
        txt = raw.decode('cp950', 'ignore')   # Alice 中文系統匯出的 Big5 報告
    want = {'Recording start': None, 'Lights out (LO)': None, 'Lights on': None, 'Sleep onset (SO)': None}
    for row in csv.reader(io.StringIO(txt)):
        if len(row) >= 3 and row[2].strip() in want and want[row[2].strip()] is None:
            want[row[2].strip()] = hms_to_s(row[1])
    rs = want['Recording start']
    rel = lambda t: None if (t is None or rs is None) else ((t - rs) % 86400)
    return {'lights_off_s': rel(want['Lights out (LO)']), 'lights_on_s': rel(want['Lights on']),
            'sleep_onset_s': rel(want['Sleep onset (SO)']), 'recording_start_clock': want['Recording start']}


def read_rml(path):
    raw = open(path, 'rb').read()
    root = ET.fromstring(raw)
    ns = root.tag.split('}')[0] + '}' if root.tag.startswith('{') else ''
    stages = []
    for el in root.iter(ns + 'Stage'):
        stages.append((float(el.get('Start')), el.get('Type')))
    stages.sort()
    events = []
    for el in root.iter(ns + 'Event'):
        events.append({'family': el.get('Family'), 'type': el.get('Type'),
                       'start': float(el.get('Start')), 'dur': float(el.get('Duration') or 0)})
    dur = None
    for el in root.iter(ns + 'Duration'):
        try:
            dur = max(dur or 0, float(el.text))
        except (TypeError, ValueError):
            pass
    return stages, events, dur


def rml_epoch_staging(change_points, n_epochs):
    out = [None] * n_epochs
    pi = 0
    for k in range(n_epochs):
        t = k * EPOCH
        while pi + 1 < len(change_points) and change_points[pi + 1][0] <= t:
            pi += 1
        if change_points and change_points[pi][0] <= t:
            out[k] = MAP.get(change_points[pi][1], None)
    return out


# ---------------------------------------------------------------- 指標
def confusion(ref, hyp):
    idx = {s: i for i, s in enumerate(STAGES)}
    M = np.zeros((len(STAGES), len(STAGES)), int)
    for r, h in zip(ref, hyp):
        M[idx[r], idx[h]] += 1
    return M


def kappa(M):
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def per_stage(M):
    rows = []
    n = M.sum()
    for i, s in enumerate(STAGES):
        tp = M[i, i]; fn = M[i].sum() - tp; fp = M[:, i].sum() - tp; tn = n - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else float('nan')
        spec = tn / (tn + fp) if tn + fp else float('nan')
        ppv = tp / (tp + fp) if tp + fp else float('nan')
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan')
        # 二元(此期 vs 其他)且只有硬標籤時,ROC 只有一個工作點,AUC = (敏感度+特異度)/2 = 平衡準確度
        auc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float('nan')
        rows.append({'stage': s, 'psg_n': int(tp + fn), 'h3_n': int(tp + fp), 'tp': int(tp),
                     'sensitivity': sens, 'specificity': spec, 'ppv': ppv, 'f1': f1, 'auc_ovr': auc,
                     'psg_pct': 100 * (tp + fn) / n, 'h3_pct': 100 * (tp + fp) / n})
    return rows


def sleep_params(stages, lights_off_ep, lights_on_ep):
    """stages: 以 PSG epoch 為索引的列表(可含 None)。回傳分鐘為單位的睡眠參數。"""
    seg = stages[lights_off_ep:lights_on_ep]
    n = len(seg)
    tib = n * EPOCH / 60
    sleep = [i for i, s in enumerate(seg) if s in ('N1', 'N2', 'N3', 'R')]
    if not sleep:
        return {'TIB_min': tib, 'TST_min': 0, 'SE_pct': 0}
    so, last = sleep[0], sleep[-1]
    tst = len(sleep) * EPOCH / 60
    cnt = {s: sum(1 for x in seg if x == s) * EPOCH / 60 for s in STAGES}
    rem = [i for i, s in enumerate(seg) if s == 'R']
    waso = sum(1 for x in seg[so:last + 1] if x == 'W') * EPOCH / 60
    return {
        'TIB_min': tib, 'TST_min': tst, 'SE_pct': 100 * tst / tib if tib else float('nan'),
        'SOL_min': so * EPOCH / 60, 'WASO_min': waso,
        'REM_latency_min': (rem[0] - so) * EPOCH / 60 if rem else None,
        'W_min': cnt['W'], 'N1_min': cnt['N1'], 'N2_min': cnt['N2'], 'N3_min': cnt['N3'], 'R_min': cnt['R'],
        'N1_pct_TST': 100 * cnt['N1'] / tst, 'N2_pct_TST': 100 * cnt['N2'] / tst,
        'N3_pct_TST': 100 * cnt['N3'] / tst, 'R_pct_TST': 100 * cnt['R'] / tst,
        'unscored_epochs': sum(1 for x in seg if x is None),
    }


def match_events(ref, hyp):
    """任意重疊即算配對。回傳 (ref 被命中數, hyp 被命中數)。"""
    def hits(A, B):
        h = 0
        for s, e in A:
            if any(bs <= e and be >= s for bs, be in B):
                h += 1
        return h
    return hits(ref, hyp), hits(hyp, ref)


# ---------------------------------------------------------------- 主程式
def main(folder):
    name = os.path.basename(folder.rstrip('/'))
    did = '#' + (re.match(r'\d+', name).group() if re.match(r'\d+', name) else name)  # 去識別化：只用 #編號
    al = json.load(open(os.path.join(folder, 'alignment_case.json')))
    icpt, slope = al['offset_intercept_s'], al['drift_slope']
    segs = al.get('segments')
    if segs:
        # 錄音不連續:依 H3 時間的分段點各用自己的線性關係(分段點換成 PSG 時間用第一段的關係)
        (i0, s0), (i1, s1) = [(g['offset_intercept_s'], g['drift_slope']) for g in segs[:2]]
        brk_psg = segs[0]['break_h3_t'] * (1 + s0) + i0
        psg_to_h3 = lambda t: (t - i0) / (1 + s0) if t < brk_psg else (t - i1) / (1 + s1)
        h3_to_psg = lambda t: t * (1 + s0) + i0 if t < segs[0]['break_h3_t'] else t * (1 + s1) + i1
    else:
        psg_to_h3 = lambda t: (t - icpt) / (1 + slope)
        h3_to_psg = lambda t: t * (1 + slope) + icpt

    stg_path = sorted(glob.glob(os.path.join(folder, '*.staging.json')))[-1]
    # 格線版(make_grid_edf.py):H3 已重取樣到 PSG 時間軸,判讀檔在格線檔上 → 恆等對映(只差 psg_t0_s)
    grid = None
    for gj in glob.glob(os.path.join(folder, '*.grid.json')):
        g = json.load(open(gj))
        gs = os.path.join(folder, g['grid_edf'].rsplit('.', 1)[0] + '.staging.json')
        if os.path.exists(gs):
            grid, stg_path = g, gs
    if grid:
        # rml_offset_s:PSG 分多段錄製時,RML 時間軸從第一段開始(含中斷),alignment/格線檔用的是主檔時間 → 需加上此差
        t0 = grid['psg_t0_s'] + grid.get('rml_offset_s', 0)
        psg_to_h3 = lambda t: t - t0
        h3_to_psg = lambda t: t + t0
        al = dict(al, h3_length_s=grid['n_epochs'] * EPOCH)
    h3 = json.load(open(stg_path))
    h3_stages = {int(e): MAP.get(s) for e, s in h3['staging']}
    # 缺頁檢查(H3 判讀檔自身編號):中間沒標籤的 epoch、非五期標籤(如 Movement)
    _lab = {int(e): s for e, s in h3['staging']}
    _ep = sorted(_lab)
    h3_gaps = sorted(set(range(_ep[0], _ep[-1] + 1)) - set(_ep)) if _ep else []
    h3_odd = {e: s for e, s in sorted(_lab.items()) if s not in MAP or MAP.get(s) is None}
    h3_arousals = [(a['start'], a['end']) for a in h3.get('arousals', [])]

    rml_path = glob.glob(os.path.join(folder, '**', '*.rml'), recursive=True)[0]
    rep_path = glob.glob(os.path.join(folder, '**', '*-report.txt'), recursive=True)
    cps, events, rml_dur = read_rml(rml_path)
    psg_len = al.get('psg_length_s') or rml_dur
    n_psg = int(psg_len // EPOCH)
    psg_stages = rml_epoch_staging(cps, n_psg)
    psg_arousals = [(e['start'], e['start'] + e['dur']) for e in events if e['type'] and 'Arousal' in e['type']]

    times = read_report_times(rep_path[0]) if rep_path else {}
    lo = times.get('lights_off_s') if times.get('lights_off_s') is not None else 0
    lon = times.get('lights_on_s') if times.get('lights_on_s') is not None else psg_len
    lo_ep, lon_ep = int(round(lo / EPOCH)), min(n_psg, int(round(lon / EPOCH)))

    # 切齊頭尾:分析視窗只取「兩邊都有錄到」的交集——起點夾到 H3 開始錄之後,
    # 終點夾到 H3 錄製結束(或 PSG 結束,取先到者)。H3 未涵蓋的 PSG 段落屬結構性缺漏,
    # 不計入比較,也不列為「未判讀」。
    h3_len = al.get('h3_length_s')
    if h3_len is None:
        h3_len = max((int(e) for e in h3_stages), default=0) * EPOCH + EPOCH
    def _h3_time(k):
        return psg_to_h3(k * EPOCH + EPOCH / 2)
    valid = [k for k in range(lo_ep, lon_ep) if 0 <= _h3_time(k) <= h3_len]
    if valid:
        lo_ep, lon_ep = min(valid), max(valid) + 1

    # 把 H3 判讀重取樣到 PSG 的 epoch 網格:取 PSG epoch 中點對應的 H3 epoch
    h3_on_psg = [None] * n_psg
    for k in range(n_psg):
        ht = psg_to_h3(k * EPOCH + EPOCH / 2)
        if ht >= 0:
            h3_on_psg[k] = h3_stages.get(int(ht // EPOCH))

    pairs = [(psg_stages[k], h3_on_psg[k]) for k in range(lo_ep, lon_ep) if psg_stages[k] and h3_on_psg[k]]
    ref = [p for p, _ in pairs]; hyp = [h for _, h in pairs]
    M = confusion(ref, hyp)
    acc = np.trace(M) / M.sum()
    k5 = kappa(M)
    # 合併 N1+N2、W/NREM/REM 等較寬鬆的比較
    def collapse(x, groups):
        for g, members in groups.items():
            if x in members:
                return g
        return x
    g3 = {'W': ['W'], 'NREM': ['N1', 'N2', 'N3'], 'R': ['R']}
    ref3 = [collapse(x, g3) for x in ref]; hyp3 = [collapse(x, g3) for x in hyp]
    lab3 = ['W', 'NREM', 'R']
    M3 = np.zeros((3, 3), int)
    for r, h in zip(ref3, hyp3):
        M3[lab3.index(r), lab3.index(h)] += 1
    acc3, k3 = np.trace(M3) / M3.sum(), kappa(M3)
    ref2 = ['S' if x != 'W' else 'W' for x in ref]; hyp2 = ['S' if x != 'W' else 'W' for x in hyp]
    M2 = np.zeros((2, 2), int)
    for r, h in zip(ref2, hyp2):
        M2[['W', 'S'].index(r), ['W', 'S'].index(h)] += 1
    acc2, k2 = np.trace(M2) / M2.sum(), kappa(M2)
    # REM vs NREM:只看兩邊都判為睡眠的 epoch
    pairs_s = [(r, h) for r, h in zip(ref, hyp) if r != 'W' and h != 'W']
    MR = np.zeros((2, 2), int)
    for r, h in pairs_s:
        MR[['NREM', 'R'].index('R' if r == 'R' else 'NREM'), ['NREM', 'R'].index('R' if h == 'R' else 'NREM')] += 1
    accR, kR = (np.trace(MR) / MR.sum(), kappa(MR)) if MR.sum() else (float('nan'), float('nan'))

    stage_rows = per_stage(M)
    sp_psg = sleep_params(psg_stages, lo_ep, lon_ep)
    sp_h3 = sleep_params(h3_on_psg, lo_ep, lon_ep)

    # Arousal:H3 事件換到 PSG 時間軸,限制在比較視窗(關燈~開燈且兩邊都有錄到;電池提早耗盡的案不能把斷電後的 PSG 事件算成漏抓)
    ar_lo, ar_hi = lo_ep * EPOCH, lon_ep * EPOCH
    h3_ar_psg = [(h3_to_psg(s), h3_to_psg(e)) for s, e in h3_arousals]
    h3_ar_psg = [(s, e) for s, e in h3_ar_psg if e >= ar_lo and s <= ar_hi]
    psg_ar = [(s, e) for s, e in psg_arousals if e >= ar_lo and s <= ar_hi]
    hit_ref, hit_hyp = match_events(psg_ar, h3_ar_psg)
    ar_sens = hit_ref / len(psg_ar) if psg_ar else float('nan')
    ar_ppv = hit_hyp / len(h3_ar_psg) if h3_ar_psg else float('nan')
    tst_h_psg = sp_psg['TST_min'] / 60; tst_h_h3 = sp_h3['TST_min'] / 60
    ar_idx_psg = len(psg_ar) / tst_h_psg if tst_h_psg else float('nan')
    ar_idx_h3 = len(h3_ar_psg) / tst_h_h3 if tst_h_h3 else float('nan')

    out = {
        'case': name, 'alignment': {k: al[k] for k in ('true_offset_s', 'header_error_s', 'drift_s_per_hour', 'fit_resid_sd_s', 'mean_r', 'windows_used')},
        'grid_aligned': bool(grid) and abs(((t0 + 15) % 30) - 15) < 1 if grid else False, 'h3_gaps': h3_gaps, 'h3_nonstage': h3_odd, 'discontinuity': al.get('discontinuity'), 'segments': al.get('segments'),
        'window': {'lights_off_s': lo, 'lights_on_s': lon, 'psg_epochs': [lo_ep, lon_ep], 'compared_epochs': len(pairs),
                   'psg_unscored': sum(1 for k in range(lo_ep, lon_ep) if not psg_stages[k]),
                   'h3_unscored_or_out_of_range': sum(1 for k in range(lo_ep, lon_ep) if psg_stages[k] and not h3_on_psg[k])},
        'agreement': {'acc_5class': acc, 'kappa_5class': k5, 'acc_W_NREM_R': acc3, 'kappa_W_NREM_R': k3, 'acc_W_S': acc2, 'kappa_W_S': k2,
                      'acc_R_NREM': accR, 'kappa_R_NREM': kR, 'n_sleep_epochs_both': int(MR.sum())},
        'confusion_R_NREM': {'labels': ['NREM', 'R'], 'rows_psg_cols_h3': MR.tolist()},
        'confusion_5class': {'labels': STAGES, 'rows_psg_cols_h3': M.tolist()},
        'confusion_W_NREM_R': {'labels': lab3, 'rows_psg_cols_h3': M3.tolist()},
        'per_stage': stage_rows,
        'sleep_params': {'psg': sp_psg, 'h3': sp_h3},
        'series': {'lo_ep': lo_ep, 'lon_ep': lon_ep, 'psg': [psg_stages[k] for k in range(lo_ep, lon_ep)], 'h3': [h3_on_psg[k] for k in range(lo_ep, lon_ep)]},
        'arousal': {'psg_n': len(psg_ar), 'h3_n': len(h3_ar_psg), 'psg_hit': hit_ref, 'h3_hit': hit_hyp,
                    'sensitivity': ar_sens, 'ppv': ar_ppv, 'index_psg_per_h': ar_idx_psg, 'index_h3_per_h': ar_idx_h3},
    }
    json.dump(out, open(os.path.join(folder, 'h3_vs_psg_metrics.json'), 'w'), ensure_ascii=False, indent=2, default=float)

    # ---------------- 圖
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['PingFang TC', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
        y = {'W': 4, 'R': 3, 'N1': 2, 'N2': 1, 'N3': 0}
        t = np.arange(lo_ep, lon_ep) * EPOCH / 3600
        fig, ax = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
        for a, series, title, col in ((ax[0], psg_stages, 'PSG (Alice 判讀)', '#1f4e79'), (ax[1], h3_on_psg, 'H3 人工判讀(校正後對到 PSG 時間軸)', '#c2185b')):
            v = [y.get(series[k], np.nan) for k in range(lo_ep, lon_ep)]
            a.step(t, v, where='post', color=col, lw=1.2)
            a.set_yticks(list(y.values())); a.set_yticklabels(list(y.keys())); a.set_ylim(-0.5, 4.5)
            a.set_title(title, loc='left', fontsize=11); a.grid(alpha=.3)
        # 不一致處標紅
        dis = [k for k in range(lo_ep, lon_ep) if psg_stages[k] and h3_on_psg[k] and psg_stages[k] != h3_on_psg[k]]
        for k in dis:
            ax[1].axvspan(k * EPOCH / 3600, (k + 1) * EPOCH / 3600, color='red', alpha=.12, lw=0)
        ax[1].set_xlabel('自 PSG 錄製開始的時間 (小時)')
        fig.suptitle(f'{did}  5 期一致率 {acc*100:.1f}%  κ={k5:.2f}   (紅底 = 不一致 epoch)', fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(folder, 'h3_vs_psg_hypnogram.png'), dpi=130); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for a, MM, labs, ttl in ((ax[0], M, STAGES, '5 期'), (ax[1], M3, lab3, 'W / NREM / REM')):
            P = MM / np.maximum(MM.sum(1, keepdims=True), 1) * 100
            a.imshow(P, cmap='Blues', vmin=0, vmax=100)
            for i in range(len(labs)):
                for j in range(len(labs)):
                    a.text(j, i, f'{MM[i,j]}\n{P[i,j]:.0f}%', ha='center', va='center', fontsize=9,
                           color='white' if P[i, j] > 55 else 'black')
            a.set_xticks(range(len(labs))); a.set_xticklabels(labs); a.set_yticks(range(len(labs))); a.set_yticklabels(labs)
            a.set_xlabel('H3 判讀'); a.set_ylabel('PSG 判讀'); a.set_title(f'混淆矩陣({ttl})', fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(folder, 'h3_vs_psg_confusion.png'), dpi=130); plt.close(fig)
    except Exception as e:  # 圖失敗不影響數字
        print('plot failed:', e)

    # ---------------- Markdown 報告
    f = lambda x, d=1: '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'
    _miss = []
    if h3_gaps: _miss.append(f'未判讀 epoch {", ".join(map(str, h3_gaps))}')
    if h3_odd: _miss.append('非五期標籤 ' + ', '.join(f'{e}={s}' for e, s in h3_odd.items()))
    L = [f'# {did}  H3 vs PSG 判讀比較', '',
         *([f'- ⚠ H3 缺頁(判讀檔 {os.path.basename(stg_path)} 的 epoch 編號):' + ';'.join(_miss) + '(不在比較範圍內者不影響數字)'] if _miss else []),
         *([f'- 格線:H3 判讀於格線版 EDF({grid["grid_edf"]},make_grid_edf.py 依 ECG 校正重取樣到 PSG 時間軸),H3 與 PSG 的 30 s epoch 完全重合,無格線錯位'] if grid else []),
         f'- 時間校正:標頭差 {al["header_offset_s"]} s → ECG 互相關實測 {al["true_offset_s"]} s(標頭誤差 {al["header_error_s"]:+} s),漂移 {al["drift_s_per_hour"]:+} s/hr,擬合殘差 SD {al["fit_resid_sd_s"]} s,r={al["mean_r"]},視窗 {al["windows_used"]}'
         + (f';⚠ 錄音不連續:H3 {al["discontinuity"]["between_h3_t"][0]:.0f}–{al["discontinuity"]["between_h3_t"][1]:.0f} s 間跳 {al["discontinuity"]["jump_s"]:+.1f} s,已分兩段對齊(漂移 {al["segments"][0]["drift_s_per_hour"]:+.2f} / {al["segments"][1]["drift_s_per_hour"]:+.2f} s/hr)' if al.get('discontinuity') else ''),
         f'- 比較範圍(切齊頭尾、兩邊都有錄到之交集):epoch {lo_ep}–{lon_ep-1},共 {lon_ep-lo_ep} 個;兩邊都有判讀的 epoch {len(pairs)} 個',
         f'- 視窗內 PSG 未判讀 {out["window"]["psg_unscored"]} 個;H3 未判讀 {out["window"]["h3_unscored_or_out_of_range"]} 個(視窗已切齊,無結構性缺漏)', '',
         '## 一致性', '',
         '| 比較方式 | 一致率 | Cohen κ |', '|---|---|---|',
         f'| 5 期 (W/N1/N2/N3/R) | {acc*100:.1f}% | {k5:.3f} |',
         f'| 3 類 (W/NREM/R) | {acc3*100:.1f}% | {k3:.3f} |',
         f'| 2 類 (醒/睡) | {acc2*100:.1f}% | {k2:.3f} |',
         f'| REM / NREM(兩邊皆睡眠之 {int(MR.sum())} epoch) | {accR*100:.1f}% | {kR:.3f} |', '',
         '## 混淆矩陣(列 = PSG,欄 = H3;epoch 數)', '',
         '| PSG \\ H3 | ' + ' | '.join(STAGES) + ' | 合計 |', '|---|' + '---|' * (len(STAGES) + 1)]
    for i, s in enumerate(STAGES):
        L.append(f'| **{s}** | ' + ' | '.join(str(x) for x in M[i]) + f' | {M[i].sum()} |')
    L.append('| 合計 | ' + ' | '.join(str(x) for x in M.sum(0)) + f' | {M.sum()} |')
    L += ['', '## 各期表現(one-vs-rest)', '',
          '| 期 | PSG 數 (%) | H3 數 (%) | 敏感度 | 特異度 | PPV | F1 | AUC* |', '|---|---|---|---|---|---|---|---|']
    for r in stage_rows:
        L.append(f'| {r["stage"]} | {r["psg_n"]} ({r["psg_pct"]:.1f}%) | {r["h3_n"]} ({r["h3_pct"]:.1f}%) | {f(r["sensitivity"],3)} | {f(r["specificity"],3)} | {f(r["ppv"],3)} | {f(r["f1"],3)} | {f(r["auc_ovr"],3)} |')
    L += ['', '*人工判讀是硬標籤、沒有機率分數,ROC 只有一個工作點,此處 AUC = (敏感度+特異度)/2,即平衡準確度。', '',
          '## 睡眠參數', '',
          '| 參數 | PSG | H3 | 差 (H3−PSG) |', '|---|---|---|---|']
    for key, lab in (('TIB_min', 'TIB (min)'), ('TST_min', 'TST (min)'), ('SE_pct', '睡眠效率 (%)'), ('SOL_min', '入睡潛伏期 (min)'),
                     ('WASO_min', 'WASO (min)'), ('REM_latency_min', 'REM 潛伏期 (min)'), ('N1_pct_TST', 'N1 %TST'),
                     ('N2_pct_TST', 'N2 %TST'), ('N3_pct_TST', 'N3 %TST'), ('R_pct_TST', 'REM %TST')):
        a, b = sp_psg.get(key), sp_h3.get(key)
        d = (b - a) if (a is not None and b is not None) else None
        L.append(f'| {lab} | {f(a)} | {f(b)} | {f(d) if d is None else f"{d:+.1f}"} |')
    L += ['', '## Arousal', '',
          f'- PSG(Alice)Arousal {len(psg_ar)} 個,指數 {f(ar_idx_psg)} /h(以 PSG TST 計)',
          f'- H3 標記 {len(h3_ar_psg)} 個,指數 {f(ar_idx_h3)} /h(以 H3 TST 計)',
          f'- 事件配對(任意重疊):H3 抓到 PSG 的 {hit_ref}/{len(psg_ar)}(敏感度 {f(ar_sens*100 if not np.isnan(ar_sens) else ar_sens)}%);H3 標記中 {hit_hyp}/{len(h3_ar_psg)} 有對應 PSG 事件(PPV {f(ar_ppv*100 if not np.isnan(ar_ppv) else ar_ppv)}%)',
          '', '![hypnogram](h3_vs_psg_hypnogram.png)', '', '![confusion](h3_vs_psg_confusion.png)', '']
    open(os.path.join(folder, 'h3_vs_psg_report.md'), 'w').write('\n'.join(L))
    print('\n'.join(L[:len(L) - 5]))


if __name__ == '__main__':
    main(sys.argv[1])
