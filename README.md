# h3-psg-staging

穿戴式額極腦波裝置(H3)與同夜多項睡眠生理檢查(PSG)睡眠分期一致性研究之分析程式碼。

Analysis code for: *Frontopolar EEG Achieves Substantial Agreement with Polysomnography for Five-Stage Sleep Scoring: A Same-Night Validation in 40 Adults with Suspected Obstructive Sleep Apnea* (JCMIT 2026, submitted).

## 分析流程(依執行順序)

| 步驟 | 腳本 | 說明 |
|---|---|---|
| 1 | `align_case.py` | 兩機時間校正:以心電圖互相關估計 PSG→H3 之時間偏移與時鐘漂移(整夜每 25 分鐘取 15 分鐘視窗,剔除低相關視窗後線性迴歸;階躍變化時分段擬合),輸出 `alignment_case.json` |
| 2 | `align_eog.py` | 以眼動通道(PSG ROC-A2 vs H3 EEG4)互相關獨立交叉驗證步驟 1 之時間校正 |
| 3 | `derive_eog_emg.py` | 由 H3 之雙額極導程離線合成水平眼動導程(dLOC,0.3–35 Hz)與肌電近似導程(dEMG,10–35 Hz),輸出 `*_derived.EDF` 供人工判讀 |
| 4 | `compare_staging.py` | 逐 epoch 比較 H3 人工判讀與 PSG 判讀(RML):五期/四類一致率與 Cohen κ、混淆矩陣、各期敏感度/特異度/PPV、整夜睡眠參數(TST、WASO、潛伏期等),輸出 `h3_vs_psg_metrics.json` 與對照圖 |
| 5 | `derivation/spectral_derivations.py` | 額極(Fp1、Fp2)與標準導程(F4、C4、O2)判期腦波特徵之分辨力:逐 epoch Welch 功率譜 → 各受試者 AUC(W vs N1:α/θ;N1 vs N2:σ;N2 vs N3:δ,另含相對功率、絕對 α 與 β 之敏感度分析) |
| 6 | `derivation/demg_vs_chin.py` | dEMG 有效性驗證:dEMG 與 PSG 下頦肌電(同以 10–35 Hz)於 NREM vs REM 之分辨力(AUC)配對比較 |

合併統計(受試者層級拔靴 95% CI、各期一對其餘 κ、AHI 分層 κ、穩定期/換期處比較、Bland–Altman 與 ICC)由論文產生階段彙整,演算法同上列腳本之輸出。

## 資料

原始記錄(H3 EDF、PSG EDF/RML)含受試者個人資料,依人體試驗委員會核准範圍(草屯療養院 IRB 115053)**不隨程式碼公開**。腳本預期之收案資料夾結構:

```
<資料根目錄>/<收案資料夾>/
├── <H3 記錄>.EDF              # H3 原始四通道
├── <H3 記錄>_derived.EDF      # 步驟 3 輸出(含 dLOC、dEMG)
├── <H3 記錄>.staging.json     # H3 人工判讀
├── alignment_case.json        # 步驟 1 輸出
└── <PSG 資料夾>/
    ├── *.edf                  # Alice 6 LDx 匯出
    ├── *.rml                  # PSG 人工判讀
    └── *-report.txt           # Alice 報告(關燈/開燈時間)
```

## 環境需求

Python ≥ 3.10,套件見 `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 授權

MIT License。
