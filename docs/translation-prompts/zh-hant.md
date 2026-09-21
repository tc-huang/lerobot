# Target language

Traditional Chinese (language tag: `zh-hant`).

- Use Traditional characters only. A Simplified form is always an error: write `選項` not
  `选项`, `設備` not `设备`, `這些` not `这些`.
- This holds everywhere: prose, headings, table cells, and translated comments inside code blocks.

# Punctuation and spacing

- Use full-width punctuation: `，。：；？！「」` and so on.
- Put one half-width space between Chinese characters and adjacent Latin letters or digits: `使用 Python 3.12 建立虛擬環境`.
- Use full-width parentheses when glossing an English term: `策略（policy）`.

# English glosses

- When you judge that an English gloss helps the reader, add it at the term's **first occurrence** in the paragraph, for example `末端執行器（end effector）`. Common cases: key terms, translations that cannot fully capture the original, and terms the reader may want to match against variable names in the code.
- Do not repeat the gloss for later occurrences in the same paragraph.

# Glossary

| English                | Translation | Do not translate |
| ---------------------- | ----------- | ---------------- |
| policy                 | 策略        |                  |
| dataset                | 資料集      |                  |
| episode                | 回合        |                  |
| frame                  | 影格        |                  |
| observation            | 觀測        |                  |
| action                 | 動作        |                  |
| reward                 | 獎勵        |                  |
| chunk                  | 區塊        |                  |
| teleoperation          | 遙操作      |                  |
| leader arm             | 主臂        |                  |
| follower arm           | 從臂        |                  |
| end effector           | 末端執行器  |                  |
| gripper                | 夾爪        |                  |
| joint                  | 關節        |                  |
| motor                  | 馬達        |                  |
| calibration            | 校正        |                  |
| camera                 | 相機        |                  |
| checkpoint             | 檢查點      |                  |
| inference              | 推論        |                  |
| fine-tuning            | 微調        |                  |
| imitation learning     | 模仿學習    |                  |
| reinforcement learning | 強化學習    |                  |
| processor              | 處理器      |                  |
| pipeline               | 管線        |                  |
| benchmark              | 基準測試    |                  |
| simulation             | 模擬        |                  |
| repository             | 儲存庫      |                  |
| Hub                    |             | X                |
| rollout                |             | X                |
