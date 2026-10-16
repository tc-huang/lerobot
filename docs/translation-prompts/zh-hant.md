# Target language

Traditional Chinese (language tag: `zh-hant`).

- Use Traditional characters only. A Simplified form is always an error: write `選項` not `选项`, `設備` not `设备`, `這些` not `这些`.
- This holds everywhere: prose, headings, table cells, and translated comments inside code blocks.
- Use these word choices: `執行` not `運行`, `程式碼` not `代碼`, `儲存` not `保存` (for saving data or files; `保存` stays for keeping or holding), `預設` not `默認`, `透過` not `通過` (for "via"), `實作` not `實現` (for "implement"), `元件` not `組件`, `使用者` not `用戶`, `執行緒` not `線程`.
- Also use Taiwan wording for these: `教學` not `教程`, `延遲` not `時延`, `文字` not `文本`, `設定` not `設置`, `資料` not `數據` (for data; `數據` stays for statistics or measurements), `影片` not `視頻`, `軟體` not `軟件`, `硬體` not `硬件`, `網路` not `網絡`, `資訊` not `信息`, `介面` not `接口`, `呼叫` not `調用`, `物件` not `對象`, `變數` not `變量`, `字串` not `字符串`, `記憶體` not `內存`, `品質` not `質量`, `專案` not `項目` (for a project; `項目` stays for an item), `支援` not `支持` (for software support).
- Address the reader as `你`, never `您`.

# Punctuation and spacing

- Use full-width punctuation: `，。：；？！「」` and so on.
- Put one half-width space between Chinese characters and adjacent Latin letters, digits or inline code: `使用 Python 3.12 建立虛擬環境`, ``執行 `lerobot-train` 指令``.
- Never put a space next to full-width punctuation: `動作（action）transformer`, not `動作（action） transformer`.
- Use full-width parentheses when glossing an English term: `策略（policy）`.
- Use `——` for a dash in prose, never a single `—`. A lone `—` that means "none" in a table cell stays as is.
- Between two Chinese words, use the full-width slash `／`: `觀測／動作`. Next to English or code, keep `/`.
- Never put a gloss inside bold text that is followed by Chinese: write `**相對動作**（relative action）會`, not `**相對動作（relative action）**會`. CommonMark does not close the bold in the second form, so the asterisks show up on the page.

# English glosses

- Gloss only technical terms, never ordinary words, phrases, or whole clauses. `資料集（dataset）` and `執行（run）` add nothing; `動作區塊（action chunk）` helps.
- Gloss a term at its first occurrence in each `##` section, and only there. Later occurrences in the same section, including in lists and tables, use the Chinese alone.
- The core terms `策略（policy）`, `動作（action）`, `觀測（observation）`, `狀態（state）` and `獎勵（reward）` are the exception: gloss each one once per page, at its first occurrence in that sense, and not again in later sections. A gloss on a compound that contains the term, such as `動作空間（action space）`, counts. Never put a gloss inside a compound: write `策略類型（policy type）`, not `策略（policy）類型`. `狀態` in another sense, such as optimizer or processor state, follows the rule above.
- Write the gloss in lowercase unless it is a proper noun or an acronym: `檢查點（checkpoint）`, `（Hub）`, `（VLA）`.
- Do not gloss inside headings or table header rows.
- A gloss never contains another pair of parentheses. If the English already has parentheses, rephrase instead of nesting.

# Headings

- Headings are in Chinese only, followed by the anchor described in the general rules. Do not add an English gloss to a heading. Acronyms and proper nouns such as `FSDP` or `LIBERO` stay as they are.

# Code blocks

- Keep ASCII box diagrams in English. Chinese characters are twice as wide, so a translated diagram no longer lines up.

# Terminology

Use these translations exactly. They are the cases where a different rendering has appeared before.

| English                                 | Translation | Do not translate | Notes                                                                          |
| --------------------------------------- | ----------- | ---------------- | ------------------------------------------------------------------------------ |
| policy                                  | 策略        |                  |                                                                                |
| inference                               | 推論        |                  | Never 推理, which is reasoning.                                                |
| reasoning                               | 推理        |                  |                                                                                |
| strategy (a `lerobot-rollout` strategy) | 執行策略    |                  | Keeps it apart from policy. In ordinary prose, "strategy" may be 做法 or 方式. |
| episode                                 | 回合        |                  |                                                                                |
| epoch                                   |             | X                | Never 回合, which is episode.                                                  |
| frame                                   | 影格        |                  | Not 幀.                                                                        |
| video                                   | 影片        |                  | Not 視訊 or 視頻.                                                              |
| sample / sampling                       | 樣本／取樣  |                  | Not 採樣 or 抽樣.                                                              |
| observation                             | 觀測        |                  |                                                                                |
| action                                  | 動作        |                  |                                                                                |
| state                                   | 狀態        |                  |                                                                                |
| reward                                  | 獎勵        |                  |                                                                                |
| teleoperation                           | 遙操作      |                  |                                                                                |
| leader arm / follower arm               | 主臂／從臂  |                  | Also for "Leader" and "Follower" on their own.                                 |
| end effector                            | 末端執行器  |                  |                                                                                |
| camera                                  | 相機        |                  | Webcam is 網路攝影機.                                                          |
| metadata                                | 中繼資料    |                  | Not 元資料.                                                                    |
| horizon                                 | 時域        |                  | Never 水平. Long-horizon is 長時程.                                            |
| embodiment                              | 具身形態    |                  | Embodied is 具身, cross-embodiment is 跨具身. Never a bare 具身 as a noun.     |
| chunk / chunking                        | 區塊／分塊  |                  | Real-time chunking is 即時分塊.                                                |
| transition (`EnvTransition`)            |             | X                | Keeps it apart from transform, which is 轉換.                                  |
| contract                                | 契約        |                  |                                                                                |
| command                                 | 指令／命令  |                  | 指令 for shell and CLI commands, 命令 for commands sent to a robot or motor.   |
| function                                | 函式／函數  |                  | 函式 for code, 函數 for math such as loss or reward functions.                 |
| process                                 | 行程        |                  | OS process. Not 進程 or 程序. The general sense, as in denoising, is 過程.     |
| config / configuration                  | 設定        |                  | 配置 only for a physical layout or placement.                                  |
| class                                   | 類別        |                  | Never a bare 類.                                                               |
| optional                                | 可選        |                  |                                                                                |
| dependency                              | 相依套件    |                  | Not 依賴. The relation itself is 相依性.                                       |
| extra (pip)                             | 額外套件    |                  |                                                                                |
| key (of a dict or observation)          | 鍵名        |                  |                                                                                |
| resume (training)                       | 接續訓練    |                  | Not 恢復.                                                                      |
| rendering                               | 渲染        |                  |                                                                                |
| checkpoint                              | 檢查點      |                  |                                                                                |
| backbone                                | 骨幹網路    |                  |                                                                                |
| embedding                               | 嵌入向量    |                  |                                                                                |
| normalization                           | 正規化      |                  | Standardization is 標準化.                                                     |
| optimizer                               | 優化器      |                  | Optimize and optimization are 最佳化.                                          |
| repository                              | 儲存庫      |                  | Not 倉庫.                                                                      |
| Docker image                            | 映像檔      |                  | Not 鏡像.                                                                      |
| backward compatibility                  | 向下相容    |                  | Not 向後相容.                                                                  |
| cheat sheet                             | 速查表      |                  | Not 小抄.                                                                      |
| access token                            | 存取權杖    |                  |                                                                                |
| token / tokenizer                       |             | X                | Tokenize is 切分為 token, never 被 tokenize.                                   |
| hook                                    |             | X                |                                                                                |
| adapter (PEFT, LoRA)                    |             | X                | A hardware adapter is 轉接器.                                                  |
| activation                              |             | X                | Not 激活.                                                                      |
| Hub                                     |             | X                |                                                                                |
| rollout                                 |             | X                |                                                                                |
