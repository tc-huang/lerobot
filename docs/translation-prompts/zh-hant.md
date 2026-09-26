# Target language

Traditional Chinese (language tag: `zh-hant`).

- Use Traditional characters only. A Simplified form is always an error: write `選項` not `选项`, `設備` not `设备`, `這些` not `这些`.
- This holds everywhere: prose, headings, table cells, and translated comments inside code blocks.

# Punctuation and spacing

- Use full-width punctuation: `，。：；？！「」` and so on.
- Put one half-width space between Chinese characters and adjacent Latin letters or digits: `使用 Python 3.12 建立虛擬環境`.
- Use full-width parentheses when glossing an English term: `策略（policy）`.

# English glosses

- Add an English gloss when it helps readers identify a technical term or match it to the source text or code. At the term's **first occurrence in each paragraph**, write the Chinese translation followed by the English term in full-width parentheses, for example `回合（episode）`.
- After glossing a term, use only its Chinese translation for later occurrences in the same paragraph.
- A gloss on a compound term also counts for a term within it when the correspondence is clear. For example, after `回合中繼資料（episode metadata）`, use `回合` without another gloss in the same paragraph. If the standalone term could still be confused with another term, gloss it explicitly.

# Terminology

| English       | Translation | Do not translate | Notes                                                           |
| ------------- | ----------- | ---------------- | --------------------------------------------------------------- |
| policy        | 策略        |                  | Use `策略（policy）` when it could be confused with `strategy`. |
| strategy      | 策略        |                  | Use `策略（strategy）` when it could be confused with `policy`. |
| episode       | 回合        |                  |                                                                 |
| frame         | 影格        |                  |                                                                 |
| observation   | 觀測        |                  |                                                                 |
| teleoperation | 遙操作      |                  |                                                                 |
| leader arm    | 主臂        |                  |                                                                 |
| follower arm  | 從臂        |                  |                                                                 |
| end effector  | 末端執行器  |                  |                                                                 |
| Hub           |             | X                |                                                                 |
| rollout       |             | X                |                                                                 |
