# Target language

Simplified Chinese (language tag: `zh-hans`).

- Use Simplified characters only. A Traditional form is always an error: write `选项` not `選項`, `设备` not `設備`, `这些` not `這些`.
- This holds everywhere: prose, headings, table cells, and translated comments inside code blocks.

# Punctuation and spacing

- Use Chinese punctuation: `，。：；？！“”（）` and so on.
- Put one half-width space between Chinese characters and adjacent Latin letters or digits: `使用 Python 3.12 创建虚拟环境`.
- Use full-width parentheses when glossing an English term: `策略（policy）`.

# English glosses

- Add an English gloss when it helps readers identify a technical term or match it to the source text or code. At the term's **first occurrence in each paragraph**, write the Chinese translation followed by the English term in full-width parentheses, for example `观测（observation）`.
- After glossing a term, use only its Chinese translation for later occurrences in the same paragraph.
- A gloss on a compound term also counts for a term within it when the correspondence is clear. For example, after `片段元数据（episode metadata）`, use `片段` without another gloss in the same paragraph. If the standalone term could still be confused with another term, gloss it explicitly.

# Terminology

| English       | Translation | Do not translate | Notes                                                           |
| ------------- | ----------- | ---------------- | --------------------------------------------------------------- |
| policy        | 策略        |                  | Use `策略（policy）` when it could be confused with `strategy`. |
| strategy      | 策略        |                  | Use `策略（strategy）` when it could be confused with `policy`. |
| frame         | 帧          |                  |                                                                 |
| observation   | 观测        |                  |                                                                 |
| teleoperation | 遥操作      |                  |                                                                 |
| leader arm    | 主臂        |                  |                                                                 |
| follower arm  | 从臂        |                  |                                                                 |
| end effector  | 末端执行器  |                  |                                                                 |
| Hub           |             | X                |                                                                 |
| rollout       |             | X                |                                                                 |
