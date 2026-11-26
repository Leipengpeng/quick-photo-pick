---
name: quick-photo-pick
description: Run Quick Photo Pick on a local RAW, JPEG, or PNG folder for automatic burst grouping, explainable within-burst ranking, scene clustering, and review export. Use for photographer photo culling or auditing shortlisted shots.
---

# Quick Photo Pick

找到包含 `scripts/pick.py` 和 `requirements.txt` 的 Quick Photo Pick 仓库。首次运行按 [安装说明](../../README.md) 准备本地 Python 依赖与校验过的人像模型。

```shell
python scripts/pick.py "原片目录" --output "结果目录"
```

默认使用 `--profile auto`。用户明确要按人像、风光或街拍评价时，可传 `portrait`、`landscape` 或 `street`。已有 `SourceFile` 元数据清单时传 `--metadata`。给独立的结果目录，源照片只读。

读取运行回执里的实际照片、连拍组、场景和推荐数量。返回实际产生的 `选片报告.html` 和 `已分组照片/`，需要细查时提供 `selection.json` 或 `selection.csv`。同一场景下的不同连拍组独立排序。普通组 10 张以内最多推荐一张，更多时最多推荐两张。解释推荐时采用实际评分、闭眼状态与复核理由。
