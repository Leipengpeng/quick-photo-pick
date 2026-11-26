# Quick Photo Pick｜自动选片

一次导入 RAW、JPG 和 PNG，把连拍自动分组，再从每组挑出最值得先看的照片。整个选片过程在本地运行，原片保持原样。

**适合拍完一大批照片后先做一轮初选。** 把重复取景聚在一起审阅，优先查看睁眼、眼部更清晰的候选，再按自己的创作需要定片。

## 能做什么

- **按拍摄分组。** 结合拍摄时间、镜头信息和画面匹配，找出每次连拍。符合条件的 RAW＋JPG 作为同一次快门的两个版本。
- **组内推荐。** 人像组优先看闭眼状态，再比眼部和整图细节。其他组按整图细节、曝光剪切等技术信号排序。
- **聚合近似场景。** 同一地点在不同时间拍下的相近构图可以集中查看，每个连拍组仍保留自己的排名和推荐结果。
- **导出原格式。** 生成离线审片网页、JSON、CSV 和分组副本。每组的推荐照片另存到 `最佳图片`，DNG 等 RAW 继续保留原格式。

## 安装和运行

安装需要 Git 和 Python。本项目在 Windows 的 Python 3.13 环境下测试。已有 ExifTool 时可用 `--exiftool` 指定它读取拍摄信息。

```shell
git clone https://github.com/Leipengpeng/quick-photo-pick.git
cd quick-photo-pick
python -m pip install -r requirements.txt
python scripts/setup_models.py
```

模型准备完成后，把输入指向原片目录，输出选在另一个目录。

```shell
python scripts/pick.py "photos" --output "photo-pick-output"
```

安装阶段从上游获取人像组件和模型，保存在仓库的 `.runtime/` 目录。后面的图片分析在本地执行。

对带有独立拍摄清单的照片可加 `--metadata metadata.json`，输入 ExifTool 风格的 JSON 数组，其中 `SourceFile` 指向照片，`DateTimeOriginal` 等字段填在同一条记录里。支持 `--profile auto|portrait|general|landscape|street`。`auto` 按组判断画面中是否经常出现人脸。

## 从哪里看结果

打开终端打印的报告路径，或在输出目录找 `latest.json`，再进入它记录的运行目录。

| 文件 | 用途 |
| --- | --- |
| `选片报告.html` | 浏览连拍组、场景、推荐照片和组内排序 |
| `已分组照片/` | 原格式副本，每组另有 `最佳图片/` |
| `selection.json`、`selection.csv` | 读取分数、排序、保留建议和复核原因 |
| `照片核验.json` | 核对输出文件与输入的 SHA-256 |

默认每组不超过 10 张时推荐 1 张，超过 10 张时推荐 2 张。存在解码、眼部状态或连拍边界问题的照片会给出复核提示。分数用于组内比较。

## 算法和实测

在有标注的 250 张 RAW 上，28 个真值连拍组全部恢复。在 360 张已分组的单人人像测试中，120 张完全闭眼图检出 119 张，240 张睁眼图误报 2 张。 [算法讲解](docs/技术原理.md) · [实验条件和完整结果](docs/实验结果.md)

使用支持 Skill 的本地 Agent 时，可加载 [Quick Photo Pick Skill](skills/quick-photo-pick/SKILL.md)，由 Agent 调用同一个本地命令。

项目由 **扫雷教你做AI** 制作，代码采用 [MIT 许可证](LICENSE)。人脸、眼部关键点和眼部锐度分析复用 Facet 的组件。模型和实验原图由各自上游提供。 [Facet](https://github.com/ncoevoet/facet)
