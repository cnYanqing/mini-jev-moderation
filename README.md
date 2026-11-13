# Mini-Jev 内容审核：图片 v6

基于 [Mini-Jev](https://github.com/agarwalastitwa027-svg/mini-jev-multimodal) 和 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 的静态图片内容审核项目。当前保留的图片 v6 分别给出色情／R18 裸露、血腥两类模型分数，并提供人工复核建议。

**模型权重与下载：** [Hugging Face · lettery/mini-jev-moderation-image-v6](https://huggingface.co/lettery/mini-jev-moderation-image-v6)。仓库中的推理代码与模型压缩包配套；Qwen 底座由程序另行下载。这里不存放权重、原始图片或个人标注。

## 项目做了什么

1. 在 1,132 张整理后的图片上按来源和相似画面隔离数据，分为 465 train、569 dev、98 test。训练和开发标签主要由 AI 初审，含待定样本；不是完整人工金标。
2. 冻结视觉语言骨干及原有 LoRA，只优化图片色情、血腥判定头；开发集选择权重，并加上复核阈值与示意图分流。
3. 用同一批回顾图片，与 Falconsai NSFW ViT 原版及轻量调优版比较。保留更好的 Mini-Jev 图片 v6，没有把效果较差的实验模型切换为正式模型。
4. 将验收后的 LoRA、判定头、策略和推理代码打包发布；外部环境默认仅供人工复核。

## 结果摘要

| 回顾样本 | Mini-Jev 图片 v6 | Falconsai 原版 | Falconsai 调优版 |
|---|---:|---:|---:|
| 用户已审、色情标签可判定的 47 张：判断正确 | **42/47（89.4%）** | 38/47（80.9%） | 35/47（74.5%） |
| 同一批图片：色情漏检／误报 | 4／1 | 0／9 | 7／5 |
| 旧 AI 初标 98 张：血腥判断正确 | 94/98（95.9%） | — | 92/98（93.9%） |

“89.4%”只描述这 47 张**已看过的回顾样本**，不能解释为真实部署正确率。用户已审的 48 张图片中，血腥标签全为阴性，故**血腥召回率尚无人工数据**。详细方法、限制与错误分析见 [实验报告](docs/REPORT.md)。

## 使用

模型包在 Hugging Face 上分为 11 个文件。下载本仓库的 `reassemble.py`、`PARTS.json`，然后自动下载和校验分卷：

```bash
pip install huggingface_hub
python reassemble.py --download
```

解压生成的 `mini-jev-moderation-image-v6.zip`，在包目录执行：

```bash
pip install -r requirements.txt
python predict.py /path/to/image.jpg --device cuda
```

`inference/` 提供与发布包对应的可审阅源码。首次运行需另行下载约 4.25 GB 的 Qwen 底座；CPU 也能运行，但会慢。输出含两类**未校准的模型分数**、`suggested_action` 和固定的 `decision: review`。图片内文字、视频、音频及文字专项分支均未在此次图片版发布中验收。

## 项目状态

- 图片 v6：已完成本地固定环境试用验收，并发布为外部人工复核版。
- 文字专项 v1：已训练候选判定头，但未完成原生测试及发布验收。
- 对照实验：[Falconsai 线性头实验版](https://huggingface.co/lettery/vision-moderation-heads-experimental-v1)仅作比较，不是当前保留模型。

代码和本项目发布权重采用 Apache-2.0；上游许可与归属见 [ATTRIBUTION.md](ATTRIBUTION.md)。
