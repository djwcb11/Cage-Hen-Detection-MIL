> > 📢 **Official Implementation Notice**  
> > This repository contains the official implementation of the paper:  
> > **"Low-Yield Hen Identification in Stacked Cages via Head Detection and Prototype-Based Multiple Instance Learning"**  
> > *Submitted to **The Visual Computer** (Springer), 2026.*  
> >
> > 🔖 If you find this work useful in your research, **please consider citing our paper**  🙏
> >
> > ---
>
> ## 📖 Abstract
>
> Precision poultry farming relies on efficient automated screening of low-yield laying hens, especially in multi-tier stacked cage systems. Manual inspection is inefficient, and existing deep learning methods depend on costly instance-level annotations. This work proposes a two-stage visual computing framework combining target detection and weakly supervised multiple instance learning for cage-level low-yield hen identification. An improved lightweight YOLOv8n with SEAM attention, SCDown downsampling, and RepHead is designed to locate hen heads accurately at 89.66\% mAP and 418.8 FPS. A prototype-guided multiple instance learning model ProtoMIL then achieves cage-level classification with 81.43\% AUC and 73.18\% accuracy under weak supervision. This framework reduces annotation costs and supports real-time edge deployment, providing a practical solution for intelligent poultry farming with broad applicability for visual computing in agricultural scenarios.
>
> ---
>
> ## ✨ Highlights
>
> Two-stage framework identifies low-yield hens at the cage level.
> Improved YOLOv8n extracts hen heads accurately in dense cages.
> ProtoMIL classifies low-yield cages using weak bag-level labels.
> Global prototype strategy overcomes few-shot challenges in MIL.
>
> ### Requirements
>
> - Python ≥ 3.10
> - PyTorch ≥ 1.13
> - CUDA ≥ 11.7
>
## 📁 目录结构

```text
.
├── detect_dataset/      # 包含鸡头检测的标注文件，需与原始图像结合使用
├── mil_dataset/         # 包含多实例学习的分类标准与数据划分信息
├── YOLOv8/              # YOLOv8 鸡头检测网络模型、配置文件及训练脚本
├── mil/                 # 多实例学习网络模型及训练脚本
└── README.md            # 项目说明文档
🚀 使用手册 (Getting Started)
本指南将帮助您从头开始准备数据集，并运行目标检测与多实例学习（MIL）模型的训练脚本。

1. 数据准备 (Data Preparation)
本项目的数据集构建分为两个主要方向：鸡头目标检测数据集和多实例学习分类数据集。所有的原始图像均来源于 Roboflow。

1.1 获取原始图像数据
请首先从以下两个 Roboflow 链接下载原始图像数据，并将其解压到本地工作目录中：

数据集 1: https://universe.roboflow.com/zhejiang-university-txg5o/
chicken-counting-o45ko
数据集 2:  https://universe.roboflow.com/zhejiang-university-txg5o/egg-fqisy
1.2 构建鸡头检测数据集 (YOLO 格式)
为了训练鸡头检测网络，您需要将下载的原始图像与本项目提供的标注文件结合：

进入 detect_dataset/ 目录。
该目录中包含了鸡头检测的标注文件（如 .txt 或 .json 格式）。
将下载的原始图像放入对应的 images/ 文件夹，确保图像文件名与标注文件名一一对应。
整理后的数据集即可直接用于 YOLOv8 网络的训练。
1.3 构建多实例学习 (MIL) 数据集
如果您需要训练多实例学习模型，请按照以下步骤对原始图像进行分类整理：

进入 mil_dataset/ 目录。
参考该目录下的分类标准（或提供的 CSV/JSON 映射文件）。
将原始图像数据按照 mil_dataset 中的类别划分，移动到对应的子文件夹中（或生成对应的数据列表文件）。
整理完成后的数据结构将作为 MIL 模型的数据输入。
2. 模型训练与运行 (Training & Inference)
本项目在 YOLOv8 和 mil 两个目录中分别提供了完整的网络模型结构与训练脚本。

2.1 YOLOv8 鸡头检测模型
进入 YOLOv8/ 目录，这里包含了用于目标检测的配置文件和脚本。
cd YOLOv8

# 示例：开始训练 YOLOv8 模型
python train.py
2.2 多实例学习 (MIL) 模型
进入 mil/ 目录，这里包含了多实例学习网络的结构定义与训练代码。
cd mil

# 示例：运行 MIL 模型的训练脚本
python train.py
