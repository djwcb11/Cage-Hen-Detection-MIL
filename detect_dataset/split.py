"""
YOLO 数据集划分脚本
按 8:1:1 的比例将数据集划分为 train / val / test
"""

import os
import random
import shutil
from pathlib import Path


# ==================== 配置参数 ====================
SOURCE_DIR = "source_dataset"           # 原始数据集路径
OUTPUT_DIR = "split_dataset"            # 输出数据集路径
TRAIN_RATIO = 0.8                       # 训练集比例
VAL_RATIO = 0.1                         # 验证集比例
TEST_RATIO = 0.1                        # 测试集比例
RANDOM_SEED = 42                        # 随机种子，保证划分可复现
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图片格式
# ================================================


def split_dataset():
    # 固定随机种子，保证每次划分结果一致
    random.seed(RANDOM_SEED)

    src_img_dir = Path(SOURCE_DIR) / "images"
    src_lbl_dir = Path(SOURCE_DIR) / "labels"

    # 1. 获取所有图片文件名（不带后缀）
    img_files = [
        f for f in os.listdir(src_img_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    ]
    print(f"共找到 {len(img_files)} 张图片")

    # 2. 打乱顺序
    random.shuffle(img_files)

    # 3. 计算划分数量
    total = len(img_files)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    # test 用剩余部分，避免因取整丢样本
    n_test = total - n_train - n_val

    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val:]

    print(f"训练集: {len(train_files)}  验证集: {len(val_files)}  测试集: {len(test_files)}")

    # 4. 创建输出目录
    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    for split in splits:
        (Path(OUTPUT_DIR) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. 复制文件
    for split, files in splits.items():
        for img_name in files:
            stem = Path(img_name).stem  # 不带后缀的文件名
            lbl_name = stem + ".txt"

            src_img = src_img_dir / img_name
            src_lbl = src_lbl_dir / lbl_name

            dst_img = Path(OUTPUT_DIR) / "images" / split / img_name
            dst_lbl = Path(OUTPUT_DIR) / "labels" / split / lbl_name

            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"警告: {img_name} 找不到对应标注文件 {lbl_name}")

    # 6. 生成 data.yaml （供 YOLOv8 训练直接使用）
    yaml_content = f"""# YOLO dataset config
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test

# 类别数和类别名称 (请根据实际情况修改)
nc: 1
names: ['hen_head']
"""
    with open(Path(OUTPUT_DIR) / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n划分完成！输出目录: {OUTPUT_DIR}")
    print(f"data.yaml 已生成，可直接用于 YOLOv8 训练。")


if __name__ == "__main__":
    split_dataset()