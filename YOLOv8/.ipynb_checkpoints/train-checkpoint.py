import random
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics import RTDETR
import os
import warnings
warnings.filterwarnings("ignore")




if __name__ == '__main__':
    # 设置随机种子
    seed = 42

    # Python 随机种子
    random.seed(seed)

    # NumPy 随机种子
    np.random.seed(seed)

    # PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 启用确定性卷积（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量（可选）
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 某些 CUDA 版本需要

    # ========== 超参数配置 ==========
    model_name = 'SEAM+SCDown+RepHead'
    batch_size = 16
    learning_rate = 0.01
    epochs = 100
    optimizer_type = 'Adam'
    imgsz = 640
    
    # 根据超参数生成唯一的实验名称
    exp_name = f"{model_name}_bs{batch_size}_imgsz{imgsz}_lr{learning_rate}_opt{optimizer_type}_ep{epochs}"
    
    print(f"实验名称: {exp_name}")
    print("=" * 60)

    model = YOLO(model=r'/root/autodl-tmp/YOLOv8.2/ultralytics/cfg/models/zdy/' + model_name + '.yaml')
    # model = RTDETR(model=r'/root/autodl-tmp/ultralytics-8.3.78/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    
    
    results = model.train(
                data=r'/root/autodl-tmp/YOLOv8.2/ultralytics/cfg/datasets/25_04_01.yaml', 
                imgsz=imgsz,
                epochs=epochs,
                batch=batch_size,
                amp=True,
                workers=16,
                seed=42,
                device='',
                optimizer=optimizer_type,
                #close_mosaic=10,
                resume=False,
                project='runs/train',
                name=exp_name,  # 使用包含超参数的唯一名称
                single_cls=True,
                cos_lr=False,
                cache=False,
                verbose=True,
                mosaic=True,
                #augment=False,
                #augment=custom_augment
                lr0=learning_rate
                )
    
    # 训练完成后，在验证集上评估模型
    print("\n开始评估模型性能...")
    
    # ========== 使用训练结果中的实际保存路径 ==========
    if hasattr(results, 'save_dir'):
        save_dir = results.save_dir
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    else:
        save_dir = os.path.join('runs/train', exp_name)
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    
    print(f"最佳模型路径: {best_model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(best_model_path):
        print(f"警告: 找不到最佳模型文件: {best_model_path}")
        print("将使用最后一个 epoch 的模型...")
        best_model_path = os.path.join(save_dir, 'weights', 'last.pt')
    
    # 加载最佳模型
    best_model = YOLO(best_model_path)
    
    # ========== 获取模型参数量 ==========
    def get_model_params(model):
        """计算模型参数量"""
        total_params = 0
        trainable_params = 0
        
        for param in model.model.parameters():
            params = param.numel()
            total_params += params
            if param.requires_grad:
                trainable_params += params
        
        return total_params, trainable_params
    
    total_params, trainable_params = get_model_params(best_model)
    params_millions = total_params / 1e6  # 转换为百万
    trainable_params_millions = trainable_params / 1e6
    
    print(f"模型参数量: {params_millions:.2f}M (可训练: {trainable_params_millions:.2f}M)")
    
    # ========== 获取GPU内存使用情况 ==========
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # 在验证集上进行评估
    metrics = best_model.val(
        data=r'/root/autodl-tmp/YOLOv8.2/ultralytics/cfg/datasets/25_04_01.yaml',
        imgsz=640,
        batch=batch_size,
        device=''
    )
    
    # 获取GPU内存使用
    gpu_memory_mb = 0
    gpu_memory_peak_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"GPU内存使用: {gpu_memory_mb:.2f} MB (峰值: {gpu_memory_peak_mb:.2f} MB)")
    
    # 获取保存路径
    metrics_file = os.path.join(save_dir, 'metrics.txt')
    
    # ========== 提取所有指标 ==========
    # 基础检测指标
    map50 = metrics.box.map50  # mAP@0.5
    map75 = metrics.box.map75  # mAP@0.75
    map = metrics.box.map  # mAP@0.5:0.95 (平均精度均值)
    
    # 精确率和召回率
    precision = metrics.box.p  # Precision (精确率)
    recall = metrics.box.r  # Recall (召回率)
    
    # 如果是多类别，计算平均值
    if isinstance(precision, (list, np.ndarray)):
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        # 获取每个类别的精确率和召回率
        if len(precision) > 0:
            precision_per_class = precision
            recall_per_class = recall
        else:
            precision_per_class = [precision_avg]
            recall_per_class = [recall_avg]
    else:
        precision_avg = precision
        recall_avg = recall
        precision_per_class = [precision]
        recall_per_class = [recall]
    
    # 计算F1-score
    if precision_avg + recall_avg > 0:
        f1_score = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
    else:
        f1_score = 0
    
    # 边界框相关指标
    # metrics.box 包含所有边界框相关的指标
    box_metrics = metrics.box
    
    # 获取所有类别的mAP
    if hasattr(box_metrics, 'ap_class_index'):
        ap_class_index = box_metrics.ap_class_index
        maps = box_metrics.maps if hasattr(box_metrics, 'maps') else [map]
    else:
        maps = [map]
    
    # ========== 速度和时间指标 ==========
    # 检测时间（包含预处理、推理、后处理）
    if hasattr(metrics, 'speed'):
        preprocess_time = metrics.speed.get('preprocess', 0)  # 预处理时间 (ms)
        inference_time = metrics.speed.get('inference', 0)    # 推理时间 (ms)
        postprocess_time = metrics.speed.get('postprocess', 0)  # 后处理时间 (ms)
        total_time = preprocess_time + inference_time + postprocess_time
        
        # 计算FPS
        fps = 1000.0 / total_time if total_time > 0 else 0
    else:
        preprocess_time = 0
        inference_time = 0
        postprocess_time = 0
        total_time = 0
        fps = 0
    
    # IoU阈值
    iou_threshold = 0.5
    
    # ========== 写入详细的指标文件 ==========
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"模型评估指标报告 - {exp_name}\n")
        f.write("=" * 70 + "\n\n")
        
        # 训练超参数
        f.write("【训练超参数】\n")
        f.write(f"  模型架构: {model_name}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Optimizer: {optimizer_type}\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Random Seed: {seed}\n")
        f.write(f"  Image Size: 640\n\n")
        
        # 模型规模指标
        f.write("【模型规模】\n")
        f.write(f"  总参数量: {total_params:,} ({params_millions:.2f}M)\n")
        f.write(f"  可训练参数: {trainable_params:,} ({trainable_params_millions:.2f}M)\n")
        f.write(f"  不可训练参数: {total_params - trainable_params:,}\n\n")
        
        # 精确率指标
        f.write("【精确率指标 (Precision)】\n")
        f.write(f"  平均精确率: {precision_avg:.4f}\n")
        if len(precision_per_class) > 1:
            for idx, prec in enumerate(precision_per_class):
                f.write(f"  类别 {idx} 精确率: {prec:.4f}\n")
        f.write("\n")
        
        # 召回率指标
        f.write("【召回率指标 (Recall)】\n")
        f.write(f"  平均召回率: {recall_avg:.4f}\n")
        if len(recall_per_class) > 1:
            for idx, rec in enumerate(recall_per_class):
                f.write(f"  类别 {idx} 召回率: {rec:.4f}\n")
        f.write("\n")
        
        # 边界框精确率 (mAP指标)
        f.write("【边界框精确率 / 平均精度均值 (mAP)】\n")
        f.write(f"  mAP@0.5:0.95 (COCO): {map:.4f}\n")
        f.write(f"  mAP@0.5 (VOC): {map50:.4f}\n")
        f.write(f"  mAP@0.75: {map75:.4f}\n")
        if len(maps) > 1:
            for idx, m in enumerate(maps):
                f.write(f"  类别 {idx} mAP: {m:.4f}\n")
        f.write("\n")
        
        # 综合指标
        f.write("【综合指标】\n")
        f.write(f"  F1-Score: {f1_score:.4f}\n")
        f.write(f"  IoU Threshold: {iou_threshold}\n\n")
        
        # 检测时间指标
        f.write("【检测时间】\n")
        f.write(f"  预处理时间: {preprocess_time:.2f} ms\n")
        f.write(f"  推理时间: {inference_time:.2f} ms\n")
        f.write(f"  后处理时间: {postprocess_time:.2f} ms\n")
        f.write(f"  总检测时间: {total_time:.2f} ms\n")
        f.write(f"  FPS (每秒帧数): {fps:.2f}\n\n")
        
        # GPU内存使用
        f.write("【GPU内存使用】\n")
        if torch.cuda.is_available():
            f.write(f"  当前内存使用: {gpu_memory_mb:.2f} MB\n")
            f.write(f"  峰值内存使用: {gpu_memory_peak_mb:.2f} MB\n")
            f.write(f"  GPU设备: {torch.cuda.get_device_name(0)}\n")
        else:
            f.write(f"  未使用GPU (CPU模式)\n")
        f.write("\n")
        
        # 模型路径信息
        f.write("【模型路径】\n")
        f.write(f"  Best Model: {best_model_path}\n")
        f.write(f"  Save Directory: {save_dir}\n\n")
        
        # 性能总结
        f.write("=" * 70 + "\n")
        f.write("【性能总结】\n")
        f.write("=" * 70 + "\n")
        f.write(f"精确率: {precision_avg:.4f} | 召回率: {recall_avg:.4f} | F1: {f1_score:.4f}\n")
        f.write(f"mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map:.4f}\n")
        f.write(f"参数量: {params_millions:.2f}M | FPS: {fps:.2f}\n")
        f.write(f"推理时间: {inference_time:.2f}ms | GPU内存: {gpu_memory_peak_mb:.2f}MB\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("注：以上指标均在验证集上评估得出\n")
        f.write("=" * 70 + "\n")
    
    # ========== 控制台输出总结 ==========
    print(f"\n评估指标已保存到: {metrics_file}")
    print("\n" + "=" * 70)
    print("指标总览:")
    print("=" * 70)
    print(f"  实验名称: {exp_name}")
    print(f"\n  【精确率】")
    print(f"    精确率 (Precision): {precision_avg:.4f}")
    print(f"    召回率 (Recall): {recall_avg:.4f}")
    print(f"    F1-Score: {f1_score:.4f}")
    print(f"\n  【边界框精确率 / 平均精度均值】")
    print(f"    mAP@0.5:0.95: {map:.4f}")
    print(f"    mAP@0.5: {map50:.4f}")
    print(f"    mAP@0.75: {map75:.4f}")
    print(f"\n  【模型规模】")
    print(f"    参数量: {params_millions:.2f}M")
    print(f"    可训练参数: {trainable_params_millions:.2f}M")
    print(f"\n  【检测时间】")
    print(f"    预处理: {preprocess_time:.2f}ms")
    print(f"    推理: {inference_time:.2f}ms")
    print(f"    后处理: {postprocess_time:.2f}ms")
    print(f"    总时间: {total_time:.2f}ms")
    print(f"    FPS: {fps:.2f}")
    print(f"\n  【GPU内存】")
    if torch.cuda.is_available():
        print(f"    当前使用: {gpu_memory_mb:.2f}MB")
        print(f"    峰值使用: {gpu_memory_peak_mb:.2f}MB")
    else:
        print(f"    CPU模式")
    print(f"\n  保存目录: {save_dir}")
    print("=" * 70)