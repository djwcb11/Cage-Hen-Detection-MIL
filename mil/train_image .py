# protomil_train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import argparse
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from torchvision import models, transforms
from PIL import Image


# ============ ResNet 特征提取器 ============

class ResNetFeatureExtractor(nn.Module):
    """使用预训练的ResNet作为特征提取器"""
    
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=2048):
        super().__init__()
        
        # 加载预训练的ResNet
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet152':
            resnet = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 移除最后的全连接层，只保留特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 如果需要调整特征维度
        if feature_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, feature_dim)
            self.output_dim = feature_dim
        else:
            self.projection = None
            self.output_dim = self.feature_dim
        
        # 冻结参数（可选，如果想要fine-tune可以设置为False）
        self.freeze_backbone = True
        if self.freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 图像张量
        Returns:
            features: [B, feature_dim] 特征向量
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.features(x)  # [B, feature_dim, 1, 1]
            features = features.view(features.size(0), -1)  # [B, feature_dim]
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features


# ============ ProtoMIL 模型定义 ============

class ProtoMIL_GlobalPrototype(nn.Module):
    """基于原型和余弦相似度的注意力 MIL 模型（集成ResNet特征提取）"""

    def __init__(self, resnet_model='resnet50', instance_dim=512, hidden_dim=256, 
                 num_classes=2, use_attention=True, pretrained=True):
        super().__init__()
        self.use_attention = use_attention
        self.instance_dim = instance_dim
        self.num_classes = num_classes

        # ResNet特征提取器
        self.feature_extractor = ResNetFeatureExtractor(
            model_name=resnet_model, 
            pretrained=pretrained,
            feature_dim=instance_dim
        )

        if use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(instance_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # 全局原型（将在训练时设置）
        self.register_buffer('global_prototypes', torch.zeros(num_classes, instance_dim))
        self.prototypes_initialized = False

    def extract_bag_features(self, images):
        """
        从一个bag的图像中提取特征
        Args:
            images: [N, 3, H, W] 一个bag中的所有图像
        Returns:
            features: [N, instance_dim] 特征向量
        """
        return self.feature_extractor(images)

    def set_global_prototypes(self, global_proto_tensor):
        """
        设置全局原型（不参与反向传播）
        Args:
            global_proto_tensor: [num_classes, D]
        """
        if global_proto_tensor.shape[0] != self.num_classes:
            raise ValueError(f"原型数量不匹配: 期望{self.num_classes}, 得到{global_proto_tensor.shape[0]}")
        if global_proto_tensor.shape[1] != self.instance_dim:
            raise ValueError(f"原型维度不匹配: 期望{self.instance_dim}, 得到{global_proto_tensor.shape[1]}")

        self.global_prototypes.copy_(global_proto_tensor.detach())
        self.prototypes_initialized = True

    def compute_prototype(self, feats):
        """
        计算bag的原型
        Args:
            feats: [B, N, D] 或 [N, D]
        Returns:
            proto: [B, D] 或 [D]
            attn_weights: [B, N] or [N] or None
        """
        # 处理单个bag的情况
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # [1, N, D]
            squeeze_output = True
        else:
            squeeze_output = False
        
        if self.use_attention:
            # 计算注意力权重
            attn_scores = self.attention_layer(feats)  # [B, N, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N, 1]

            # 加权聚合
            proto = (attn_weights * feats).sum(dim=1)  # [B, D]

            attn_weights = attn_weights.squeeze(-1)  # [B, N]
        else:
            # 简单平均
            proto = feats.mean(dim=1)  # [B, D]
            attn_weights = None
        
        if squeeze_output:
            proto = proto.squeeze(0)  # [D]
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)  # [N]
        
        return proto, attn_weights

    def compute_similarity(self, proto):
        """
        使用余弦相似度计算与全局原型的相似度
        Args:
            proto: [B, D] 或 [D] bag原型
        Returns:
            similarity: [B, num_classes] 或 [num_classes] 相似度得分
        """
        if not self.prototypes_initialized:
            raise ValueError("全局原型尚未初始化！请先调用 set_global_prototypes()")

        # 处理单个原型的情况
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)  # [1, D]
            squeeze_output = True
        else:
            squeeze_output = False

        # L2归一化
        proto_norm = F.normalize(proto, p=2, dim=1)  # [B, D]
        class_proto_norm = F.normalize(self.global_prototypes, p=2, dim=1)  # [num_classes, D]

        # 余弦相似度 (等价于归一化后的点积)
        similarity = torch.matmul(proto_norm, class_proto_norm.t())  # [B, num_classes]

        # 缩放相似度以便更好地训练（可选）
        similarity = similarity * 10.0  # 温度参数

        if squeeze_output:
            similarity = similarity.squeeze(0)  # [num_classes]

        return similarity

    def forward(self, images):
        """
        前向传播
        Args:
            images: [B, N, 3, H, W] 批次的bags，每个bag包含N个图像
                   或 [N, 3, H, W] 单个bag的N个图像
        Returns:
            logits: [B, num_classes] 或 [num_classes]
            attn_weights: [B, N] or [N] or None
        """
        # 判断是批次输入还是单个bag输入
        if images.dim() == 5:  # [B, N, 3, H, W]
            batch_mode = True
            B, N, C, H, W = images.shape
            # 合并batch和instance维度
            images_flat = images.view(B * N, C, H, W)
        elif images.dim() == 4:  # [N, 3, H, W]
            batch_mode = False
            N, C, H, W = images.shape
            images_flat = images
        else:
            raise ValueError(f"输入维度错误: 期望4或5维，得到{images.dim()}维")
        
        # 提取特征
        features = self.feature_extractor(images_flat)  # [B*N, D] 或 [N, D]
        
        if batch_mode:
            # 恢复batch维度
            features = features.view(B, N, -1)  # [B, N, D]
        
        # 计算bag原型
        proto, attn_weights = self.compute_prototype(features)

        # 计算与全局原型的相似度
        logits = self.compute_similarity(proto)

        return logits, attn_weights


# ============ 数据集定义 ============

class MILHenDataset(Dataset):
    """
    从图像目录加载MIL数据集
    目录结构:
    root_dir/
        normal_hen/
            bag_001/
                image_001.jpg
                image_002.jpg
                ...
            bag_002/
                ...
        spent_hen/
            bag_001/
                image_001.jpg
                ...
    """
    
    def __init__(self, root_dir, transform=None, max_instances=None, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            max_instances: 每个bag最多使用的instance数量（None表示全部使用）
            image_extensions: 支持的图像文件扩展名
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_instances = max_instances
        self.image_extensions = image_extensions
        self.samples = []

        # 默认的图像变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

        # 扫描数据集
        for label_dir, label in [('normal_hen', 0), ('spent_hen', 1)]:
            dir_path = os.path.join(root_dir, label_dir)
            if not os.path.exists(dir_path):
                print(f"⚠️ 警告: 目录不存在: {dir_path}")
                continue

            # 获取所有bag目录
            bag_dirs = [d for d in os.listdir(dir_path) 
                       if os.path.isdir(os.path.join(dir_path, d))]
            
            print(f"📁 {label_dir}: 找到 {len(bag_dirs)} 个bag目录")

            for bag_dir in bag_dirs:
                bag_path = os.path.join(dir_path, bag_dir)
                
                # 获取bag中的所有图像文件
                image_files = [f for f in os.listdir(bag_path) 
                             if f.lower().endswith(self.image_extensions)]
                
                if len(image_files) == 0:
                    print(f"⚠️ 警告: bag {bag_path} 中没有图像文件")
                    continue
                
                # 完整路径
                image_paths = [os.path.join(bag_path, f) for f in sorted(image_files)]
                
                # 限制instance数量
                if self.max_instances is not None and len(image_paths) > self.max_instances:
                    image_paths = image_paths[:self.max_instances]
                
                self.samples.append((image_paths, label))

        print(f"✅ 数据集初始化完成: 共 {len(self.samples)} 个bags")
        
        # 统计信息
        num_instances = [len(paths) for paths, _ in self.samples]
        print(f"📊 每个bag的图像数量统计:")
        print(f"  - 最小: {min(num_instances) if num_instances else 0}")
        print(f"  - 最大: {max(num_instances) if num_instances else 0}")
        print(f"  - 平均: {np.mean(num_instances) if num_instances else 0:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]

        try:
            # 加载所有图像
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            
            # 堆叠成tensor: [N, C, H, W]
            images_tensor = torch.stack(images)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return images_tensor, label_tensor

        except Exception as e:
            print(f"❌ 错误: 加载bag失败 {image_paths[0]}: {e}")
            raise


def protomil_collate_fn(batch):
    """
    自定义collate函数，处理不同大小的bags
    Args:
        batch: list of (images, label), where images is [N_i, C, H, W]
    Returns:
        images: [B, max_N, C, H, W] 填充后的图像tensor
        labels: [B] 标签tensor
        lengths: [B] 每个bag的实际图像数量
    """
    images_list, labels = zip(*batch)
    
    # 获取最大的instance数量
    max_instances = max(imgs.size(0) for imgs in images_list)
    B = len(images_list)
    C, H, W = images_list[0].size()[1:]
    
    # 创建填充后的tensor
    padded_images = torch.zeros(B, max_instances, C, H, W)
    lengths = []
    
    for i, imgs in enumerate(images_list):
        N = imgs.size(0)
        padded_images[i, :N] = imgs
        lengths.append(N)
    
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    
    return padded_images, labels, lengths


# ============ 指标计算函数 ============

def compute_metrics(y_true, y_pred, y_prob):
    """
    计算分类指标
    Args:
        y_true: 真实标签 (numpy array)
        y_pred: 预测标签 (numpy array)
        y_prob: 预测概率 (numpy array, shape=[N, num_classes])
    Returns:
        metrics: dict 包含 acc, f1, auc
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 计算AUC（二分类）
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = 0.0

    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc
    }


# ============ 原型计算函数 ============

def compute_global_prototypes(model, loader, device, num_classes=2):
    """
    计算每个类的全局原型
    Args:
        model: ProtoMIL模型
        loader: DataLoader
        device: 计算设备
        num_classes: 类别数
    Returns:
        global_prototypes: [num_classes, D]
    """
    print(f"\n🔧 开始计算全局原型...")

    model.eval()

    # 存储每个类的所有原型
    class_prototypes = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (images, labels, lengths) in enumerate(tqdm(loader, desc="计算原型")):
            images, labels = images.to(device), labels.to(device)

            # 处理每个bag
            for i in range(images.size(0)):
                bag_images = images[i, :lengths[i]]  # [N, C, H, W]
                bag_label = labels[i].item()
                
                # 提取特征
                features = model.feature_extractor(bag_images)  # [N, D]
                
                # 计算原型
                proto, _ = model.compute_prototype(features)  # [D]
                
                class_prototypes[bag_label].append(proto)
                class_counts[bag_label] += 1

    # 打印每个类的样本数
    print(f"类别样本分布:")
    for class_id in range(num_classes):
        print(f"  - 类别 {class_id}: {class_counts[class_id]} 个样本")

    # 计算每个类的平均原型
    global_prototypes = []
    for class_id in range(num_classes):
        if len(class_prototypes[class_id]) > 0:
            class_proto = torch.stack(class_prototypes[class_id]).mean(dim=0)
            global_prototypes.append(class_proto)
        else:
            # 如果某个类没有样本，使用随机初始化
            print(f"⚠️ 警告: 类别 {class_id} 没有样本，使用随机初始化")
            random_proto = torch.randn(model.instance_dim).to(device)
            global_prototypes.append(random_proto)

    global_prototypes = torch.stack(global_prototypes)  # [num_classes, D]

    print(f"✅ 全局原型计算完成: {global_prototypes.shape}\n")

    return global_prototypes


# ============ 训练与验证函数 ============

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0

    all_labels = []
    all_preds = []
    all_probs = []

    loop = tqdm(loader, desc=f"Training Epoch {epoch}")
    for step, (images, labels, lengths) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        try:
            # 处理每个bag（因为bag大小不同）
            batch_logits = []
            batch_labels = []
            
            for i in range(images.size(0)):
                bag_images = images[i, :lengths[i]]  # [N, C, H, W]
                bag_label = labels[i]
                
                # 前向传播
                logits, attn_weights = model(bag_images)  # [num_classes]
                
                batch_logits.append(logits)
                batch_labels.append(bag_label)
            
            # 堆叠成batch
            batch_logits = torch.stack(batch_logits)  # [B, num_classes]
            batch_labels = torch.stack(batch_labels)  # [B]

            # 计算损失
            loss = criterion(batch_logits, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()

            probs = F.softmax(batch_logits, dim=1)
            preds = torch.argmax(batch_logits, dim=1)

            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            loop.set_postfix(loss=loss.item())

            # 记录到 TensorBoard
            global_step = epoch * len(loader) + step
            writer.add_scalar("Train/StepLoss", loss.item(), global_step)

        except Exception as e:
            print(f"❌ 训练错误 (Epoch {epoch}, Step {step}): {e}")
            import traceback
            traceback.print_exc()
            raise

    avg_loss = total_loss / len(loader)

    # 计算指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", metrics['accuracy'], epoch)
    writer.add_scalar("Train/F1Score", metrics['f1_score'], epoch)
    writer.add_scalar("Train/AUC", metrics['auc'], epoch)

    return avg_loss, metrics


def evaluate(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Evaluating Epoch {epoch}")
        for step, (images, labels, lengths) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)

            try:
                # 处理每个bag
                batch_logits = []
                batch_labels = []
                
                for i in range(images.size(0)):
                    bag_images = images[i, :lengths[i]]  # [N, C, H, W]
                    bag_label = labels[i]
                    
                    # 前向传播
                    logits, attn_weights = model(bag_images)  # [num_classes]
                    
                    batch_logits.append(logits)
                    batch_labels.append(bag_label)
                
                # 堆叠成batch
                batch_logits = torch.stack(batch_logits)  # [B, num_classes]
                batch_labels = torch.stack(batch_labels)  # [B]

                # 计算损失
                loss = criterion(batch_logits, batch_labels)
                total_loss += loss.item()

                # 收集预测结果
                probs = F.softmax(batch_logits, dim=1)
                preds = torch.argmax(batch_logits, dim=1)

                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                loop.set_postfix(loss=loss.item())

            except Exception as e:
                print(f"❌ 验证错误 (Epoch {epoch}, Step {step}): {e}")
                import traceback
                traceback.print_exc()
                raise

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    # 计算指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", metrics['accuracy'], epoch)
    writer.add_scalar("Val/F1Score", metrics['f1_score'], epoch)
    writer.add_scalar("Val/AUC", metrics['auc'], epoch)

    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n[Eval] Epoch {epoch} 详细结果:")
    print(f"  - Loss: {avg_loss:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    print(f"  - AUC: {metrics['auc']:.4f}")
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Spent']))

    return avg_loss, metrics


# ============ 主函数 ============

def main(train_dir, val_dir, log_dir='runs/protomil_experiment', 
         resnet_model='resnet50', feature_dim=512, batch_size=8, 
         epochs=50, lr=1e-4, update_proto_freq=5, max_instances=None):
    print(f"\n{'=' * 70}")
    print(f"🚀 开始训练 ProtoMIL_GlobalPrototype 模型 (图像版本)")
    print(f"{'=' * 70}")
    print(f"📝 配置参数:")
    print(f"  - 训练集目录: {train_dir}")
    print(f"  - 验证集目录: {val_dir}")
    print(f"  - 日志目录: {log_dir}")
    print(f"  - ResNet模型: {resnet_model}")
    print(f"  - 特征维度: {feature_dim}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - 原型更新频率: 每 {update_proto_freq} epochs")
    print(f"  - 最大Instance数: {max_instances if max_instances else '无限制'}")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

    # 数据加载
    print("📊 加载数据集...")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MILHenDataset(train_dir, transform=train_transform, max_instances=max_instances)
    val_dataset = MILHenDataset(val_dir, transform=val_transform, max_instances=max_instances)

    print(f"✅ 训练集样本数: {len(train_dataset)}")
    print(f"✅ 验证集样本数: {len(val_dataset)}\n")

    if len(train_dataset) == 0:
        raise ValueError("训练集为空，请检查数据路径！")
    if len(val_dataset) == 0:
        raise ValueError("验证集为空，请检查数据路径！")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=protomil_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=protomil_collate_fn, num_workers=4)

    print(f"✅ DataLoader 创建完成")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 验证批次数: {len(val_loader)}\n")

    # 模型构建
    print("🔧 构建模型...")
    model = ProtoMIL_GlobalPrototype(
        resnet_model=resnet_model,
        instance_dim=feature_dim,
        num_classes=2,
        use_attention=True,
        pretrained=True
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}\n")

    # 初始化全局原型
    print("🔧 初始化全局原型...")
    global_prototypes = compute_global_prototypes(model, train_loader, device, num_classes=2)
    model.set_global_prototypes(global_prototypes)
    print(f"✅ 全局原型初始化完成\n")

    # 优化器 & 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"✅ 优化器: Adam (lr={lr})")
    print(f"✅ 损失函数: CrossEntropyLoss\n")

    # 日志记录器
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"📝 TensorBoard 日志保存到: {log_dir}\n")

    # 训练循环
    print(f"{'=' * 70}")
    print(f"🎯 开始训练")
    print(f"{'=' * 70}\n")

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'=' * 70}")

        # 定期更新全局原型
        if epoch > 1 and epoch % update_proto_freq == 0:
            print(f"🔄 更新全局原型...")
            global_prototypes = compute_global_prototypes(model, train_loader, device, num_classes=2)
            model.set_global_prototypes(global_prototypes)

        # 训练阶段
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, writer, epoch
        )

        # 验证阶段
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, writer, epoch
        )

        print(f"\n📊 Epoch {epoch} 总结:")
        print(f"  训练集:")
        print(f"    - Loss: {train_loss:.4f}")
        print(f"    - Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"    - F1 Score: {train_metrics['f1_score']:.4f}")
        print(f"    - AUC: {train_metrics['auc']:.4f}")
        print(f"  验证集:")
        print(f"    - Loss: {val_loss:.4f}")
        print(f"    - Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    - F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"    - AUC: {val_metrics['auc']:.4f}")

        # 保存最佳模型（基于F1分数）
        if val_metrics['f1_score'] > best_val_f1:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1_score']
            best_val_auc = val_metrics['auc']
            best_epoch = epoch

            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = "checkpoints/protomil_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_prototypes': model.global_prototypes,
                'val_metrics': val_metrics,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 保存最佳模型到 {checkpoint_path} (F1: {val_metrics['f1_score']:.4f})")

        print(f"  当前最佳 (Epoch {best_epoch}):")
        print(f"    - Accuracy: {best_val_acc:.4f}")
        print(f"    - F1 Score: {best_val_f1:.4f}")
        print(f"    - AUC: {best_val_auc:.4f}")

    # 保存最终模型
    final_path = "protomil_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_prototypes': model.global_prototypes,
    }, final_path)
    print(f"\n💾 保存最终模型到 {final_path}")

    writer.close()

    print(f"\n{'=' * 70}")
    print(f"✅ ProtoMIL 模型训练完成！")
    print(f"{'=' * 70}")
    print(f"📊 最佳结果 (Epoch {best_epoch}):")
    print(f"  - Accuracy: {best_val_acc:.4f}")
    print(f"  - F1 Score: {best_val_f1:.4f}")
    print(f"  - AUC: {best_val_auc:.4f}")
    print(f"  - 模型保存位置: checkpoints/protomil_best.pth")
    print(f"  - TensorBoard日志: {log_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProtoMIL_GlobalPrototype Model with Images")
    parser.add_argument("--train_dir", type=str, default="/root/autodl-tmp/mil_image_dataset/meta_train",
                        help="Path to training dataset directory")
    parser.add_argument("--val_dir", type=str, default="/root/autodl-tmp/mil_image_dataset/meta_val",
                        help="Path to validation dataset directory")
    parser.add_argument("--log_dir", type=str, default="runs/protomil_image_experiment",
                        help="Path to TensorBoard log directory")
    parser.add_argument("--resnet_model", type=str, default="resnet18",
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help="ResNet model to use for feature extraction")
    parser.add_argument("--feature_dim", type=int, default=512,
                        help="Dimension of output features")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--update_proto_freq", type=int, default=1,
                        help="Frequency of updating global prototypes (in epochs)")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Maximum number of instances per bag (None for unlimited)")

    args = parser.parse_args()
    main(args.train_dir, args.val_dir, args.log_dir, args.resnet_model,
         args.feature_dim, args.batch_size, args.epochs, args.lr, 
         args.update_proto_freq, args.max_instances)