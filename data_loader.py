"""
WM-811K 数据集预处理脚本
功能：
1. 解析 LSWMD.pkl，提取有标签数据
2. 对 None 类下采样
3. 划分 train / val / test 集
4. 将晶圆图矩阵转为 3 通道 PNG 图像，按 ImageFolder 格式保存
"""
import os
import pickle
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil

# ============================================================
# 自定义 Unpickler（兼容旧版 pandas）
# ============================================================
import pandas as pd

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes', 1)
        return super().find_class(module, name)


# ============================================================
# 配置区
# ============================================================
PKL_PATH = os.path.join(os.path.dirname(__file__), "archive", "LSWMD.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# 类别映射
LABEL_MAP = {
    'Center': 0,
    'Donut': 1,
    'Edge-Loc': 2,
    'Edge-Ring': 3,
    'Loc': 4,
    'Random': 5,
    'Scratch': 6,
    'Near-full': 7,
    'none': 8,
}

# 输出图像尺寸
IMG_SIZE = 64  # 64x64，也可以改为 96

# None 类下采样数量（None 原始有 147,431 张，保留约 20,000）
NONE_MAX_SAMPLES = 20000

# 划分比例（train / val / test）
# 使用现成的 WM-811K_semiconductor_wafer_map_pattern_classified 划分索引，
# 如果没有则自动随机划分
USE_EXTERNAL_SPLIT = False  # 设为 True 使用 external CSV 划分

SPLIT_RATIO = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
}

# ============================================================
# 主流程
# ============================================================
def load_pkl(pkl_path):
    print(f"正在加载 {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        return CompatUnpickler(f, encoding='latin1').load()


def extract_label(x):
    """从嵌套 ndarray 中提取标签字符串"""
    if isinstance(x, np.ndarray) and x.size > 0:
        return x[0][0]
    return 'unknown'


def filter_and_balance(df):
    """过滤 unknown 标签 + 对 None 类下采样"""
    print(f"\n原始数据: {len(df)} 条")

    # 提取标签
    df['label_str'] = df['failureType'].apply(extract_label)

    # 过滤 unknown
    labeled = df[df['label_str'] != 'unknown'].copy()
    print(f"有标签数据: {len(labeled)} 条")
    print("类别分布:")
    print(labeled['label_str'].value_counts())

    # 对 None 类下采样
    none_mask = labeled['label_str'] == 'none'
    none_count = none_mask.sum()
    if none_count > NONE_MAX_SAMPLES:
        print(f"\nNone 类过多 ({none_count} 张)，下采样至 {NONE_MAX_SAMPLES} 张...")
        none_indices = labeled[none_mask].index
        drop_indices = np.random.choice(none_indices, none_count - NONE_MAX_SAMPLES, replace=False)
        labeled = labeled.drop(drop_indices)
        print(f"下采样后 None 类: {NONE_MAX_SAMPLES} 张")

    # 映射为数字标签
    labeled['label'] = labeled['label_str'].map(LABEL_MAP)
    # 过滤掉未知类别（理论上不应出现）
    labeled = labeled.dropna(subset=['label'])
    labeled['label'] = labeled['label'].astype(int)

    print(f"\n最终用于训练的数据: {len(labeled)} 条")
    print("最终类别分布:")
    print(labeled['label_str'].value_counts().sort_index())

    return labeled


def split_data(labeled_df):
    """划分 train / val / test"""
    print(f"\n正在划分数据集...")

    # 按类别分层划分
    train_parts, val_parts, test_parts = [], [], []

    for label_str, group in labeled_df.groupby('label_str'):
        n = len(group)
        indices = np.random.permutation(n)

        n_train = int(n * SPLIT_RATIO['train'])
        n_val = int(n * SPLIT_RATIO['val'])

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_parts.append(group.iloc[train_idx])
        val_parts.append(group.iloc[val_idx])
        test_parts.append(group.iloc[test_idx])

    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)
    test_df = pd.concat(test_parts)

    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")

    return train_df, val_df, test_df


def wafer_map_to_image(wm_array, img_size=64):
    """
    将晶圆图矩阵转为 3 通道 PIL Image
    wm_array: 原始 2D uint8 矩阵 (值: 0=无die, 1=pass, 2=fail)
    """
    # 缩放到目标尺寸
    wm_resized = np.array(Image.fromarray(wm_array).resize((img_size, img_size), Image.NEAREST))

    # 归一化到 0-255: 0→0(黑), 1→128(灰), 2→255(白)
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    img[wm_resized == 1] = 128
    img[wm_resized == 2] = 255

    # 转为 3 通道（适配 ResNet）
    img_rgb = np.stack([img, img, img], axis=-1)
    return Image.fromarray(img_rgb, mode='RGB')


def save_images(df, split_name, output_dir):
    """将 DataFrame 中的晶圆图保存为 PNG，按 ImageFolder 格式组织"""
    split_dir = os.path.join(output_dir, split_name)

    # 按类别创建子目录
    for label_str, label_id in LABEL_MAP.items():
        os.makedirs(os.path.join(split_dir, str(label_id)), exist_ok=True)

    print(f"\n正在保存 {split_name} 集图像到 {split_dir} ...")

    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wm = row['waferMap']
        if not isinstance(wm, np.ndarray) or wm.size == 0:
            continue

        label_id = int(row['label'])
        img = wafer_map_to_image(wm, IMG_SIZE)

        img_path = os.path.join(split_dir, str(label_id), f"{idx:07d}.png")
        img.save(img_path)
        count += 1

    print(f"  {split_name} 集保存了 {count} 张图像")
    return count


def save_label_map(output_dir):
    """保存类别映射 JSON"""
    import json
    # 数字 → 类别名（反向映射）
    cla_dict = {str(v): k for k, v in LABEL_MAP.items()}
    json_path = os.path.join(output_dir, "class_indices.json")
    with open(json_path, 'w') as f:
        json.dump(cla_dict, f, indent=4)
    print(f"\n类别映射已保存到 {json_path}")
    print(f"  {cla_dict}")


def main():
    print("=" * 60)
    print("WM-811K 数据集预处理")
    print("=" * 60)

    # 1. 加载数据
    df = load_pkl(PKL_PATH)
    print(f"加载完成, DataFrame shape: {df.shape}")

    # 2. 过滤 + 平衡
    labeled_df = filter_and_balance(df)

    # 3. 划分数据集
    train_df, val_df, test_df = split_data(labeled_df)

    # 4. 保存图像
    if os.path.exists(OUTPUT_DIR):
        print(f"\n删除旧数据集目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    total = 0
    total += save_images(train_df, 'train', OUTPUT_DIR)
    total += save_images(val_df, 'val', OUTPUT_DIR)
    total += save_images(test_df, 'test', OUTPUT_DIR)

    # 5. 保存类别映射
    save_label_map(OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"预处理完成！共保存 {total} 张图像到 {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # 打印目录结构示例
    print(f"\n目录结构示例:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(split_path):
            dirs = os.listdir(split_path)
            print(f"  {OUTPUT_DIR}/{split}/ → {len(dirs)} 个类别目录")


if __name__ == "__main__":
    main()
