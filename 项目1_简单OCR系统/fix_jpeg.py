"""
修复JPEG文件：重新编码去除多余的字节
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

def fix_jpeg_files(src_dir, dst_dir=None):
    """
    重新编码所有JPEG文件

    Args:
        src_dir: 源目录
        dst_dir: 目标目录（如果为None，则覆盖源文件）
    """
    if dst_dir is None:
        dst_dir = src_dir

    os.makedirs(dst_dir, exist_ok=True)

    # 获取所有JPEG文件
    jpeg_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

    print(f"找到 {len(jpeg_files)} 个JPEG文件")

    # 编码参数
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    fixed_count = 0
    for filename in tqdm(jpeg_files, desc="修复JPEG文件"):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 读取图像
        with open(src_path, 'rb') as f:
            data = f.read()

        img_bytes = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print(f"无法读取: {filename}")
            continue

        # 重新编码
        success, encoded = cv2.imencode('.jpg', img, encode_params)

        if success:
            # 保存
            with open(dst_path, 'wb') as f:
                f.write(encoded.tobytes())
            fixed_count += 1
        else:
            print(f"编码失败: {filename}")

    print(f"成功修复: {fixed_count} / {len(jpeg_files)}")

    return fixed_count


if __name__ == "__main__":
    src_dir = "data/ICDAR-2019-SROIE/img"
    dst_dir = "data/ICDAR-2019-SROIE/img_fixed"

    print("开始修复JPEG文件...")
    fix_jpeg_files(src_dir, dst_dir)
    print("修复完成!")

    # 测试修复后的文件
    print("\n测试修复后的文件...")
    test_files = ['000.jpg', '001.jpg', '002.jpg']
    for f in test_files:
        path = os.path.join(dst_dir, f)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"  {f}: {img.shape} - OK")
            else:
                print(f"  {f}: 读取失败")
