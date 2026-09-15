"""
上传表格识别模型到魔搭 ModelScope（单仓库，两个模型）
仓库: Brilliantccc/table-detection-and-structure
"""
import os
from modelscope.hub.api import HubApi

# ============ 配置 ============
ACCESS_TOKEN = '***REMOVED-LEAKED-TOKEN***'
REPO_ID = 'Brilliantccc/table-detection-and-structure'

FILES = [
    # 检测模型
    ('项目2_表格识别工具/runs/detection/best_model.pth', 'detection/best_model.pth'),
    # 结构识别模型
    ('项目2_表格识别工具/runs/structure/best_model.pth', 'structure/best_model.pth'),
    # 代码
    ('项目2_表格识别工具/model.py', 'code/model.py'),
    ('项目2_表格识别工具/config.py', 'code/config.py'),
    ('项目2_表格识别工具/utils.py', 'code/utils.py'),
    ('项目2_表格识别工具/dataset.py', 'code/dataset.py'),
    ('项目2_表格识别工具/inference.py', 'code/inference.py'),
    ('项目2_表格识别工具/evaluate.py', 'code/evaluate.py'),
    ('项目2_表格识别工具/requirements.txt', 'code/requirements.txt'),
]

README_PATH = '项目2_表格识别工具/runs/README.md'
# ==============================


def main():
    api = HubApi()
    api.login(ACCESS_TOKEN)
    base = os.path.dirname(os.path.abspath(__file__))

    # 创建仓库
    try:
        api.create_repo(repo_id=REPO_ID, exist_ok=True)
        print(f'仓库就绪: {REPO_ID}')
    except Exception as e:
        print(f'仓库创建提示: {e}')

    # 上传 README
    readme_path = os.path.join(base, README_PATH)
    if os.path.exists(readme_path):
        print(f'[UPLOAD] README.md')
        api.upload_file(repo_id=REPO_ID, path_or_fileobj=readme_path, path_in_repo='README.md')
        print(f'[OK] README.md')

    # 上传所有文件
    for local_path, repo_path in FILES:
        full_path = os.path.join(base, local_path)
        if not os.path.exists(full_path):
            print(f'[SKIP] 不存在: {local_path}')
            continue
        size_mb = os.path.getsize(full_path) / 1024 / 1024
        print(f'[UPLOAD] {repo_path} ({size_mb:.1f} MB)...')
        api.upload_file(repo_id=REPO_ID, path_or_fileobj=full_path, path_in_repo=repo_path)
        print(f'[OK] {repo_path}')

    print(f'\n全部上传完成!')
    print(f'模型页面: https://modelscope.cn/models/{REPO_ID}')


if __name__ == '__main__':
    main()
