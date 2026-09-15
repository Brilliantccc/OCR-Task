"""
上传OCR模型权重到魔搭 ModelScope
"""
import os
from modelscope.hub.api import HubApi

# ============ 配置 ============
ACCESS_TOKEN = '***REMOVED-LEAKED-TOKEN***'
MODEL_ID = 'Brilliantccc/OCR-CRNN-v2'

WEIGHTS = [
    ('项目1_简单OCR系统/runs/v2/best_model.pth', 'model/best_model.pth'),
    ('项目1_简单OCR系统/runs/v2/last.pth', 'model/last.pth'),
    ('项目1_简单OCR系统/runs/v2/history.json', 'logs/history.json'),
    ('项目1_简单OCR系统/runs/v2/history.csv', 'logs/history.csv'),
    ('项目1_简单OCR系统/model.py', 'code/model.py'),
    ('项目1_简单OCR系统/config.py', 'code/config.py'),
    ('项目1_简单OCR系统/utils.py', 'code/utils.py'),
]

README_PATH = '项目1_简单OCR系统/runs/README.md'
# ==============================

def main():
    api = HubApi()
    api.login(ACCESS_TOKEN)

    base = os.path.dirname(os.path.abspath(__file__))

    # 上传README
    readme_full = os.path.join(base, README_PATH)
    if os.path.exists(readme_full):
        print(f'[UPLOAD] README.md')
        api.upload_file(
            repo_id=MODEL_ID,
            path_or_fileobj=readme_full,
            path_in_repo='README.md',
        )
        print(f'[OK] README.md 上传成功')

    # 上传权重和代码
    for local_path, repo_path in WEIGHTS:
        full_path = os.path.join(base, local_path)
        if not os.path.exists(full_path):
            print(f'[SKIP] 文件不存在: {full_path}')
            continue
        size_mb = os.path.getsize(full_path) / 1024 / 1024
        print(f'[UPLOAD] {local_path} -> {repo_path} ({size_mb:.1f} MB)')
        api.upload_file(
            repo_id=MODEL_ID,
            path_or_fileobj=full_path,
            path_in_repo=repo_path,
        )
        print(f'[OK] {repo_path} 上传成功')

    print('\n全部上传完成!')
    print(f'模型页面: https://modelscope.cn/models/{MODEL_ID}')

if __name__ == '__main__':
    main()
