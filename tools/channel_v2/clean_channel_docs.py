
import os, shutil

# Create archive directory
archive_dir = '.archive/historical_channel_models'
os.makedirs(archive_dir, exist_ok=True)

files_to_clean = [
    'docs/信道模型草稿_系统与硬件设定_20261005.md',
    'docs/信道速率模型_学术汇报版_论文引用版_检查说明.md',
    'docs/论文草稿_第2节_系统与信道模型.md',
    'docs/速率模型v2公式依据核查_20260930.md',
    'docs/速率模型v2文献对比_20260929.md',
    'docs/速率模型v2验证_20260929.md',
    'Ddestopwork_spacework_for_reseachqkd_rl.md',
    'rate-model-v2.md'
]

cleaned = []
for f in files_to_clean:
    if os.path.exists(f):
        dest = os.path.join(archive_dir, os.path.basename(f))
        shutil.move(f, dest)
        cleaned.append(f)
        print(f'Archived & cleaned: {f} -> {dest}')

print(f'Total cleaned files: {len(cleaned)}')
