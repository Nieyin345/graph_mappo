
import os, shutil

# Create archive directory for obsolete paper drafts
archive_dir = '.archive/historical_paper_drafts'
os.makedirs(archive_dir, exist_ok=True)

# List of obsolete drafts to archive (v1, v2, v3, v4, ancient drafts)
drafts_to_archive = [
    'docs/论文草稿_20260927.md',
    'docs/论文初稿_中文_20260930.md',
    'docs/论文初稿_中文_v2_20260930.md',
    'docs/论文初稿_中文_v3_20260930.md',
    'docs/论文初稿_中文_v4_20260930.md',
    'docs/论文大纲_20260930.md',
    'docs/论文主线与参数_20260929.md',
    'docs/论文实验计划_20260927.md',
    'docs/论文第四章草稿_20261003.md'
]

archived = []
for f in drafts_to_archive:
    if os.path.exists(f):
        dest = os.path.join(archive_dir, os.path.basename(f))
        shutil.move(f, dest)
        archived.append(f)
        print(f'Archived: {f} -> {dest}')

print(f'Total archived drafts: {len(archived)}')
