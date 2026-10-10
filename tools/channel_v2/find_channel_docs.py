
import os, glob

# Check docs directory
doc_files = sorted(glob.glob('docs/**/*.md', recursive=True) + glob.glob('*.md'))
channel_related = []

keywords = ['信道', '速率模型', 'channel', 'rate', 'Ddestopwork', 'rate-model']

for f in doc_files:
    fname = os.path.basename(f)
    if any(k in fname or k in f for k in keywords):
        size = os.path.getsize(f)
        channel_related.append((f, size))

print(f'Total channel-related files found: {len(channel_related)}')
for path, size in channel_related:
    print(f'{size:>8} bytes | {path}')
