
import os, glob

files = sorted(glob.glob('docs/*'))
print(f'Total active files in docs/: {len(files)}')
for f in files:
    if os.path.isfile(f):
        print(f'{os.path.getsize(f):>8} bytes | {f}')
