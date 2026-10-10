
import os, glob

files = sorted(glob.glob('docs/*'))
for f in files:
    if os.path.isfile(f):
        print(f'{os.path.getsize(f):>8} bytes | {f}')
