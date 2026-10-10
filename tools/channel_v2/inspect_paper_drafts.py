
import os, glob, time

files = sorted(glob.glob('docs/*论文*') + glob.glob('docs/*审阅*') + glob.glob('docs/*润色*') + glob.glob('docs/*草稿*'))
for f in files:
    stat = os.stat(f)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
    size = stat.st_size
    print(f'{mtime} | {size:>8} bytes | {f}')
