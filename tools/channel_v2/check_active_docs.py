
import os, glob

files = sorted(glob.glob('docs/*'))
channel_active = [f for f in files if any(k in f for k in ['信道', '速率', 'channel', 'rate', 'DOI'])]
for f in channel_active:
    print(f'{os.path.getsize(f):>8} bytes | {f}')
