"""用 PowerPoint COM 把生成的 pptx 导出为 PNG，便于逐页检查版式。"""

import os
import sys

import win32com.client

SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "QKD-SAGIN生产端调度_组会汇报.pptx")
OUT = os.path.abspath(".tmp/preview")

if os.path.isdir(OUT):
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
else:
    os.makedirs(OUT)

app = win32com.client.Dispatch("PowerPoint.Application")
try:
    pres = app.Presentations.Open(SRC, ReadOnly=True, Untitled=False, WithWindow=False)
    # 18 = ppSaveAsPNG，导出后每页一个 PNG
    pres.SaveAs(OUT, 18)
    pres.Close()
finally:
    app.Quit()

files = sorted(os.listdir(OUT))
print("导出 %d 个文件到 %s" % (len(files), OUT))
for f in files[:5]:
    print("  ", f)