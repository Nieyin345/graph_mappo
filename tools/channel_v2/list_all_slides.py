
import zipfile, xml.etree.ElementTree as ET, re

with zipfile.ZipFile('docs/信道速率模型_学术汇报版_论文引用版.pptx', 'r') as z:
    for s_idx in range(1, 41):
        try:
            tree = ET.fromstring(z.read(f'ppt/slides/slide{s_idx}.xml'))
            texts = [elem.text.strip() for elem in tree.iter() if elem.tag.endswith('}t') and elem.text and elem.text.strip()]
            t0 = texts[0] if texts else '(Empty)'
            t1 = texts[1] if len(texts)>1 else ''
            print(f'Slide {s_idx:02d}: {t0[:35]:<35} | {t1[:30]}')
        except Exception as e:
            print(f'Slide {s_idx:02d}: Error {e}')
