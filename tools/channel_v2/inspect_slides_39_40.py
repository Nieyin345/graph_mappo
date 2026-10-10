
import zipfile, xml.etree.ElementTree as ET

with zipfile.ZipFile('docs/信道速率模型_学术汇报版_论文引用版.pptx', 'r') as z:
    for s_idx in [39, 40]:
        tree = ET.fromstring(z.read(f'ppt/slides/slide{s_idx}.xml'))
        print(f'=== SLIDE {s_idx} XML TEXTS ===')
        for elem in tree.iter():
            if elem.tag.endswith('}t') and elem.text:
                print(elem.text)
