
import zipfile, xml.etree.ElementTree as ET, re

with zipfile.ZipFile('docs/信道速率模型_学术汇报版_论文引用版.pptx', 'r') as z:
    slide_files = sorted([f for f in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml', f)],
                         key=lambda x: int(re.search(r'\d+', x).group()))
    print(f'Total slide XMLs found: {len(slide_files)}')
    for idx, sfile in enumerate(slide_files, 1):
        tree = ET.fromstring(z.read(sfile))
        texts = [elem.text.strip() for elem in tree.iter() if elem.tag.endswith('}t') and elem.text and elem.text.strip()]
        title = texts[0] if texts else '(Empty)'
        print(f'\n=== Slide {idx}: {title} ===')
        for t in texts[1:10]:
            print(f'   * {t}')
