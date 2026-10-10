
import zipfile, xml.etree.ElementTree as ET

with zipfile.ZipFile('docs/信道速率模型_学术汇报版_论文引用版.pptx', 'r') as z:
    for s_target in ['slides/slide44.xml', 'slides/slide45.xml']:
        s_xml = ET.fromstring(z.read('ppt/' + s_target))
        print(f'=== {s_target} ===')
        for p in s_xml.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}p'):
            t = ''.join([e.text for e in p.iter() if e.tag.endswith('}t') and e.text])
            if t.strip():
                print(f'  {t.strip()}')
