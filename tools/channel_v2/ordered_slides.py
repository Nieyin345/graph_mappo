
import zipfile, xml.etree.ElementTree as ET

with zipfile.ZipFile('docs/信道速率模型_学术汇报版_论文引用版.pptx', 'r') as z:
    rels = ET.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
    r_map = {elem.attrib['Id']: elem.attrib['Target'] for elem in rels}
    pres = ET.fromstring(z.read('ppt/presentation.xml'))
    for idx, sldId in enumerate(pres.iter('{http://schemas.openxmlformats.org/presentationml/2006/main}sldId'), 1):
        rId = sldId.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target = r_map[rId]
        # Read the slide target
        s_xml = ET.fromstring(z.read('ppt/' + target))
        texts = [elem.text.strip() for elem in s_xml.iter() if elem.tag.endswith('}t') and elem.text and elem.text.strip()]
        t0 = texts[0] if texts else '(Empty)'
        print(f'{idx:02d} | {target:<20} | {t0}')
