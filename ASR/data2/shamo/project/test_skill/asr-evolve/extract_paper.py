"""Extract paper content from docx to markdown using zipfile + xml parsing."""
import zipfile
import xml.etree.ElementTree as ET
import os
import sys

src = r'C:\Users\shamo\Desktop\专业学位研究生学位申请成果开题报告_已修改_文献更新.docx'
out_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(src):
    with open(os.path.join(out_dir, 'paper_content.md'), 'w', encoding='utf-8') as f:
        f.write(f'ERROR: File not found: {src}')
    sys.exit(1)

# Extract document.xml from docx
with zipfile.ZipFile(src, 'r') as z:
    # Save document.xml
    with z.open('word/document.xml') as doc_xml:
        with open(os.path.join(out_dir, 'paper_document.xml'), 'wb') as f:
            f.write(doc_xml.read())
    # List all files in zip
    names = z.namelist()
    with open(os.path.join(out_dir, 'paper_zip_listing.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(names))

# Parse and extract text
ns = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

tree = ET.parse(os.path.join(out_dir, 'paper_document.xml'))
root = tree.getroot()

lines = []
for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = []
    for t in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    line = ''.join(texts)
    lines.append(line)

with open(os.path.join(out_dir, 'paper_content.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# Also try to extract footnotes/endnotes for references
for fname in ['word/footnotes.xml', 'word/endnotes.xml']:
    if fname in names:
        with zipfile.ZipFile(src, 'r') as z:
            with z.open(fname) as fx:
                with open(os.path.join(out_dir, f'paper_{os.path.basename(fname)}'), 'wb') as f:
                    f.write(fx.read())
