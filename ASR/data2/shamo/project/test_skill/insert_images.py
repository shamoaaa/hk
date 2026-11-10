# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

DESKTOP = Path("C:/Users/shamo/Desktop")
INPUT_DOCX = DESKTOP / "kaiti_v2.docx"
OUTPUT_DOCX = DESKTOP / "kaiti_images_v2.docx"

IMG_MAP = {
    "\u56fe1": [("00a_\u5e73\u53f0\u7ba1\u7406\u5458\u7528\u4f8b\u56fe.png", "\u56fe1-1 \u5e73\u53f0\u7ba1\u7406\u5458\u7528\u4f8b\u56fe"), ("00b_\u9879\u76eeOwner\u7528\u4f8b\u56fe.png", "\u56fe1-2 \u9879\u76eeOwner\u7528\u4f8b\u56fe"), ("00c_\u9879\u76eeMember\u7528\u4f8b\u56fe.png", "\u56fe1-3 \u9879\u76eeMember\u7528\u4f8b\u56fe")],
    "\u56fe2": [("01_\u7cfb\u7edf\u603b\u4f53\u67b6\u6784\u56fe.png", "\u56fe2 \u7cfb\u7edf\u603b\u4f53\u67b6\u6784\u56fe")],
    "\u56fe3": [("07_\u90e8\u7f72\u67b6\u6784\u56fe.png", "\u56fe3 \u90e8\u7f72\u67b6\u6784\u56fe")],
    "\u56fe4": [("02_Agent\u51b3\u7b56\u5f15\u64ce\u5185\u90e8\u67b6\u6784.png", "\u56fe4 Agent\u51b3\u7b56\u5f15\u64ce\u5185\u90e8\u67b6\u6784\u56fe")],
    "\u56fe5": [("03_Agent\u4f18\u5316\u5faa\u73af\u6d41\u7a0b\u56fe.png", "\u56fe5 Agent\u4f18\u5316\u5faa\u73af\u6d41\u7a0b\u56fe")],
    "\u56fe6": [("04_\u8bc4\u6d4b\u4e0e\u5206\u6790\u6d41\u7a0b\u56fe.png", "\u56fe6 \u8bc4\u6d4b\u4e0e\u5206\u6790\u6d41\u7a0b\u56fe")],
    "\u56fe7": [("05_Trace\u94fe\u8def\u793a\u4f8b.png", "\u56fe7 Trace\u94fe\u8def\u89c2\u6d4b\u6a21\u5757\u56fe")],
    "\u56fe8": [("06_\u7248\u672c\u7ba1\u7406\u4e0e\u9879\u76ee\u521d\u59cb\u5316.png", "\u56fe8 \u7248\u672c\u7ba1\u7406\u5173\u8054\u56fe")],
}

doc = Document(str(INPUT_DOCX))
body = doc.element.body

placeholders = []
for para in doc.paragraphs:
    text = para.text.strip()
    if text.startswith("[") and "\u63d2\u5165" in text:
        for key in IMG_MAP:
            if key in text:
                placeholders.append((para, key))
                break

for para, key in reversed(placeholders):
    imgs = IMG_MAP[key]
    para_elem = para._element
    parent = para_elem.getparent()
    idx = list(parent).index(para_elem)
    parent.remove(para_elem)

    elems = []
    for img_file, caption_text in imgs:
        img_path = DESKTOP / img_file
        if not img_path.exists():
            print(f"SKIP: {img_file}")
            continue
        ip = doc.add_paragraph()
        ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ip.paragraph_format.first_line_indent = Cm(0)
        ip.add_run().add_picture(str(img_path), width=Inches(5.2))
        elems.append(ip._element)
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent = Cm(0)
        cp.paragraph_format.space_after = Pt(10)
        r = cp.add_run(caption_text)
        r.font.size = Pt(9)
        r.font.name = "\u5b8b\u4f53"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")
        elems.append(cp._element)

    for e in reversed(elems):
        parent.remove(e)
        parent.insert(idx, e)

doc.save(str(OUTPUT_DOCX))
print(f"Done! {OUTPUT_DOCX}")
