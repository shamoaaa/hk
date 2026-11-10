# -*- coding: utf-8 -*-
import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUTPUT = Path("C:/Users/shamo/Desktop/kaiti_v2.docx")
MD = Path(__file__).parent / "\u5f00\u9898\u62a5\u544a-\u57fa\u4e8eAgent\u67b6\u6784\u7684\u8bed\u97f3\u8bc6\u522b\u6a21\u578b\u81ea\u52a8\u5316\u4f18\u5316\u6846\u67b6\u8bbe\u8ba1\u4e0e\u5b9e\u73b0.md"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "\u5b8b\u4f53"
style.font.size = Pt(10.5)
style._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.first_line_indent = Cm(0.74)

def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.name = "\u5b8b\u4f53"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")

def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.name = "\u5b8b\u4f53"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")

def body(text):
    doc.add_paragraph(text)

def add_table(headers, rows):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        c = table.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = "\u5b8b\u4f53"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = table.rows[ri+1].cells[ci]
            c.text = ""
            r = c.paragraphs[0].add_run(val)
            r.font.size = Pt(9)
            r.font.name = "\u5b8b\u4f53"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "\u5b8b\u4f53")

md = MD.read_text(encoding="utf-8")
lines = md.split("\n")
i = 0
in_table = False
table_headers = []
table_rows = []

while i < len(lines):
    line = lines[i]
    s = line.strip()
    if s.startswith("# "):
        i += 1; continue
    if s.startswith("## "):
        h1(s[3:]); i += 1; continue
    if s.startswith("### "):
        h1(s[4:]); i += 1; continue
    if s.startswith("#### "):
        h2(s[5:]); i += 1; continue
    if s.startswith("|") and not in_table:
        table_headers = [x.strip() for x in s.strip("|").split("|")]
        i += 1
        if i < len(lines) and "---" in lines[i]:
            i += 1
        in_table = True
        table_rows = []
        continue
    elif in_table and s.startswith("|"):
        table_rows.append([x.strip() for x in s.strip("|").split("|")])
        i += 1
        continue
    elif in_table and not s.startswith("|"):
        add_table(table_headers, table_rows)
        in_table = False
        continue
    if not s:
        i += 1; continue
    if s.startswith("```"):
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            i += 1
        i += 1; continue
    if s.startswith("**") and s.endswith("**"):
        h2(s.strip("*")); i += 1; continue
    if s.startswith("[") and "\u63d2\u5165" in s:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(s)
        r.bold = True
        r.font.size = Pt(10)
        i += 1; continue
    if "\u7b7e\u540d" in s:
        p = doc.add_paragraph(s)
        p.paragraph_format.first_line_indent = Cm(0)
        i += 1; continue
    body(s)
    i += 1

if in_table:
    add_table(table_headers, table_rows)

doc.save(str(OUTPUT))
print(f"Done! {OUTPUT}")
