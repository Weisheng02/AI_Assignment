#!/usr/bin/env python3
"""Export the generated DOCX as copy-friendly Markdown for Google Docs."""

from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "AI Report - Final.docx"
OUTPUT_PATH = ROOT / "Google Docs Copy - Report Content.md"


def iter_blocks(document):
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def clean(value):
    return " ".join(str(value).replace("|", "\\|").split())


def export():
    document = Document(INPUT_PATH)
    lines = [
        "<!-- Copy this file into Google Docs, or upload AI Report - Final.docx directly to Google Drive. -->",
        "",
    ]
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = block.style.name if block.style else ""
            if style.startswith("Heading "):
                try:
                    level = min(int(style.split()[-1]), 6)
                except ValueError:
                    level = 2
                lines.extend([f"{'#' * level} {text}", ""])
            elif "List Bullet" in style:
                lines.extend([f"- {text}", ""])
            elif "List Number" in style:
                lines.extend([f"1. {text}", ""])
            elif "Caption" in style:
                lines.extend([f"*{text}*", ""])
            else:
                lines.extend([text, ""])
        else:
            rows = [[clean(cell.text) for cell in row.cells] for row in block.rows]
            if not rows:
                continue
            width = len(rows[0])
            lines.append("| " + " | ".join(rows[0]) + " |")
            lines.append("| " + " | ".join(["---"] * width) + " |")
            for row in rows[1:]:
                row = (row + [""] * width)[:width]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    export()
