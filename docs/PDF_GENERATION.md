# Generating the Project Report PDF

This guide explains how to generate **PROJECT_REPORT.pdf** from the Markdown report and diagram images.

## Option 1: Pandoc (Recommended)

1. **Install Pandoc and a PDF engine**
   - Windows: `winget install pandoc` or download from https://pandoc.org/
   - Install a LaTeX engine for PDF: MiKTeX or TeX Live (for `--pdf-engine=xelatex` or `pdflatex`)

2. **Generate PDF from Markdown**
   ```bash
   cd docs
   pandoc PROJECT_REPORT.md -o PROJECT_REPORT.pdf --pdf-engine=xelatex -V geometry:margin=1in
   ```

3. **If you have diagram images** (see Option 3), insert them in the Markdown where the Mermaid blocks are, then run the same pandoc command.

## Option 2: VS Code / Browser

1. Open `docs/PROJECT_REPORT.md` in VS Code.
2. Install "Markdown PDF" extension (yzane.markdown-pdf) or "Markdown Preview Enhanced".
3. Right-click in the editor → "Markdown PDF: Export (pdf)" to export the current file to PDF.
4. Or: use "Markdown: Open Preview" and use browser Print → Save as PDF.

## Option 3: Export Diagram Images

The report includes **Data Flow Diagram** and **Class Diagram** as Mermaid code. To get PNG/SVG images:

1. **Mermaid Live Editor**
   - Go to https://mermaid.live
   - Copy the contents of:
     - `docs/diagrams/data_flow_diagram.mmd`
     - `docs/diagrams/class_diagram.mmd`
   - Paste, then use "Actions" → "Export" → PNG or SVG.
   - Save as `docs/diagrams/data_flow_diagram.png` and `docs/diagrams/class_diagram.png`.

2. **VS Code**
   - Install "Mermaid" or "Markdown Preview Mermaid Support" extension.
   - Open the `.mmd` files and use the export/preview to save as image.

3. **Use in report**
   - In `PROJECT_REPORT.md`, you can replace the Mermaid code blocks with:
     ```markdown
     ![Data Flow Diagram](diagrams/data_flow_diagram.png)
     ![Class Diagram](diagrams/class_diagram.png)
     ```
   - Then run Pandoc (Option 1) to get a single PDF with embedded images.

## Option 4: Python script (pypandoc)

If you have Pandoc installed and Python:

```bash
pip install pypandoc
python scripts/generate_report_pdf.py
```

The script `scripts/generate_report_pdf.py` (see below) uses `pypandoc` to convert `docs/PROJECT_REPORT.md` to `docs/PROJECT_REPORT.pdf`.

## Summary

| Method              | Requirement              | Output                    |
|---------------------|--------------------------|---------------------------|
| Pandoc              | Pandoc + LaTeX           | PROJECT_REPORT.pdf        |
| VS Code / Browser   | VS Code or browser       | PDF from preview/print    |
| Diagram export      | Mermaid Live or VS Code  | PNG/SVG for report        |
| Python (pypandoc)   | pip install pypandoc, Pandoc | PROJECT_REPORT.pdf   |

For **50+ references and full formatting**, use **Pandoc** with a LaTeX engine; then add page numbers and adjust margins in the pandoc command or via a custom LaTeX template if needed.




