"""
Generate PROJECT_REPORT.pdf from docs/PROJECT_REPORT.md using pypandoc.
Requires: pip install pypandoc, and Pandoc + LaTeX installed on the system.
"""
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MD_FILE = DOCS / "PROJECT_REPORT.md"
PDF_FILE = DOCS / "PROJECT_REPORT.pdf"


def main():
    if not MD_FILE.exists():
        print(f"Error: {MD_FILE} not found.")
        sys.exit(1)

    try:
        import pypandoc
    except ImportError:
        print("Install pypandoc: pip install pypandoc")
        sys.exit(1)

    # Ensure pandoc is available
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Pandoc not found. Install from https://pandoc.org/")
        sys.exit(1)

    extra_args = [
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
    ]

    print(f"Converting {MD_FILE} -> {PDF_FILE}")
    os.chdir(ROOT)
    pypandoc.convert_file(
        str(MD_FILE),
        "pdf",
        outputfile=str(PDF_FILE),
        extra_args=extra_args,
    )
    print(f"Done: {PDF_FILE}")


if __name__ == "__main__":
    main()




