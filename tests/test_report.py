import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


def test_report_builder_emits_pdf(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "status.pdf"
    result = subprocess.run(
        [sys.executable, "scripts/build_report.py", "--output", str(output)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 5000
    reader = PdfReader(str(output))
    assert len(reader.pages) >= 3
    assert "Egg Vision Assignment" in "\n".join(page.extract_text() or "" for page in reader.pages)
