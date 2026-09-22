"""Bridge to the existing report renderer.

Rendering and metric calculation stay in ``scripts/sim/generate_final_report``
as required by the console contract.  This adapter only adds provenance text
to the self-contained HTML returned by that renderer.
"""
from __future__ import annotations

from html import escape
from typing import Any
from urllib.parse import quote


def build_report_html(
    generator: Any,
    *args: Any,
    provenance: str,
    run_id: str,
    source_paths: list[str],
) -> str:
    html = generator.build_html(*args)
    return append_provenance_html(html, provenance=provenance, run_id=run_id, source_paths=source_paths)


def append_provenance_html(
    html: str,
    *,
    provenance: str,
    run_id: str,
    source_paths: list[str],
) -> str:
    links = []
    for source in source_paths:
        href = f"/api/runs/{quote(run_id, safe='')}/artifacts/{quote(source, safe='/')}"
        links.append(f'<li><a href="{href}">{escape(source)}</a></li>')
    marker = (
        '<section class="dasol-provenance" aria-label="DASOL provenance">'
        f"<p>{escape(provenance)}</p>"
        "<p>원본 근거 (run snapshot)</p>"
        f"<ul>{''.join(links)}</ul>"
        "</section>"
    )
    if "</body>" in html:
        return html.replace("</body>", f"{marker}</body>", 1)
    return html + marker
