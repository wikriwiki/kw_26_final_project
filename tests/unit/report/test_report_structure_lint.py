"""생성된 보고서 HTML 의 구조 검사.

숫자가 맞는지는 `consistency` 가 본다. 여기서는 **그 숫자가 그림으로 제대로
나갔는지**를 본다. 그림이 하나 깨져도 보고서는 여전히 열리기 때문에, 눈으로
보지 않으면 모르고 지나간다. 그 자리를 기계가 대신 본다.

- 모든 SVG 에 viewBox 가 있는가 (없으면 인쇄·확대에서 잘린다)
- 좌표에 NaN/Infinity 가 섞이지 않았는가 (한 점만 깨져도 선 전체가 사라진다)
- 축·범례 글자가 남아 있는가 (숫자 없는 그림은 장식이다)
- 표의 머리 열 수와 본문 열 수가 같은가
- 계산이 비었을 때 0 이나 undefined 로 메우지 않았는가
"""
from __future__ import annotations

import math
import re
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from scripts.report import analytics, consistency, narrator, render_v2

from . import _demo_run

COORD_ATTRS = {"x", "y", "cx", "cy", "width", "height", "x1", "y1", "x2", "y2", "r"}
PATH_ATTRS = {"d", "points"}
NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?|NaN|Infinity|-Infinity")


class ReportStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-lint-")
        root = _demo_run.build(Path(cls.temp.name) / "out_LINT")
        policy = _demo_run.policy()
        bundle = analytics.build_bundle(run_id="LINT", run_root=root, policy=policy)
        checks = consistency.run_checks(bundle)
        narration = narrator.narrate_report(bundle, checks, enabled=False)
        cls.html = render_v2.build_html(
            bundle, checks, narration, policy=policy, run_id="LINT", source_paths=["events.jsonl"]
        )
        cls.svgs = re.findall(r"<svg\b.*?</svg>", cls.html, re.S)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_every_svg_parses_as_xml(self) -> None:
        """문자열을 이어 붙여 만드는 SVG 라 태그가 어긋나면 그림 하나가 통째로 사라진다."""
        self.assertTrue(self.svgs)
        for index, svg in enumerate(self.svgs):
            with self.subTest(svg=index):
                ElementTree.fromstring(svg)

    def test_every_svg_declares_a_viewbox(self) -> None:
        for index, svg in enumerate(self.svgs):
            with self.subTest(svg=index):
                root = ElementTree.fromstring(svg)
                self.assertTrue(root.get("viewBox"), "viewBox 없음")

    def test_no_coordinate_is_nan_or_infinite(self) -> None:
        for index, svg in enumerate(self.svgs):
            root = ElementTree.fromstring(svg)
            for element in root.iter():
                for key, value in element.attrib.items():
                    if key in COORD_ATTRS:
                        with self.subTest(svg=index, attr=key):
                            self.assertTrue(
                                math.isfinite(float(value)), f"{key}={value}"
                            )
                    elif key in PATH_ATTRS:
                        for token in NUMBER.findall(value):
                            with self.subTest(svg=index, attr=key):
                                self.assertNotIn(token, ("NaN", "Infinity", "-Infinity"))

    def test_no_rectangle_has_negative_size(self) -> None:
        """음수 크기의 사각형은 브라우저가 그리지 않는다 — 막대가 조용히 사라진다."""
        for index, svg in enumerate(self.svgs):
            root = ElementTree.fromstring(svg)
            for element in root.iter():
                for key in ("width", "height"):
                    if key in element.attrib:
                        with self.subTest(svg=index):
                            self.assertGreaterEqual(float(element.attrib[key]), 0.0)

    def test_every_chart_keeps_its_axis_labels(self) -> None:
        for index, svg in enumerate(self.svgs):
            root = ElementTree.fromstring(svg)
            texts = [
                (element.text or "").strip()
                for element in root.iter()
                if element.tag.endswith("text")
            ]
            with self.subTest(svg=index):
                self.assertTrue([t for t in texts if t], "글자가 하나도 없는 그림")

    def test_table_header_and_body_column_counts_agree(self) -> None:
        for index, table in enumerate(re.findall(r"<table\b.*?</table>", self.html, re.S)):
            if "</thead>" not in table or "<tbody>" not in table:
                continue
            head = len(re.findall(r"<th\b", table.split("</thead>")[0]))
            rows = re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[-1], re.S)
            widths = {len(re.findall(r"<t[dh]\b", row)) for row in rows if row.strip()}
            with self.subTest(table=index):
                self.assertTrue(widths <= {head}, f"머리 {head}열, 본문 {sorted(widths)}열")

    def test_no_placeholder_leaks_into_the_document(self) -> None:
        for token in ("NaN", "undefined", "Infinity", "null원", "[object Object]"):
            self.assertNotIn(token, self.html)

    def test_the_document_stays_offline(self) -> None:
        """단일 HTML 이어야 한다 — 외부 리소스를 하나라도 부르면 오프라인에서 깨진다."""
        self.assertNotIn("<link", self.html)
        self.assertNotIn("<img", self.html)
        self.assertNotIn("<script src", self.html)


if __name__ == "__main__":
    unittest.main()
