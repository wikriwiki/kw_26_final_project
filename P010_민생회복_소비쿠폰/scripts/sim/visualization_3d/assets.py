import re


DECK_JS_URL = "https://unpkg.com/deck.gl@9.3.3/dist.min.js"
MAPLIBRE_JS_URL = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"
MAPLIBRE_CSS_URL = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css"
CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.js"
GOOGLE_FONTS_URL = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"


def style_tag(css: str) -> str:
    safe_css = _escape_closing_tag(css, "style")
    return f"<style>\n{safe_css}\n</style>"


def script_tag(js: str) -> str:
    safe_js = _escape_closing_tag(js, "script")
    return f"<script>\n{safe_js}\n</script>"


def _escape_closing_tag(text: str, tag_name: str) -> str:
    return re.sub(
        rf"</{tag_name}",
        lambda match: r"<\/" + match.group(0)[2:],
        text,
        flags=re.IGNORECASE,
    )
