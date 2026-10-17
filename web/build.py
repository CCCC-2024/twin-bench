"""Build the local page: wrap page.html (the published fragment) into a full HTML document.

page.html is a fragment without <!doctype>/<head>; opening the page locally or on GitHub Pages needs
them here. Output: index.html (open it in a browser; replay of the real data works locally).
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent
frag = (WEB / "page.html").read_text(encoding="utf-8")
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n')
(WEB / "index.html").write_text(head + frag + "\n</body>\n</html>\n", encoding="utf-8")
print("wrote", WEB / "index.html")
