#!/usr/bin/env python3
"""
Extract GDPR Articles 5-43 from the official EUR-Lex XHTML file and save as
clean plain text to data/gdpr/gdpr_art5_43.txt.

Input:  data/gdpr/L_2016119EN.01000101.html  (EUR-Lex Official Journal XHTML)
Output: data/gdpr/gdpr_art5_43.txt
"""

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "data/gdpr/L_2016119EN.01000101.html"
OUTPUT_PATH = ROOT / "data/gdpr/gdpr_art5_43.txt"
ARTICLE_RANGE = range(5, 44)  # Articles 5-43 inclusive


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _walk(element, lines: list[str]) -> None:
    """Recursively extract text, handling EUR-Lex list tables specially."""
    if isinstance(element, NavigableString):
        return
    if element.name == "table":
        # Two-column layout: narrow marker col (a)/(b)/1/2 + wide content col
        for row in element.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                marker = _clean(cells[0].get_text())
                content = _clean(cells[1].get_text())
                if marker and content:
                    lines.append(f"  {marker} {content}")
                elif content:
                    lines.append(f"  {content}")
        return  # don't descend further into the table
    if element.name == "p":
        text = _clean(element.get_text())
        if text:
            lines.append(text)
        return  # don't descend into <p> children
    for child in element.children:
        _walk(child, lines)


def extract_articles(soup: BeautifulSoup) -> dict[int, list[str]]:
    articles: dict[int, list[str]] = {}
    for n in ARTICLE_RANGE:
        div = soup.find(id=f"art_{n}")
        if div is None:
            print(f"  Warning: Article {n} not found in HTML")
            continue
        lines: list[str] = []
        _walk(div, lines)
        articles[n] = lines
    return articles


def main() -> None:
    print(f"Parsing {HTML_PATH.name} ...")
    with open(HTML_PATH, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    articles = extract_articles(soup)

    output_lines: list[str] = []
    for n, lines in sorted(articles.items()):
        output_lines.append(f"{'=' * 60}")
        output_lines.extend(lines)
        output_lines.append("")

    OUTPUT_PATH.write_text("\n".join(output_lines), encoding="utf-8")

    total_chars = sum(len(l) for lines in articles.values() for l in lines)
    print(f"Extracted {len(articles)} articles → {OUTPUT_PATH.name}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total lines:      {sum(len(l) for l in articles.values()):,}")


if __name__ == "__main__":
    main()
