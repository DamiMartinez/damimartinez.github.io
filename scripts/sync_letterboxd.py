#!/usr/bin/env python3
"""Snapshot the Letterboxd RSS feed into _data/movies.json for Jekyll to render."""

import json
import re
import sys
import urllib.request
from pathlib import Path

import defusedxml.ElementTree as ET  # stdlib ElementTree is vulnerable to XXE/billion-laughs

LETTERBOXD_USERNAME = "damiavr"
RSS_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/rss/"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "_data" / "movies.json"

NAMESPACES = {"letterboxd": "https://letterboxd.com"}
POSTER_RE = re.compile(r'<img src="([^"]+)"')

# Letterboxd 403s on generic/bot user agents; a browser UA works.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def fetch_feed(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def parse_items(feed_xml: bytes) -> list[dict]:
    root = ET.fromstring(feed_xml)
    movies = []
    for item in root.findall("./channel/item"):
        title = item.findtext("letterboxd:filmTitle", namespaces=NAMESPACES)
        year = item.findtext("letterboxd:filmYear", namespaces=NAMESPACES)
        rating = item.findtext("letterboxd:memberRating", namespaces=NAMESPACES)
        watched_date = item.findtext("letterboxd:watchedDate", namespaces=NAMESPACES)
        link = item.findtext("link")
        description = item.findtext("description") or ""

        poster_match = POSTER_RE.search(description)

        movies.append(
            {
                "title": title,
                "year": int(year) if year else None,
                "rating": float(rating) if rating else None,
                "watched_date": watched_date,
                "poster": poster_match.group(1) if poster_match else None,
                "link": link,
            }
        )
    return movies


def main() -> None:
    try:
        feed_xml = fetch_feed(RSS_URL)
    except Exception as exc:  # network/HTTP errors shouldn't crash the whole Action run
        print(f"Failed to fetch Letterboxd RSS feed: {exc}", file=sys.stderr)
        sys.exit(1)

    movies = parse_items(feed_xml)
    OUTPUT_PATH.write_text(json.dumps(movies, indent=2) + "\n")
    print(f"Wrote {len(movies)} movies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
