#!/usr/bin/env python3
"""Pull clean article text out of Wikipedia and write it to a file.

No navigation menus, no table of contents, no citation lists — just the prose,
which is what generate.py and stats.py want.

    python wiki.py Cat                          # one article -> input.txt
    python wiki.py Cat Dog Lion Tiger           # several, concatenated
    python wiki.py --search cat --limit 40      # 40 articles about cats
    python wiki.py --random 50                  # 50 random articles
    python wiki.py Chat --lang fr -o corpus.txt

Then:

    python stats.py
    python generate.py -p "The cat"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = ("ngram-playground/1.0 (educational n-gram demo; "
              "https://www.mediawiki.org/wiki/API:Etiquette)")

# Sections that are lists of links or citations rather than prose. Everything
# from the first one of these onwards is dropped.
TAIL_SECTIONS = {
    "en": ("see also", "notes", "references", "external links", "further reading",
           "bibliography", "sources", "citations", "footnotes", "works cited"),
    "fr": ("voir aussi", "notes et r\u00e9f\u00e9rences", "notes", "r\u00e9f\u00e9rences",
           "liens externes", "bibliographie", "annexes", "articles connexes",
           "sources"),
    "es": ("v\u00e9ase tambi\u00e9n", "referencias", "notas", "enlaces externos",
           "bibliograf\u00eda"),
    "de": ("siehe auch", "literatur", "weblinks", "einzelnachweise", "anmerkungen"),
}

HEADING_RE = re.compile(r"^\s*(=+)\s*(.+?)\s*=+\s*$", re.M)


# ----------------------------------------------------------------- api ---

def api(params, lang="en", delay=0.5, tries=5):
    """One call to the MediaWiki API, with backoff on rate limiting."""
    params = dict(params, format="json", formatversion="2")
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                time.sleep(delay)  # be polite, the API is free
                return json.load(response)
        except urllib.error.HTTPError as err:
            if err.code in (429, 503) and attempt < tries - 1:
                wait = 3 * 2 ** attempt
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
        except urllib.error.URLError as err:
            sys.exit(f"network error: {err.reason}")
    return {}


def fetch_extract(title, lang, delay):
    """Plain text of one article. Full extracts are one page per request."""
    data = api({"action": "query", "prop": "extracts", "explaintext": "1",
                "redirects": "1", "titles": title}, lang, delay)
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None, None
    return pages[0].get("title", title), pages[0].get("extract", "")


def fetch_links(title, lang, limit, delay):
    """Titles of the articles a page links to, in page order."""
    titles, cont = [], {}
    while len(titles) < limit:
        data = api({"action": "query", "prop": "links", "titles": title,
                    "plnamespace": "0", "pllimit": "500", **cont}, lang, delay)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            break
        titles += [link["title"] for link in pages[0].get("links", [])]
        cont = data.get("continue", {})
        if not cont:
            break
    return titles[:limit]


def fetch_search(query, lang, limit, delay):
    """Article titles matching a search, most relevant first."""
    titles, offset = [], 0
    while len(titles) < limit:
        data = api({"action": "query", "list": "search", "srsearch": query,
                    "srnamespace": "0", "srlimit": str(min(50, limit - len(titles))),
                    "sroffset": str(offset)}, lang, delay)
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        titles += [h["title"] for h in hits]
        offset += len(hits)
    return titles[:limit]


def fetch_random(count, lang, delay):
    """Random article titles."""
    titles = []
    while len(titles) < count:
        batch = min(50, count - len(titles))
        data = api({"action": "query", "list": "random", "rnnamespace": "0",
                    "rnlimit": str(batch)}, lang, delay)
        titles += [p["title"] for p in data.get("query", {}).get("random", [])]
    return titles[:count]


# --------------------------------------------------------------- clean ---

def strip_tail_sections(text, lang):
    """Drop 'References', 'External links' and friends, and everything after."""
    names = TAIL_SECTIONS.get(lang, TAIL_SECTIONS["en"])
    for match in HEADING_RE.finditer(text):
        if len(match.group(1)) == 2 and match.group(2).strip().lower() in names:
            return text[:match.start()]
    return text


def clean(text, lang, headings="plain"):
    """Tidy one extract. `headings` is plain, markup or drop."""
    text = strip_tail_sections(text, lang)

    if headings == "drop":
        text = HEADING_RE.sub("", text)
    elif headings == "plain":
        text = HEADING_RE.sub(lambda m: m.group(2), text)

    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)  # no runs of blank lines
    text = re.sub(r"\u00a0", " ", text)  # non-breaking spaces
    return text.strip()


# ---------------------------------------------------------------- main ---

def build_parser():
    p = argparse.ArgumentParser(
        description="Download Wikipedia articles as plain text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("titles", nargs="*", help="article titles to fetch")
    p.add_argument("-o", "--out", default="input.txt", help="file to write")
    p.add_argument("-a", "--append", action="store_true",
                   help="add to the file instead of replacing it")
    p.add_argument("--lang", default="en", help="Wikipedia language edition")
    p.add_argument("--search", metavar="QUERY",
                   help="fetch the articles matching a search, most relevant "
                        "first; use with --limit")
    p.add_argument("--limit", type=int, default=20, metavar="N",
                   help="how many articles --search returns")
    p.add_argument("--links", type=int, default=0, metavar="N",
                   help="also fetch the first N articles linked from the first "
                        "title; note that Wikipedia returns links in "
                        "alphabetical order, so this is a broad sample of what "
                        "the page mentions, not the most relevant pages")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="fetch N random articles")
    p.add_argument("--headings", choices=("plain", "markup", "drop"),
                   default="plain",
                   help="section titles as plain lines, with == markers, or removed")
    p.add_argument("--separator", default="\n\n\n",
                   help="text inserted between articles")
    p.add_argument("--delay", type=float, default=0.5,
                   help="seconds between API calls")
    p.add_argument("--min-chars", type=int, default=400,
                   help="skip articles shorter than this (stubs)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    titles = list(args.titles)
    if args.search:
        print(f"searching for {args.search!r}", file=sys.stderr)
        found = fetch_search(args.search, args.lang, args.limit, args.delay)
        titles += [t for t in found if t not in titles]
    if args.links:
        if not titles:
            sys.exit("--links needs a title to start from")
        print(f"listing links from {titles[0]}", file=sys.stderr)
        linked = fetch_links(titles[0], args.lang, args.links, args.delay)
        titles += [t for t in linked if t not in titles]
    if args.random:
        titles += fetch_random(args.random, args.lang, args.delay)
    if not titles:
        sys.exit("nothing to fetch — give a title, or use --random N")

    pieces, kept, skipped = [], 0, 0
    for i, title in enumerate(titles, 1):
        resolved, extract = fetch_extract(title, args.lang, args.delay)
        if extract is None:
            print(f"  [{i}/{len(titles)}] {title}: no such page", file=sys.stderr)
            skipped += 1
            continue
        body = clean(extract, args.lang, args.headings)
        if len(body) < args.min_chars:
            print(f"  [{i}/{len(titles)}] {resolved}: too short, skipped",
                  file=sys.stderr)
            skipped += 1
            continue
        pieces.append(body)
        kept += 1
        print(f"  [{i}/{len(titles)}] {resolved}: {len(body):,} characters",
              file=sys.stderr)

    if not pieces:
        sys.exit("nothing was written")

    text = args.separator.join(pieces) + "\n"
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
        if args.append:
            f.write(args.separator)
        f.write(text)

    words = len(re.findall(r"\w+", text))
    print(f"\n{kept} article(s), {skipped} skipped, {len(text):,} characters, "
          f"{words:,} words {'appended to' if args.append else 'written to'} "
          f"{args.out}", file=sys.stderr)
    print(f"next: python stats.py -i {args.out}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
