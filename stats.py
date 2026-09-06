#!/usr/bin/env python3
"""Everything worth knowing about a text file before you generate from it.

    python stats.py                         # full report on input.txt
    python stats.py -i corpus.txt --top 15 --max-order 5
    python stats.py --context "the"         # what follows a given context
    python stats.py --level word            # measure words instead of characters

Companion script: generate.py, which generates text from the same file.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from collections import Counter

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master"
    "/data/tinyshakespeare/input.txt"
)

WORDS_RE = re.compile(r"\w+(?:['\u2019]\w+)*")
TOKEN_RE = re.compile(r"\n|\w+(?:['\u2019]\w+)*|[^\w\s]")

BAR, RULE = "\u2588", "\u2500"  # replaced by ASCII with --ascii


# --------------------------------------------------------------- input ---

def download(url, dest):
    import urllib.request

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"downloading {url}\n         -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)


def load_text(path, allow_download=False):
    if allow_download and not os.path.isfile(path):
        download(TINY_SHAKESPEARE, path)
    if not os.path.isfile(path):
        sys.exit(
            f"'{path}' does not exist.\n"
            "Paste some text into it, or run again with --download to grab "
            "tinyshakespeare."
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        sys.exit(f"'{path}' is empty — paste some text into it first.")
    return text


# ------------------------------------------------------------ printing ---

def section(title):
    print()
    print(title)
    print(RULE * len(title))


def field(label, value, note=""):
    print(f"  {label:<26} {value:>14}{'  ' + note if note else ''}")


def show(tok):
    """Printable form of a token: whitespace becomes \\n, \\t, ' '."""
    if tok == " ":
        return "' '"
    return tok if tok.strip() else repr(tok)[1:-1]


def bar_line(label, count, total, top, width=26):
    frac = count / total if total else 0
    length = round((count / top) * width) if top else 0
    print(f"  {label:<16.16} {count:>9,}  {frac:6.2%}  {BAR * length}")


def print_counter(counter, n, total=None, key=show):
    total = total or sum(counter.values())
    items = counter.most_common(n)
    if not items:
        print("  (nothing)")
        return
    top = items[0][1]
    for item, count in items:
        bar_line(key(item), count, total, top)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# --------------------------------------------------------------- maths ---

def entropy(counts, total=None):
    """Shannon entropy of a collection of counts, in bits."""
    counts = list(counts)
    total = total or sum(counts)
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def ngram_table(tokens, order):
    """{context of length `order`: {next token: count}}."""
    table = {}
    for gram in zip(*(tokens[j:] for j in range(order + 1))):
        context, nxt = gram[:-1], gram[-1]
        row = table.get(context)
        if row is None:
            row = table[context] = {}
        row[nxt] = row.get(nxt, 0) + 1
    return table


def predictability(tokens, max_order):
    """How much a context of each length tells you about the next token."""
    rows = []
    n = len(tokens)
    for order in range(max_order + 1):
        table = ngram_table(tokens, order)
        if not table:
            break
        total = n - order
        bits = 0.0
        branching = 0
        deterministic = 0
        for row in table.values():
            m = sum(row.values())
            bits += (m / total) * entropy(row.values(), m)
            branching += len(row)
            deterministic += len(row) == 1
        rows.append({
            "order": order,
            "bits": bits,
            "perplexity": 2 ** bits,
            "contexts": len(table),
            "branching": branching / len(table),
            "deterministic": deterministic / len(table),
        })
        del table  # these get big, let them go before building the next one
    return rows


def next_after(tokens, context):
    """Count what follows a given context anywhere in the text."""
    k = len(context)
    context = tuple(context)
    counter = Counter()
    if k == 0:
        return Counter(tokens)
    for gram in zip(*(tokens[j:] for j in range(k + 1))):
        if gram[:-1] == context:
            counter[gram[-1]] += 1
    return counter


# ------------------------------------------------------------ sections ---

def report_counts(text, path, words, lines):
    non_empty = [ln for ln in lines if ln.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = re.findall(r"[.!?]+(?=\s|$)", text)
    unique_words = {w.lower() for w in words}

    section("File")
    field("path", path)
    field("size", human_bytes(os.path.getsize(path)))

    section("Counts")
    field("characters", f"{len(text):,}")
    field("distinct characters", f"{len(set(text)):,}")
    field("words", f"{len(words):,}")
    field("distinct words", f"{len(unique_words):,}", "(case folded)")
    field("sentences", f"{len(sentences):,}", "(. ! ? endings)")
    field("lines", f"{len(lines):,}")
    field("non-empty lines", f"{len(non_empty):,}")
    field("paragraphs", f"{len(paragraphs):,}", "(blank-line separated)")
    if words:
        field("mean word length", f"{statistics.fmean(len(w) for w in words):.2f}")
    if non_empty:
        field("mean line length", f"{statistics.fmean(len(l) for l in non_empty):.1f}",
              "(non-empty lines)")


def report_vocabulary(text, per_row=16):
    chars = sorted(set(text))
    section(f"Character vocabulary ({len(chars)})")
    cells = [show(c) for c in chars]
    width = max(len(c) for c in cells) + 2
    for i in range(0, len(cells), per_row):
        print("  " + "".join(c.ljust(width) for c in cells[i:i + per_row]))

    classes = Counter()
    for c in text:
        if c.isalpha():
            classes["letters"] += 1
        elif c.isdigit():
            classes["digits"] += 1
        elif c.isspace():
            classes["whitespace"] += 1
        else:
            classes["punctuation & symbols"] += 1
    section("Character classes")
    print_counter(classes, 10, len(text), key=str)

    non_ascii = [c for c in chars if ord(c) > 127]
    if non_ascii:
        counts = Counter({c: text.count(c) for c in non_ascii})
        section(f"Non-ASCII characters ({len(non_ascii)})")
        print_counter(counts, 20)


def report_characters(text, top):
    counts = Counter(text)
    section(f"Most frequent characters (top {top})")
    print_counter(counts, top, len(text))
    rare = counts.most_common()[:-top - 1:-1]
    section(f"Rarest characters (bottom {len(rare)})")
    for tok, n in rare:
        print(f"  {show(tok):<16.16} {n:>9,}  {n / len(text):6.3%}")


def report_words(words, top):
    if not words:
        return
    lowered = [w.lower() for w in words]
    counts = Counter(lowered)
    section(f"Most frequent words (top {top}, case folded)")
    print_counter(counts, top, len(words), key=str)

    hapax = sum(1 for c in counts.values() if c == 1)
    section("Vocabulary")
    field("distinct words", f"{len(counts):,}")
    field("used exactly once", f"{hapax:,}", f"{hapax / len(counts):.1%} of the vocabulary")
    field("type/token ratio", f"{len(counts) / len(words):.4f}",
          "(1.0 = every word is new)")
    longest = max(set(words), key=len)
    field("longest word", longest[:24])

    lengths = Counter(len(w) for w in words)
    section("Word lengths")
    top_count = max(lengths.values())
    for length in sorted(lengths):
        if length > 18:
            continue
        bar_line(str(length), lengths[length], len(words), top_count)


def report_lines(lines, top):
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return
    lengths = [len(ln) for ln in non_empty]
    section("Lines")
    field("shortest / median / longest",
          f"{min(lengths)} / {int(statistics.median(lengths))} / {max(lengths)}")
    field("empty lines", f"{len(lines) - len(non_empty):,}",
          f"{(len(lines) - len(non_empty)) / len(lines):.1%}")

    buckets = Counter()
    for n in lengths:
        buckets[min(n // 10 * 10, 100)] += 1
    top_count = max(buckets.values())
    for lo in sorted(buckets):
        label = "100+" if lo == 100 else f"{lo}-{lo + 9}"
        bar_line(label, buckets[lo], len(non_empty), top_count)

    repeats = Counter(ln.strip() for ln in non_empty)
    repeats = Counter({k: v for k, v in repeats.items() if v > 1})
    if repeats:
        section(f"Most repeated lines (top {min(top, len(repeats))})")
        print_counter(repeats, top, len(non_empty), key=str)


def report_ngrams(text, words, top):
    section(f"Most frequent character pairs and triples (top {top})")
    for size in (2, 3):
        grams = Counter(text[i:i + size] for i in range(len(text) - size + 1))
        print(f"  {size}-grams")
        print_counter(grams, top, len(text) - size + 1, key=repr)
        print()

    if len(words) > 2:
        lowered = [w.lower() for w in words]
        pairs = Counter(zip(lowered, lowered[1:]))
        section(f"Most frequent word pairs (top {top})")
        print_counter(pairs, top, len(words) - 1, key=" ".join)


def report_predictability(tokens, level, max_order):
    section(f"Predictability ({level} level, orders 0 to {max_order})")
    print("  How much a context of N tokens narrows down the next one.")
    print()
    print(f"  {'order':>5} {'bits/token':>11} {'perplexity':>11} {'contexts':>11}"
          f" {'avg next':>9} {'only 1 option':>14} {'text @ rate':>12}")
    for row in predictability(tokens, max_order):
        size = human_bytes(row["bits"] * len(tokens) / 8)
        print(f"  {row['order']:>5} {row['bits']:>11.3f} {row['perplexity']:>11.2f}"
              f" {row['contexts']:>11,} {row['branching']:>9.2f}"
              f" {row['deterministic']:>13.1%} {size:>12}")
    print()
    print("  bits/token    average surprise: how many yes/no questions the next")
    print("                token is worth once you know the context")
    print("  perplexity    2^bits, i.e. the effective number of choices left")
    print("  contexts      how many rows generate.py builds at that order")
    print("  text @ rate   size of the file if coded at that rate — but careful,")
    print("                this is measured on the text itself, so a high order")
    print("                looks great mostly because it has memorised it")


def report_context(tokens, level, context_arg, top):
    if context_arg:
        context = list(context_arg) if level == "char" else TOKEN_RE.findall(context_arg)
    else:  # default: the most common triple of tokens the text is long enough for
        context = []
        for size in (3, 2, 1):
            grams = Counter(zip(*(tokens[j:] for j in range(size))))
            if grams:
                context = list(grams.most_common(1)[0][0])
                break
    if not context:
        return

    counts = next_after(tokens, context)
    joined = "".join(context) if level == "char" else " ".join(context)
    section(f"What follows '{repr(joined)[1:-1]}'")
    total = sum(counts.values())
    if not total:
        print("  nothing ever follows that context in this text")
        return
    print(f"  seen {total:,} times, followed by {len(counts)} different tokens, "
          f"{entropy(counts.values()):.2f} bits of surprise")
    print_counter(counts, top, total)


def report_encoding(text):
    """The tokenizer from the original snippet: characters in, integers out."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    sample = text[:100]
    section("Encoding sample (first 100 characters)")
    print(f"  vocabulary size: {len(chars)}")
    print(f"  encoded: {encode(sample)}")
    print(f"  decoded: {decode(encode(sample))!r}")
    assert decode(encode(sample)) == sample


# ---------------------------------------------------------------- main ---

def build_parser():
    p = argparse.ArgumentParser(
        description="Print statistics about a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default="input.txt", help="text file to describe")
    p.add_argument("--download", action="store_true",
                   help="fetch tinyshakespeare if the input file is missing")
    p.add_argument("--top", type=int, default=12,
                   help="how many entries in each top-N list")
    p.add_argument("--level", choices=("char", "word"), default="char",
                   help="token type for the predictability table")
    p.add_argument("--max-order", type=int, default=4,
                   help="longest context to measure (memory grows fast above 5)")
    p.add_argument("--context",
                   help="inspect what follows this exact context "
                        "(default: the most common one)")
    p.add_argument("--ascii", action="store_true",
                   help="ASCII bars and rules instead of box drawing characters")
    p.add_argument("--skip", default="",
                   help="comma separated sections to skip: counts, vocab, chars, "
                        "words, lines, ngrams, predict, context, encoding")
    return p


SECTIONS = ("counts", "vocab", "chars", "words", "lines", "ngrams", "predict",
            "context", "encoding")


def main(argv=None):
    global BAR, RULE
    args = build_parser().parse_args(argv)
    if args.ascii:
        BAR, RULE = "#", "-"
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    for name in skip - set(SECTIONS):
        print(f"warning: '{name}' is not a section, expected one of "
              f"{', '.join(SECTIONS)}", file=sys.stderr)

    text = load_text(args.input, args.download)
    words = WORDS_RE.findall(text)
    lines = text.splitlines()
    tokens = list(text) if args.level == "char" else TOKEN_RE.findall(text)

    if "counts" not in skip:
        report_counts(text, args.input, words, lines)
    if "vocab" not in skip:
        report_vocabulary(text)
    if "chars" not in skip:
        report_characters(text, args.top)
    if "words" not in skip:
        report_words(words, args.top)
    if "lines" not in skip:
        report_lines(lines, args.top)
    if "ngrams" not in skip:
        report_ngrams(text, words, args.top)
    if "predict" not in skip:
        report_predictability(tokens, args.level, args.max_order)
    if "context" not in skip:
        report_context(tokens, args.level, args.context, args.top)
    if "encoding" not in skip:
        report_encoding(text)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
