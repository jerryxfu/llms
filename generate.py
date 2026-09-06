#!/usr/bin/env python3
"""n-gram text generator — character or word level, standard library only.

It reads a plain text file, counts which token follows which context, then
continues whatever prompt you give it.

    python generate.py --download              # fetch tinyshakespeare into input.txt
    python generate.py                         # interactive session
    python generate.py -p "ROMEO:" -n 400 -t 0.8
    python generate.py --order 6 --explain -p "First Cit"
    python generate.py --level word -p "To be" -n 60

Companion script: stats.py, which describes the same input file.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master"
    "/data/tinyshakespeare/input.txt"
)


# --------------------------------------------------------------- input ---

def download(url, dest):
    """Fetch `url` into `dest`. Uses urllib so there is nothing to pip install."""
    import urllib.request

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"downloading {url}\n         -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)


def load_text(path, allow_download=False, lower=False):
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
    return text.lower() if lower else text


# ---------------------------------------------------------- tokenizing ---

# A word token is a word (apostrophes included), a lone punctuation mark, or a
# newline. Newlines are kept because they carry the shape of the text.
WORD_RE = re.compile(r"\n|\w+(?:['\u2019]\w+)*|[^\w\s]")

NO_SPACE_BEFORE = set(",.;:!?)]}%-\u2026'\u2019\"\u00bb")
NO_SPACE_AFTER = set("([{-\u00ab\u201c\u00bf\u00a1$")


def tokenize(text, level):
    """Split text into the units the model works with."""
    return list(text) if level == "char" else WORD_RE.findall(text)


def needs_space(prev, tok):
    """Whether a space goes between two word-level tokens."""
    if prev is None or prev == "\n" or tok == "\n":
        return False
    if tok in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER:
        return False
    return True


def detokenize(tokens, level):
    """Glue tokens back into readable text."""
    if level == "char":
        return "".join(tokens)
    out, prev = [], None
    for tok in tokens:
        if needs_space(prev, tok):
            out.append(" ")
        out.append(tok)
        prev = tok
    return "".join(out)


def show(tok):
    """Printable form of a token, for the --explain output."""
    if tok == " ":
        return "' '"
    return tok if tok.strip() else repr(tok)[1:-1]


# --------------------------------------------------------------- model ---

class NGramModel:
    """For every context of length 0..order, how often each token follows it.

    Keeping the shorter contexts around is what makes backing off possible:
    if the last `order` tokens were never seen together, we ask a shorter
    context instead of giving up.
    """

    def __init__(self, order):
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = order
        self.tables = [{} for _ in range(order + 1)]  # tables[k][context] = {token: count}
        self.vocab = []
        self.vocab_set = frozenset()

    def train(self, tokens):
        self.vocab = sorted(set(tokens))
        self.vocab_set = frozenset(self.vocab)
        for k in range(self.order + 1):
            table = self.tables[k]
            # zip of shifted views walks every (context, next token) pair
            for gram in zip(*(tokens[j:] for j in range(k + 1))):
                context, nxt = gram[:-1], gram[-1]
                row = table.get(context)
                if row is None:
                    row = table[context] = {}
                row[nxt] = row.get(nxt, 0) + 1
        return self

    def row_for(self, context, min_count=1):
        """Longest known context that matches, i.e. stupid backoff.

        Returns (context length actually used, {token: count}).
        """
        for k in range(min(len(context), self.order), 0, -1):
            row = self.tables[k].get(tuple(context[-k:]))
            if row is not None and sum(row.values()) >= min_count:
                return k, row
        return 0, self.tables[0][()]

    @property
    def n_contexts(self):
        return sum(len(t) for t in self.tables)


# ------------------------------------------------------------ sampling ---

def sample(row, vocab, rng, temperature=1.0, top_k=0, top_p=0.0, smoothing=0.0):
    """Pick one token out of a {token: count} row. Returns (token, top candidates)."""
    if smoothing > 0:
        counts = {tok: smoothing for tok in vocab}
        for tok, c in row.items():
            counts[tok] += c
    else:
        counts = row

    total = sum(counts.values())
    dist = sorted(((tok, c / total) for tok, c in counts.items()),
                  key=lambda kv: kv[1], reverse=True)
    preview = dist[:8]

    if temperature <= 0:  # greedy: always the most likely token
        return dist[0][0], preview

    if temperature != 1.0:
        inv = 1.0 / temperature
        dist = [(t, p ** inv) for t, p in dist]  # order is preserved
        s = sum(p for _, p in dist)
        if s <= 0:  # everything underflowed, fall back to greedy
            return dist[0][0], preview
        dist = [(t, p / s) for t, p in dist]

    if top_k > 0:
        dist = dist[:top_k]

    if 0 < top_p < 1:  # nucleus: smallest set of tokens worth top_p of the mass
        kept, acc = [], 0.0
        for t, p in dist:
            kept.append((t, p))
            acc += p
            if acc >= top_p:
                break
        dist = kept

    tokens = [t for t, _ in dist]
    weights = [p for _, p in dist]
    return rng.choices(tokens, weights=weights)[0], preview


# ---------------------------------------------------------- generation ---

def line_starts(tokens):
    """Indices right after a newline, so a random start lands on a fresh line."""
    return [i + 1 for i, t in enumerate(tokens) if t == "\n" and i + 1 < len(tokens)]


def random_context(tokens, starts, order, rng):
    i = rng.choice(starts) if starts else rng.randrange(len(tokens))
    return tokens[max(0, i - order):i]


def generate(model, context, length, rng, *, level, temperature, top_k, top_p,
             smoothing, min_count, stop=None, explain=False, delay=0.0,
             out=sys.stdout):
    """Emit `length` tokens, streaming them as they are produced."""
    context = list(context)
    prev = context[-1] if context else None
    orders_used = Counter()
    tail = ""

    for _ in range(length):
        k, row = model.row_for(context, min_count)
        orders_used[k] += 1
        tok, preview = sample(row, model.vocab, rng, temperature, top_k, top_p,
                              smoothing)
        piece = (" " if (level == "word" and needs_space(prev, tok)) else "") + tok
        out.write(piece)
        out.flush()

        if explain:
            candidates = ", ".join(f"{show(t)} {p:.0%}" for t, p in preview[:5])
            print(f"\n    [context {k} | {len(row)} candidates | {candidates}]",
                  file=sys.stderr)
        if delay:
            time.sleep(delay)

        context.append(tok)
        prev = tok
        if stop:
            tail = (tail + piece)[-len(stop):]
            if tail == stop:
                break

    return orders_used


def run_once(model, tokens, starts, args, rng):
    if args.prompt:
        prompt_tokens = tokenize(args.prompt, args.level)
        unknown = sorted({t for t in prompt_tokens if t not in model.vocab_set})
        if unknown and not args.quiet:
            print(f"note: not in the corpus, ignored as context: "
                  f"{', '.join(show(t) for t in unknown)}", file=sys.stderr)
        sys.stdout.write(args.prompt)
    else:
        prompt_tokens = random_context(tokens, starts, args.order, rng)
        if not args.quiet:
            seed_text = detokenize(prompt_tokens, args.level)
            print(f"note: no prompt, starting from {seed_text!r} taken from the corpus",
                  file=sys.stderr)

    orders_used = generate(
        model, prompt_tokens, args.length, rng,
        level=args.level, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, smoothing=args.smoothing, min_count=args.min_count,
        stop=args.stop, explain=args.explain, delay=args.delay,
    )
    print()

    if not args.quiet:
        total = sum(orders_used.values())
        full = orders_used.get(args.order, 0)
        spread = " ".join(f"{k}:{orders_used[k]}" for k in sorted(orders_used))
        print(f"[{total} tokens | full context {full / total:.0%} of the time "
              f"| context length used {spread}]", file=sys.stderr)


# --------------------------------------------------------- interactive ---

HELP = """\
commands
  /help                  this list
  /config                current settings
  /temp 0.8              temperature (0 = always the most likely token)
  /len 400               how many tokens to generate
  /order 5               context length (rebuilds the model)
  /topk 10   /topp 0.9   candidate filtering (0 = off)
  /smooth 0.1            add-k smoothing
  /mincount 2            a context must be this frequent to be trusted
  /explain               toggle the per-token explanation
  /seed 42               reseed the random generator
  /quit                  leave

anything else is used as a prompt; an empty line starts from a random line
of the corpus.\
"""


def config_summary(args, model):
    return (f"input={args.input} level={args.level} order={args.order} "
            f"length={args.length} temperature={args.temperature} "
            f"top_k={args.top_k} top_p={args.top_p} smoothing={args.smoothing} "
            f"min_count={args.min_count} explain={args.explain}\n"
            f"vocabulary={len(model.vocab)} contexts={model.n_contexts:,}")


def handle_command(line, args, tokens, model, rng):
    """Run a /command. Returns a new model when the order changed, else None."""
    parts = line[1:].split()
    if not parts:
        return None
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
    try:
        if cmd in ("q", "quit", "exit"):
            sys.exit(0)
        if cmd in ("h", "help", "?"):
            print(HELP)
        elif cmd == "config":
            print(config_summary(args, model))
        elif cmd == "explain":
            args.explain = not args.explain
            print(f"explain = {args.explain}")
        elif cmd == "order":
            args.order = int(arg)
            t0 = time.time()
            model = NGramModel(args.order).train(tokens)
            print(f"order = {args.order} ({model.n_contexts:,} contexts, "
                  f"{time.time() - t0:.1f}s)")
            return model
        elif cmd == "seed":
            rng.seed(int(arg))
            print(f"seed = {arg}")
        elif cmd == "temp":
            args.temperature = float(arg)
            print(f"temperature = {args.temperature}")
        elif cmd == "len":
            args.length = int(arg)
            print(f"length = {args.length}")
        elif cmd == "topk":
            args.top_k = int(arg)
            print(f"top_k = {args.top_k}")
        elif cmd == "topp":
            args.top_p = float(arg)
            print(f"top_p = {args.top_p}")
        elif cmd == "smooth":
            args.smoothing = float(arg)
            print(f"smoothing = {args.smoothing}")
        elif cmd == "mincount":
            args.min_count = int(arg)
            print(f"min_count = {args.min_count}")
        else:
            print(f"unknown command '{cmd}' — /help for the list")
    except (TypeError, ValueError):
        print(f"'/{cmd}' needs a number, e.g. /{cmd} 2")
    return None


def repl(model, tokens, starts, args, rng):
    print("n-gram playground — /help for commands, Ctrl-D to quit.\n",
          file=sys.stderr)
    while True:
        try:
            line = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if line.startswith("/"):
            model = handle_command(line, args, tokens, model, rng) or model
            continue
        args.prompt = line
        run_once(model, tokens, starts, args, rng)
        print()


# ---------------------------------------------------------------- main ---

def build_parser():
    p = argparse.ArgumentParser(
        description="Generate text with an n-gram model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("input")
    g.add_argument("-i", "--input", default="input.txt",
                   help="text file to learn from")
    g.add_argument("--download", action="store_true",
                   help="fetch tinyshakespeare if the input file is missing")
    g.add_argument("--level", choices=("char", "word"), default="char",
                   help="what counts as a token")
    g.add_argument("--lower", action="store_true",
                   help="lowercase the corpus before training")

    g = p.add_argument_group("model")
    g.add_argument("-k", "--order", type=int, default=4,
                   help="context length, in tokens")
    g.add_argument("--min-count", type=int, default=1,
                   help="a context must have been seen this often to be used, "
                        "otherwise a shorter one is used instead")
    g.add_argument("--smoothing", type=float, default=0.0,
                   help="add-k smoothing: pretend every token of the vocabulary "
                        "was seen this many extra times")

    g = p.add_argument_group("sampling")
    g.add_argument("-t", "--temperature", type=float, default=1.0,
                   help="0 = always the most likely token, 1 = the raw counts, "
                        "above 1 = wilder")
    g.add_argument("--top-k", type=int, default=0,
                   help="only sample from the k most likely tokens (0 = off)")
    g.add_argument("--top-p", type=float, default=0.0,
                   help="nucleus sampling threshold (0 or 1 = off)")
    g.add_argument("--seed", type=int, help="random seed, for reproducible runs")

    g = p.add_argument_group("output")
    g.add_argument("-p", "--prompt",
                   help="text to continue; omit it for an interactive session")
    g.add_argument("-n", "--length", type=int, default=500,
                   help="how many tokens to generate")
    g.add_argument("--stop", help=r"stop as soon as this string appears (\n works)")
    g.add_argument("--delay", type=float, default=0.0,
                   help="seconds between tokens, for live demos")
    g.add_argument("--explain", action="store_true",
                   help="show the context length used and the top candidates "
                        "for every token")
    g.add_argument("-q", "--quiet", action="store_true",
                   help="print the generated text and nothing else")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stop:
        args.stop = args.stop.encode().decode("unicode_escape")

    rng = random.Random(args.seed)
    text = load_text(args.input, args.download, args.lower)
    tokens = tokenize(text, args.level)
    if not tokens:
        sys.exit("nothing to train on")

    t0 = time.time()
    model = NGramModel(args.order).train(tokens)
    if not args.quiet:
        print(f"{len(tokens):,} {args.level} tokens, {len(model.vocab):,} distinct, "
              f"{model.n_contexts:,} contexts up to order {args.order} "
              f"({time.time() - t0:.1f}s)\n", file=sys.stderr)

    starts = line_starts(tokens)
    if args.prompt is None:
        repl(model, tokens, starts, args, rng)
    else:
        run_once(model, tokens, starts, args, rng)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
