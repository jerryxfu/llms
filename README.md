# n-gram playground

Two small scripts that learn from a plain text file: one prints everything worth knowing about the text, the other generates new text from it.

- `stats.py` — describes `input.txt`: counts, vocabulary, frequencies, and how predictable the next character is.
- `generate.py` — trains an n-gram model on `input.txt` and continues whatever prompt you give it.

Python 3.8+, standard library only. Nothing to install.

## Quick start

```bash
python generate.py --download     # writes tinyshakespeare to input.txt
python stats.py                   # describe it
python generate.py -p "ROMEO:"    # continue a prompt
python generate.py                # interactive session
```

Or skip the download and paste your own text into `input.txt`: a book, a transcript, your own writing, a chat log. Anything above ~100 kB starts to sound like
itself. Below ~10 kB the model mostly quotes.

Both scripts read `input.txt` from the current directory by default; use
`-i other.txt` for anything else.

## How it works

An n-gram model is a table of counts: for every context of *n* tokens, how often each token came next. To generate, you look up the last *n* tokens, and draw
the next one at random with those counts as weights. That is the whole idea. No training, no gradients, no neural network — just counting.

Three details make it usable:

**Backoff.** A prompt you typed may contain a context the text never had. If the last 4 characters were never seen together, the model falls back to the last 3,
then 2, then 1, then the raw character frequencies. The generator reports how often it had to do that:

```
[300 tokens | full context 100% of the time | context length used 4:300]
```

**Order.** The context length, `--order`. Low order sounds like soup, high order recites the source text from memory. On tinyshakespeare, 3 gives English-shaped
nonsense, 5 gives plausible lines, 8 gives Shakespeare.

**Temperature.** `--temperature 0` always picks the most likely token, which loops forever ("I will be so shall the shall the shall the"). 1 uses the raw
counts. Above 1 flattens the distribution and gets stranger. `--top-k` and
`--top-p` cut off the tail before sampling instead.

## generate.py

```bash
python generate.py -p "First Citizen:" -n 400 --order 5 -t 0.8
python generate.py --order 3 --explain -p "The " -n 6
python generate.py --level word --order 2 -p "To be"
python generate.py -p "ROMEO:" --stop '\n\n'
```

| option               | what it does                                                                                 | default     |
|----------------------|----------------------------------------------------------------------------------------------|-------------|
| `-i, --input`        | text file to learn from                                                                      | `input.txt` |
| `--download`         | fetch tinyshakespeare if the input file is missing                                           | off         |
| `--level char\|word` | what counts as one token                                                                     | `char`      |
| `--lower`            | lowercase the corpus before training                                                         | off         |
| `-k, --order`        | context length, in tokens                                                                    | `4`         |
| `--min-count`        | how often a context must have been seen to be trusted; below that, back off to a shorter one | `1`         |
| `--smoothing`        | add-k smoothing: give every token of the vocabulary this many extra counts                   | `0`         |
| `-t, --temperature`  | 0 = always the most likely token, 1 = the raw counts, above 1 = wilder                       | `1.0`       |
| `--top-k`            | sample only from the k most likely tokens                                                    | off         |
| `--top-p`            | nucleus sampling threshold                                                                   | off         |
| `--seed`             | random seed, for reproducible runs                                                           | none        |
| `-p, --prompt`       | text to continue; omit it for an interactive session                                         | none        |
| `-n, --length`       | how many tokens to generate                                                                  | `500`       |
| `--stop`             | stop as soon as this string appears (`\n` works)                                             | none        |
| `--delay`            | seconds between tokens, for live demos                                                       | `0`         |
| `--explain`          | per token: the context length used and the top candidates                                    | off         |
| `-q, --quiet`        | print the generated text and nothing else                                                    | off         |

`--explain` is the interesting one for showing what the model is doing:

```
    [context 3 | 9 candidates | h 62%, o 26%, r 5%, i 3%, e 2%]
```

### Interactive session

Run without `-p` and you get a prompt. Type text to continue it, an empty line to start from a random line of the corpus, or a command:

```
/help                  the list
/config                current settings
/temp 0.8              temperature
/len 400               tokens to generate
/order 5               context length (rebuilds the model)
/topk 10   /topp 0.9   candidate filtering (0 = off)
/smooth 0.1            add-k smoothing
/mincount 2            minimum times a context must be seen
/explain               toggle the per-token explanation
/seed 42               reseed the random generator
/quit                  leave
```

## stats.py

```bash
python stats.py --top 20
python stats.py --context "the "        # what follows a given context
python stats.py --level word --max-order 3
python stats.py --skip vocab,chars,words,lines   # just the model-related parts
```

| option               | what it does                                                                                               | default         |
|----------------------|------------------------------------------------------------------------------------------------------------|-----------------|
| `-i, --input`        | text file to describe                                                                                      | `input.txt`     |
| `--download`         | fetch tinyshakespeare if the input file is missing                                                         | off             |
| `--top`              | how many entries in each top-N list                                                                        | `12`            |
| `--level char\|word` | token type for the predictability table                                                                    | `char`          |
| `--max-order`        | longest context to measure                                                                                 | `4`             |
| `--context`          | inspect what follows this exact context                                                                    | most common one |
| `--ascii`            | ASCII bars instead of box drawing characters                                                               | off             |
| `--skip`             | sections to skip: `counts`, `vocab`, `chars`, `words`, `lines`, `ngrams`, `predict`, `context`, `encoding` | none            |

Sections: file size and counts, the character vocabulary, character classes (with the non-ASCII characters listed separately, handy for accented text), most and
least frequent characters, word frequencies and lengths, line lengths and repeated lines, character pairs and triples, word pairs, the predictability table,
what follows one chosen context, and the character-to-integer encoding demo.

The predictability table is the link between the two scripts. On tinyshakespeare:

```
  order  bits/token  perplexity    contexts  avg next  only 1 option  text @ rate
      0       4.779       27.46           1     65.00          0.0%     650.7 KB
      1       3.538       11.62          65     21.58          4.6%     481.8 KB
      2       2.752        6.74       1,403      8.24         18.8%     374.7 KB
      3       2.161        4.47      11,556      4.39         34.0%     294.2 KB
      4       1.767        3.40      50,712      2.78         50.3%     240.6 KB
```

- **bits/token** — the average surprise of the next character once you know the context.
- **perplexity** — 2^bits, the effective number of choices left. At order 4, knowing the last 4 characters cuts 65 possible characters down to about 3.
- **contexts** — how many rows `generate.py` builds at that order.
- **only 1 option** — the share of contexts with exactly one continuation. Half of them at order 4: that is memorisation starting.
- **text @ rate** — the size of the file if it were coded at that rate. Careful:
  it is measured on the same text the counts came from, so a high order looks impressive mostly because it has learned the text by heart.

## Things to try

- Same seed, different order: `--seed 1 --order 2` then `3`, `4`, `6`, `8`. Watch words appear, then grammar, then whole memorised lines.
- Same order, different temperature: `-t 0` (a loop), `-t 0.5`, `-t 1`, `-t 2`.
- `--min-count 10` at order 8: forces the model to back off instead of reciting, so it stays fluent without copying.
- Feed it something other than English. It has no idea what language is; it works on chat logs, source code, or a CSV just as well.
- Compare `stats.py --level char` and `--level word` on the same file:
  characters are far more predictable than words.

## Limits

The model has no memory beyond `--order` tokens, no meaning, and no way to plan a sentence. It cannot answer a question — it only continues text that looks like
what it was given. A modern language model differs in scale and in mechanism: it learns a compressed representation instead of a lookup table, so it can handle
contexts it has never seen. But the last step is the same one you can read in `sample()`: a probability for every possible next token, and a draw from that
distribution.

Memory grows fast with `--order`, since the table stores every context that appears. On a 1.1 MB file: about 90 MB at order 4, 400 MB at order 7. Above that,
use a smaller text.