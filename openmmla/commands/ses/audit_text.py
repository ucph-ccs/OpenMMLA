"""The text of a transcription audit (mmla ses-code --audit-sample ID --audit-task transcript): the grammar
of an auditor's transcript, which the page checks before it saves one and the scorer reads again, the
two normalisers that put a transcript and the ASR's text into the same words, and the aligner that
counts the errors between them.

A transcript is at most MAX_LINES lines (one per turn) of at most MAX_LINE characters, MAX_TEXT in all,
each its tag, a space and the words: M: a pupil of this group, T: the teacher or another adult, O: a
pupil of another group, ?: cannot tell. The words are letters (with their combining marks), spaces,
. , ? ! ' and -, the markers [x] (a stretch that cannot be written down) and [bg] (background talk that
cannot be understood), and a guess in braces, {ord} (letters, spaces, ' and - only, never nested). A
marker or a guess stands apart: a space or the line's start before it, a space, . , ? ! or the line's
end after it. Numbers are written as words. The text is taken in NFC, each line trimmed and its runs
of spaces (any Unicode space) made one, a typographic apostrophe made ', empty lines dropped; anything
else is refused, the line and the kind of character or the rule named (parse_reference raises
TranscriptError, a ValueError), so the page can say it as it is. A word token is a run of letters
outside the markers, a fragment (mi-) and a guessed word included, and a line holds at least one word
or marker.

N1, the primary normaliser, runs on both sides in this order:
  1. fragments out: a token that ends in a hyphen, punctuation after the hyphen aside (the reference's
     tags and markers are never tokens, and {w} is w, before this)
  2. NFKC and casefold; ä→æ, ö→ø, é/è/ê→e, á/à→a, ü→u; æ, ø and å kept, aa left as it is
  3. a thousands dot before exactly three digits out (1.000 is 1000), a decimal comma ' komma ', and
     each run of digits from 0 to 9999 spelled as a Danish cardinal (da_number); ordinals and clock
     times are not handled, and count as errors (number_tokens counts the tokens with digits)
  4. punctuation and symbols (Unicode P*, S*) a space, apostrophes deleted (hva' is hva)
  5. split on whitespace
  6. the variant map (VARIANTS): several tokens at once, the longest match first, never applied twice
  7. the fillers (FILLERS) deleted; ja, jo, nej, næ, nå and okay are words
N0, the sensitivity, is steps 1, NFKC with casefold, and 4 alone. Content words are N1 tokens less
Snowball's Danish stopwords (STOPWORDS_DA) and the response words (RESPONSE_WORDS). The LLM form of a
transcript (llm_form), which the content model is run again on, is its lines' words without tags,
markers or fragments, a guess without its braces, the auditor's casing and punctuation kept, one line
per turn.

The aligner (align) is a Levenshtein alignment of two token lists in plain Python with unit costs and
a backtrace, its ties broken from the end the same way every time: a hit or a substitution first, then
a deletion, then an insertion. It returns the hits, substitutions, deletions and insertions with the
operations, so each reference token's tag can carry its own count (by_tag). A hypothesis token may be
optional: left out it costs nothing, matched it is a hit and substituted a substitution. CER (cer) is
the same alignment over the characters of the N1 tokens joined by single spaces, the spaces counted.
split_merge joins two adjacent tokens of one side whose concatenation is a token of the other side
(klasse værelse against klasseværelse), and bag counts the tokens two multisets share.

Frozen at sampling: the plan records NORMALISER_VERSION and this file's sha256. The variant map, the
fillers and the stopwords are constants here, built before sampling from Retskrivningsordbogen's
variants, the spec's list and the spellings counted in the ASR's own text (never in an auditor's); a
later addition is a map v2, given to normalise as `variants=` and reported as a sensitivity, never an
edit of VARIANTS.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, NamedTuple

from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

NORMALISER_VERSION = 1
LEVELS = ('N1', 'N0')
TAGS = ('M', 'T', 'O', '?')
MARKERS = ('[x]', '[bg]')
STATUSES = ('speech', 'unintelligible', 'none')
MAX_LINES = 12
MAX_LINE = 300
MAX_TEXT = 2000
LINE = re.compile(r'(M|T|O|\?): (.+)')
# what may follow a marker or a guess, and the punctuation a line's words may carry
SENTENCE_PUNCTUATION = '.,?!'
WORD_PUNCTUATION = "'-"
# the brackets refused by name (Unicode Ps and Pe are refused as brackets too)
BRACKETS = '<>'
# the apostrophe, the right single quotation mark and the modifier letter apostrophe: a transcript takes each
# as an apostrophe, and N1 and N0 delete them
APOSTROPHES = "'\u2019\u02bc"
# the hyphen-minus, the hyphen and the no-break hyphen
HYPHENS = '-\u2010\u2011'
# the characters refused anywhere in a transcript, by Unicode category: (rule, what the page says, advice)
REFUSED = {'Cc': ('control', 'a control character', ''), 'Cf': ('format', 'an invisible formatting character', ''),
           'Co': ('private', 'a private-use character', ''), 'Cs': ('surrogate', 'a lone surrogate', ''),
           'Cn': ('unassigned', 'an unassigned character', ''),
           'Zl': ('separator', 'a line separator', ': start a new line with Enter'),
           'Zp': ('separator', 'a paragraph separator', ': start a new line with Enter')}
# N1 step 2, after casefold
LETTER_MAP = str.maketrans({'ä': 'æ', 'ö': 'ø', 'é': 'e', 'è': 'e', 'ê': 'e', 'á': 'a', 'à': 'a', 'ü': 'u'})
THOUSANDS = re.compile(r'(?<=[0-9])\.(?=[0-9]{3}(?![0-9]))')
DECIMAL = re.compile(r'(?<=[0-9]),(?=[0-9])')
DIGITS = re.compile(r'[0-9]+')
UNITS = ('nul', 'en', 'to', 'tre', 'fire', 'fem', 'seks', 'syv', 'otte', 'ni', 'ti', 'elleve', 'tolv', 'tretten',
         'fjorten', 'femten', 'seksten', 'sytten', 'atten', 'nitten')
TENS = {2: 'tyve', 3: 'tredive', 4: 'fyrre', 5: 'halvtreds', 6: 'tres', 7: 'halvfjerds', 8: 'firs', 9: 'halvfems'}
# map v1 (N1 step 6): each key's tokens, in N1's form up to step 5, become the value's tokens
VARIANTS = {
    # reduced spoken spellings, the unambiguous ones
    'hva': 'hvad', 'ik': 'ikke', 'ikk': 'ikke', 'ska': 'skal', 'mej': 'mig', 'dej': 'dig',
    # response words and their spellings
    'jah': 'ja', 'jaa': 'ja', 'jaaa': 'ja', 'nåh': 'nå', 'næh': 'næ', 'ja men': 'jamen',
    'ok': 'okay', 'okej': 'okay', 'okey': 'okay', 'o k': 'okay',
    # abbreviations
    'fx': 'for eksempel', 'f eks': 'for eksempel', 'osv': 'og så videre', 'bl a': 'blandt andet',
    'kl': 'klokken', 'ca': 'cirka',
    # the task material, its number and definite forms kept apart
    'micro bit': 'microbit', 'micro bits': 'microbit', 'microbits': 'microbit', 'mikrobit': 'microbit',
    'mikrobits': 'microbit', 'mikro bit': 'microbit', 'mikro bits': 'microbit',
    'microbiten': 'microbitten', 'mikrobiten': 'microbitten', 'mikrobitten': 'microbitten',
    'micro biten': 'microbitten', 'micro bitten': 'microbitten',
    'mikrobittene': 'microbittene', 'microbitsene': 'microbittene', 'mikrobitsene': 'microbittene',
    'microbitterne': 'microbittene', 'mikrobitterne': 'microbittene',
}
# N1 step 7: hesitations, deleted on both sides (the guide asks auditors to leave them out)
FILLERS = frozenset('øh øhm øhh æh æhm eh ehm ehh uh um hm hmm mm mmm mhm hmhm'.split())
# the response words: written by the auditors, kept by N1, left out of the content words
RESPONSE_WORDS = ('ja', 'jo', 'nej', 'næ', 'nå', 'okay')
# Snowball's Danish stop word list (snowballstem.org/algorithms/danish/stop.txt), in its order
STOPWORDS_DA = (
    'og', 'i', 'jeg', 'det', 'at', 'en', 'den', 'til', 'er', 'som', 'på', 'de', 'med', 'han', 'af', 'for', 'ikke',
    'der', 'var', 'mig', 'sig', 'men', 'et', 'har', 'om', 'vi', 'min', 'havde', 'ham', 'hun', 'nu', 'over', 'da',
    'fra', 'du', 'ud', 'sin', 'dem', 'os', 'op', 'man', 'hans', 'hvor', 'eller', 'hvad', 'skal', 'selv', 'her',
    'alle', 'vil', 'blev', 'kunne', 'ind', 'når', 'være', 'dog', 'noget', 'ville', 'jo', 'deres', 'efter', 'ned',
    'skulle', 'denne', 'end', 'dette', 'mit', 'også', 'under', 'have', 'dig', 'anden', 'hende', 'mine', 'alt',
    'meget', 'sit', 'sine', 'vor', 'mod', 'disse', 'hvis', 'din', 'nogle', 'hos', 'blive', 'mange', 'ad', 'bliver',
    'hendes', 'været', 'thi', 'jer', 'sådan',
)
NOT_CONTENT = frozenset(STOPWORDS_DA) | frozenset(RESPONSE_WORDS)


class TranscriptError(ValueError):
    """a transcript refused: the message names the line and the kind of character or the rule (`rule`)"""

    def __init__(self, message: str, line: int | None = None, rule: str = ''):
        super().__init__(message)
        self.line, self.rule = line, rule


class Line(NamedTuple):
    """a line of a transcript: its tag, its words after the tag as cleaned, its word tokens (a guess's braces
    gone, the markers out, punctuation still on) and its markers, in order"""
    tag: str
    body: str
    tokens: tuple
    markers: tuple

    @property
    def text(self) -> str:
        return f'{self.tag}: {self.body}'


class Op(NamedTuple):
    """an operation of an alignment: H a hit, S a substitution, D a deletion (a reference token the
    hypothesis lacks), I an insertion, X an optional hypothesis token left out at no cost; `ref` and
    `hyp` are the tokens' indices, None on the side the operation has no token of"""
    kind: str
    ref: int | None
    hyp: int | None


class Alignment(NamedTuple):
    """the counts of an alignment and its operations in order; N = H + S + D (the reference's tokens),
    E = S + D + I (the errors)"""
    H: int
    S: int
    D: int
    I: int
    ops: tuple

    @property
    def N(self) -> int:
        return self.H + self.S + self.D

    @property
    def E(self) -> int:
        return self.S + self.D + self.I


class Merged(NamedTuple):
    """the two sides after split_merge, each new token's indices on its side before the joins"""
    ref: list
    hyp: list
    ref_from: list
    hyp_from: list


class Bag(NamedTuple):
    """the tokens two multisets share (hits) and each one's size"""
    hits: int
    ref: int
    hyp: int


# ---- the grammar of a transcript ----

def _code(ch: str) -> str:
    return f'U+{ord(ch):04X}'


def _shown(ch: str) -> str:
    """a character as the page can show it: itself in quotes when visible, its code point when not"""
    return _code(ch) if unicodedata.category(ch)[0] in 'CZM' else f"'{ch}' ({_code(ch)})"


def _is_letter(ch: str) -> bool:
    """a letter or a combining mark (on a letter, as NFC leaves a letter that has no composed form)"""
    return unicodedata.category(ch)[0] in 'LM'


def _has_letter(text: str) -> bool:
    return any(unicodedata.category(c)[0] == 'L' for c in text)


def _word_character(ch: str) -> bool:
    """a character the words of a line may hold outside the markers and braces"""
    return _is_letter(ch) or ch == ' ' or ch in WORD_PUNCTUATION or ch in SENTENCE_PUNCTUATION


def _refused_character(ch: str, kind: str) -> tuple[str, str]:
    """(rule, why) of a character the words of a line may not hold"""
    if kind in ('Ps', 'Pe') or ch in BRACKETS:
        return 'bracket', (f'the bracket {_shown(ch)}: a guess goes in {{ }}, a stretch that cannot be written '
                           'down is [x]')
    if kind[0] == 'P':
        return 'punctuation', f"the character {_shown(ch)}: the punctuation is . , ? ! ' and - only"
    if kind[0] == 'S':
        return 'symbol', f'the symbol {_shown(ch)}: write it as words'
    return 'character', f'the character {_shown(ch)} is not allowed'


def _cleaned(raw: str, number: int) -> str:
    """a line's characters checked, its spaces and typographic apostrophes made plain, trimmed and its runs
    of spaces made one"""
    chars = []
    for ch in raw:
        if ch == '\t':
            raise TranscriptError(f'line {number}: a tab: separate the words with spaces', number, 'tab')
        kind = unicodedata.category(ch)
        if kind in REFUSED:
            rule, what, advice = REFUSED[kind]
            raise TranscriptError(f'line {number}: {what} ({_code(ch)}){advice}', number, rule)
        chars.append(' ' if kind == 'Zs' else "'" if ch in APOSTROPHES else ch)
    return re.sub(' +', ' ', ''.join(chars)).strip()


def _apart_before(body: str, at: int) -> bool:
    return at == 0 or body[at - 1] == ' '


def _apart_after(body: str, end: int) -> bool:
    return end == len(body) or body[end] == ' ' or body[end] in SENTENCE_PUNCTUATION


def _body_error(body: str) -> tuple[str, str] | None:
    """(rule, why) of the first thing in a line's words the grammar refuses, or None"""
    group, at = None, 0
    while at < len(body):
        ch = body[at]
        kind = unicodedata.category(ch)
        if kind[0] == 'N':
            return 'digit', f'a digit, {_shown(ch)}: write numbers as words'
        if ch == '[' and group is None:
            marker = next((m for m in MARKERS if body.startswith(m, at)), None)
            if marker is None:
                return 'marker', 'square brackets only as the markers [x] and [bg]'
            if not _apart_before(body, at) or not _apart_after(body, at + len(marker)):
                return 'marker', f'{marker} stands apart: a space before it, a space or . , ? ! after it'
            at += len(marker)
            continue
        if ch == ']' and group is None:
            return 'marker', 'square brackets only as the markers [x] and [bg]'
        if ch == '{':
            if group is not None:
                return 'group', 'a { } inside another'
            if not _apart_before(body, at):
                return 'group', 'a guess in { } stands apart: a space before it, a space or . , ? ! after it'
            group = at
        elif ch == '}':
            if group is None:
                return 'group', 'a } without its {'
            if not _has_letter(body[group + 1:at]):
                return 'group', 'an empty { }'
            if not _apart_after(body, at + 1):
                return 'group', 'a guess in { } stands apart: a space before it, a space or . , ? ! after it'
            group = None
        elif group is not None and not (_is_letter(ch) or ch == ' ' or ch in WORD_PUNCTUATION):
            return 'group', "only letters, spaces, ' and - inside { }"
        elif not _word_character(ch):
            return _refused_character(ch, kind)
        at += 1
    if group is not None:
        return 'group', 'a { without its }'
    return None


def _tokens(body: str) -> tuple[list[str], list[str]]:
    """a line's word tokens (the braces gone, punctuation still on) and its markers, in order"""
    tokens, markers = [], []
    for token in body.replace('{', '').replace('}', '').split(' '):
        core = token.rstrip(SENTENCE_PUNCTUATION)
        if core in MARKERS:
            markers.append(core)
        elif _has_letter(token):
            tokens.append(token)
    return tokens, markers


def parse_reference(text) -> list[Line]:
    """the lines of an auditor's transcript, or TranscriptError (a ValueError) naming the line and the kind
    of character or the rule it breaks; an empty transcript has no lines"""
    if not isinstance(text, str):
        raise TranscriptError('the transcript is not text', rule='text')
    lines = []
    for number, raw in enumerate(unicodedata.normalize('NFC', text).replace('\r\n', '\n').split('\n'), 1):
        line = _cleaned(raw, number)
        if not line:
            continue
        if len(line) > MAX_LINE:
            raise TranscriptError(f'line {number} is longer than {MAX_LINE} characters', number, 'length')
        match = LINE.fullmatch(line)
        if match is None:
            if line[:-1] in TAGS and line.endswith(':'):
                raise TranscriptError(f'line {number}: write the words after the tag', number, 'empty')
            raise TranscriptError(f'line {number}: start the line with its tag and a space: M:, T:, O: or ?:',
                                  number, 'tag')
        tag, body = match.groups()
        error = _body_error(body)
        if error is not None:
            raise TranscriptError(f'line {number}: {error[1]}', number, error[0])
        tokens, markers = _tokens(body)
        if not tokens and not markers:
            raise TranscriptError(f'line {number}: write at least one word, [x] or [bg] after the tag', number,
                                  'empty')
        lines.append(Line(tag, body, tuple(tokens), tuple(markers)))
    if len(lines) > MAX_LINES:
        raise TranscriptError(f'at most {MAX_LINES} lines, one per turn', rule='lines')
    if len(reference_text(lines)) > MAX_TEXT:
        raise TranscriptError(f'the transcript is longer than {MAX_TEXT} characters', rule='length')
    return lines


def reference_text(lines: Iterable[Line]) -> str:
    """a transcript's text as parse_reference cleaned it, one line per turn (parsed again, the same lines)"""
    return '\n'.join(line.text for line in lines)


def status_error(status, lines: list[Line]) -> str | None:
    """why a status disagrees with a transcript's lines, or None: none has no lines, unintelligible no word
    (its lines only [x] and [bg]), speech at least one word"""
    if status not in STATUSES:
        return f"status is one of {', '.join(STATUSES)}"
    words = sum(len(line.tokens) for line in lines)
    if status == 'none' and lines:
        return 'no one speaks: leave the transcript empty'
    if status == 'unintelligible' and words:
        return 'unintelligible: write only [x] or [bg], or choose speech'
    if status == 'speech' and not words:
        return 'speech: write at least one word, or choose unintelligible'
    return None


def _lines(reference) -> list[Line]:
    return parse_reference(reference) if isinstance(reference, str) else list(reference)


def llm_form(reference) -> str:
    """the text the content model is run again on, from a transcript (its text or its lines): each line's
    words without its tag, markers and fragments, a guess without its braces, the auditor's casing and
    punctuation kept; lines with no word left out, the rest joined by newlines"""
    out = []
    for line in _lines(reference):
        words = []
        for token in line.body.replace('{', '').replace('}', '').split(' '):
            core = token.rstrip(SENTENCE_PUNCTUATION)
            if core in MARKERS or _fragment(core):
                # a marker or a fragment goes; the punctuation after it ends the word before it instead
                tail = token[len(core):]
                if tail and words:
                    words[-1] = words[-1].rstrip(SENTENCE_PUNCTUATION) + tail
            elif token:
                words.append(token)
        text = ' '.join(words).lstrip(' ' + SENTENCE_PUNCTUATION)
        if _has_letter(text):
            out.append(text)
    return '\n'.join(out)


# ---- the normalisers ----

def da_number(n: int) -> str:
    """n (0-9999) as a Danish cardinal: 1 en, 21 enogtyve, 101 hundrede og en, 1100 tusind et hundrede,
    2500 to tusind fem hundrede"""
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 9999:
        raise ValueError(f'da_number spells the whole numbers from 0 to 9999, not {n!r}')
    if n < 100:
        return _below_hundred(n)
    thousands, rest = divmod(n, 1000)
    hundreds, below = divmod(rest, 100)
    parts = []
    if thousands:
        parts.append('tusind' if thousands == 1 else f'{UNITS[thousands]} tusind')
    if hundreds:
        parts.append(('et hundrede' if thousands else 'hundrede') if hundreds == 1 else f'{UNITS[hundreds]} hundrede')
    if below:
        parts.append(f'og {_below_hundred(below)}')
    return ' '.join(parts)


def _below_hundred(n: int) -> str:
    if n < 20:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    return TENS[tens] if unit == 0 else f'{UNITS[unit]}og{TENS[tens]}'


def _spelled(match) -> str:
    value = int(match.group())
    return f' {da_number(value)} ' if value <= 9999 else match.group()


def _numbers(text: str) -> str:
    """N1 step 3: the thousands dot out, a decimal comma ' komma ', digit runs 0-9999 as words"""
    return DIGITS.sub(_spelled, DECIMAL.sub(' komma ', THOUSANDS.sub('', text)))


def _fragment(token: str) -> bool:
    """whether a token is a cut-off word: letters ending in a hyphen, with punctuation after it or not"""
    end = len(token)
    while end and token[end - 1] not in HYPHENS and unicodedata.category(token[end - 1])[0] in 'PS':
        end -= 1
    return end > 0 and token[end - 1] in HYPHENS and _has_letter(token[:end])


def _unpunctuated(text: str) -> str:
    """N1 step 4: punctuation and symbols a space, apostrophes and invisible formatting characters deleted,
    other separators and control characters a space"""
    out = []
    for ch in text:
        kind = unicodedata.category(ch)
        if ch in APOSTROPHES or kind == 'Cf':
            continue
        out.append(' ' if kind[0] in 'PSZC' else ch)
    return ''.join(out)


def _compiled(variants: dict) -> tuple[dict, int]:
    table = {tuple(key.split()): tuple(value.split()) for key, value in variants.items()}
    return table, max(map(len, table), default=0)


_VARIANT_TABLE = _compiled(VARIANTS)


def _mapped(pairs: list, compiled: tuple[dict, int]) -> list:
    """N1 step 6 on (token, origin) pairs: the longest key at each place, its value's tokens in its stead,
    their origin the key's tokens' together"""
    table, longest = compiled
    out, at = [], 0
    while at < len(pairs):
        for size in range(min(longest, len(pairs) - at), 0, -1):
            key = tuple(token for token, _ in pairs[at:at + size])
            if key in table:
                origin = tuple(sorted({i for _, o in pairs[at:at + size] for i in o}))
                out.extend((token, origin) for token in table[key])
                at += size
                break
        else:
            out.append(pairs[at])
            at += 1
    return out


def normalise_tracked(tokens, level: str = 'N1', variants: dict | None = None) -> tuple[list[str], list[tuple]]:
    """normalise's tokens and, for each, the indices of the input tokens it came from (a number spelled
    out gives several tokens of one input token, a variant one token of several)"""
    if level not in LEVELS:
        raise ValueError(f'the level is N1 or N0, not {level!r}')
    if isinstance(tokens, str):
        tokens = tokens.split()
    pairs = []
    for at, token in enumerate(tokens):
        text = unicodedata.normalize('NFKC', token)
        if _fragment(text):
            continue
        text = text.casefold()
        if level == 'N1':
            text = _numbers(text.translate(LETTER_MAP))
        for piece in _unpunctuated(text).split():
            # a piece of nothing but combining marks is no word
            if any(unicodedata.category(c)[0] in 'LN' for c in piece):
                pairs.append((piece, (at,)))
    if level == 'N1':
        compiled = _VARIANT_TABLE if variants is None else _compiled(variants)
        pairs = [(token, origin) for token, origin in _mapped(pairs, compiled) if token not in FILLERS]
    return [token for token, _ in pairs], [origin for _, origin in pairs]


def normalise(tokens, level: str = 'N1', variants: dict | None = None) -> list[str]:
    """tokens (a list, or a text split on whitespace) normalised at `level`, N1 or N0; `variants` replaces
    the variant map (a map v2 for a sensitivity) and is N1's only"""
    return normalise_tracked(tokens, level, variants)[0]


def reference_tokens(reference, level: str = 'N1', variants: dict | None = None) -> tuple[list[str], list[str]]:
    """a transcript's (its text or its lines) tokens normalised line by line, and each token's tag"""
    tokens, tags = [], []
    for line in _lines(reference):
        words = normalise(line.tokens, level, variants)
        tokens.extend(words)
        tags.extend([line.tag] * len(words))
    return tokens, tags


def content_words(tokens: Iterable[str]) -> list[str]:
    """the N1 tokens that are neither Snowball's Danish stopwords nor response words"""
    return [token for token in tokens if token not in NOT_CONTENT]


def number_tokens(tokens) -> int:
    """the tokens (a list, or a text split on whitespace) that hold a digit, before normalising"""
    if isinstance(tokens, str):
        tokens = tokens.split()
    return sum(1 for token in tokens if any(unicodedata.category(c) == 'Nd' for c in token))


# ---- the aligner ----

def _optional(optional, size: int) -> frozenset:
    """the optional hypothesis tokens' indices, from indices or from one flag per token"""
    if optional is None:
        return frozenset()
    optional = list(optional)
    if optional and all(isinstance(x, bool) for x in optional):
        if len(optional) != size:
            raise ValueError('optional flags come one per hypothesis token')
        return frozenset(at for at, flag in enumerate(optional) if flag)
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < size for x in optional):
        raise ValueError('optional holds indices of hypothesis tokens, or one flag per token')
    return frozenset(optional)


def align(ref, hyp, optional=None) -> Alignment:
    """the least-cost alignment of a reference's tokens with a hypothesis's (unit costs; an optional
    hypothesis token left out costs 0), its ties broken from the end: a hit or a substitution, then a
    deletion, then an insertion"""
    ref, hyp = list(ref), list(hyp)
    free = _optional(optional, len(hyp))
    insert = [0 if at in free else 1 for at in range(len(hyp))]
    # cost[i][j]: the least cost of ref[:i] against hyp[:j]
    first = [0]
    for j in range(len(hyp)):
        first.append(first[j] + insert[j])
    cost = [first]
    for i, token in enumerate(ref, 1):
        above, row = cost[-1], [i]
        for j, other in enumerate(hyp, 1):
            row.append(min(above[j - 1] + (token != other), above[j] + 1, row[j - 1] + insert[j - 1]))
        cost.append(row)
    ops, i, j = [], len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(Op('H' if ref[i - 1] == hyp[j - 1] else 'S', i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(Op('D', i - 1, None))
            i -= 1
        else:
            ops.append(Op('I' if insert[j - 1] else 'X', None, j - 1))
            j -= 1
    ops.reverse()
    counts = Counter(op.kind for op in ops)
    return Alignment(counts['H'], counts['S'], counts['D'], counts['I'], tuple(ops))


def cer(ref, hyp) -> Alignment:
    """the alignment of the characters of two token lists each joined by single spaces (a text is taken
    as it is), the spaces counted"""
    ref_text = ref if isinstance(ref, str) else ' '.join(ref)
    hyp_text = hyp if isinstance(hyp, str) else ' '.join(hyp)
    return align(ref_text, hyp_text)


def by_tag(alignment: Alignment, tags: list[str]) -> dict[str, dict[str, int]]:
    """per tag, N, H, S and D of the reference tokens of that tag (`tags` one per reference token); an
    insertion has no reference token, so no tag"""
    out = {tag: {'N': 0, 'H': 0, 'S': 0, 'D': 0} for tag in TAGS}
    for op in alignment.ops:
        if op.ref is not None:
            row = out.setdefault(tags[op.ref], {'N': 0, 'H': 0, 'S': 0, 'D': 0})
            row[op.kind] += 1
            row['N'] += 1
    return out


def _joined(side: list, other: list) -> tuple[list, list]:
    wanted = set(other)
    pairs = set(zip(other, other[1:]))
    out, origin, at = [], [], 0
    while at < len(side):
        pair = tuple(side[at:at + 2])
        # the other side's same two tokens side by side would match as they are
        if len(pair) == 2 and pair[0] + pair[1] in wanted and pair not in pairs:
            out.append(pair[0] + pair[1])
            origin.append((at, at + 1))
            at += 2
        else:
            out.append(side[at])
            origin.append((at,))
            at += 1
    return out, origin


def split_merge(ref, hyp) -> Merged:
    """the split/merge tolerance before aligning: on each side, two adjacent tokens whose concatenation is
    a token of the other side (as it was) become that token, left to right, unless the other side has
    the same two tokens side by side; a join may cross the reference's lines"""
    ref, hyp = list(ref), list(hyp)
    new_ref, ref_from = _joined(ref, hyp)
    new_hyp, hyp_from = _joined(hyp, ref)
    return Merged(new_ref, new_hyp, ref_from, hyp_from)


def bag(ref, hyp) -> Bag:
    """the multiset overlap of two token lists: shared tokens (each counted as often as both sides have
    it), and each side's size; precision is hits / hyp, recall hits / ref"""
    ref_counts, hyp_counts = Counter(ref), Counter(hyp)
    return Bag(sum((ref_counts & hyp_counts).values()), sum(ref_counts.values()), sum(hyp_counts.values()))
