"""The agreement page of an audit (mmla ses-code --audit ID): beside each audit's page, at <page>/agreement
(/sensing/agreement, /transcription/agreement), how far two auditors agree, from the answers as they are at each load.
It is reached by its address only: no page links to it.

It is the agreement of the auditors with each other, never the system's accuracy: like the rest of the audit's server
it reads the plan, the views and the answers files (and campaign.yml, whose claimed links say whose answers count),
never a pipeline*.json. It shows counts and statistics only, never a transcript, a note or any other text an auditor
typed but their names.

The answers are those the scores count (audit_score.read_answers): the lines a claimed audit link saved for its own
auditor and, in an open audit, the lines the open page saved, under the name typed; of an item's phase the line that
counts is the scores' (an identity's first, any other phase's last, a transcription's last). Practice items and
flagged answers are left out. The page lists each name with answers, the scope its answers chose (full, reliability,
or mixed) and how many items it answered (and how many of them it flagged), then compares two names chosen on the page
(by default the name that chose the full audit with the most items against the reliability name it shares most items
with, else the two names that share most items) on the items both answered:

  sensing        per question of audit_score.interauditor (the roster, who is in each box, the members without a box,
                 where the gaze lands, who speaks): the pairs, the share agreeing, Cohen's kappa over the answers either
                 gave (code.kappa, the scores' value), Gwet's AC1 over the question's scale, the most frequent answer's
                 share and the confusion matrix (rows the first name, columns the second); the gaze over single
                 classes, its between and cannot-tell answers counted apart (the scores' gaze row counts them as
                 answers, so its pairs, share and kappa are not the page's)
  transcription  the WER of each name's transcript against the other's as the reference (N1 and the alignment of
                 audit_score_transcript.item_pair_counts, pooled over the items whose reference has words, with the
                 substitutions, deletions and insertions, and the CER over the same items), then the share agreeing,
                 kappa, AC1 and the confusion matrix of status, who speaks, adult, peer, other_offtask and overlap; a
                 transcript the grammar now refuses, or its status now disagrees with, leaves its item out, counted,
                 as it leaves the scores' units

No interval is given: the scores give them.
"""
from __future__ import annotations

import json
from collections import Counter

from openmmla.commands.ses import audit_page as P
from openmmla.commands.ses import audit_score as S
from openmmla.commands.ses import audit_score_transcript as T
from openmmla.commands.ses import audit_text as X
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

# the questions of a sensing audit, as audit_score.interauditor names them, in the page's order, with their titles
SENSING_QUESTIONS = (('roster: wears the badge', 'Roster: wears the badge'),
                     ('identity: who is in a box', 'Identity: who is in a box'),
                     ('identity: members without a box', 'Identity: members without a box'),
                     ('gaze: where', 'Gaze: where, single classes'),
                     ('who speaks', 'Who speaks'))
# the questions of a transcription audit after its transcript, with their titles
TRANSCRIPT_FIELDS = (('status', 'Whether anyone speaks'), ('who', 'Who speaks'), ('adult', 'Whether an adult speaks'),
                     ('peer', 'Whether pupils of this group talk'),
                     ('other_offtask', "Whether another group's pupils talk off the task"),
                     ('overlap', 'Whether voices overlap'))
TITLES = {'sensing': 'Sensing audit', 'transcription': 'Transcription audit'}


# ---- the statistics ----

def ac1(matrix: list[list[int]]) -> float | None:
    """Gwet's AC1 of a square confusion matrix (rows one auditor, columns the other) over its q categories, the
    question's scale: (pa - pe) / (1 - pe), pa the share agreeing, pe the sum over the categories of pi (1 - pi)
    over q - 1, pi the category's share of both auditors' answers. None when the matrix is empty or q < 2. Unlike
    kappa it stays near the share agreeing when one answer is far the most frequent (as the roster's yes)."""
    q, n = len(matrix), sum(map(sum, matrix))
    if not n or q < 2:
        return None
    pa = sum(matrix[i][i] for i in range(q)) / n
    shares = [(sum(matrix[k]) + sum(row[k] for row in matrix)) / (2 * n) for k in range(q)]
    pe = sum(p * (1 - p) for p in shares) / (q - 1)
    return (pa - pe) / (1 - pe) if pe < 1 else None


def _round(value) -> float | None:
    return None if value is None else round(float(value), 4)


def table(pairs, scale) -> dict:
    """two auditors' agreement on one question from its pairs (the first's answer, the second's): the pairs (n), the
    share agreeing, Cohen's kappa where both gave more than one answer between them (code.kappa: the scores' value,
    whatever the categories never given), Gwet's AC1 over the question's scale, the most frequent answer and its
    share of both auditors' answers, and the confusion matrix over the scale, an answer outside it after it"""
    values = [(str(x), str(y)) for x, y in pairs]
    seen = {v for pair in values for v in pair}
    categories = [str(c) for c in scale] + sorted(seen - {str(c) for c in scale})
    at = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in categories]
    for x, y in values:
        matrix[at[x]][at[y]] += 1
    n = len(values)
    given = Counter(v for pair in values for v in pair)
    top = max(categories, key=lambda c: (given[c], -at[c])) if n else None
    return {'n': n, 'agree': _round(sum(1 for x, y in values if x == y) / n) if n else None,
            'kappa': _round(C.kappa(matrix)) if len(seen) > 1 else None, 'ac1': _round(ac1(matrix)),
            'top': None if top is None else {'answer': top, 'share': _round(given[top] / (2 * n))},
            'categories': categories, 'matrix': matrix}


# ---- the names ----

def _plan(audit: dict) -> dict:
    """what audit_score.read_answers reads of the plan: the audit's id and its sessions"""
    return {'audit_id': audit['audit_id'],
            'sessions': [{'id': audit['sessions'][alias]['id']} for alias in audit['order']]}


def _counted(phases: dict, phase: str | None) -> list[dict]:
    """an item's records the page counts: those of `phase` (any phase when None), none of a practice item"""
    records = [r for p, r in phases.items() if phase is None or p == phase]
    return [] if any(r.get('practice') for r in records) else records


def _items(answers: dict, name: str, phase: str | None) -> set:
    return {item for item, phases in (answers.get(name) or {}).items() if _counted(phases, phase)}


def names_of(answers: dict, scopes: dict, phase: str | None) -> list[dict]:
    """each name with answers: the scope its answers chose, the items it answered (practice left out) and how many of
    them it flagged, by the most items first"""
    rows = []
    for name in answers:
        items = flagged = 0
        for phases in answers[name].values():
            records = _counted(phases, phase)
            if records:
                items += 1
                flagged += any((r.get('answer') or {}).get('flag') is True for r in records)
        if items:
            rows.append({'name': name, 'scope': S._scope_of(scopes.get(name, set())), 'items': items,
                         'flagged': flagged})
    return sorted(rows, key=lambda r: (-r['items'], r['name']))


def default_pair(names: list[dict], shared) -> dict | None:
    """the two names compared at first: the name that chose the full audit with the most items against the reliability
    name it shares most items with; else the two names that share most items (`shared(a, b)` counts them)"""
    order = [r['name'] for r in names]
    full = [r['name'] for r in names if r['scope'] == 'full']
    reliable = [r['name'] for r in names if r['scope'] == 'reliability']
    if full and reliable:
        a = full[0]
        return {'a': a, 'b': min(reliable, key=lambda b: (-shared(a, b), order.index(b)))}
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    if not pairs:
        return None
    a, b = min(pairs, key=lambda p: (-shared(*p), order.index(p[0]), order.index(p[1])))
    return {'a': a, 'b': b}


# ---- the two kinds ----

def _sensing_scales(audit: dict) -> dict:
    """each question's scale: the answers its page offers"""
    views = [s['view'] for s in audit['sessions'].values()]
    letters = sorted({letter for view in views for letter in view.get('pupils') or []})
    most = max([1] + [int(view.get('group_size') or 0) for view in views])
    return {'roster: wears the badge': list(P.WEARS), 'identity: who is in a box': letters + list(P.BOX_VALUES),
            'identity: members without a box': [str(i) for i in range(most + 1)], 'gaze: where': list(P.GAZE_VALUES),
            'who speaks': list(P.SPEAKERS)}


def sensing_pair(audit: dict, answers: dict, a: str, b: str) -> dict:
    """two names' agreement per question of the sensing audit (audit_score.interauditor_pairs)"""
    pairs = S.interauditor_pairs(answers[a], answers[b])
    scales = _sensing_scales(audit)
    questions = []
    for key, title in SENSING_QUESTIONS:
        values = pairs.get(key, [])
        row = {'question': key, 'title': title}
        if key == 'gaze: where':
            # single classes only: a between or a cannot tell is counted apart, for each name
            single = [(x, y) for x, y in values if x in P.GAZE_VALUES and y in P.GAZE_VALUES]
            row['apart'] = {'pairs': len(values) - len(single),
                            'between': [sum(1 for x, _ in values if x == 'between'),
                                        sum(1 for _, y in values if y == 'between')],
                            'cannot_tell': [sum(1 for x, _ in values if x == 'cannot_tell'),
                                            sum(1 for _, y in values if y == 'cannot_tell')]}
            values = single
        questions.append({**row, **table(values, scales[key])})
    return {'questions': questions}


def _yes_no(value) -> str:
    return 'yes' if value is True else 'no' if value is False else str(value)


def transcript_pair(answers: dict, a: str, b: str) -> dict:
    """two names' agreement on the transcriptions both saved (practice and flagged ones left out, and as the scores'
    units leave them out, a transcript the grammar now refuses and one its status now disagrees with): the WER of
    each against the other as the reference, then the questions after the transcript"""
    from openmmla.commands.ses import audit_transcript_page as TP
    units, left = [], Counter()
    for item in sorted(_items(answers, a, 'transcribe') & _items(answers, b, 'transcribe')):
        mine, theirs = (answers[name][item]['transcribe'].get('answer') or {} for name in (a, b))
        # in the order audit_score_transcript.build_units checks each auditor's: the flag, the grammar, the status
        if mine.get('flag') or theirs.get('flag'):
            left['flagged'] += 1
            continue
        try:
            lines = [X.parse_reference(said.get('transcript') or '') for said in (mine, theirs)]
        except X.TranscriptError:
            left['refused by the grammar'] += 1
            continue
        if any(X.status_error(said.get('status'), parsed) for said, parsed in zip((mine, theirs), lines)):
            left['at odds with its status'] += 1
            continue
        units.append((mine, theirs, T.item_pair_counts(answers, a, b, item)))
    wer = []
    for key, reference, hypothesis in (('ab', a, b), ('ba', b, a)):
        kept = [counts[key] for _, _, counts in units if counts[key]['N'] >= 1]
        sums = {k: sum(c[k] for c in kept) for k in ('N', 'H', 'S', 'D', 'I', 'E', 'cN', 'cE')}
        wer.append({'reference': reference, 'hypothesis': hypothesis, 'items': len(kept), **sums,
                    'wer': _round(sums['E'] / sums['N']) if sums['N'] else None,
                    'cer': _round(sums['cE'] / sums['cN']) if sums['cN'] else None})
    scales = {'status': list(X.STATUSES), **{f: list(v) for f, v in TP.QUESTIONS.items()}, 'overlap': ['yes', 'no']}
    questions = [{'question': field, 'title': title,
                  **table([(_yes_no(x.get(field)), _yes_no(y.get(field))) for x, y, _ in units], scales[field])}
                 for field, title in TRANSCRIPT_FIELDS]
    return {'items': len(units), 'left': dict(left), 'wer': wer, 'questions': questions}


def agreement(kind: str, audit: dict, artifacts, campaign: dict, typed: bool, a: str | None = None,
              b: str | None = None) -> tuple[int, dict]:
    """(status, the agreement page's data): the names with answers, the pair compared first and, for `a` and `b` (else
    that pair), their agreement. `kind` is 'sensing' or 'transcription', `campaign` campaign.yml's data and `typed`
    whether the open page's lines count (an open audit)"""
    answers, left, scopes = S.read_answers(artifacts, _plan(audit), campaign, typed)
    phase = 'transcribe' if kind == 'transcription' else None
    names = names_of(answers, scopes, phase)
    listed = {r['name'] for r in names}

    def shared(x: str, y: str) -> int:
        return len(_items(answers, x, phase) & _items(answers, y, phase))
    default = default_pair(names, shared)
    out = {'kind': kind, 'audit_id': audit['audit_id'], 'open': bool(typed), 'names': names, 'default': default,
           'left_out': dict(left), 'pair': None}
    if a is None and b is None:
        if default is None:
            return 200, out
        a, b = default['a'], default['b']
    if a not in listed or b not in listed:
        return 404, {'error': 'choose two names with answers'}
    if a == b:
        return 400, {'error': 'choose two different names'}
    pair = {'a': a, 'b': b, 'shared': shared(a, b),
            'shared_with_a': {r['name']: shared(a, r['name']) for r in names if r['name'] != a}}
    pair.update(transcript_pair(answers, a, b) if kind == 'transcription' else sensing_pair(audit, answers, a, b))
    out['pair'] = pair
    return 200, out


# ---- the page ----

AGREEMENT_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title><!-- title --> agreement</title>
<style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--text:#eee;--dim:#999;--accent:#ffd400;--good:#5fd38d;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif}
header{display:flex;gap:10px 16px;align-items:center;flex-wrap:wrap;padding:8px 16px;border-bottom:1px solid var(--line)}
header b{font-size:15px}header label{display:flex;gap:6px;align-items:center;min-width:0;max-width:100%}
select{font:inherit;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:4px;padding:4px 8px;min-width:0;max-width:62vw}
main{padding:4px 16px 24px;max-width:1100px}h2{font-size:15px;margin:20px 0 8px}
.meta{color:var(--dim);font-size:13px;max-width:760px}.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:13px}th,td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:top}
th{color:var(--dim);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
td.diag{background:#1f3a26}td.zero{color:#555}td.wrap{white-space:normal;min-width:9em}details{margin:8px 0}summary{cursor:pointer}
#status{color:var(--good)}#status.fail{color:var(--bad);font-weight:600}
</style></head><body>
<header><b><!-- title -->: agreement</b>
<label>Auditor A <select id="a"></select></label>
<label>Auditor B <select id="b"></select></label>
<span id="status"></span></header>
<main>
<p class="meta">How far two auditors agree with each other on the items both answered, never how far the system is right. Practice items and flagged answers are left out, and of each item the answer that counts is the one the scores count. κ is Cohen's kappa over the answers either auditor gave, the value the scores give, but for the gaze: this page compares the gaze on single classes, while the scores also count between and cannot tell as answers. AC1 is Gwet's over the answers the question offers, which stays near the share agreeing when one answer is far the most frequent. No answer, transcript or note is shown. The numbers are those of the answers saved when the page was loaded.</p>
<div id="names"></div>
<div id="summary"></div>
<div id="wer"></div>
<div id="questions"></div>
<div id="matrices"></div>
</main>
<script>
const $ = id => document.getElementById(id);
// where the server serves this audit's routes, set by the server
const BASE = '';
let names = [], generation = 0;
function recall(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }
function esc(text) { return String(text ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function said(text, failed) { $('status').textContent = text; $('status').className = failed ? 'fail' : ''; }
async function api(path) {
  const r = await fetch(BASE + path), data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `status ${r.status}`);
  return data;
}
function pct(x) { return x == null ? '–' : `${(100 * x).toFixed(1)} %`; }
function num(x) { return x == null ? '–' : (Math.abs(x) < 0.005 ? 0 : x).toFixed(2); }
function answer(text) { return String(text).replace(/_/g, ' '); }
function namesTable(data) {
  if (!data.names.length) return '<h2>Names</h2><p class="meta">No answers yet.</p>';
  const a = data.pair ? data.pair.a : null, shared = data.pair ? data.pair.shared_with_a : {};
  let html = '<h2>Names with answers</h2><div class="scroll"><table><tr><th>name</th><th>scope</th><th class="num">items</th><th class="num">flagged</th>' +
    `<th class="num">shared with ${esc(a ?? 'A')}</th></tr>`;
  html += data.names.map(r => `<tr><td>${esc(r.name)}</td><td>${esc(r.scope)}</td><td class="num">${r.items}</td><td class="num">${r.flagged}</td>` +
    `<td class="num">${r.name === a ? '' : shared[r.name] ?? '–'}</td></tr>`).join('');
  return html + '</table></div>';
}
function werTable(pair) {
  if (!pair.wer) return '';
  let html = '<h2>Transcripts</h2><p class="meta">The WER of B against A takes A\'s transcript as the reference: both normalised and aligned as the scores do it, pooled over the items where the reference has words. The CER is over the characters of the same items.</p>' +
    '<div class="scroll"><table><tr><th>hypothesis</th><th>reference</th><th class="num">items</th><th class="num">words</th><th class="num">sub</th><th class="num">del</th><th class="num">ins</th><th class="num">WER</th><th class="num">CER</th></tr>';
  html += pair.wer.map(r => `<tr><td>${esc(r.hypothesis)}</td><td>${esc(r.reference)}</td><td class="num">${r.items}</td><td class="num">${r.N}</td><td class="num">${r.S}</td>` +
    `<td class="num">${r.D}</td><td class="num">${r.I}</td><td class="num">${pct(r.wer)}</td><td class="num">${pct(r.cer)}</td></tr>`).join('');
  const left = Object.entries(pair.left || {}).map(([why, n]) => `${n} ${esc(why)}`).join(', ');
  return html + `</table></div><p class="meta">${pair.items} item${pair.items === 1 ? '' : 's'} compared${left ? `; left out: ${left}` : ''}.</p>`;
}
function questionTable(pair) {
  let html = '<h2>Questions</h2><div class="scroll"><table><tr><th>question</th><th class="num">pairs</th><th class="num">agree</th>' +
    '<th class="num" title="Cohen\'s kappa">κ</th><th class="num" title="Gwet\'s AC1">AC1</th><th>most frequent</th></tr>';
  html += pair.questions.map(q => `<tr><td class="wrap">${esc(q.title)}</td><td class="num">${q.n}</td><td class="num">${pct(q.agree)}</td><td class="num">${num(q.kappa)}</td>` +
    `<td class="num">${num(q.ac1)}</td><td>${q.top ? `${esc(answer(q.top.answer))} (${pct(q.top.share)})` : '–'}</td></tr>`).join('');
  html += '</table></div>';
  const apart = (pair.questions.find(q => q.apart) || {}).apart;
  if (apart) html += `<p class="meta">The gaze is compared on single classes; ${apart.pairs} box${apart.pairs === 1 ? '' : 'es'} left out: between two classes ${apart.between[0]} for ${esc(pair.a)} and ${apart.between[1]} for ${esc(pair.b)}, cannot tell ${apart.cannot_tell[0]} and ${apart.cannot_tell[1]}. The scores' gaze row counts these as answers.</p>`;
  return html;
}
function matrix(q, pair) {
  const cols = q.categories.map((_, j) => q.matrix.reduce((sum, r) => sum + r[j], 0));
  let html = `<details><summary>${esc(q.title)} (${q.n} pair${q.n === 1 ? '' : 's'}): rows ${esc(pair.a)}, columns ${esc(pair.b)}</summary><div class="scroll"><table><tr><th></th>` +
    q.categories.map(c => `<th class="num">${esc(answer(c))}</th>`).join('') + '<th class="num">total</th></tr>';
  html += q.matrix.map((r, i) => `<tr><th>${esc(answer(q.categories[i]))}</th>` +
    r.map((v, j) => `<td class="num${i === j ? ' diag' : v ? '' : ' zero'}">${v}</td>`).join('') + `<td class="num">${r.reduce((x, y) => x + y, 0)}</td></tr>`).join('');
  return html + `<tr><th>total</th>${cols.map(v => `<td class="num">${v}</td>`).join('')}<td class="num">${q.n}</td></tr></table></div></details>`;
}
function draw(data) {
  $('names').innerHTML = namesTable(data);
  const pair = data.pair;
  if (!pair) {
    $('summary').innerHTML = `<p class="meta">${data.names.length < 2 ? 'Agreement needs a second name with answers.' : 'Choose two different names.'}</p>`;
    $('wer').innerHTML = $('questions').innerHTML = $('matrices').innerHTML = ''; return;
  }
  $('summary').innerHTML = `<h2>${esc(pair.a)} and ${esc(pair.b)}</h2><p class="meta">${pair.shared} item${pair.shared === 1 ? '' : 's'} both answered.</p>`;
  $('wer').innerHTML = werTable(pair);
  $('questions').innerHTML = questionTable(pair);
  $('matrices').innerHTML = '<h2>Confusion matrices</h2>' + pair.questions.filter(q => q.n).map(q => matrix(q, pair)).join('');
}
async function show() {
  const a = $('a').value, b = $('b').value, gen = ++generation;
  remember(`${BASE}/agreementA`, a); remember(`${BASE}/agreementB`, b);
  if (!a || !b || a === b) { draw({names, pair: null}); return; }
  let data;
  try { data = await api(`/api/agreement?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`); }
  catch (e) { if (gen === generation) said(`not loaded: ${e.message}`, true); return; }
  if (gen !== generation) return;
  said(''); names = data.names; draw(data);
}
(async () => {
  let data;
  try { data = await api('/api/agreement'); }
  catch (e) { said(`not loaded: ${e.message}`, true); return; }
  names = data.names;
  const option = r => `<option value="${esc(r.name)}">${esc(r.name)} (${esc(r.scope)}, ${r.items})</option>`;
  $('a').innerHTML = $('b').innerHTML = names.map(option).join('');
  const listed = names.map(r => r.name), pick = (key, fallback) => { const v = recall(key); return v && listed.includes(v) ? v : fallback; };
  const first = data.default || {};
  $('a').value = pick(`${BASE}/agreementA`, first.a || ''); $('b').value = pick(`${BASE}/agreementB`, first.b || '');
  for (const id of ['a', 'b']) $(id).onchange = show;
  // the pair the server compared first, unless this browser remembers another
  if (data.pair && $('a').value === data.pair.a && $('b').value === data.pair.b) draw(data); else show();
})();
</script></body></html>
"""
# the line of the page's script that says where the audit's routes are, which the server fills in
BASE_LINE = "const BASE = '';"


def page(kind: str, base: str) -> str:
    """the agreement page of an audit of `kind` ('sensing' or 'transcription') served under the path `base`"""
    out = AGREEMENT_PAGE.replace('<!-- title -->', TITLES[kind])
    if out.count(BASE_LINE) != 1:
        raise RuntimeError(f'the agreement page changed: {BASE_LINE!r} is not in it once; update audit_agreement.page')
    return out.replace(BASE_LINE, f'const BASE = {json.dumps(base)};')
