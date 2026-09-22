"""the compared neural variant of the 10 s interaction classifier: set encoders over persons and
pairs (so no tag id or slot order reaches the model), a window fusion layer, and a short temporal
convolution over the session's windows; torch is imported only here, and only for -m net.

A session is one sequence over all its T windows, coded or not, read from the scaled tokens of
layout.py (a value whose mask is 0 is already 0 there):

    G           (T, 11)     the group token: 7 speech values, m_asr, m_transcription, m_dia, group_size / 3
    avail       (T, 3)      speech_ran, ips_ran, vfa_ran (the order of MODALITIES)
    P           (T, 3, 19)  the person slots: 13 values, then 6 masks (P_COLUMNS)
    P_exists    (T, 3)      1 where a slot holds a kept person
    Q           (T, 3, 13)  the pair slots: 8 values symmetric in a<->b, then 5 masks (Q_COLUMNS)
    Q_exists    (T, 3)      1 where a slot holds a kept pair
    pair_index  (3, 2)      the person slots of each pair slot: (0, 1), (0, 2), (1, 2)

with a label per window: 0 individual, 1 social, 2 collaborative, anything else (uncoded, unclear,
None, NaN) none. `session_tensors` turns a session into the dict every other function reads; the
pooled control (PooledNet) reads the 82-column pooled view instead of the tokens.

The training part follows the recipe fixed before any result: tempered class weights,
cross-entropy with label smoothing on coded windows only (uncoded and unclear windows are context),
48-window crops around coded windows, run-wise modality dropout, person dropout and noise, AdamW.
Only the epoch count is tuned (the median of the inner splits' best epochs), and the refit is a
5-seed ensemble whose mean softmax is the prediction; no seed is ever picked.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openmmla.analytics.interaction import layout as _layout

D_G, D_P, D_Q, N_AVAIL, N_CLASSES = 11, 19, 13, 3, 3


class SetEncoder(nn.Module):
    """shared phi over the slots of a set, then masked mean || masked max; an empty set gives zeros.
    mean and max are symmetric, so the output does not depend on slot order or group size."""

    def __init__(self, d_in, d_hidden, d_cond=0):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in + d_cond, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))

    def forward(self, x, exists):  # x (B,T,N,d), exists (B,T,N) in {0,1}
        z = self.phi(x)
        w = exists.unsqueeze(-1)
        count = w.sum(2)
        mean = (z * w).sum(2) / count.clamp(min=1)
        top = z.masked_fill(w == 0, float('-inf')).max(2).values
        top = torch.where(count > 0, top, torch.zeros_like(top))
        return torch.cat([mean, top], -1), z


class TemporalBlock(nn.Module):
    """depthwise conv over windows, gelu, pointwise mix, dropout, residual, layernorm; causal pads
    on the left only, so a window never reads the ones after it."""

    def __init__(self, d, kernel=5, dilation=1, causal=False, dropout=0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.causal = causal
        self.depthwise = nn.Conv1d(d, d, kernel, dilation=dilation, groups=d)
        self.pointwise = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):  # (B,T,d)
        x = h.transpose(1, 2)
        left = self.pad if self.causal else self.pad // 2
        x = F.pad(x, (left, self.pad - left))
        x = self.pointwise(F.gelu(self.depthwise(x))).transpose(1, 2)
        return self.norm(h + self.drop(x))


class InteractionNet(nn.Module):
    """speech token, person set and pair set per window, fused, then the temporal blocks and the
    head. `d_g` is the extension point of the deferred +lexicon ablation (one more G value)."""

    def __init__(self, d_set=24, d_speech=16, d_window=48, kernel=5, dilations=(1, 2), causal=False,
                 pair_conditioned=False, d_g=D_G):
        super().__init__()
        self.pair_conditioned = pair_conditioned
        self.speech = nn.Sequential(nn.Linear(d_g, d_speech), nn.GELU())
        self.persons = SetEncoder(D_P, d_set)
        # the ladder's pair-conditioned rung sees both persons' states, symmetric in a<->b
        self.pairs = SetEncoder(D_Q, d_set, d_cond=2 * d_set if pair_conditioned else 0)
        d_cat = d_speech + 4 * d_set + N_AVAIL
        self.window = nn.Sequential(nn.LayerNorm(d_cat), nn.Linear(d_cat, d_window), nn.GELU(), nn.Dropout(0.3))
        self.temporal = nn.Sequential(*[TemporalBlock(d_window, kernel, d, causal) for d in dilations])
        self.head = nn.Linear(d_window, N_CLASSES)

    def forward(self, g, avail, persons, person_exists, pairs, pair_exists, pair_index, return_persons=False):
        pooled_p, h = self.persons(persons, person_exists)
        if self.pair_conditioned:
            ha, hb = h[:, :, pair_index[:, 0]], h[:, :, pair_index[:, 1]]
            pairs = torch.cat([pairs, ha + hb, ha * hb], -1)
        pooled_q, _ = self.pairs(pairs, pair_exists)
        e = self.window(torch.cat([self.speech(g), pooled_p, pooled_q, avail], -1))
        logits = self.head(self.temporal(e))  # (B,T,3)
        # per-person embeddings are kept for the 60 s role-function layer (layer b)
        return (logits, h) if return_persons else logits


class PooledNet(nn.Module):
    """the control: the 82-column pooled view through the same window, temporal and head layers."""

    def __init__(self, d_in=82, d_window=48, kernel=5, dilations=(1, 2), causal=False):
        super().__init__()
        self.window = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_window), nn.GELU(), nn.Dropout(0.3))
        self.temporal = nn.Sequential(*[TemporalBlock(d_window, kernel, d, causal) for d in dilations])
        self.head = nn.Linear(d_window, N_CLASSES)

    def forward(self, x):
        return self.head(self.temporal(self.window(x)))


# ---- the recipe (fixed before any result) ----

CLASSES = ('individual', 'social', 'collaborative')  # the class indices of labels.CLASSES
MODALITIES = ('speech', 'space', 'body_gaze')  # the order of the availability bits
CROP = 48  # windows per crop: 8 minutes
BATCH = 16  # crops per step
WINDOWS_PER_CROP = 30  # an epoch is ceil(coded training windows / 30) crops
LR, WEIGHT_DECAY, CLIP = 2e-3, 1e-2, 1.0
NOISE = 0.1  # sigma of the Gaussian noise on observed scaled values
SMOOTHING = 0.05
MAX_EPOCHS, PATIENCE = 300, 25
SEEDS = (0, 1, 2, 3, 4)
# the small configuration, for an outer-training fold with fewer coded non-unclear windows than SMALL_BELOW
SMALL = {'d_set': 16, 'd_speech': 12, 'd_window': 32}
SMALL_BELOW = 3000
VARIANTS = ('pooled-net', 'net-notcn', 'net', 'net-pair')  # ladder rungs a-d; rung e (pretraining) is deferred

# the token columns, taken from the layout that builds the tokens (values in the order of the
# layout tables, then the masks), so the two cannot drift apart: the noise and modality dropout
# find their columns by these names
G_COLUMNS, P_COLUMNS, Q_COLUMNS = _layout.G_COLUMNS, _layout.P_COLUMNS, _layout.Q_COLUMNS
# the mask that says a value was observed, and the modality of each mask
MASK_OF = {v.name: v.mask for v in _layout.GROUP_VALUES + _layout.PERSON_VALUES + _layout.PAIR_VALUES}
MODALITY_OF_MASK = dict(_layout.MASK_MODALITY)
# the pooled view's columns that are masks, flags or shares rather than scaled values: no noise on them
POOLED_FLAGS = ('group_size', 'ips_ran', 'vfa_ran')
POOLED_FLAG_PREFIXES = ('m_', 'share_')

# the model's arguments, in order, and the tokens' field each one comes from
TOKEN_FIELDS = {'g': 'G', 'avail': 'avail', 'persons': 'P', 'person_exists': 'P_exists', 'pairs': 'Q',
                'pair_exists': 'Q_exists'}
# what a crop cuts along the windows; pair_index and pooled_blocks are shared by every crop
SEQUENCES = tuple(TOKEN_FIELDS) + ('y', 'pooled', 'pooled_observed')


def token_layout(columns):
    """where a token's values, the mask of each value and each modality's columns sit, from its
    column names: the noise touches a value only where its mask is on, and modality dropout zeroes
    a modality's values together with its masks. A column with no mask (group_size) is neither."""
    index = {name: i for i, name in enumerate(columns)}
    values = [index[name] for name in columns if name in MASK_OF]
    masks = [index[MASK_OF[name]] for name in columns if name in MASK_OF]
    by_modality = {modality: [] for modality in MODALITIES}
    for name in columns:
        mask = MASK_OF.get(name, name)
        if mask in MODALITY_OF_MASK:
            by_modality[MODALITY_OF_MASK[mask]].append(index[name])
    return {'values': values, 'masks': masks, 'modality': by_modality}


LAYOUT = {'g': token_layout(G_COLUMNS), 'persons': token_layout(P_COLUMNS), 'pairs': token_layout(Q_COLUMNS)}


def make_model(variant='net', small=False, causal=False, d_in=82):
    """a fresh model of a ladder rung: (a) pooled-net, (b) net-notcn (DeepSets only), (c) net,
    (d) net-pair (the pair-conditioned encoder); `small` takes the pre-declared small configuration."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown network variant {variant!r}: one of {', '.join(VARIANTS)}")
    if variant == 'pooled-net':
        return PooledNet(d_in=d_in, causal=causal, **({'d_window': SMALL['d_window']} if small else {}))
    return InteractionNet(dilations=() if variant == 'net-notcn' else (1, 2), causal=causal,
                          pair_conditioned=variant == 'net-pair', **(SMALL if small else {}))


def use_small(n_coded, small='auto'):
    """whether a fold trains the small configuration: `--small yes|no`, or with auto when it has
    fewer than SMALL_BELOW coded non-unclear windows."""
    return small == 'yes' or small is True or (small == 'auto' and n_coded < SMALL_BELOW)


def count_parameters(model):
    """the trainable parameters, the number the design fixed (13,625 default, 6,673 small)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- sessions and batches ----

def _field(tokens, name):
    return tokens[name] if isinstance(tokens, dict) else getattr(tokens, name)


def _float(array):
    # a NaN left in the tokens would reach every window the convolution sees
    array = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(np.ascontiguousarray(array))


def _label_codes(y, length):
    """class indices for the windows, -1 where a window has no label (uncoded, unclear, None,
    NaN, or anything that is not a class)."""
    codes = np.full(length, -1, dtype=np.int64)
    if y is None:
        return codes
    y = list(y)
    if len(y) != length:
        raise ValueError(f"{len(y)} labels for {length} windows")
    for t, value in enumerate(y):
        if isinstance(value, str):
            codes[t] = CLASSES.index(value) if value in CLASSES else -1
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number in (0.0, 1.0, 2.0):
            codes[t] = int(number)
    return codes


def _is_flag(name):
    return name in POOLED_FLAGS or str(name).startswith(POOLED_FLAG_PREFIXES)


def _position(names, column):
    """a pooled column's position, from its name or as given."""
    if isinstance(column, (int, np.integer)):
        return int(column)
    if names is None:
        raise ValueError(f"the pooled view has no column names to find {column!r} by")
    return names.index(str(column))


def session_tensors(tokens, y=None, pooled=None, blocks=None, session=None):
    """one session as tensors named after the model's arguments (g, avail, persons, person_exists,
    pairs, pair_exists, pair_index) plus y, the class index per window with -1 for no label.
    `tokens` is layout's Tokens (or a dict with its field names); with `pooled` (the 82-column
    view, a DataFrame or array, NaN where no slot qualified) the dict also holds the PooledNet
    input, NaN as 0, and `blocks` (modality -> its pooled columns, names or positions, as
    layout.block_columns gives them) tells modality dropout where each modality sits in it."""
    out = {'session': session}
    length = None
    if tokens is not None:
        for key, field in TOKEN_FIELDS.items():
            out[key] = _float(_field(tokens, field))
        out['pair_index'] = torch.as_tensor(np.asarray(_field(tokens, 'pair_index'), dtype=np.int64))
        length = out['g'].shape[0]
    if pooled is not None:
        names = [str(c) for c in pooled.columns] if hasattr(pooled, 'columns') else None
        array = np.asarray(pooled, dtype=np.float32)
        if length is not None and len(array) != length:
            raise ValueError(f"the pooled view has {len(array)} windows, the tokens {length}")
        observed = np.isfinite(array)
        # noise lands only on observed scaled values, never on a mask, flag or share
        values = np.ones(array.shape[1], dtype=bool) if names is None else np.array([not _is_flag(n) for n in names])
        out['pooled'] = _float(np.where(observed, array, 0.0))
        out['pooled_observed'] = torch.from_numpy((observed & values).astype(np.float32))
        out['pooled_blocks'] = {modality: [_position(names, c) for c in columns]
                                for modality, columns in (blocks or {}).items()}
        length = len(array)
    if length is None:
        raise ValueError("give the tokens, the pooled view, or both")
    out['y'] = torch.from_numpy(_label_codes(y, length))
    return out


def _clone(batch):
    return {key: (value.clone() if torch.is_tensor(value) else value) for key, value in batch.items()}


def _single(session):
    """a session as a batch of one, for full-session inference."""
    return {key: (value.unsqueeze(0) if torch.is_tensor(value) and key != 'pair_index' else value)
            for key, value in session.items()}


def _slice(batch, part):
    return {key: (value[part] if key in SEQUENCES and torch.is_tensor(value) else value) for key, value in batch.items()}


def _forward(model, batch):
    if isinstance(model, PooledNet):
        return model(batch['pooled'])
    return model(*(batch[key] for key in TOKEN_FIELDS), batch['pair_index'])


def crops(sessions, length=CROP, rng=None, n=None):
    """one epoch of training crops, stacked as one batch: each crop is `length` windows placed at a
    random offset around a coded window drawn uniformly from all coded windows (so a session
    counts by its labels, not its length), clipped to the session; a session shorter than a crop
    is padded after its end with zero tokens and no labels. `n` defaults to ceil(L / 30) crops for
    L coded windows."""
    rng = rng if rng is not None else np.random.default_rng()
    if not sessions:
        raise ValueError("no session to crop")
    owners = np.concatenate([np.full(int((s['y'] >= 0).sum()), i, dtype=np.int64) for i, s in enumerate(sessions)])
    times = np.concatenate([np.flatnonzero(s['y'].numpy() >= 0) for s in sessions])
    if not len(times):
        raise ValueError("no coded window to place a crop around")
    first = sessions[0]
    if 'pair_index' in first and any(not torch.equal(s['pair_index'], first['pair_index']) for s in sessions):
        raise ValueError("the sessions disagree on pair_index; the layout fixes it to (0, 1), (0, 2), (1, 2)")
    n = n or math.ceil(len(times) / WINDOWS_PER_CROP)
    keys = [key for key in SEQUENCES if all(key in s for s in sessions)]
    rows = {key: [] for key in keys}
    for pick in rng.integers(0, len(times), size=n):
        session, t = sessions[owners[pick]], int(times[pick])
        size = len(session['y'])
        # the coded window lands anywhere in the crop, which is then pushed back inside the session
        start = min(max(t - int(rng.integers(0, length)), 0), max(size - length, 0))
        for key in keys:
            piece = session[key][start:start + length]
            if len(piece) < length:
                pad = torch.full((length - len(piece),) + tuple(piece.shape[1:]), -1 if key == 'y' else 0,
                                 dtype=piece.dtype)
                piece = torch.cat([piece, pad])
            rows[key].append(piece)
    batch = {key: torch.stack(pieces) for key, pieces in rows.items()}
    for key in ('pair_index', 'pooled_blocks'):
        if key in first:
            batch[key] = first[key]
    return batch


# ---- augmentation ----

def modality_dropout(batch, p=0.15, run=(6, 24), rng=None, layout=None):
    """per crop and per modality, with probability p, a run of U[6, 24] consecutive windows loses
    that modality: its values and masks in G, the person and the pair slots, and its availability
    bit, are zeroed, which is what a real outage looks like in the tokens. For the pooled control
    the modality's pooled block is zeroed instead."""
    rng = rng if rng is not None else np.random.default_rng()
    layout = layout or LAYOUT
    out = _clone(batch)
    n, length = out['y'].shape
    blocks = out.get('pooled_blocks') or {}
    for i in range(n):
        for m, modality in enumerate(MODALITIES):
            if rng.random() >= p:
                continue
            size = int(rng.integers(run[0], run[1] + 1))
            start = int(rng.integers(0, max(length - size, 0) + 1))
            span = slice(start, start + size)
            if 'g' in out:
                out['avail'][i, span, m] = 0
                for key in ('g', 'persons', 'pairs'):
                    columns = layout[key]['modality'][modality]
                    if columns:
                        out[key][i, span, ..., columns] = 0
            if 'pooled' in out and blocks.get(modality):
                out['pooled'][i, span, blocks[modality]] = 0
                out['pooled_observed'][i, span, blocks[modality]] = 0
    return out


def person_dropout(batch, p=0.1, rng=None):
    """per window and existing person, with probability p, that person goes unobserved: their
    values and masks are zeroed (the slot is kept), and so are their pairs', since a pair is only
    observed when both of its persons are. The pooled control has no persons and is left as is."""
    if 'persons' not in batch:
        return batch
    rng = rng if rng is not None else np.random.default_rng()
    out = _clone(batch)
    drop = torch.from_numpy(rng.random(tuple(out['person_exists'].shape)) < p) & (out['person_exists'] > 0)
    out['persons'][drop] = 0
    index = out['pair_index']
    out['pairs'][drop[:, :, index[:, 0]] | drop[:, :, index[:, 1]]] = 0
    return out


def add_noise(batch, sigma=NOISE, rng=None, layout=None):
    """Gaussian noise on the observed scaled values only: a masked value stays 0, a mask stays 0 or 1."""
    rng = rng if rng is not None else np.random.default_rng()
    layout = layout or LAYOUT
    out = _clone(batch)
    for key in ('g', 'persons', 'pairs'):
        if key in out:
            x = out[key]
            observed = x[..., layout[key]['masks']]
            noise = torch.from_numpy(rng.standard_normal(tuple(observed.shape)).astype(np.float32))
            x[..., layout[key]['values']] += sigma * noise * observed
    if 'pooled' in out:
        noise = torch.from_numpy(rng.standard_normal(tuple(out['pooled'].shape)).astype(np.float32))
        out['pooled'] += sigma * noise * out['pooled_observed']
    return out


def _augment(batch, rng):
    return add_noise(person_dropout(modality_dropout(batch, rng=rng), rng=rng), rng=rng)


# ---- loss ----

def class_weights(y, power=0.5):
    """w_c = (N / (3 N_c)) ** power over the coded windows, normalised to mean 1: tempered, so the
    rare class counts more without its few windows dominating; calibration restores the prior
    afterwards. `y` is one label array or a list of them (a session's y tensor will do); a class
    with no window gets weight 0."""
    arrays = y if isinstance(y, (list, tuple)) else [y]
    labels = np.concatenate([np.asarray(a).reshape(-1) for a in arrays]) if arrays else np.zeros(0)
    counts = np.array([(labels == c).sum() for c in range(N_CLASSES)], dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError("no coded window to weight the classes by")
    weights = np.zeros(N_CLASSES)
    present = counts > 0
    weights[present] = (counts.sum() / (N_CLASSES * counts[present])) ** power
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def masked_loss(logits, y, weights=None, smoothing=SMOOTHING):
    """class-weighted cross-entropy with label smoothing over the coded windows only: an uncoded or
    unclear window (any label outside 0..2) is context for its neighbours, never a target. The
    weighted mean divides by the targets' weights, so a batch's loss does not grow with how many
    rare-class windows it happens to hold; a batch with no target gives 0."""
    flat = logits.reshape(-1, logits.shape[-1])
    target = y.reshape(-1).long()
    target = torch.where((target >= 0) & (target < N_CLASSES), target, torch.full_like(target, -1))
    if not bool((target >= 0).any()):
        return flat.sum() * 0.0
    return F.cross_entropy(flat, target, weight=weights, ignore_index=-1, label_smoothing=smoothing)


def _held_out_nll(model, sessions, weights):
    """the class-weighted NLL (no smoothing) of full-session predictions, pooled over the sessions'
    coded windows; NaN when none is coded."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([_forward(model, _single(s))[0] for s in sessions])
        y = torch.cat([s['y'] for s in sessions])
    if not bool(((y >= 0) & (y < N_CLASSES)).any()):
        return float('nan')
    return float(masked_loss(logits, y, weights, smoothing=0.0))


# ---- training ----

class AdamW:
    """the update of torch.optim.AdamW (decoupled weight decay, bias-corrected moments, eps 1e-8),
    written out: torch.optim loads torch._dynamo with the first optimizer, which needs sympy, and a
    torch installed without it could not train a model of 14k parameters for that alone."""

    def __init__(self, params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.weight_decay, self.betas, self.eps = lr, weight_decay, betas, eps
        self.steps = 0
        self.first = [torch.zeros_like(p) for p in self.params]
        self.second = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.steps += 1
        beta1, beta2 = self.betas
        correction1, correction2 = 1 - beta1 ** self.steps, 1 - beta2 ** self.steps
        for p, first, second in zip(self.params, self.first, self.second):
            if p.grad is None:
                continue
            p.mul_(1 - self.lr * self.weight_decay)
            first.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            second.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            p.addcdiv_(first / correction1, (second / correction2).sqrt().add_(self.eps), value=-self.lr)


def train_epochs(model, train_sessions, epochs, seed, val_sessions=None, weights=None, patience=None,
                 augment=True):
    """train for `epochs` epochs of crops, in batches of 16, with AdamW and a clipped gradient;
    seeded, and on one thread, so a reported run repeats. With `val_sessions` the held-out
    class-weighted NLL is tracked after every epoch, training stops `patience` epochs after its
    best (when a patience is given), and the model is left at its best epoch. `weights` default to
    the training sessions' tempered class weights. The history holds the mean training loss and
    the held-out NLL per epoch, and the best epoch (counted from 1) with its NLL."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = class_weights([s['y'] for s in train_sessions])
    optimizer = AdamW(model.parameters())
    history = {'loss': [], 'val_nll': [], 'best_epoch': None, 'best_val_nll': None}
    best_state, since_best = None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_crops = crops(train_sessions, CROP, rng)
        losses = []
        for start in range(0, len(epoch_crops['y']), BATCH):
            batch = _slice(epoch_crops, slice(start, start + BATCH))
            if augment:
                batch = _augment(batch, rng)
            loss = masked_loss(_forward(model, batch), batch['y'], weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            losses.append(float(loss.detach()))
        history['loss'].append(float(np.mean(losses)))
        if not val_sessions:
            continue
        nll = _held_out_nll(model, val_sessions, weights)
        history['val_nll'].append(nll)
        if math.isnan(nll):
            continue
        if history['best_val_nll'] is None or nll < history['best_val_nll']:
            history['best_epoch'], history['best_val_nll'], since_best = epoch, nll, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif val_sessions:
        # no held-out window was coded: nothing to pick by, so the last epoch stands
        history['best_epoch'] = len(history['loss'])
    model.eval()
    return history


def _positions(sessions, part):
    """positions in `sessions` of an inner split's part, given as positions or session names."""
    names = [s.get('session') for s in sessions]
    return [int(item) if isinstance(item, (int, np.integer)) else names.index(item) for item in part]


def select_epochs(outer_train, inner_folds, make=None, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=0,
                  weights=None):
    """the one tuned hyperparameter, the epoch count. Seed 0 trains on each inner split for up to
    `max_epochs` epochs, tracking the held-out lessons' class-weighted NLL (patience 25), and E* is
    the median of the splits' best epochs. The inner models, each back at its best epoch, give the
    out-of-fold logits the calibrator and the HMM's gamma are fit on. `inner_folds` holds (train,
    held) pairs of positions in `outer_train` or of session names; `make` builds a fresh model
    (default InteractionNet); the class weights come from the whole outer-training fold. The
    logits come back in `outer_train`'s order, (T, 3) arrays, None for a session no split held out."""
    make = make or InteractionNet
    if weights is None:
        weights = class_weights([s['y'] for s in outer_train])
    best, oof = [], [None] * len(outer_train)
    for train_part, held_part in inner_folds:
        train_at, held_at = _positions(outer_train, train_part), _positions(outer_train, held_part)
        if not any(bool((outer_train[i]['y'] >= 0).any()) for i in held_at):
            # a split with nothing to score would stop at its last epoch and drag the median up
            raise ValueError("an inner split holds out no coded window; only lessons with coded windows form folds")
        torch.manual_seed(seed)
        model = make()
        history = train_epochs(model, [outer_train[i] for i in train_at], max_epochs, seed,
                               val_sessions=[outer_train[i] for i in held_at], weights=weights, patience=patience)
        best.append(history['best_epoch'])
        with torch.no_grad():
            for i in held_at:
                oof[i] = _forward(model, _single(outer_train[i]))[0].numpy().astype(np.float64)
    if not best:
        raise ValueError("no inner split to select the epoch count on")
    # the median of an even count rounds half up
    return int(np.floor(np.median(best) + 0.5)), oof


def fit_ensemble(outer_train, epochs, seeds=SEEDS, make=None, weights=None):
    """the refit: every outer-training session, `epochs` (E*) epochs, one model per seed; the mean
    of their softmaxes is the prediction, and no seed is ever picked over another."""
    make = make or InteractionNet
    if weights is None:
        weights = class_weights([s['y'] for s in outer_train])
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = make()
        train_epochs(model, outer_train, epochs, seed, weights=weights)
        models.append(model.eval())
    return models


def predict_net(models, session):
    """the (T, 3) log-probabilities of a session: the mean of the models' softmaxes over the full
    session (the temporal blocks are local, so this equals stitching crops), as its log."""
    models = [models] if isinstance(models, nn.Module) else list(models)
    batch = _single(session)
    with torch.no_grad():
        probabilities = torch.stack([torch.softmax(_forward(model.eval(), batch)[0], -1) for model in models]).mean(0)
    return torch.log(probabilities.clamp_min(1e-12)).numpy().astype(np.float64)


def pretrain_masked_modality(model, sessions, epochs=50):
    """ladder rung (e), deferred by decision: hide one modality in runs (p = 0.3) on every window of
    the training sessions, never the test ones, and regress its standardised pooled values from
    z_t with linear heads 48 -> 7 / 15 / 48 (MSE on observed columns) for 50 epochs, then fine-tune
    with train_epochs. It would hide with modality_dropout (p = 0.3) and needs the model to expose
    z_t, the temporal blocks' output, which forward does not return yet."""
    raise NotImplementedError("masked-modality pretraining (ladder rung e) is deferred")
