import os
import re
import stat
import logging

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

MASTER_KEY_DIR = os.path.join(os.path.expanduser("~"), ".openmmla")
MASTER_KEY_PATH = os.path.join(MASTER_KEY_DIR, "master.key")

ENC_PREFIX = "ENC("
ENC_SUFFIX = ")"
ENC_RE = re.compile(r"^ENC\((.+)\)$")
# an ENC(...) value anywhere in a text or a string: Fernet's urlsafe base64
ENC_TOKEN_RE = re.compile(r"ENC\(([A-Za-z0-9_\-=]+)\)")

SENSITIVE_KEYS = {"api_key", "token", "hf_token", "password", "secret", "secret_key", "subscription_key"}


def _ensure_key_dir():
    os.makedirs(MASTER_KEY_DIR, mode=0o700, exist_ok=True)


def _create_key_file(path: str, key: bytes) -> None:
    """write `key` to `path` only if there is no file there yet, readable by
    its owner alone (0600): it is written beside it first and then linked
    into place, so two that make a key at once never write over each other,
    and the one that comes second finds the first one's key (FileExistsError).
    A file system without hard links gets an exclusive create instead."""
    tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.new"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(key + b"\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
        except OSError:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(key + b"\n")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def generate_master_key(force=False) -> str:
    """generate a new Fernet master key and save to ~/.openmmla/master.key.

    A key that is there already is never replaced unless `force` is given
    (FileExistsError otherwise, also when another process made one a moment
    before). Returns the path to the key file.
    """
    _ensure_key_dir()
    key = Fernet.generate_key()
    if force:
        with open(MASTER_KEY_PATH, "wb") as f:
            f.write(key)
        os.chmod(MASTER_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    else:
        if os.path.exists(MASTER_KEY_PATH):
            raise FileExistsError(
                f"Master key already exists at {MASTER_KEY_PATH}. "
                f"Use force=True to overwrite (this will invalidate all encrypted values)."
            )
        _create_key_file(MASTER_KEY_PATH, key)
    logger.info(f"Master key generated at {MASTER_KEY_PATH}")
    return MASTER_KEY_PATH


def load_master_key() -> bytes:
    """load the Fernet master key from ~/.openmmla/master.key."""
    if not os.path.exists(MASTER_KEY_PATH):
        raise FileNotFoundError(
            f"Master key not found at {MASTER_KEY_PATH}. "
            f"Run 'openmmla crypto init' to generate one."
        )
    with open(MASTER_KEY_PATH, "rb") as f:
        key = f.read().strip()
    return key


def is_encrypted(value) -> bool:
    """check if a string value has the ENC(...) wrapper."""
    if not isinstance(value, str):
        return False
    return bool(ENC_RE.match(value))


def encrypt_value(plaintext: str, key: bytes | None = None) -> str:
    """encrypt a plaintext string and return ENC(ciphertext)."""
    if key is None:
        key = load_master_key()
    f = Fernet(key)
    encrypted = f.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    return f"{ENC_PREFIX}{encrypted}{ENC_SUFFIX}"


def decrypt_value(wrapped: str, key: bytes | None = None) -> str:
    """decrypt an ENC(ciphertext) string back to plaintext."""
    m = ENC_RE.match(wrapped)
    if not m:
        raise ValueError(f"Value is not encrypted (missing ENC(...) wrapper): {wrapped!r}")
    if key is None:
        key = load_master_key()
    f = Fernet(key)
    return f.decrypt(m.group(1).encode("utf-8")).decode("utf-8")


def is_sensitive_key(key_name: str) -> bool:
    """check if a yaml key name represents a sensitive value."""
    return key_name.lower() in SENSITIVE_KEYS


def ensure_master_key() -> bytes:
    """load the master key, generating one automatically on first use. Every
    machine has a key of its own, made here the first time something is
    encrypted on it; one that is there is never replaced, and of two
    processes that make one at once, both end up with the one that won."""
    try:
        return load_master_key()
    except FileNotFoundError:
        try:
            generate_master_key()
            logger.info("No master key found; generated a new one.")
        except FileExistsError:
            pass  # made by another process a moment ago: that one is the key
        return load_master_key()


def sensitive_plaintext(value) -> str | None:
    """return the plaintext to encrypt for a sensitive key, or None to skip it.

    yaml coerces unquoted scalars, so an all-digit password loads as an int and
    would silently stay in plaintext under a str-only check. Non-string scalars
    are stringified the same way the config consumers already read them.
    Empty values, template placeholders (<...>) and values that are already
    ENC(...) are left alone."""
    if value is None or isinstance(value, (dict, list)):
        return None
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value)
    if not text or text.startswith("<") or is_encrypted(text):
        return None
    return text


def encrypt_sensitive_values(data, key: bytes | None = None) -> int:
    """recursively encrypt plaintext sensitive values in-place.

    Walks nested dicts and lists, so secrets held in a list of entries (e.g.
    the ssh profile store) are covered too. Existing ENC(...) values and
    template placeholders (<...>) are left untouched. Returns the number of
    values encrypted."""
    if key is None:
        key = load_master_key()
    count = 0
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                count += encrypt_sensitive_values(item, key)
        return count
    if not isinstance(data, dict):
        return count
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            count += encrypt_sensitive_values(v, key)
        elif is_sensitive_key(k):
            plaintext = sensitive_plaintext(v)
            if plaintext is not None:
                data[k] = encrypt_value(plaintext, key)
                count += 1
    return count


def process_config_dict(data, key: bytes | None = None) -> tuple:
    """recursively walk a config structure, decrypt ENC() values in-place, and
    detect plaintext sensitive values that need encryption.

    Nested dicts and lists are both walked, so secrets held in a list of
    entries are covered too.

    Returns (decrypted_data_for_runtime, needs_rewrite) where:
    - decrypted_data_for_runtime: a copy with all sensitive values decrypted
    - needs_rewrite: True if any plaintext sensitive values were found and encrypted in data
    """
    if key is None:
        key = load_master_key()
    needs_rewrite = False

    if isinstance(data, list):
        runtime_list = []
        for item in data:
            if isinstance(item, (dict, list)):
                sub_runtime, sub_rewrite = process_config_dict(item, key)
                runtime_list.append(sub_runtime)
                if sub_rewrite:
                    needs_rewrite = True
            else:
                runtime_list.append(item)
        return runtime_list, needs_rewrite

    if not isinstance(data, dict):
        return data, needs_rewrite

    runtime_data = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            sub_runtime, sub_rewrite = process_config_dict(v, key)
            runtime_data[k] = sub_runtime
            if sub_rewrite:
                needs_rewrite = True
        elif is_encrypted(v):
            try:
                runtime_data[k] = decrypt_value(v, key)
            except Exception:
                logger.warning(
                    f"Could not decrypt config value '{k}' — wrong or missing master key? "
                    f"Keeping the encrypted form; sync ~/.openmmla/master.key from the "
                    f"machine that encrypted it."
                )
                runtime_data[k] = v
        elif is_sensitive_key(k):
            plaintext = sensitive_plaintext(v)
            if plaintext is not None:
                data[k] = encrypt_value(plaintext, key)
                runtime_data[k] = plaintext
                needs_rewrite = True
            else:
                runtime_data[k] = v
        else:
            runtime_data[k] = v
    return runtime_data, needs_rewrite


# ── one key per machine: values sealed again for the machine they go to ──
#
# Every machine encrypts with its own ~/.openmmla/master.key, and a value at
# rest on a machine is sealed with that machine's key. What travels from one
# machine to another is opened with the key it was sealed with and sealed
# again with the key of the machine it goes to; the plaintext lives in memory
# only, in between.

def enc_tokens(data) -> set[str]:
    """every ENC(...) value in `data`: a text, or the strings of a nested
    structure of dicts and lists (keys included)."""
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            found.update(m.group(0) for m in ENC_TOKEN_RE.finditer(node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(k)
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(data)
    return found


def open_value(wrapped: str, keys) -> str | None:
    """the plaintext of an ENC(...) value, opened with the first of `keys`
    that opens it (None and malformed keys are passed over); None when none
    does."""
    m = ENC_RE.match(wrapped) if isinstance(wrapped, str) else None
    if not m:
        return None
    for key in keys:
        if not key:
            continue
        try:
            return Fernet(key).decrypt(m.group(1).encode("utf-8")).decode("utf-8")
        except Exception:
            continue
    return None


def plan_reseal(data, dest_key: bytes | None, candidates, keep=()) -> tuple[dict[str, str], list[str]]:
    """what sealing `data` for a machine whose key is `dest_key` takes:
    ({value: plaintext} of the ENC(...) values to seal again, the values
    none of the keys opens). A value `dest_key` opens is the machine's own
    already and stays as it is; any other is opened with the first of
    `candidates` that opens it; one of `keep` that none opens (the machine
    holds it already) stays as it is too."""
    opened: dict[str, str] = {}
    unopened: list[str] = []
    for token in sorted(enc_tokens(data)):
        if dest_key and open_value(token, [dest_key]) is not None:
            continue
        plaintext = open_value(token, candidates)
        if plaintext is not None:
            opened[token] = plaintext
        elif token not in keep:
            unopened.append(token)
    return opened, unopened


def apply_reseal(data, opened: dict[str, str], dest_key: bytes):
    """`data` (a text, or a nested structure, which is copied) with each
    value of `opened` sealed again with `dest_key`; everything else, the
    layout and comments of a text included, stays as it was."""
    if not opened:
        return data
    sealed = {token: encrypt_value(plaintext, dest_key) for token, plaintext in opened.items()}

    def swap(text: str) -> str:
        return ENC_TOKEN_RE.sub(lambda m: sealed.get(m.group(0), m.group(0)), text)

    def walk(node):
        if isinstance(node, str):
            return swap(node)
        if isinstance(node, dict):
            return {walk(k): walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(data)


def reseal(data, dest_key: bytes, candidates, keep=()) -> tuple[object, list[str]]:
    """`data` with every ENC(...) value sealed with `dest_key` (see
    plan_reseal), and the values none of the keys opens; while there are
    any, `data` comes back as it was."""
    opened, unopened = plan_reseal(data, dest_key, candidates, keep)
    if unopened:
        return data, unopened
    return apply_reseal(data, opened, dest_key), []
