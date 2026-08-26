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

SENSITIVE_KEYS = {"api_key", "token", "password", "secret", "secret_key", "subscription_key"}


def _ensure_key_dir():
    os.makedirs(MASTER_KEY_DIR, mode=0o700, exist_ok=True)


def generate_master_key(force=False) -> str:
    """generate a new Fernet master key and save to ~/.openmmla/master.key.

    Returns the path to the key file.
    """
    _ensure_key_dir()
    if os.path.exists(MASTER_KEY_PATH) and not force:
        raise FileExistsError(
            f"Master key already exists at {MASTER_KEY_PATH}. "
            f"Use force=True to overwrite (this will invalidate all encrypted values)."
        )
    key = Fernet.generate_key()
    with open(MASTER_KEY_PATH, "wb") as f:
        f.write(key)
    os.chmod(MASTER_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
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
    """load the master key, generating one automatically on first use."""
    try:
        return load_master_key()
    except FileNotFoundError:
        generate_master_key()
        logger.info("No master key found; generated a new one.")
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
