import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        prog="openmmla crypto",
        description="Manage encryption keys for sensitive config values.",
    )
    sub = parser.add_subparsers(dest="action")

    init_parser = sub.add_parser("init", help="Generate a new master encryption key.")
    init_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing master key (WARNING: invalidates all encrypted values).",
    )

    sub.add_parser("status", help="Check if master key exists and is usable.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.action == "init":
        _cmd_init(args)
    elif args.action == "status":
        _cmd_status()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_init(args):
    from openmmla.utils.crypto import generate_master_key, MASTER_KEY_PATH
    try:
        path = generate_master_key(force=args.force)
        print(f"Master key generated at {path}")
        print("Keep this file safe — it is required to decrypt config values.")
    except FileExistsError:
        print(f"Master key already exists at {MASTER_KEY_PATH}")
        print("Use --force to regenerate (this will invalidate all encrypted values).")
        sys.exit(1)


def _cmd_status():
    from openmmla.utils.crypto import MASTER_KEY_PATH
    import os
    if not os.path.exists(MASTER_KEY_PATH):
        print(f"No master key found at {MASTER_KEY_PATH}")
        print("Run 'openmmla crypto init' to generate one.")
        sys.exit(1)
    st = os.stat(MASTER_KEY_PATH)
    mode = oct(st.st_mode & 0o777)
    print(f"Master key exists at {MASTER_KEY_PATH}")
    print(f"File permissions: {mode}")
    if st.st_mode & 0o077:
        print("WARNING: key file is readable by group/others. Run: chmod 600 " + MASTER_KEY_PATH)
    else:
        print("Permissions OK (owner-only access).")

    try:
        from cryptography.fernet import Fernet
        with open(MASTER_KEY_PATH, "rb") as f:
            key = f.read().strip()
        Fernet(key)
        print("Key format: valid Fernet key.")
    except Exception as e:
        print(f"Key format: INVALID ({e})")
        sys.exit(1)
