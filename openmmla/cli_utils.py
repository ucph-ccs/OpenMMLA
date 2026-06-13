import importlib
import importlib.metadata
import sys

from openmmla.cli_config import COMMANDS, HIDDEN_COMMANDS, OPTIONAL_DEP_MAP


def load_func(entry: str):
    """Lazy import a function like 'module.submodule:func'"""
    module_path, func_name = entry.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def print_main_help(show_all=False):
    print("\n📦 OpenMMLA CLI\n")
    print("OpenMMLA is a toolkit for multimodal learning analytics, providing various built-in pipelines for "
          "different tasks.\n")
    print(f"usage: mmla [-h] [-V] or mmla COMMAND [options]\n")
    print(f'options: \n'
          f'    -h, --help       show this help message and exit (add --all for internal commands)\n'
          f'    -V --version     show version number and exit\n'
          f'    COMMAND          subcommand to run (e.g. asr-base)\n')

    sections = [
        ("Management Console", "tui"),
        ("ASR (Automatic Speech Recognition)", "asr-"),
        ("IPS (Indoor Positioning System)", "ips-"),
        ("VFA (Video Frame Analyzer)", "vfa-"),
        ("Raw Data Collection", "collect-"),
        ("Session-level Tools", "ses-"),
        ("Security & Encryption", "crypto"),
    ]

    print("🛠️  Available Commands:\n")
    for title, prefix in sections:
        rows = [
            (name, desc)
            for name, (_, desc) in COMMANDS.items()
            if name.startswith(prefix) and (show_all or name not in HIDDEN_COMMANDS)
        ]
        if not rows:
            continue
        print(f"🔸 {title}")
        print("     COMMAND                  DESCRIPTION")
        for name, desc in rows:
            print(f"  🔹 {name:<24} {desc}")
        print("")

    if not show_all:
        hidden_count = len([name for name in COMMANDS if name in HIDDEN_COMMANDS])
        if hidden_count:
            print(f"ℹ️  {hidden_count} internal/dev commands hidden; run `mmla --help --all` to list them.\n")
    print("📘 Tip: run `mmla <command> -h` for detailed usage of a command. "
          "The TUI (`mmla tui`) is the recommended way to configure and launch everything.\n")


def run_cli():
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print_main_help(show_all="--all" in argv)
        return

    if argv[0] in ("-V", "--version"):
        try:
            from openmmla import __version__
        except ImportError:
            try:
                __version__ = importlib.metadata.version("openmmla")
            except importlib.metadata.PackageNotFoundError:
                __version__ = "(dev)"
        print(f"openmmla version {__version__}")
        return

    command = argv[0]
    remaining = argv[1:]

    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        print_main_help()
        sys.exit(1)

    entry = COMMANDS[command][0]
    module_path = entry.split(":")[0]

    # help 模式
    if "-h" in remaining or "--help" in remaining:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "get_parser"):
                parser = module.get_parser()
                parser.print_help()
                return
        except Exception as e:
            print(f"(Warning) Cannot fully import '{command}': {e}")
            return

    # 正常执行子命令
    try:
        func = load_func(entry)
        sys.argv = [f"{sys.argv[0]} {command}"] + remaining
        func()
    except ModuleNotFoundError as e:
        print(f"❌ Missing dependency: {e}")
        dep_group = OPTIONAL_DEP_MAP.get(command, "dev")
        print(f'\n💡 Try installing optional dependencies:\n  pip install -e ".[{dep_group}]"\n')

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "get_parser"):
                print("\nShowing help for this command:")
                parser = module.get_parser()
                parser.print_help()
        except Exception:
            pass
