import importlib
import importlib.metadata
import sys

from openmmla.cli_config import COMMANDS, OPTIONAL_DEP_MAP


def load_func(entry: str):
    """Lazy import a function like 'module.submodule:func'"""
    module_path, func_name = entry.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def print_main_help():
    print("\n📦 OpenMMLA CLI\n")
    print("OpenMMLA is a toolkit for multimodal learning analytics, providing various built-in pipelines for "
          "different tasks.\n")
    print(f"usage: mmla [-h] [-V] or mmla COMMAND [options]\n")
    print(f'options: \n'
          f'    -h, --help       show this help message and exit\n'
          f'    -V --version     show version number and exit\n'
          f'    COMMAND          subcommand to run (e.g. asr-base)\n')

    sections = [
        ("ASR (Automatic Speech Recognition)", "asr-"),
        ("IPS (Indoor Positioning System)", "ips-"),
        ("VFA (Video Frame Analyzer)", "vfa-"),
        ("Session-level Tools", "ses-")
    ]

    print("🛠️  Available Commands:\n")
    for title, prefix in sections:
        print(f"🔸 {title}")
        print("     COMMAND                  DESCRIPTION")
        for name, (_, desc) in COMMANDS.items():
            if name.startswith(prefix):
                print(f"  🔹 {name:<24} {desc}")
        print("")

    print("📘 Tip: run `mmla <command> -h` for detailed usage of a command.\n")


def run_cli():
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print_main_help()
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
