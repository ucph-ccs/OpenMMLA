from __future__ import annotations

import asyncio
import os
import re
import subprocess

from rich.markup import escape as rich_escape
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, DataTable, Button, Select, Label

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, ssh_run_async,
    git_remote_url, wrap_remote,
)
from openmmla.tui.widgets.command_session import CommandSession, _list_conda_envs_sync, _parse_conda_envs


# Env/group metadata only. Required packages are read dynamically from the
# project's pyproject.toml ([project.optional-dependencies].<group>), so this
# table no longer carries a hardcoded package list.
ENV_GROUPS = [
    # NOTE: asr-server-* and vfa-server envs were removed — those services are
    # dockerized now (see docker/README.md); their dependencies live in per-
    # service images, not conda envs.
    {"group": "asr-base", "env": "asr-base", "python": "3.10",
     "description": "ASR base station"},
    {"group": "vfa-base", "env": "vfa-base", "python": "3.10",
     "description": "VFA base station"},
    {"group": "vfa-vllm-runtime", "env": "vfa-vllm", "python": "3.12",
     "description": "VFA local vLLM runtime (MLLM Server)"},
    {"group": "ips-base", "env": "ips-base", "python": "3.10",
     "description": "IPS base station"},
    {"group": "uber-base", "env": "uber-base", "python": "3.10",
     "description": "Analysis framework"},
    {"group": "uber-server", "env": "uber-server", "python": "3.10",
     "description": "Dashboard & infrastructure services"},
    {"group": "tui", "env": "tui", "python": "3.10",
     "description": "TUI management console"},
]


def _target_options() -> list[tuple[str, str]]:
    from openmmla.tui.ssh import target_options
    return target_options()


def _normalize_target(value) -> str:
    if value is Select.BLANK or value is None:
        return "local"
    text = str(value)
    return text if text else "local"


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_conda_packages(output: str) -> set[str]:
    packages: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            packages.add(_normalize_package_name(parts[0]))
    return packages


def _list_conda_packages_sync(env_name: str) -> set[str]:
    try:
        result = subprocess.run(
            ["conda", "list", "-n", env_name],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return _parse_conda_packages(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return set()


def _requirement_to_name(req: str) -> str:
    """Reduce a PEP 508 requirement string to its normalized distribution name.

    Handles version specifiers, extras, environment markers and direct
    references, e.g. ``modelscope[framework]==1.16.1`` -> ``modelscope`` and
    ``wespeaker @ git+https://...`` -> ``wespeaker``.
    """
    req = req.split(";", 1)[0]   # drop environment markers
    req = req.split("@", 1)[0]   # drop direct URL references (name @ url)
    req = req.split("[", 1)[0]   # drop extras
    req = re.split(r"[<>=!~\s]", req, 1)[0]  # drop version specifiers / whitespace
    return _normalize_package_name(req)


def _extract_array_items(text: str, section: str, key: str) -> list[str] | None:
    """Fallback TOML array extractor used when no toml parser is available.

    Scans the given ``section`` header for ``key = [ ... ]`` and returns the
    quoted string items, correctly ignoring brackets that appear *inside*
    quoted strings (e.g. ``modelscope[framework]``).
    """
    sec = text.find(section)
    if sec == -1:
        return None
    rest = text[sec + len(section):]
    nxt = re.search(r"\n\[", rest)  # stop at the next table header
    if nxt:
        rest = rest[:nxt.start()]
    km = re.search(r'(?m)^\s*"?' + re.escape(key) + r'"?\s*=\s*', rest)
    if not km:
        return None
    i = km.end()
    while i < len(rest) and rest[i] in " \t\r\n":
        i += 1
    if i >= len(rest) or rest[i] != "[":
        return None
    i += 1
    items: list[str] = []
    buf: list[str] = []
    in_str = False
    quote = ""
    while i < len(rest):
        c = rest[i]
        if in_str:
            if c == quote:
                items.append("".join(buf))
                buf = []
                in_str = False
            else:
                buf.append(c)
        elif c in ("\"", "'"):
            in_str = True
            quote = c
            buf = []
        elif c == "]":
            break
        i += 1
    return items


def _parse_optional_deps_from_text(text: str, group: str) -> set[str] | None:
    """Return normalized package names for ``[project.optional-dependencies].<group>``.

    Returns ``None`` if the group can't be found so callers can fall back to a
    hardcoded sentinel list.
    """
    deps: list[str] | None = None
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(text)
        deps = list(data["project"]["optional-dependencies"][group])
    except Exception:
        deps = _extract_array_items(text, "[project.optional-dependencies]", group)
    if deps is None:
        return None
    names = {_requirement_to_name(d) for d in deps}
    names.discard("")
    return names


def _missing_packages(installed_packages: set[str], required: set[str]) -> list[str]:
    missing = sorted(pkg for pkg in required if pkg not in installed_packages)
    if "opencv-python" in missing and "cv2" in installed_packages:
        missing.remove("opencv-python")
    if "cv2" in missing and "opencv-python" in installed_packages:
        missing.remove("cv2")
    return missing


def _format_env_status(
    entry: dict,
    conda_envs: set[str],
    env_packages: dict[str, set[str]],
    required_by_group: dict[str, set[str]],
) -> str:
    env_name = entry["env"]
    if env_name not in conda_envs:
        return "Missing"
    required = required_by_group.get(entry["group"])
    if not required:
        # No pyproject deps resolved for this group (file unreadable or group
        # absent) -> we have nothing to check against.
        return "Unknown"
    missing = _missing_packages(env_packages.get(env_name, set()), required)
    if missing:
        suffix = ", ".join(missing[:2])
        if len(missing) > 2:
            suffix += f" +{len(missing) - 2}"
        return f"Partial: {suffix}"
    return "Ready"


class EnvironmentPanel(Widget):

    DEFAULT_CSS = """
    EnvironmentPanel {
        width: 1fr;
        height: 1fr;
    }
    #env-header {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }
    #env-table {
        height: 1fr;
    }
    #env-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #env-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #env-target-bar Select {
        width: 1fr;
    }
    #env-target-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    #env-actions {
        height: auto;
        padding: 1;
    }
    #env-actions Button {
        margin: 0 1;
        min-width: 16;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._conda_envs: set[str] = set()
        self._env_packages: dict[str, set[str]] = {}
        self._required_packages: dict[str, set[str]] = {}
        self._selected_group: str | None = None
        self._pending_delete: tuple[str, str] | None = None
        self._target = "local"

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="env-target-bar"):
                yield Label("Host:")
                yield Select(_target_options(), value="local", id="env-target-select")
                yield Button("↻", variant="primary", compact=True, id="env-target-refresh")
            yield Static(
                "Conda Environment Manager — select a row, then use actions",
                id="env-header",
            )
            yield DataTable(id="env-table")
            with Horizontal(id="env-actions"):
                yield Button("Connect", variant="primary", id="btn-connect")
                yield Button("Refresh", variant="primary", id="btn-env-refresh")
                yield Button("Git Clone", variant="warning", id="btn-git-clone")
                yield Button("Git Pull", variant="warning", id="btn-git-pull")
                yield Button("Create Env", variant="success", id="btn-create-env")
                yield Button("Install Deps", variant="success", id="btn-install-deps")
                yield Button("Delete Env", variant="error", id="btn-delete-env")
            yield CommandSession(show_target=False, id="env-cmd-session")

    @property
    def _cmd(self) -> CommandSession:
        return self.query_one("#env-cmd-session", CommandSession)

    @property
    def _is_remote(self) -> bool:
        return self._get_target() != "local"

    def _get_target(self) -> str:
        return self._target

    def _get_selected_target(self) -> str:
        try:
            sel = self.query_one("#env-target-select", Select)
            return _normalize_target(sel.value)
        except Exception:
            return self._target

    def _set_target(self, target: str) -> None:
        normalized = _normalize_target(target)
        if normalized != self._target:
            self._pending_delete = None
        self._target = normalized
        try:
            self._cmd.set_target(self._target)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        self._cmd.log(msg)

    def on_mount(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.add_columns("Conda Env", "Dep Group", "Python", "Status", "Description")
        table.cursor_type = "row"
        self._refresh_table()

    def on_show(self) -> None:
        self._refresh_target_options()

    def _refresh_target_options(self) -> None:
        options = _target_options()
        try:
            sel = self.query_one("#env-target-select", Select)
            current = sel.value
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
                self._set_target(_normalize_target(current))
            else:
                sel.value = "local"
                self._set_target("local")
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "env-target-select":
            if getattr(self, "_suppress_select", False):
                return
            from openmmla.tui.ssh import REFRESH_TARGETS_OPTION, TARGET_STATES, is_select_sentinel
            if is_select_sentinel(event.value):
                return
            if str(event.value) == REFRESH_TARGETS_OPTION:
                self._cmd.log("[yellow]Testing connections to all hosts...[/yellow]")
                self._revert_select(event.select)
                self.run_worker(self._async_probe_hosts(), group="env-host-probe", exclusive=True)
                return
            target = _normalize_target(event.value)
            if target != "local" and TARGET_STATES.get(target) == "offline":
                self._cmd.log(
                    f"[red]Host '{target}' is offline; staying on '{self._target}'. "
                    f"Re-testing it now...[/red]"
                )
                self._revert_select(event.select)
                self.run_worker(self._async_test_single_host(target), group="env-host-probe", exclusive=False)
                return
            self._set_target(target)
            self._refresh_table()

    def _revert_select(self, select: Select) -> None:
        self._suppress_select = True
        select.value = self._target
        self.call_after_refresh(self._clear_select_suppression)

    def _clear_select_suppression(self) -> None:
        self._suppress_select = False

    async def _async_test_single_host(self, name: str) -> None:
        from openmmla.tui.ssh import test_profile_by_name
        success, msg = await asyncio.to_thread(test_profile_by_name, name)
        self._refresh_target_options()
        color = "green" if success else "red"
        self._cmd.log(f"[{color}]'{name}': {msg}[/{color}]")

    async def _async_probe_hosts(self) -> None:
        from openmmla.tui.ssh import probe_all_profiles, summarize_states
        states = await asyncio.to_thread(probe_all_profiles)
        self._refresh_target_options()
        self._cmd.log(f"[green]Connection test finished: {summarize_states(states)}.[/green]")

    # -- table -----------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._set_target(self._get_selected_target())
        target = self._get_target()
        if target != "local":
            self.run_worker(self._refresh_table_remote(target), exclusive=True)
        else:
            self.run_worker(self._refresh_table_local(target), exclusive=True)

    async def _refresh_table_local(self, target: str) -> None:
        if target != self._get_target():
            return
        conda_envs = await asyncio.to_thread(_list_conda_envs_sync)
        env_packages = await asyncio.to_thread(self._list_selected_conda_packages, conda_envs)
        if target != self._get_target():
            return
        self._conda_envs = conda_envs
        self._env_packages = env_packages
        self._required_packages = self._build_required_map(self._read_local_pyproject())
        self._populate_table()

    def _read_local_pyproject(self) -> str | None:
        path = os.path.join(self._root, "pyproject.toml")
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def _build_required_map(self, pyproject_text: str | None) -> dict[str, set[str]]:
        """Build {group: required package names} from pyproject text.

        Falls back to an empty map (callers then use the hardcoded sentinel
        list) when the text is missing or a group can't be parsed.
        """
        result: dict[str, set[str]] = {}
        if not pyproject_text:
            return result
        for entry in ENV_GROUPS:
            deps = _parse_optional_deps_from_text(pyproject_text, entry["group"])
            if deps is not None:
                deps = set(deps)
                deps.add("openmmla")  # the editable project itself is always installed
                result[entry["group"]] = deps
        return result

    def _list_selected_conda_packages(self, conda_envs: set[str]) -> dict[str, set[str]]:
        packages: dict[str, set[str]] = {}
        for entry in ENV_GROUPS:
            env_name = entry["env"]
            if env_name in conda_envs:
                packages[env_name] = _list_conda_packages_sync(env_name)
        return packages

    def _populate_table(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.clear()
        for eg in ENV_GROUPS:
            status = _format_env_status(
                eg, self._conda_envs, self._env_packages, self._required_packages
            )
            table.add_row(
                eg["env"], eg["group"], eg["python"], status, eg["description"],
            )

    async def _refresh_table_remote(self, profile_name: str) -> None:
        if profile_name != self._get_target():
            return
        profile = get_profile_by_name(profile_name)
        if profile is None:
            if profile_name == self._get_target():
                self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[yellow]Checking conda envs on '{profile_name}'...[/yellow]")
        proc = await ssh_run_async(profile, wrap_remote("conda env list"))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        if profile_name != self._get_target():
            return
        self._conda_envs = _parse_conda_envs(output)
        env_packages = await self._list_remote_conda_packages(profile_name, self._conda_envs)
        pyproject_text = await self._read_remote_pyproject(profile)
        if profile_name != self._get_target():
            return
        self._env_packages = env_packages
        self._required_packages = self._build_required_map(pyproject_text)
        self._populate_table()
        self._log("[green]Remote env list refreshed.[/green]")

    async def _read_remote_pyproject(self, profile) -> str | None:
        remote_path = profile.remote_project_path
        proc = await ssh_run_async(
            profile, wrap_remote(f"cat {remote_path}/pyproject.toml")
        )
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode(errors="replace")
        rc = await proc.wait()
        if rc != 0 or not output.strip():
            return None
        return output

    async def _list_remote_conda_packages(
        self,
        profile_name: str,
        conda_envs: set[str],
    ) -> dict[str, set[str]]:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return {}
        env_names = [
            entry["env"] for entry in ENV_GROUPS
            if entry["env"] in conda_envs
        ]
        if not env_names:
            return {}
        cmd_parts = []
        for env_name in env_names:
            cmd_parts.append(
                f"echo __OPENMMLA_ENV__{env_name}; "
                f"conda list -n {env_name} 2>/dev/null || true"
            )
        proc = await ssh_run_async(profile, wrap_remote("; ".join(cmd_parts)))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        packages: dict[str, set[str]] = {}
        current_env: str | None = None
        current_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith("__OPENMMLA_ENV__"):
                if current_env is not None:
                    packages[current_env] = _parse_conda_packages("\n".join(current_lines))
                current_env = line.replace("__OPENMMLA_ENV__", "", 1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_env is not None:
            packages[current_env] = _parse_conda_packages("\n".join(current_lines))
        return packages

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#env-table", DataTable)
        try:
            row = table.get_row(event.row_key)
            self._selected_group = str(row[1])
            self._pending_delete = None
        except Exception:
            pass

    # -- buttons ---------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "env-target-refresh":
            self._cmd.log("[yellow]Testing connections to all hosts...[/yellow]")
            self.run_worker(self._async_probe_hosts(), group="env-host-probe", exclusive=True)
            return
        if bid == "btn-connect":
            self._refresh_target_options()
            self._set_target(self._get_selected_target())
            self._cmd.connect()
            self._refresh_table()
        elif bid == "btn-env-refresh":
            self._refresh_target_options()
            self._set_target(self._get_selected_target())
            self._refresh_table()
        elif bid == "btn-create-env":
            self._create_selected_env()
        elif bid == "btn-install-deps":
            self._install_selected_deps()
        elif bid == "btn-delete-env":
            self._delete_selected_env()
        elif bid == "btn-git-clone":
            self._git_clone()
        elif bid == "btn-git-pull":
            self._git_pull()

    # -- git operations -------------------------------------------------------

    def _git_clone(self) -> None:
        if not self._is_remote:
            self._log("[yellow]Git Clone is for remote targets only.[/yellow]")
            return
        profile = get_profile_by_name(self._get_target())
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        repo_url = git_remote_url()
        if not repo_url:
            self._log("[red]Could not determine git remote URL.[/red]")
            return
        self._log(f"[green]Cloning repo on '{profile.name}'...[/green]")
        self.run_worker(self._run_git_clone(profile.name, repo_url), exclusive=True)

    async def _run_git_clone(self, profile_name: str, repo_url: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        cmd = f"git clone {repo_url} {remote_path}"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log("[green]Git clone completed.[/green]")
        else:
            self._log(f"[red]Git clone failed (exit {rc}).[/red]")

    def _git_pull(self) -> None:
        if not self._is_remote:
            self._log("[yellow]Git Pull is for remote targets only.[/yellow]")
            return
        profile = get_profile_by_name(self._get_target())
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[green]Pulling latest on '{profile.name}'...[/green]")
        self.run_worker(self._run_git_pull(profile.name), exclusive=True)

    async def _run_git_pull(self, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        cmd = f"cd {remote_path} && git pull"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log("[green]Git pull completed.[/green]")
        else:
            self._log(f"[red]Git pull failed (exit {rc}).[/red]")

    # -- conda operations -----------------------------------------------------

    def _get_selected_entry(self) -> dict | None:
        if not self._selected_group:
            self._log("[yellow]Select a row first.[/yellow]")
            return None
        entry = next(
            (e for e in ENV_GROUPS if e["group"] == self._selected_group), None,
        )
        if entry is None:
            self._log(f"[red]Unknown group: {self._selected_group}[/red]")
        return entry

    def _create_selected_env(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        python_ver = entry["python"]
        if env_name in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' already exists.[/yellow]")
            return
        self._log(f"[green]Creating conda env '{env_name}' (python={python_ver})...[/green]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_create(self._get_target(), env_name, python_ver),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_create(env_name, python_ver), exclusive=True)

    async def _run_create(self, env_name: str, python_ver: str) -> None:
        cmd = ["conda", "create", "-n", env_name, f"python={python_ver}", "-y"]
        self._log(f"  $ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' created successfully.[/green]")
        else:
            self._log(f"[red]Failed to create env '{env_name}' (exit {rc}).[/red]")
        self._refresh_table()

    async def _run_remote_create(self, profile_name: str, env_name: str, python_ver: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cmd = f"conda create -n {env_name} python={python_ver} -y"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' created remotely.[/green]")
        else:
            self._log(f"[red]Remote env creation failed (exit {rc}).[/red]")
        self._refresh_table()

    def _delete_selected_env(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        if env_name not in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' does not exist.[/yellow]")
            return
        delete_key = (self._get_target(), env_name)
        if self._pending_delete != delete_key:
            self._pending_delete = delete_key
            target_label = "local" if delete_key[0] == "local" else f"remote target '{delete_key[0]}'"
            self._log(
                f"[yellow]Press Delete Env again to permanently delete '{env_name}' on {target_label}.[/yellow]"
            )
            return
        self._pending_delete = None
        self._log(f"[red]Deleting conda env '{env_name}'...[/red]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_delete(self._get_target(), env_name),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_delete(env_name), exclusive=True)

    async def _run_delete(self, env_name: str) -> None:
        cmd = ["conda", "env", "remove", "-n", env_name, "-y"]
        self._log(f"  $ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' deleted.[/green]")
        else:
            self._log(f"[red]Failed to delete env '{env_name}' (exit {rc}).[/red]")
        self._refresh_table()

    async def _run_remote_delete(self, profile_name: str, env_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cmd = f"conda env remove -n {env_name} -y"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' deleted remotely.[/green]")
        else:
            self._log(f"[red]Remote env deletion failed (exit {rc}).[/red]")
        self._refresh_table()

    def _install_selected_deps(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        group = entry["group"]
        if env_name not in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' does not exist. Create it first.[/yellow]")
            return
        self._log(f"[green]Installing deps '{rich_escape(f'[{group}]')}' into env '{env_name}'...[/green]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_install(self._get_target(), env_name, group),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_install(env_name, group), exclusive=True)

    async def _run_install(self, env_name: str, group: str) -> None:
        cmd = [
            "conda", "run", "-n", env_name,
            "pip", "install", "-e", f".[{group}]",
        ]
        self._log(f"  $ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self._root,
        )
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Dependencies '{rich_escape(f'[{group}]')}' installed successfully.[/green]")
        else:
            self._log(f"[red]Installation failed (exit {rc}).[/red]")
        self._refresh_table()

    async def _run_remote_install(self, profile_name: str, env_name: str, group: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        # --no-capture-output: conda run buffers child stdout/stderr by default
        # and only flushes on exit; this streams pip output line-by-line instead.
        cmd = (
            f"cd {remote_path} && "
            f"conda run --no-capture-output -n {env_name} pip install -e '.[{group}]'"
        )
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Remote install of '{rich_escape(f'[{group}]')}' completed.[/green]")
        else:
            self._log(f"[red]Remote install failed (exit {rc}).[/red]")
        self._refresh_table()
