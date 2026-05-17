from types import SimpleNamespace

from openmmla.tui.screens import environment


def test_environment_groups_use_current_dependency_extra_names():
    groups = {entry["group"]: entry for entry in environment.ENV_GROUPS}

    assert "asr-server" not in groups
    assert groups["asr-server-nemo"]["env"] == "asr-server-nemo"
    assert groups["asr-server-wespeaker"]["env"] == "asr-server-wespeaker"
    assert groups["vfa-server"]["env"] == "vfa-server"
    assert groups["vfa-vllm-runtime"]["env"] == "vfa-vllm"
    assert groups["vfa-vllm-runtime"]["python"] == "3.12"


def test_environment_target_options_reload_ssh_profiles(monkeypatch):
    monkeypatch.setattr(
        environment,
        "load_ssh_profiles",
        lambda: [SimpleNamespace(name="server-01"), SimpleNamespace(name="mac-01")],
    )

    assert environment._target_options() == [
        ("Local", "local"),
        ("server-01", "server-01"),
        ("mac-01", "mac-01"),
    ]


def test_environment_refresh_options_syncs_hidden_command_target(monkeypatch):
    monkeypatch.setattr(
        environment,
        "load_ssh_profiles",
        lambda: [SimpleNamespace(name="dell-01")],
    )

    class FakeSelect:
        value = "local"

        def set_options(self, options):
            self.options = options

    class FakeCommand:
        def __init__(self):
            self.targets = []

        def set_target(self, target):
            self.targets.append(target)

    panel = environment.EnvironmentPanel()
    panel._target = "dell-01"
    select = FakeSelect()
    command = FakeCommand()

    def fake_query_one(selector, *args):
        if selector == "#env-target-select":
            return select
        if selector == "#env-cmd-session":
            return command
        raise LookupError(selector)

    monkeypatch.setattr(panel, "query_one", fake_query_one)

    panel._refresh_target_options()

    assert panel._get_target() == "local"
    assert command.targets[-1] == "local"
