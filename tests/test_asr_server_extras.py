import ast
from pathlib import Path
import unittest


def load_optional_dependencies(project_root):
    optional_dependencies = {}
    for line in (project_root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if " = [ " not in line:
            continue
        name, value = line.split(" = ", 1)
        optional_dependencies[name] = ast.literal_eval(value)
    return optional_dependencies


class ASRServerExtrasTest(unittest.TestCase):
    def test_nemo_and_wespeaker_server_extras_are_explicit(self):
        project_root = Path(__file__).resolve().parents[1]
        optional_dependencies = load_optional_dependencies(project_root)

        self.assertIn("asr-server-nemo", optional_dependencies)
        self.assertIn("asr-server-wespeaker", optional_dependencies)

        self.assertEqual(optional_dependencies["asr-server"], optional_dependencies["asr-server-nemo"])
        self.assertIn("nemo-toolkit[asr]<=1.23.0", optional_dependencies["asr-server-nemo"])
        self.assertTrue(
            any(dep.startswith("wespeaker @ git+") for dep in optional_dependencies["asr-server-wespeaker"])
        )


if __name__ == "__main__":
    unittest.main()
