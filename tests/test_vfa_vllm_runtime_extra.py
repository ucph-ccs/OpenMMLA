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


class VFAVLLMRuntimeExtraTest(unittest.TestCase):
    def test_vfa_vllm_runtime_extra_contains_local_vllm_runtime(self):
        project_root = Path(__file__).resolve().parents[1]
        optional_dependencies = load_optional_dependencies(project_root)

        self.assertIn("vfa-vllm-runtime", optional_dependencies)
        self.assertNotIn("vfa-server-local", optional_dependencies)
        dependencies = optional_dependencies["vfa-vllm-runtime"]

        self.assertIn("qwen-vl-utils==0.0.14", dependencies)
        self.assertIn("transformers>=4.57.0", dependencies)
        self.assertIn("vllm>=0.11.0", dependencies)


if __name__ == "__main__":
    unittest.main()
