import ast
from pathlib import Path
import unittest


class VFAVLLMCommandTest(unittest.TestCase):
    def test_vfa_vllm_command_uses_multi_angle_analyzer(self):
        project_root = Path(__file__).resolve().parents[1]
        command_path = project_root / "openmmla" / "commands" / "vfa" / "vllm.py"
        tree = ast.parse(command_path.read_text(encoding="utf-8"))

        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "openmmla.services.vfa"
            for alias in node.names
        }
        self.assertIn("MultiAngleVLLMFrameAnalyzer", imported_names)
        self.assertNotIn("VLLMFrameAnalyzer", imported_names)

        class_type_names = [
            keyword.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "class_type" and isinstance(keyword.value, ast.Name)
        ]
        self.assertIn("MultiAngleVLLMFrameAnalyzer", class_type_names)
        self.assertNotIn("VLLMFrameAnalyzer", class_type_names)


if __name__ == "__main__":
    unittest.main()
