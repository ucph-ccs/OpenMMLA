import ast
from pathlib import Path
import unittest


class VFAOptionalImportsTest(unittest.TestCase):
    def test_zhipuai_sdk_is_not_imported_at_module_load(self):
        project_root = Path(__file__).resolve().parents[1]
        analyzer_path = (
            project_root
            / "openmmla"
            / "services"
            / "vfa"
            / "multi_angle_vllm_frame_analyzer.py"
        )
        tree = ast.parse(analyzer_path.read_text(encoding="utf-8"))

        top_level_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

        imported_modules = set()
        for node in top_level_imports:
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("zai", imported_modules)


if __name__ == "__main__":
    unittest.main()
