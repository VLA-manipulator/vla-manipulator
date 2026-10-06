from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from openpi_ext.install_into_openpi import DATA_BEGIN, IMPORT_LINE, TRAIN_BEGIN, install


MINIMAL_CONFIG = """import openpi.policies.libero_policy as libero_policy


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    pass

_CONFIGS = [
]
"""


class InstallerTest(unittest.TestCase):
    def test_install_is_guarded_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "src/openpi/training/config.py"
            policies = root / "src/openpi/policies"
            policies.mkdir(parents=True)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(MINIMAL_CONFIG, encoding="utf-8")

            self.assertTrue(install(root, repo_id="test-user/so101-data"))
            installed = config.read_text(encoding="utf-8")
            self.assertIn(IMPORT_LINE, installed)
            self.assertIn(DATA_BEGIN, installed)
            self.assertIn(TRAIN_BEGIN, installed)
            self.assertIn('repo_id="test-user/so101-data"', installed)
            self.assertTrue((policies / "so101_policy.py").is_file())
            self.assertFalse(install(root, repo_id="test-user/so101-data"))
            self.assertEqual(config.read_text(encoding="utf-8"), installed)

    def test_refuses_unknown_openpi_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "src/openpi/training/config.py"
            (root / "src/openpi/policies").mkdir(parents=True)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("_CONFIGS = []\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing anchor"):
                install(root)


if __name__ == "__main__":
    unittest.main()
