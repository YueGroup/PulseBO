import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"


class ReadmeScriptReferenceTests(unittest.TestCase):
    def test_readme_script_commands_reference_existing_scripts(self):
        readme_text = README.read_text(encoding="utf-8")
        script_refs = re.findall(r"python\s+scripts[\\/](\S+?\.py)", readme_text)

        self.assertTrue(script_refs, "README.md should document script commands.")

        missing = [
            script_ref
            for script_ref in script_refs
            if not (PROJECT_ROOT / "scripts" / script_ref).exists()
        ]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
