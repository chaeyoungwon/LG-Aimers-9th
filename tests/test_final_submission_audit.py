import unittest

import pandas as pd

from scripts.audit_final_submission import (
    CHAMPION,
    DEFAULT_DATA_DIR,
    EXPECTED_SHA256,
    EXPECTED_SIZE,
    inspect_zip,
    model_component_audit,
    sha256_path,
    smoke_rows,
    static_independence_audit,
    validate_member_name,
    validate_submission,
)


class FinalSubmissionAuditTest(unittest.TestCase):
    def test_frozen_champion_identity_and_inventory(self):
        self.assertEqual(sha256_path(CHAMPION), EXPECTED_SHA256)
        self.assertEqual(CHAMPION.stat().st_size, EXPECTED_SIZE)
        audit, inventory = inspect_zip(CHAMPION)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["file_count"], 13)
        self.assertIn("script.py", inventory)
        self.assertFalse(audit["absolute_path_hits"])

    def test_zip_member_path_traversal_is_rejected(self):
        for name in ("../secret", "/absolute/path", "", "model/"):
            with self.assertRaises(ValueError):
                validate_member_name(name)

    def test_static_audit_has_no_test_aggregation_or_target_access(self):
        audit = static_independence_audit(CHAMPION)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["test_wide_prediction_aggregation_found"])
        self.assertFalse(audit["executable_test_target_access"])

    def test_every_serialized_component_is_repository_independent(self):
        components = model_component_audit(CHAMPION)
        self.assertEqual(len(components), 8)
        self.assertTrue(
            all(not row["repository_module_required"] for row in components)
        )

    def test_output_contract_and_extreme_case_matrix(self):
        output = pd.DataFrame(
            {"row_id": ["B", "A"], "control_success": [0.4, 0.6]}
        )
        self.assertTrue(validate_submission(output, ["B", "A"])["passed"])
        self.assertFalse(validate_submission(output, ["A", "B"])["passed"])
        sample = pd.read_csv(DEFAULT_DATA_DIR / "test.csv", encoding="utf-8-sig")
        frame, cases = smoke_rows(sample)
        self.assertEqual(len(frame), 11)
        self.assertEqual(
            set(cases),
            {
                "unseen_pitcher",
                "unseen_batter",
                "missing_categorical",
                "missing_numeric",
                "rare_category",
                "game_type_f",
                "game_type_r",
                "full_count",
                "bases_empty",
                "bases_full",
                "extreme_asof_count",
            },
        )


if __name__ == "__main__":
    unittest.main()
