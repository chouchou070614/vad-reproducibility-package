import argparse
import unittest

import Evaluate
import Train


class EntryPointTest(unittest.TestCase):
    def test_ped2_prediction_training_is_frozen(self):
        args = argparse.Namespace(
            branch="P", dataset="ped2", dataset_root="dataset", output="out",
            gpu="0", seed=2020, device="auto",
        )
        command = Train.build_command(args)
        joined = " ".join(command)
        self.assertIn("--kmeans_warmup_epochs 26", joined)
        self.assertIn("--loss_compact 0.1", joined)
        self.assertIn("--lr 0.0002", joined)

    def test_reconstruction_uses_original_branch_scale(self):
        args = argparse.Namespace(
            branch="R", dataset="avenue", dataset_root="dataset", output="out",
            gpu="0", seed=2020, device="auto",
        )
        joined = " ".join(Train.build_command(args))
        self.assertIn("--kmeans_warmup_epochs 27", joined)
        self.assertIn("--loss_compact 0.01", joined)
        self.assertIn("--lr 2e-05", joined)

    def test_evaluation_uses_branch_specific_alpha(self):
        args = argparse.Namespace(
            branch="P", split="testing", dataset="avenue", dataset_root="dataset",
            model="model.pth", memory="keys.pt", output="scores.npz",
            batch_size=4, num_workers=0, feature_sample_count=0, seed=2020, device="cpu",
        )
        joined = " ".join(Evaluate.build_command(args))
        self.assertIn("--alpha 0.7", joined)
        self.assertIn("Export_Components.py", joined)


if __name__ == "__main__":
    unittest.main()
