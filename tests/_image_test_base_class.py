import os
import glob
import unittest
import shutil
import c3po.tests.compareImage as compareImage

class ImageTestBaseClass(unittest.TestCase):
    """Reusable base for tests that compare generated plot images vs. reference images."""

    def setUp(self):
        """Initialise a per-test output subdirectory."""
        OUTPUT_DIR = f"test_figs"
        base_output_dir = OUTPUT_DIR
        os.makedirs(base_output_dir, exist_ok=True)
        self.output_dir = os.path.join(base_output_dir, self._testMethodName)
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                fp = os.path.join(self.output_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def _run_test(self, test_data_path: str, test_data_directory: str | None = ""):
        """Compare plotting output with KGO files."""
        # Test output
        # ---------------------------------------------

        sourceFigDir = os.path.join(
            os.path.dirname(__file__), test_data_path, test_data_directory
        )
        script_output_images = glob.glob(os.path.join(self.output_dir, '*.png'))
        reference_images = glob.glob(os.path.join(sourceFigDir, 'REF_*.png'))

        # Ensure both generated and reference image directories are non-empty before comparing
        if not script_output_images:
            self.fail(f"No output images produced in {self.output_dir}")
        if not reference_images:
            self.fail(f"No reference images found in /tests/{test_data_path}{test_data_directory}/")

        expected = {os.path.basename(filename)[4:] for filename in reference_images}  # strip 'REF_'
        generated = {os.path.basename(filename) for filename in script_output_images}

        if generated != expected:
            missing = sorted(expected - generated)
            extra = sorted(generated - expected)
            parts = []
            if missing:
                parts.append("Missing:\n - " + "\n - ".join(missing))
            if extra:
                parts.append("Unexpected:\n + " + "\n - ".join(extra))
            self.fail("Mismatch in generated images versus reference images:\n" + "\n".join(parts))

        for generated_image in script_output_images:
            figName = os.path.basename(generated_image)
            reference_image = os.path.join(sourceFigDir, "REF_" + figName)

            result = compareImage.compareImage(generated_image, reference_image)

            # only test result from 'SsimCompare' when comparing images
            if result["SsimCompare"] < 0.95:
                self.fail(f"output differs between: \n {generated_image}\n {reference_image}\n\n")
            else:
                os.remove(generated_image)
                pass

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
