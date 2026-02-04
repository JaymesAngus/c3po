"""Unit test to compare VAR and JADA Dirac perturbations."""
# pylint: disable=invalid-name

import unittest
from c3po.plot_cost_values import main as cost_values
from c3po.tests._image_test_base_class import ImageTestBaseClass

test_data_path = "data/cost_values/"
class TestCostValues(ImageTestBaseClass):
    """Tests using the unittest package infrastructure."""

    def test_plot_cost_values(self):
        jadafile = "input_data/job.out"
        varfile = "input_data/job.stats"
        output_dir = self.output_dir

        cost_values(jadafile, varfile, output_dir)

        # test outputs figures
        self._run_test(test_data_path)

def runTest():
    unittest.main(__name__)

if __name__ == "__main__":
    runTest()
