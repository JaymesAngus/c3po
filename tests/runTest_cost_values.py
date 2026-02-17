"""Unit test to compare VAR and JADA Dirac perturbations."""
# pylint: disable=invalid-name

import unittest
import os
import matplotlib
from matplotlib import pyplot as plt

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 100,
})

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
