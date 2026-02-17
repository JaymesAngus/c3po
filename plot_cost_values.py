import argparse
import copy
import matplotlib.pyplot as plt
import os
from lib.costLib import *
import matplotlib as mpl

# Make rendering deterministic across environments
mpl.use("Agg", force=True)
mpl.rcdefaults()
mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "savefig.bbox": "standard",       # ensure no content-driven resizing
    "savefig.pad_inches": 0.0,
    "figure.constrained_layout.use": False,
    "figure.autolayout": False,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

"""
Example usage:
python plot_cost_values.py --jadafile=$HOME/cylc-run/malak/log/job/20210701T1200Z/glu_jada/NN/job.out \
                           --varfile=$HOME/cylc-run/malak/log/job/20210701T1200Z/glu_var_jopa_anal_high/NN/job.stats \
                           --outputdir=$HOME/myplots
Notes:
(1) Requires the use of python 3.
(2) The jadafile should be the output file from a jada run e.g. job.out
(3) The varfile should be the stats file produced by a run of var e.g. job.stats
(4) The output directory to put the plots.
"""

def init_argparse():
    """Parse the python call arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-j", "--jadafile",
        type=str,
        required='--debug' not in os.sys.argv
        )

    parser.add_argument(
        "-v", "--varfile",
        type=str,
        required='--debug' not in os.sys.argv
        )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        help="Output directory to save the figures (default: %(default)s)",
        default=f"{os.environ['SCRATCH']}/c3po_output/figs/cost_values"
        )

    parser.add_argument(
        '-d', '--debug', action=argparse.BooleanOptionalAction,
        help='Enable debug mode (default: %(default)s).',
        default=False,
    )

    return parser

def write_cost_data(data, outname):
    outfile = open(outname, 'w')
    outfile.write("iter, j, jojc, jb \n")
    for aa in range(0, len(data.J)):
        getdataout = f'{data.iter[aa]}, {data.J[aa]}, {data.JoJc[aa]}, {data.Jb[aa]}\n'
        outfile.write(getdataout)
    outfile.close()


def plot_cost_data(data, outputname,
                   ylabel="Cost function value",
                   hline=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(data.iter, data.J, '-+', label='J')
    ax.plot(data.iter, data.JoJc, '-+', label='JoJc')
    ax.plot(data.iter, data.Jb, '-+', label='Jb')
    if hline is not None:
        ax.axhline(hline, color="black")
    ax.legend(loc="best")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(ylabel)

    # Avoid tight_layout; keep fixed canvas size
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.92)
    fig.savefig(outputname, dpi=100, bbox_inches=None, pad_inches=0.0, transparent=False)
    plt.close(fig)


def plot_cost_difference(jada, var, outputname):
    localdata = copy.deepcopy(jada)
    localdata.J = var.J - jada.J
    localdata.JoJc = var.JoJc - jada.JoJc
    localdata.Jb = var.Jb - jada.Jb
    plot_cost_data(localdata, outputname,
                   ylabel="Cost function difference (VAR - JADA)",
                   hline=0)


def plot_cost_difference_over_mean(jada, var, outputname):
    localdata = copy.deepcopy(jada)
    localdata.J = 200.0 * (var.J - jada.J) / (var.J + jada.J)
    localdata.JoJc = 200.0 * (var.JoJc - jada.JoJc) / (var.J + jada.J)
    localdata.Jb = 200.0 * (var.Jb - jada.Jb) / (var.J + jada.J)
    plot_cost_data(localdata, outputname,
                   ylabel="Cost function difference (VAR - JADA) \n divided by mean total (%)",
                   hline=0)


def plot_cost_data_side_by_side(jada, var, outputname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100, sharey=True)

    ax1.plot(jada.iter, jada.J, '-+', label='J')
    ax1.plot(jada.iter, jada.JoJc, '-+', label='JoJc')
    ax1.plot(jada.iter, jada.Jb, '-+', label='Jb')
    ax1.set_title("JADA cost function values")
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylabel("Cost function value")
    ax1.legend(loc="best")

    ax2.plot(var.iter, var.J, '-+', label='J')
    ax2.plot(var.iter, var.JoJc, '-+', label='JoJc')
    ax2.plot(var.iter, var.Jb, '-+', label='Jb')
    ax2.set_title("VAR cost function values")
    ax2.set_xlabel("Number of iterations")
    ax2.legend(loc="best")

    # Fixed spacing to avoid any auto layout affecting saved size
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.35)
    fig.savefig(outputname, dpi=100, bbox_inches=None, pad_inches=0.0, transparent=False)
    plt.close(fig)

def main(jadafile, varfile, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    jada = cost_function_info_for_jada(jadafile)
    var = cost_function_info_for_var(varfile)
    plot_cost_data(jada, f"{output_dir}/cost_jada.png")
    plot_cost_data(var, f"{output_dir}/cost_var.png")
    plot_cost_difference(jada, var, f"{output_dir}/cost_var_minus_jada.png")
    plot_cost_difference_over_mean(jada, var, f"{output_dir}/cost_var_minus_jada_over_mean_total.png")
    plot_cost_data_side_by_side(jada, var, f"{output_dir}/cost_side_by_side.png")

if __name__ == "__main__":
    args = init_argparse().parse_args()

    if args.debug:
        args.jadafile = "input_data/job.out"
        args.varfile = "input_data/job.stats"
        args.output_dir = f"c3po_output/figs/cost_values"
    main(args.jadafile, args.varfile, args.output_dir)
