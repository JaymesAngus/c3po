#!/usr/bin/env python
"""
dirac_plot is designed to compare analysis increment fields from
var from a 3DVar_PseudoOb run (with the ob at an ENDGAME gridpoint) or a TestCov
run (Dirac test) and jada from a Dirac test, both specified by arguments
x -> lon, y -> lat, z -> lev, f -> dfield.
It assumes all VAR increments are on the ENDGAME global grid, while JADA
increments are on the LFRic cubed-sphere grid. We specify the ENDGAME grid
indices (x, y and z) for a point whose value will be used to normalise the field
for VAR pseudo-observation test output in order to compare to JADA Dirac test
output. Call with
python -vi <VAR increment> -ji <JADA increment> -e <expt name>
-o <output directory> -loni <Dirac longitude> -lati <Dirac latitude>
-verti <model level> -field <Dirac field>
"""

import argparse
import os
from pprint import pprint
import matplotlib as mpl
from iris.exceptions import IrisError
from c3po.lib.covariance import covLib
import c3po.lib.JadaVarNamingConvention as nameconv
import c3po.lib.covariance.covPlotLib as plotLib
import iris
import re
import sys

mpl.use("Agg")

def init_argparse():
    """Parse the python call arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-vi", "--var_inc",
        nargs="+",
        default=[],
        help="Increment file(s) produced by VAR",
    )
    parser.add_argument(
        "-ji",
        "--jada_inc",
        nargs="+",
        default=[],
        help="Increment file(s) produced by JADA",
    )
    parser.add_argument(
        "-s", "--save", action=argparse.BooleanOptionalAction,
        help="Save the plot, rather than display interactively",
        default=True
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        help="Output directory to save the figures (default: %(default)s)",
        default=f"{os.environ['SCRATCH']}/c3po_output/figs/dirac"
    )
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        required='--debug' not in os.sys.argv,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-lon", "--longitude",
        type=float,
        required='--debug' not in os.sys.argv,
        help="Dirac point longitude"
    )
    parser.add_argument(
        "-lat", "--latitude",
        type=float,
        required='--debug' not in os.sys.argv,
        help="Dirac point latitude"
    )
    parser.add_argument(
        "-nomlon", "--nominal_longitude",
        default=None,
        type=float,
        help="Nominal dirac point longitude for geostationary satellite projection"
    )
    parser.add_argument(
        "-nomlat", "--nominal_latitude",
        type=float,
        default=None,
        help="Nominal dirac point latitude for NearsidePerspective projection"
    )
    parser.add_argument(
        "-verti", "--vert_index",
        type=int,
        required='--debug' not in os.sys.argv,
        help="Dirac vertical level index"
    )
    parser.add_argument(
        "-field", "--field_name",
        type=str,
        required='--debug' not in os.sys.argv,
        help="Dirac variable",
        choices=["u", "v", "theta", "q", "qcl", "qcf", "exner", "density"],
    )
    parser.add_argument(
        "-rfields", "--response_fields",
        nargs="+",
        type=str,
        default=None,
        help="List of fields to plot (e.g. -rfields u v theta). If not set, all fields are plotted.",
    )
    parser.add_argument(
        "-n", "--normalise", action=argparse.BooleanOptionalAction,
        help="Normalise non-dirac test data to value at specified lat-lon-lev (default: %(default)s).",
        default=False
    )
    parser.add_argument(
        "-v", "--verbose", action=argparse.BooleanOptionalAction,
        help="Output diagnostics from covLib.py",
        default=False
    )
    parser.add_argument(
        '-d', '--debug', action=argparse.BooleanOptionalAction,
        help='Enable debug mode (default: %(default)s).',
        default=False,
    )
    return parser


def main(
    jada_input_files, var_input_files, save, output_dir, dlon, dlat, dlevel, expt_string, dfield, nom_lon, nom_lat, rfields, normalise, verbose,
):  # pylint: disable=too-many-arguments, too-many-locals
    """
    Args:
        jada_input_files (list): Increment file produced by JADA
        var_input_files  (list): Increment file produced by VAR
        output_dir      (str): Output directory to save the figures
        x               (int): Dirac longitude index
        y               (int): Dirac latitude index
        z               (int): Dirac vertical level index
        expt_string     (str): Name of the experiment
        dfield          (str): Dirac variable

    Raises:
        Exception: FileNotFoundError (jada_input_files and var_input_files)
        Exception: IrisError (requested variable absent from file)
    """

    if not jada_input_files and not var_input_files:
        sys.exit("UserError: no input files provided.")
    # Validate provided files independently
    for jada_file in jada_input_files:
        if not jada_file or not os.path.isfile(jada_file):
            raise FileNotFoundError(jada_file)
    for var_file in var_input_files:
        if not var_file or not os.path.isfile(var_file):
            raise FileNotFoundError(var_file)

    suffix = "D" if not normalise else "P"

    # make list of fields to process
    plotfields = list(nameconv.name_convention.keys())
    if not rfields:
        fields = [dfield] + [f for f in plotfields if f != dfield]
    else:
        fields = rfields
    print(f"fields to plot = {fields}")
    template_cube = iris.load("/data/users/jopa/test_data/c3po/cov/dirac/template_grid.nc")[0]

    for field in fields:
        print("\n --------------------------------------------------------- ")
        print(f" Dirac variable: {dfield} - Response variable {field}")
        print(" --------------------------------------------------------- \n")
        var_increments = {}
        jada_increments = {}
        exit_field_loop = False

        for i in range(len(var_input_files)):
            # VAR
            # ----------------------------------------------------------------------
            try:
                var = covLib.localIncrementClass(
                    dlon, dlat, dlevel, dfield, normalise, var_input_files[i], field,
                    increment=True
                )

            except (IrisError, KeyError):
                exit_field_loop = True
                print(f"\n [WARNING] no variable {field} in {var_input_files[i]}")
                break
            var.regrid(template_cube)
            var.locate_indices_after_regrid()
            var.applyNormFactor(dfield, normalise)
            var.slice(verbose)
            filename = re.search(r'[^/]+$', var_input_files[i]).group(0)
            dict_key = f"Var{str(i)} ({str(filename)})"
            var_increments[dict_key] = var

        for j in range(len(jada_input_files)):
            # JADA
            # ----------------------------------------------------------------------
            try:
                jada = covLib.localIncrementClass(
                    dlon, dlat, dlevel, dfield, normalise, jada_input_files[j], field,
                    increment=True
                )
            except (IrisError, KeyError):
                exit_field_loop = True
                print(f"\n [WARNING] no variable {field} in {jada_input_files[j]}")
                break
            
            jada.regrid(template_cube)
            jada.locate_indices_after_regrid()
            jada.applyNormFactor(dfield, normalise)
            jada.slice(verbose)
            filename = re.search(r'[^/]+$', jada_input_files[j]).group(0)
            dict_key = f"Jada{str(j)} ({str(filename)})"
            jada_increments[dict_key] = jada
        if exit_field_loop:
            continue


        # Instantiate plotting routines
        # ----------------------------------------------------------------------
        plot = plotLib.cPlot(
            var_increments,
            jada_increments,
            save = save,
            output_dir=output_dir,
            normalise=normalise,
            expt_string=expt_string,
            suffix=suffix,
        )

        # plot 1D, 2D
        plot.plotXsegments()
        plot.plotYsegments()
        plot.plotXYplane()
        plot.plotXZplane()
        plot.plotYZplane()
        plot.plotProfile()
        plot.plotNearsideProjection(nom_lon, nom_lat)

    print("\nEnd of script")


if __name__ == "__main__":
    args = init_argparse().parse_args()

    if args.nominal_longitude is None:
        args.nominal_longitude = args.longitude
    if args.nominal_latitude is None:
        args.nominal_latitude = args.latitude

    if args.debug:
        args.var_inc = ["/data/users/jopa/test_data/c3po/cov/dirac/var_dirac_u_10L_1E_45N"]
        args.jada_inc = ["/data/users/jopa/test_data/c3po/cov/dirac/dirac_parametricB_C224_u_10L_1E_45N.nc"]
        args.save = True
        args.output_dir = f"{os.environ['SCRATCH']}/c3po_output/figs/dirac"
        args.longitude = 1.
        args.latitude = 45.
        args.vert_index = 10
        args.exp_name = "u_10L_1E_45N_n320"
        args.field_name = "u"
        args.nominal_longitude = 1.0
        args.nominal_latitude = 45.0
        args.normalise = False
        args.verbose = False

    pprint(vars(args))

    main(
        args.jada_inc,
        args.var_inc,
        args.save,
        args.output_dir,
        args.longitude,
        args.latitude,
        args.vert_index,
        args.exp_name,
        args.field_name,
        args.nominal_longitude,
        args.nominal_latitude,
        args.response_fields,
        args.normalise,
        args.verbose,
    )
