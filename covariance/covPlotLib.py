"""
Module for plotting comparison diagnostics for JADA and VAR Dirac and
pseudo-observation test increments.
"""
# pylint: disable=invalid-name

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import numpy as np
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from iris.util import squeeze
import c3po.lib.JadaVarNamingConvention as nameconv
import matplotlib.cm
from matplotlib.ticker import MaxNLocator


mpl.use("Agg")
class cPlot:

    """Class for producing diagnostic comparison plots.

    Notes
    -----
    All arguments except for cVAR and cLFric must be provided as keyword arguments
    to avoid potential mix-ups between output_dir and expt_string.
    """
    def __init__(self, cVAR, cLFric, *, save, output_dir, normalise=False, expt_string="", suffix=""):
        self.VAR = cVAR
        self.LFric = cLFric
        self.save = save
        self.output_dir = output_dir
        self.normalise = bool(normalise)
        self.expt_string = expt_string
        self.suffix = suffix
        self.title_key = ""

        if not cVAR:
            #next(iter(...)) returns the first value in a dict,
            #i.e. the first LFRic cube if no VAR cubes were provided
            self.generic_source = next(iter(self.LFric.values()))
        else:
            self.generic_source = next(iter(self.VAR.values()))
        self.EW = "W" if self.generic_source.dlon <= 0 else "E"
        self.NS = "S" if self.generic_source.dlat <= 0 else "N"

        os.makedirs(self.output_dir, exist_ok=True)

    def plotProfile(self):
        """Plot a vertical profile comparison."""
        for var_object in self.VAR:
            x_Var = squeeze(self.VAR[var_object].sz.data)
            y_Var = self.VAR[var_object].coord["levs"]
            if self.normalise:
                label_Var = "V" + self.suffix + " N"
            else:
                label_Var = var_object
            plt.plot(x_Var, y_Var, label=label_Var)

        for jada_object in self.LFric:
            x_LFric = squeeze(self.LFric[jada_object].sz.data)
            y_LFric = self.LFric[jada_object].coord["levs"]
            if self.normalise:
                label_LFric = "J  N"
            else:
                label_LFric = jada_object
            plt.plot(x_LFric, y_LFric, label=label_LFric)

        self.title_key = "_ew"

        plt.xlabel(nameconv.name_convention[self.generic_source.variable]["unit"])
        plt.ylabel("model level number")

        title = (
            f"{self.expt_string}\n"
            f"Response variable: {self.generic_source.variable}"
            f" vertical profile at"
            f" {abs(int(self.generic_source.dlat))}{self.NS},"
            f" {abs(int(self.generic_source.dlon))}{self.EW}"
        )

        plt.title(title)
        plt.legend(prop={"size": 6})

        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + "_z"
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close()
        else:
            plt.show()

    def plotYsegments(self):
        """Prepare a longitude segment comparison."""
        var_segment_inputs = []
        for var_object in self.VAR:
            x_Var = self.VAR[var_object].yseglat
            x_Var %= 360
            y_Var = self.VAR[var_object].yseg
            # Sort longitude values to and reorder y values accordingly
            x_Var, y_Var = zip(*sorted(zip(x_Var, y_Var)))

            if self.VAR[var_object].normalise:
                label_Var = "V" + self.suffix + " N"
            else:
                label_Var = var_object
            var_segment_inputs.append((x_Var, y_Var, label_Var))

        jada_segment_inputs = []
        for jada_object in self.LFric:
            x_LFric = self.LFric[jada_object].yseglat
            x_LFric %= 360
            y_LFric = self.LFric[jada_object].yseg
            x_LFric, y_LFric = zip(*sorted(zip(x_LFric, y_LFric)))

            if self.LFric[jada_object].normalise:
                label_LFric = "J  N"
            else:
                label_LFric = jada_object
            self.title_key = "_fns"
            jada_segment_inputs.append((x_LFric, y_LFric, label_LFric))

        xlabel = (
            "segment of NS great-circle (longitude="
            + str(self.generic_source.coord["lons"][self.generic_source.x - 1])
            + ")"
        )

        title = (
            f"{self.expt_string}\n"
            f"Response variable: {self.generic_source.variable}\n"
            f" y-segment at longitude {abs(int(self.generic_source.dlon))}{self.EW}"
        )

        self._plotSegments(
            var_segment_inputs, jada_segment_inputs, xlabel, title
        )

    def plotXsegments(self):
        """Prepare a latitude segment comparison."""
        var_segment_inputs = []
        for var_object in self.VAR:
            x_Var = self.VAR[var_object].xseglon
            x_Var = (x_Var + 180) % 360 - 180
            y_Var = self.VAR[var_object].xseg
            x_Var, y_Var = zip(*sorted(zip(x_Var, y_Var)))
            if self.VAR[var_object].normalise:
                label_Var = "V" + self.suffix + " N"
            else:
                label_Var = var_object
            var_segment_inputs.append((x_Var, y_Var, label_Var))

        jada_segment_inputs = []
        for jada_object in self.LFric:
            x_LFric = self.LFric[jada_object].xseglon
            x_LFric = (x_LFric + 180) % 360 - 180
            y_LFric = self.LFric[jada_object].xseg
            x_LFric, y_LFric = zip(*sorted(zip(x_LFric, y_LFric)))
            if self.LFric[jada_object].normalise:
                label_LFric = "J  N"
            else:
                label_LFric = jada_object
            self.title_key = "_ew"
            jada_segment_inputs.append((x_LFric, y_LFric, label_LFric))

        xlabel = (
            "longitude (for latitude "
            + str(self.generic_source.dlat)
            + ")"
        )
        title = (
            f"{self.expt_string}\n"
            f"Response variable: {self.generic_source.variable}\n"
            f" x-segment at latitude {abs(int(self.generic_source.dlat))}{self.NS}"
        )

        self._plotSegments(
            var_segment_inputs, jada_segment_inputs, xlabel, title
        )

    def _plotSegments(
        self, var_segment_inputs, jada_segment_inputs, xlabel, title
    ):
        """Plot a latitude or longitude segment comparison."""
        for x, y, label in var_segment_inputs:
            plt.plot(x, y, label=label)
        for x, y, label in jada_segment_inputs:
            plt.plot(x, y, label=label)
        plt.ylabel(nameconv.name_convention[self.generic_source.variable]["unit"])
        plt.xlabel(xlabel)

        plt.title(title)
        plt.legend(prop={"size": 6})
        plt.tight_layout(w_pad=0)

        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + self.title_key
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close()
        else:
            plt.show()

    def _getColourbarLevels(self, var_data_list, jada_data_list):
        """Compute colourbar levels for any number of VAR and JADA input arrays."""
        # Avoid concatenating large arrays; stream min/max across inputs
        absmax = 0.0
        for arr in var_data_list + jada_data_list:
            if arr is None:
                continue
            # use nan-aware min/max; cast to float to avoid dtype surprises
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            absmax = max(absmax, abs(arr_min), abs(arr_max))
        if absmax == 0.0:
            absmax = 1.0
            print("\n [WARNING] All provided fields contain only zero values\n")
        n_levels = 10
        locator = MaxNLocator(nbins=n_levels)
        levels = locator.tick_values(-absmax, absmax)
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256, clip=False)
        sm = matplotlib.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        return sm, norm

    def _populate2DAxes(self, cClass, figname, ax, norm, slice_type, cClass2=None):
        pcolormesh_kwargs = dict(
            cmap='coolwarm',
            shading='nearest',  # avoid extra interpolation work
        )
        match slice_type:
            case "sxy":
                coords=[cClass.coord["lons"], cClass.coord["lats"]]
                texty = -0.12
                pcolormesh_kwargs["transform"] = ccrs.PlateCarree()
                ax.coastlines(resolution='110m', linewidth=0.5)
            case "sxz":
                coords=[cClass.coord["lons"], cClass.coord["levs"]]
                texty = -0.25
            case "syz":
                texty = -0.25
                coords=[cClass.coord["lats"], cClass.coord["levs"]]
            case _:
                raise ValueError(f"Unknown cube slice type: {slice_type}")
        
        cClass_slice = getattr(cClass, slice_type)
        x = coords[0]
        y = coords[1]
        # the presence of cClass2 implies a difference plot
        if cClass2 is not None:
            data1 = squeeze(cClass_slice.data)
            data2 = squeeze(getattr(cClass2, slice_type).data)
            # handle differing vertical level numbers
            min_level_number = min(data1.shape[0], data2.shape[0])
            data1 = data1[:min_level_number, ...]
            data2 = data2[:min_level_number, ...]
            field_data = (data1 - data2).astype(np.float32, copy=False)
            y = y[:min_level_number]
        else:
            field_data = cClass_slice.data.astype(np.float32, copy=False)

        plt.sca(ax)
        lon_grid, lat_grid = np.meshgrid(x, y)
        contour_plot = ax.pcolormesh(
            lon_grid, lat_grid, squeeze(field_data),
            norm=norm,
            **pcolormesh_kwargs
        )

        title_font = font.FontProperties("monospace")
        ax.set_title(figname, fontproperties=title_font)

        caption = (
            f"min = {field_data.min():.3g},"
            f" max = {field_data.max():.3g}"
        )
        ax.text(
            0.5, texty,
            caption,
            ha='center', va='top',
            transform=ax.transAxes,
            fontsize=12
        )
        return contour_plot

    def _generate2DPlot(self, subplot_args, cbar_args, slice_type):
        var_data_list = [getattr(v, slice_type).data for v in self.VAR.values()]
        jada_data_list = [getattr(j, slice_type).data for j in self.LFric.values()]
        sm, norm = self._getColourbarLevels(var_data_list, jada_data_list)
        fig, axes = plt.subplots(1, 3, **subplot_args,)

        for idx, (key, var_increment) in enumerate(self.VAR.items()):
            self._populate2DAxes(var_increment, key, axes[idx],
                                      norm, slice_type)
        offset = len(self.VAR)
        for idx, (key, jada_increment) in enumerate(self.LFric.items()):
            self._populate2DAxes(jada_increment, key, axes[offset + idx],
                                      norm, slice_type)
        if len(self.VAR) > 0 and len(self.LFric) > 0:
            self._populate2DAxes(
                list(self.VAR.values())[0], "Difference (left-centre)", axes[2],
                norm, slice_type, list(self.LFric.values())[0]
            )
        elif len(self.LFric) > 1:
            self._populate2DAxes(
                list(self.LFric.values())[0], "Difference (left-centre)", axes[2],
                norm, slice_type, list(self.LFric.values())[1]
            )
        else:
            axes[2].set_title("No difference plot available")

        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', **cbar_args)
        unit = nameconv.name_convention[self.generic_source.variable]["unit"]
        if unit == "1":
            unit = "(dimensionless)"
        cbar.set_label(unit)
        cbar.ax.minorticks_off()
        return fig, axes, cbar

    def plotXYplane(self):
        """Plot of XY plane at dirac point level"""
        xy_subplot_args = {
            'figsize': (14,5),
            'constrained_layout': True,
            'subplot_kw': {'projection': ccrs.PlateCarree()}
            }
        xy_cbar_args = {'fraction': 0.08, 'pad': 0.18}
        fig, axes, cbar = self._generate2DPlot(xy_subplot_args, xy_cbar_args, "sxy")

        fig.suptitle(f"{self.expt_string}\n"
                     f"Response variable: {self.generic_source.variable}\n"
                     f"XY contour at model level {str(self.generic_source.z)}")
        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + "_xy"
                + str(self.generic_source.z)
                + "L"
                + "_"
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close(fig)
        else:
            plt.show()
          

    def plotXZplane(self):
        """Plot of XZ plane at dirac point latitude"""
        xz_subplot_args = {
            'figsize': (14,6),
            'constrained_layout': True,
            }
        xz_cbar_args = {'fraction': 0.08, 'pad': 0.08}
        fig, axes, cbar = self._generate2DPlot(xz_subplot_args, xz_cbar_args, "sxz")


        fig.supxlabel(self.generic_source.coord["lon"], x=0.54, y=0.25)
        fig.supylabel(self.generic_source.coord["lev"], y=0.6)
        fig.suptitle(f"{self.expt_string}\n"
                     f"Response variable: {self.generic_source.variable}\n"
                     f"XZ contour at latitude {str(self.generic_source.dlat)}{self.NS}")
        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + "_xz"
                + str(int(self.generic_source.dlat))
                + self.NS
                + "_"
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plotYZplane(self):
        """Plot of YZ plane at dirac point longitude"""
        yz_subplot_args = {
            'figsize': (14,6),
            'constrained_layout': True,
            }
        yz_cbar_args = {'fraction': 0.08, 'pad': 0.08}
        fig, axes, cbar = self._generate2DPlot(yz_subplot_args, yz_cbar_args, "syz")

        fig.supxlabel(self.generic_source.coord["lat"], x=0.54, y=0.25)
        fig.supylabel(self.generic_source.coord["lev"], y=0.6)
        fig.suptitle(f"{self.expt_string}\n"
                     f"Response variable: {self.generic_source.variable}\n"
                     f"YZ contour at longitude {str(self.generic_source.dlon)}{self.EW}")

        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + "_yz"
                + str(int(self.generic_source.dlon))
                + self.EW
                + "_"
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plotNearsideProjection(self, nom_lon, nom_lat):
        """Nearside persepctive plot"""
        nearside_subplot_args = {
            'figsize': (14,6),
            'constrained_layout': True,
            'subplot_kw': {'projection': ccrs.NearsidePerspective(nom_lon, nom_lat)}
            }
        nearside_cbar_args = {'fraction': 0.05, 'pad': 0.05}
        fig, axes, cbar = self._generate2DPlot(nearside_subplot_args, nearside_cbar_args, "sxy")

        fig.suptitle(f"{self.expt_string}\n"
                     f"Response variable: {self.generic_source.variable}\n"
                     f"Nearside projection at model level {str(self.generic_source.z)}",
                     fontsize=16)
        if self.save:
            outfile = os.path.join(
                self.output_dir,
                self.expt_string
                + "_"
                + self.generic_source.variable
                + "_nearside"
                + str(self.generic_source.z)
                + "L"
                + "_"
                + self.suffix
                + ".png",
            )
            plt.savefig(outfile, dpi=200)
            plt.close(fig)
        else:
            plt.show()
