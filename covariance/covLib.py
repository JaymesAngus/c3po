"""
Module providing a class for preparing data assimilation increments for
plotting.
"""
# pylint: disable=invalid-name

import math
import numpy as np
import iris
import c3po.lib.fieldLib as field


class localIncrementClass(field.cField):
    """
    Class to prepare data assimilation increments for plotting to compare
    outputs from VAR and JADA Dirac and pseudo-observation tests.
    """
    def __init__(self, dlon, dlat, dlev, dirac_variable, normalise, filename, variable,
                 increment):
        field.cField.__init__(self, filename, variable, increment)
        self.dirac_variable = dirac_variable # Variable being perturbed
        self.normalise = normalise           # True if a scaling factor is to be applied
        self.normfactor = 0.0                # Normalisation scaling factor
        self.s0 = np.empty(0)                # 0D single point
        self.sx = np.empty(0)                # 1D longitude slice
        self.sxy = np.empty(0)               # 2D horizontal latitude-longitude data slice
        self.sxz = np.empty(0)               # 2D vertical longitude-altitude data slice
        self.sy = np.empty(0)                # 1D latitude slice
        self.sya = np.empty(0)               # 0D antipode longitude index
        self.syz = np.empty(0)               # 2D vertical latitude-altitude data slice
        self.sz = np.empty(0)                # 1D altitude slice
        self.dlon = dlon                     # Dirac point longitude
        self.xseg = np.empty(0)              # 1D longitude slice segment
        self.xseglon = np.empty(0)           # 1D longitude slice longitude coordinates
        self.xsegpm = 0                      # Number of longitude points each side of central point
        self.dlat = dlat                     # Dirac point latitude
        self.yseg = np.empty(0)              # 1D latitude slice segment
        self.yseglat = np.empty(0)           # 1D latitude slice latitude coordinates
        self.ysegpm = 0                      # Number of latitude points each side of central point
        self.z = dlev                        # Vertical level index

    def _read_jada_field(self):
        """
            Redefine _read_jada_field() from c3po.lib.fieldLib to include a callback function
            during the read to remove the standard_name.
            Doing this unifies field names as some have CF compliant names and others do not.
        """
        def callback(cube, field, filename):
            # pylint: disable=redefined-outer-name, unused-argument
            cube.standard_name = None

        self.system = "Jada"

        # read file appropriately
        if self.variable:
            variable = self._get_name()["name"]
            self.field = iris.load_cube(
                self.filename, variable, callback=callback)
        else:
            self.field = iris.load(self.filename, callback=callback)

        # update coordinate
        self._update_jada_coord()

        # extract the wanted model level if necessary
        if self.kwargs:
            constraint = iris.Constraint(**self.kwargs)
            self.field = self.field.extract(constraint)

    def _latlonlev(self):
        """
        find coordinates for lat, lon, lev
        """
        dim_names = {}
        for dim in range(len(self.field.shape)):
            dim_coords = self.field.coords(
                contains_dimension=dim, dim_coords=True)

            if dim_coords:
                for coord in dim_coords:
                    match coord.name():
                        case "latitude":
                            dim_names["lat"] = coord.name()
                        case "longitude":
                            dim_names["lon"] = coord.name()
                        case "level":
                            dim_names["level"] = coord.name()

        lats = self.field.coord(dim_names["lat"]).points
        lons = self.field.coord(dim_names["lon"]).points
        levs = self.field.coord("model_level_number").points

        coords = {
            "lat": dim_names["lat"],
            "lats": lats,
            "lon": dim_names["lon"],
            "lons": lons,
            "lev": "model_level_number",
            "levs": levs,
        }
        return coords

    def locate_indices_after_regrid(self):
        self.coord = self._latlonlev()
        lons = self.field.coord('longitude').points
        lats = self.field.coord('latitude').points

        def find_nearest(array, value):
            return int((np.abs(array - value)).argmin()) + 1

        self.x = find_nearest(lons, self.dlon)
        self.y = find_nearest(lats, self.dlat)

    def applyNormFactor(self, dirac_variable, normalise):
        """

        Args:
            dirac_variable (str): variable name
            normalise (bool): if True, the normalize factor is applied.
        """
        self.dirac_variable = dirac_variable
        self.normalise = normalise

        if self.variable == self.dirac_variable:
            s0 = self.field.extract(
                iris.Constraint(
                    coord_values={
                        self.coord["lon"]: self.coord["lons"][self.x - 1],
                        self.coord["lat"]: self.coord["lats"][self.y - 1],
                        self.coord["lev"]: self.coord["levs"][self.z - 1],
                    }
                )
            )

            self.normfactor = s0.data
            if self.normfactor > 0.0:
                self.normfactor = 1.0 / self.normfactor
        else:
            self.normfactor = 1.0

        if self.normalise:
            self.field.data *= self.normfactor

    def slice(self, verbose=False):
        """
        Extract 2D and 1D slices of 3D cube, containing point x,y,z.
        """

        name = self.field.name()
        wind_comp = name in ('u', 'u.inc', 'v', 'v.inc', 'x_wind', 'y_wind')

        lat = self.coord["lat"]
        lats = self.coord["lats"]
        lon = self.coord["lon"]
        lons = self.coord["lons"]
        lev = self.coord["lev"]
        levs = self.coord["levs"]

        nx = len(lons)
        ny = len(lats)

        pt_at_pole = ny % 2 == 1
        if pt_at_pole:
            ny_circ = (ny - 1) * 2
        else:
            ny_circ = ny * 2

        # set up segments with x,y at mid-point
        segment_fraction = 1.0
        self.ysegpm = int(ny_circ * segment_fraction * 0.5)
        self.xsegpm = min(
            int(
                nx
                * segment_fraction
                * 0.5
                / max(math.cos(math.radians(lats[self.y - 1])), 0.001)
            ),
            nx // 2,
        )
        self.yseg = np.zeros([1 + 2 * self.ysegpm])
        self.yseglat = np.zeros([1 + 2 * self.ysegpm])

        # convert from fortran index 1:n to python index 0:n-1
        self.sxz = self.field.extract(
            iris.Constraint(coord_values={lat: lats[self.y - 1]})
        )
        self.syz = self.field.extract(
            iris.Constraint(coord_values={lon: lons[self.x - 1]})
        )
        self.sxy = self.field.extract(
            iris.Constraint(coord_values={lev: levs[self.z - 1]})
        )

        self.sx = self.sxy.extract(
            iris.Constraint(coord_values={lat: lats[self.y - 1]})
        )
        self.sy = self.sxy.extract(
            iris.Constraint(coord_values={lon: lons[self.x - 1]})
        )
        self.sz = self.syz.extract(
            iris.Constraint(coord_values={lat: lats[self.y - 1]})
        )

        # also extract antipode, to complete great circle of longitude
        xan = (self.x - 1 + len(lons) // 2) % len(lons)
        self.sya = self.sxy.extract(
            iris.Constraint(coord_values={lon: lons[xan]}))
        self.sya = iris.util.squeeze(self.sya)
        self.sx = iris.util.squeeze(self.sx)
        self.sy = iris.util.squeeze(self.sy)
        # also extract 0D cube with single point
        self.s0 = self.sz.extract(iris.Constraint(
            coord_values={lev: levs[self.z - 1]}))

        # now extract segment of longitude circle with x at midpoint
        indices = [i % nx for i in range(self.x - self.xsegpm - 1,
                                         self.x + self.xsegpm)]

        self.xseg = self.sx.data[indices]

        if self.xseg.dtype is not np.float64:
            self.xseg = self.xseg.astype(np.float64)
        if isinstance(self.xseg, np.ma.MaskedArray):
            self.xseg = np.ma.getdata(self.xseg)

        # Transform longitude coordinates into [-180, 180] range to avoid
        # non-monotonicity across the prime meridian.
        self.xseglon = (lons[indices] + 180) % 360 - 180
        self.xseglon += 360

        # Transform xz plane coordinates into [-180, 180] range to avoid
        # non-monotonicity across the prime meridian.
        sxz_lon_coord = self.sxz.coord(lon)
        sxz_lons = np.array(sxz_lon_coord.points)
        sxz_lons_180 = (sxz_lons + 180) % 360 - 180
        sort_idx = np.argsort(sxz_lons_180)
        sxz_lon_coord.points = sxz_lons_180[sort_idx]
        self.sxz.data = np.array(self.sxz.data)[..., sort_idx]

        # now extract segment of great circle with y at midpoint
        for i in range(self.y - self.ysegpm - 1, self.y + self.ysegpm):
            if i >= 0 and i < ny:
                self.yseg[i - (self.y - self.ysegpm - 1)] = self.sy.data[i]
                self.yseglat[i - (self.y - self.ysegpm - 1)] = lats[i]

            else:  # cope with point over pole in antipode slice
                if i < 0:
                    if pt_at_pole:
                        ii = -i
                    else:
                        ii = -i - 1
                else:
                    if pt_at_pole:
                        ii = 2 * ny - i - 2
                    else:
                        ii = 2 * ny - i - 1

                if wind_comp:  # swap sign over pole
                    self.yseg[i - (self.y - self.ysegpm - 1)
                              ] = -self.sya.data[ii]
                else:
                    self.yseg[i - (self.y - self.ysegpm - 1)
                              ] = self.sya.data[ii]
                lata = self.sya.coord(lat).points[ii]  # antipodal lat

                # convert to extrapolated lat so plot axis is monotonic
                if lata > 0:
                    self.yseglat[i - (self.y - self.ysegpm - 1)] = 180.0 - lata
                else:
                    self.yseglat[i - (self.y - self.ysegpm - 1)
                                 ] = -180.0 - lata
        if verbose:
            self._outputDiagnostics()

    def _outputDiagnostics(self):
        print(" --------------------------------------------------------- ")
        print(f"          outputs for system: {self.system}")
        print(" --------------------------------------------------------- ")

        print(" \n")
        print("self.sxy.  max=", np.amax(self.sxy.data))
        print(self.sxy)

        print("-----\n")
        print("self.sxz.  max=", np.amax(self.sxz.data))
        print(self.sxz)

        print("-----\n")
        print("self.syz.  max=", np.amax(self.syz.data))
        print(self.syz)

        print("-----\n")
        print("sx.  max, x =", np.amax(self.sx.data),
              np.argmax(self.sx.data) + 1)
        print(self.sx)
        print(self.sx.data)

        print("-----\n")
        print("sy.   max, y =", np.amax(self.sy.data),
              np.argmax(self.sy.data) + 1)
        print(self.sy)
        print(self.sy.data)

        print("-----\n")
        print("sya.  max, y =", np.amax(self.sya.data),
              np.argmax(self.sya.data) + 1)
        print(self.sya)
        print(self.sya.data)

        print("-----\n")
        print("sz.  max, z =", np.amax(self.sz.data),
              np.argmax(self.sz.data) + 1)
        print(self.sz)

        print("-----\n")
        print(
            self.field.name(),
            "max, s0",
            np.amax(self.field.data),
            self.s0.data,
            self.s0,
        )

        print("-----\n")
        print(
            "self.xseg[",
            self.x - self.xsegpm - 1,
            ":",
            self.x + self.xsegpm,
            "]=",
            self.xseg,
        )

        print("-----\n")
        print("self.xseglon", self.xseglon)

        print("-----\n")
        print(
            "self.yseg[",
            self.y - self.ysegpm - 1,
            ":",
            self.y + self.ysegpm,
            "]=",
            self.yseg,
        )

        print("-----\n")
        print("self.yseglat", self.yseglat)
        print("\n")
