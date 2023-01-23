import numpy as np
import h5py
import healpy as hp
from astropy.io import fits
from .plot_utils import show_plt
from astropy import units

def is_fits(filepath):
    """
    Check if file is a FITS file
    Returns True of False

    Parameters
    ----------
    filepath: str
        Path to file
    """
    FITS_SIGNATURE = (b"\x53\x49\x4d\x50\x4c\x45\x20\x20\x3d\x20\x20\x20\x20\x20"
                      b"\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"
                      b"\x20\x54")
    with open(str(filepath),'rb') as f:
        try:
            return f.read(30) == FITS_SIGNATURE
        except FileNotFoundError as e:
            print(e)
            return False


class BaseSkyModel(object):
    """ Global sky model (GSM) class for generating sky models.
    """
    def __init__(self, name, filepath, freq_unit, data_unit, basemap):
        """ Initialise basic sky model class

        Parameters
        ----------
        name (str):      Name of GSM
        filepath (str):    Path to HDF5 data / FITS data (healpix) to load
        freq_unit (str): Frequency unit (MHz / GHz / Hz)
        data_unit (str): Unit for pixel scale (e.g. K)
        basemap (str):   Map used as a basis for spatial structure in PCA fit.

        Notes
        -----
        Any GSM needs to supply a generate() function
        """
        self.name = name
        if h5py.is_hdf5(filepath):
            self.h5 = h5py.File(filepath, "r")
        elif is_fits(filepath):
            self.fits = fits.open(filepath, "readonly")
        else:
            raise RuntimeError(f"Cannot read HDF5/FITS file {filepath}")
        self.basemap = basemap
        self.freq_unit = freq_unit
        self.data_unit = data_unit

        self.generated_map_data = None
        self.generated_map_freqs = None

    def generate(self, freqs, reset_cache = False):
        """ Generate a global sky model at a given frequency or frequencies

        Parameters
        ----------
        freqs: float or np.array
            Frequency for which to return GSM model

        reset_cache: bool
            Remove the existing caches from memory for this model

        Returns
        -------
        gsm: np.array
            Global sky model in healpix format, with NSIDE=1024. Output map
            is in galactic coordinates, ring format.

        """

        if reset_cache:
            del self.generated_map_data
            del self.generated_map_freqs
            self.generated_map_data = None
            self.generated_map_freqs = None


        # Convert value into the model's set frequency units
        freqs = np.array([freqs]).ravel() * units.Unit(self.freq_unit)

        if self.generated_map_freqs is None:
            exisiting_map_freqs = 0
            freqs_to_generate = freqs.copy()
        else:
            exisiting_map_freqs = self.generated_map_freqs.size
            freqs_to_generate = []
            for freq in freqs:
                if freq.value in self.generated_map_freqs.value:
                    continue
                freqs_to_generate.append(freq.value)
            freqs_to_generate = np.array(freqs_to_generate).ravel() * units.Unit(self.freq_unit)

        if len(freqs_to_generate):
            freqs_mhz = freqs_to_generate.to('MHz').value
            map_out = self._generate(freqs_mhz).squeeze()

            if exisiting_map_freqs:
                self.generated_map_data = np.resize(self.generated_map_data, (exisiting_map_freqs + len(freqs_to_generate), self.generated_map_data.shape[-1]))
                self.generated_map_freqs = np.resize(self.generated_map_freqs, (exisiting_map_freqs + len(freqs_to_generate)))
            else:
                self.generated_map_data = np.zeros_like(map_out)
                self.generated_map_freqs = np.zeros_like(freqs_to_generate)

            if self.generated_map_data.ndim == 2:
                self.generated_map_data[exisiting_map_freqs:, :] = map_out
                self.generated_map_freqs[exisiting_map_freqs:] = freqs_to_generate
            else:
                self.generated_map_data = map_out.copy()
                self.generated_map_freqs = freqs_to_generate.copy()

        if freqs.size > 1 or self.generated_map_freqs.size > 1:
            map_out = self.generated_map_data[[np.argwhere(self.generated_map_freqs == freq).item() for freq in freqs], :].squeeze()
        else:
            map_out = self.generated_map_data

        return map_out

    def _generate(self, freq_mhz):
        raise NotImplementedError

    def view(self, idx=0, logged=False, show=False):
        """ View generated map using healpy's mollweide projection.

        Parameters
        ----------
        idx: int
            index of map to view. Only required if you generated maps at
            multiple frequencies.
        logged: bool
            Take the log of the data before plotting. Defaults to False.

        """

        if self.generated_map_data is None:
            raise RuntimeError("No GSM map has been generated yet. Run generate() first.")

        if self.generated_map_data.ndim == 2:
            gmap = self.generated_map_data[idx]
            freq = self.generated_map_freqs[idx]
        else:
            gmap = self.generated_map_data
            freq = self.generated_map_freqs

        if logged:
            gmap = np.log2(gmap)

        hp.mollview(gmap, coord='G', title='%s %s, %s' % (self.name, str(freq), self.basemap))

        if show:
            show_plt()
    
    def get_sky_temperature(self, coords, freqs=None, include_cmb=True):
        """ Get sky temperature at given coordinates.

        Returns sky temperature at a given SkyCoord (e.g. Ra/Dec or galactic l/b).
        Useful for estimating sky contribution to system temperature.

        Parameters
        ----------
        coords (astropy.coordinates.SkyCoord): 
            Sky Coordinates to compute temperature for.
        freqs (None, float, or np.array):
            frequencies to evaluate. If not set, will default to those supplied 
            when generate() was called.
        include_cmb (bool): 
            Include a 2.725 K contribution from the CMB (default True).
        """
        
        T_cmb = 2.725 if include_cmb else 0
        
        if freqs is not None:
            map_data = self.generate(freqs)

        pix = hp.ang2pix(self.nside, coords.galactic.l.deg, coords.galactic.b.deg, lonlat=True)
        if map_data.ndim == 2:
            return map_data[:, pix] + T_cmb
        else:
            return map_data[pix] + T_cmb


    def write_fits(self, filename):
        """ Write out map data as FITS file.

        Parameters
        ----------
        filename: str
            file name for output FITS file
        """
        hp.write_map(filename, self.generated_map_data, column_units=self.data_unit)