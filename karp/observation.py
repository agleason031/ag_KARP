"""
Quantities that come from a frame's header and the observatory: the heliocentric
velocity correction and the mid-exposure airmass.

Both use the frame's RA/DEC (unless overridden), the observatory latitude/longitude
from the header, the site height (not in the headers; APO = 2788 m), and the middle of
the exposure (DATE-OBS is its start, in the time system given by TIMESYS).
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time


def _target_and_site(header, site_height, ra, dec):
    ra = ra if ra is not None else str(header["RA"])
    dec = dec if dec is not None else str(header["DEC"])
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    site = EarthLocation.from_geodetic(lon=float(header["LONGITUD"]) * u.deg,
                                       lat=float(header["LATITUDE"]) * u.deg,
                                       height=site_height * u.m)
    return target, site


def _mid_exposure(header):
    """(exposure time in s, Time at the middle of the exposure)."""
    scale = str(header.get("TIMESYS", "UTC")).strip().lower()
    exptime = float(header.get("EXPTIME", 0.0))
    start = Time(header["DATE-OBS"], scale=scale if scale in ("tai", "utc") else "utc")
    return exptime, start + exptime / 2 * u.s


def heliocentric_correction(header, site_height=2788.0, ra=None, dec=None):
    """Heliocentric radial-velocity correction (km/s), to be applied as lam * (1 + v/c)."""
    target, site = _target_and_site(header, site_height, ra, dec)
    _, t = _mid_exposure(header)
    corr = target.radial_velocity_correction("heliocentric", obstime=t, location=site)
    return float(corr.to(u.km / u.s).value)


def exposure_info(header, site_height=2788.0, ra=None, dec=None):
    """
    (exposure time s, mid-exposure airmass, mid-exposure MJD). The airmass is computed
    from the coordinates, site and time; the header AIRMASS is for the start of the exposure.
    """
    target, site = _target_and_site(header, site_height, ra, dec)
    exptime, t = _mid_exposure(header)
    alt = target.transform_to(AltAz(obstime=t, location=site)).alt
    return exptime, float(1 / np.sin(alt.radian)), float(t.utc.mjd)
