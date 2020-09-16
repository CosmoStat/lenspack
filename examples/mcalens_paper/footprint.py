import numpy as np

COSMOS_VERTICES = [(149.508, 2.880),
                   (149.767, 2.836),
                   (149.780, 2.887),
                   (150.040, 2.842),
                   (150.051, 2.893),
                   (150.363, 2.840),
                   (150.376, 2.890),
                   (150.746, 2.826),
                   (150.737, 2.774),
                   (150.790, 2.765),
                   (150.734, 2.449),
                   (150.787, 2.441),
                   (150.730, 2.125),
                   (150.785, 2.118),
                   (150.758, 2.013),
                   (150.768, 2.010),
                   (150.747, 1.910),
                   (150.799, 1.897),
                   (150.740, 1.580),
                   (150.481, 1.625),
                   (150.466, 1.572),
                   (150.211, 1.619),
                   (150.196, 1.567),
                   (149.887, 1.621),
                   (149.872, 1.571),
                   (149.617, 1.615),
                   (149.602, 1.566),
                   (149.493, 1.584),
                   (149.504, 1.637),
                   (149.450, 1.646),
                   (149.488, 1.855),
                   (149.433, 1.862),
                   (149.491, 2.178),
                   (149.436, 2.186),
                   (149.484, 2.445),
                   (149.431, 2.455),
                   (149.508, 2.880)]


def draw_footprint(ax, c='w', lw=1, **kwargs):
    """Draw the COSMOS field footprint on a plot.

    Parameters
    ----------
    ax : matplotlib.axes
        Space to draw on.
    c : str
        Color of the line. Default is white.
    lw : float
        Width of the line. Default is 1.
    """
    ra, dec = np.array(COSMOS_VERTICES).T
    ax.plot(ra, dec, c=c, lw=lw, **kwargs)


def draw_massey_outline(ax, c='w', lw=1, **kwargs):
    """Draw the border of the reconstruction field from Massey et al. 2007.

    Parameters
    ----------
    ax : matplotlib.axes
        Space to draw on.
    c : str
        Color of the line. Default is white.
    lw : float
        Width of the line. Default is 1.
    """
    ra_min, ra_max = (149.425, 150.8)
    dec_min, dec_max = (1.57, 2.9)
    ra = [ra_min, ra_max, ra_max, ra_min, ra_min]
    dec = [dec_min, dec_min, dec_max, dec_max, dec_min]
    ax.plot(ra, dec, c=c, lw=lw, **kwargs)
