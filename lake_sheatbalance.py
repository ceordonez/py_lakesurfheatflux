import numpy as np


## FIX Assuming all inputs are pandas series!!
def atm_emmissivity(temp, cc, ew):
    """Calculate atmospere emmisivity

    Parameters
    ----------
    temp : array
        Air temperature at 2 meter in degC
    cc: array
        Cloud cover [-]
    ew: array
         Water vapor pressure of the air at the screen-level in hPa

    Returns
    -------
    emi : array
        Atmosphere emmisivity
    """
    A1 = 0.98
    A2 = 0.17
    A3 = 1.24
    atemp_k = temp + 273.15
    emi = A1 * (1 + A2 * cc**2) * A3 * (ew / atemp_k) ** (1 / 7)
    return emi


def swrad_in(rad, albedo):
    """Calculate incoming solar radiation absorved by the lake.

    Parameters
    ----------
    rad : array
        Incoming solar radiation in Wm2
    albedo : array
        Water albedo [-]

    Returns
    -------
    hs : array
        Absorved solar radiation in Wm-2
    """

    rad_out = 0*rad.copy()
    rad_out.loc[rad > 5] = rad.loc[rad>5] * albedo.loc[rad>5]
    hs = rad - rad_out
    return hs


def absorved_lw(temp, emi):
    """Calculate absorved longwave radiation.

    Parameters
    ----------
    temp : array
        Air temperature at 2 meter in degC
    emi : array
        Atmosphere emmisivity

    Returns
    -------
    ha : array
        Absorved longwave radiation in Wm-2
    """

    A1 = 0.03  # Reflection of IR from water surface
    sigma = 5.67e-8  # [Wm-2K-4]
    atemp_k = temp + 273.15
    ha = (1 - A1) * emi * sigma * atemp_k**4
    return ha


def emmited_lw(wtemp):
    """Calculate emmited longwave radiation.

    Parameters
    ----------
    wtemp : array
        Surface water temperature in degC

    Returns
    -------
    hw = array
        Longwave radiation emmited by the lake in Wm-2.
    """
    A1 = 0.972  # LW water emmissivity
    sigma = 5.67e-8  # [Wm-2K-4]
    wtemp_k = wtemp + 273.15
    hw = -A1 * sigma * wtemp_k**4
    return hw


def sat_vaporpress(temp):
    """Calculate saturated vapor pressure.

    Parameters
    ----------
    temp: array
        Temperature in degC

    Returns
    -------
    ew : array
        Saturated vapor pressure in hPa
    """
    A1 = 6.112
    A2 = 17.62
    A3 = 243.12
    ew = A1 * np.exp((A2 * temp) / (A3 + temp))
    return ew


def transfer_function(wind10, wtemp, temp):
    """Calculate the transfer function.

    Parameters
    ----------
    wind10 : array
        Wind velocity at 10 m in ms-1
    wtemp : array
        Surface water temperature in degC
    temp: array
        Air temperature at 2m in degC

    Returns
    -------
    f1 : array
        Transfer function (REF) in Wm-2hPa-1
    """

    A1 = 4.8
    A2 = 1.98
    A3 = 0.28
    f1 = A1 + A2 * wind10 + A3 * (wtemp - temp)
    return f1


def latent_heat(f1, ew, ea):
    """Calculate latent heat.

    Parameters
    ----------
    f1 : array
        Transfer function in Wm-2hPa-1
    ew : array
        Saturated vapor pressure in hPa
    ea: array
        Actual vapor pressure in hPa

    Returns
    -------
    he : array
        Latent heat in Wm-2
    """

    he = -f1 * (ew - ea)  ##CHECK THIS EQUATION IN THE PAPER!!!!
    return he


def vapor_pressure(temp, rh):
    """Calculate vapor pressure from relative humidity.

    Parameters
    ----------
    temp : array
        Air temperature at 2m in degC
    rh : array
        Relative humidity at 2m in percent

    Return
    ------
    es : array
        Vapor pressure in hPa
    """

    es = sat_vaporpress(temp)
    ew = rh * es/100
    return ew


def psychrometric_constant(airp):
    """Calculate psychrometric constant.

    Parameters
    ----------
    airp : array
        Air pressure in hPa

    Returns
    -------
    psy : array
        psychrometric constant in hPaK-1
    """
    CP = 1005  # Air heat capacity [J Kg'1 K-1] at 20degC
    LV = 2.47e6  # Latent heat of vaporization [JKg-1]
    MV = 0.622  # MV ratio
    psy = CP * airp / (LV * MV)
    return psy


def sensible_heat(psy, f1, wtemp, temp):
    """Calculate sensible heat

    Parameters
    ----------
    psy : array
        psychrometric constant in hPaK-1
    f1 : array
        Transfer function in Wm-2hPa-1
    wtemp : array
        Surface water temperature in degC
    temp : array
        Air temperature at 2m in degC

    Returns
    -------
    hc : array
        Sensible heat flux in Wm-2
    """
    hc = -psy * f1 * (wtemp - temp)
    return hc


def heat_balance(hs, ha, hw, he, hc):
    """Calculate net heat.

    Parameters
    ----------
    hs : array
        Absorved solar radiation in Wm-2.
    ha : array
        Absorved longwave radiation in Wm-2.
    hw : array
        Longwave radiation emmited by the lake in Wm-2.
    he : array
        Latent heat in Wm-2.
    hc : array
        Sensible heat flux in Wm-2.

    Return
    ------
    hnet : array
        Net heat flux in Wm-2.
    """
    hnet = hs + ha + hw + he + hc
    return hnet
