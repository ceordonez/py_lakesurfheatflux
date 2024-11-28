import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylake

import lake_sheatbalance as lhb

plt.style.use("~/.config/matplotlib/aslo-paper.mplstyle")

BATHFILE = "~/Dropbox/Cesar/PostDoc/Projects/OMP-Daily/Data/Bretaye/Bathymetry/BRE_BATHYMETRY.csv"
TEMPFILE = "~/Dropbox/Cesar/PostDoc/Projects/OMP-Daily/Data/Bretaye/Mooring/Mooring_LacBretaye_M1_10min.csv"
DEPTHFILE = "~/Dropbox/Cesar/PostDoc/Projects/OMP-Daily/Data/Bretaye/Mooring/Depth_Mooring_LacBretaye_M1_10min.csv"


def main():
    bath_data = pd.read_csv(
        BATHFILE, skiprows=5, usecols=[0, 2], names=["Depth_m", "Area_m2"]
    )
    temp_data = pd.read_csv(TEMPFILE, parse_dates=[0], index_col=[0], na_values="NaN")
    depth_data = pd.read_csv(DEPTHFILE, parse_dates=[0], index_col=[0])
    temp_data = temp_data.resample("h").mean().interpolate(limit_direction="both")
    depth_data = depth_data.resample("h").mean().interpolate(limit_direction="both")

    meteo_data = pd.read_csv(
        "Data.csv", parse_dates=[0], index_col=[0], dtype=float, na_values="NAN"
    )
    mindate = max([meteo_data.index.min(), temp_data.index.min()])
    maxdate = min([meteo_data.index.max(), temp_data.index.max()])

    meteo_data = meteo_data[mindate:maxdate]
    temp_data = temp_data[mindate:maxdate]
    depth_data = depth_data[mindate:maxdate]

    meteo_data["ew"] = lhb.sat_vaporpress(meteo_data["AirTemp_degC"])
    meteo_data["emi"] = lhb.atm_emmissivity(
        meteo_data["AirTemp_degC"], meteo_data["Clouds_Tot"], meteo_data.ew
    )
    meteo_data["f1"] = lhb.transfer_function(
        meteo_data.WS10_ms, temp_data["0.45"], meteo_data["AirTemp_degC"]
    )
    meteo_data["ea"] = lhb.vapor_pressure(
        meteo_data["AirTemp_degC"], meteo_data["RH_%"]
    )
    meteo_data["psy"] = lhb.psychrometric_constant(meteo_data["AirPress_hPa"])
    he = lhb.latent_heat(meteo_data["f1"], meteo_data["ew"], meteo_data["ea"])
    hc = lhb.sensible_heat(
        meteo_data["psy"],
        meteo_data["f1"],
        temp_data["0.45"],
        meteo_data["AirTemp_degC"],
    )
    ha = lhb.absorved_lw(meteo_data["AirTemp_degC"], meteo_data["emi"])
    hs = lhb.swrad_in(meteo_data["Rad_Wm2"], meteo_data["Albedo"])
    hw = lhb.emmited_lw(temp_data["0.45"])
    hnet = lhb.heat_balance(hs, ha, hw, he, hc)

    heatc_data = heat_content(temp_data, depth_data, bath_data)
    dhdt = heatc_data.diff() / heatc_data.index.to_series().diff().dt.total_seconds()
    dhw = -(dhdt - hnet) * bath_data.Area_m2.max()

    wtemp = pd.Series(
        np.linspace(2, temp_data["0"].max(), len(dhw)),
        name="flow_temp",
        index=dhw.index,
    )
    wheat = pd.Series(
        pylake.dens0(wtemp) * 4186 * wtemp,
        name="flow_heat",
        index=wtemp.index,
    )
    # rho*cp*Q*T = dhw 
    # kg/m3 * J/kgC m3/s * C = J/s
    # -> Q = dhw/rho*cp*T 
    # (J/s) / (kg/m3 * J/kgC * C)
    # -> Q m3/s
    flow = pd.Series(dhw.values / wheat.values, name="flow", index=dhw.index)

    dhdt = dhdt.resample("d").mean()
    hnet = hnet.resample("d").mean()  # *bath_data.iloc[0].Area_m2
    dhw = dhw.resample("d").mean()
    # heatc_data = heatc_data.resample("d").mean()
    flow = flow.resample("d").mean()*60*60

    fig, axs = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
    dhdt.plot(
        ax=axs[0],
        label=r"$\frac{1}{A_\text{surf}}\frac{\partial \text{HC}}{\partial t}$",
    )
    hnet.plot(ax=axs[0], color="r", label="Heat balance")
    axs[0].legend()
    axs[0].set_ylabel(r"H$_\text{net}$ (\si{\watt\per\square\meter})")
    dhw.plot(ax=axs[1])
    axs[1].axhline(0)
    axs[1].set_ylabel(r"Missing heat (\si{\watt})")
    flow.plot(ax=axs[2])
    axs[2].set_ylabel(r"Flow (\si{\cubic\meter\per\hour})")
    axs[2].axhline(0)
    fig.savefig('Flow_heatbudget.png', format='png')
    plt.show()


def heat_content(temp_data, depth_data, bath_data):

    heatc_data = []
    temp_data["0"] = temp_data["0.45"]
    depth_data["0"] = 0
    depth_data.loc[temp_data["0.45"].isna(), "0"] = np.nan
    for i, _ in enumerate(temp_data.index):
        data_p = pd.DataFrame(
            {"Depth_m": depth_data.iloc[i].values, "Temp_C": temp_data.iloc[i].values}
        )
        if data_p.Temp_C.isna().all():
            heatc_data.append(np.nan)
        else:
            hct = pylake.heat_content(
                data_p.Temp_C.values,
                bath_data.Area_m2.values,
                bath_data.Depth_m.values,
                data_p.Depth_m.values,
            )
            heatc_data.append(hct.values[0])

    heatc_data = pd.Series(heatc_data, name="HC_Jm2", index=temp_data.index)
    return heatc_data


def interp_temp(data_p, date, newz):

    # newz = pd.Series(np.arange(0, data_p.Depth_m.max()+1, 1), name='Depth_m')
    newdata_p = pd.merge(data_p, newz, on="Depth_m", how="outer")
    newdata_p = newdata_p.set_index("Depth_m")
    newdata_p = newdata_p.interpolate(
        method="slinear", limit_direction="both", limit=2
    ).bfill()
    newdata_p.reset_index(inplace=True)
    newdata_p["Datetime"] = date
    newdata_p.set_index("Datetime", inplace=True)
    return newdata_p


if __name__ == "__main__":
    main()
