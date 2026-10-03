
import io
import zipfile
import time
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# Optional analysis/plotting packages
import matplotlib.pyplot as plt
import numpy as np

try:
    import metpy
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    # Import through metpy.calc so the app is less sensitive to changes
    # in the list of names exposed by different MetPy versions.
    parcel_profile = mpcalc.parcel_profile
    cape_cin = mpcalc.cape_cin
    lcl = mpcalc.lcl
    lifted_index = mpcalc.lifted_index
    k_index = mpcalc.k_index
    precipitable_water = mpcalc.precipitable_water
    wind_components = mpcalc.wind_components
    wind_speed = mpcalc.wind_speed
    pressure_to_height_std = mpcalc.pressure_to_height_std
    METPY_AVAILABLE = True
    METPY_VERSION = metpy.__version__
    METPY_IMPORT_ERROR = ""
except Exception as exc:
    METPY_AVAILABLE = False
    METPY_VERSION = ""
    METPY_IMPORT_ERROR = repr(exc)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Radiosonde Analytics System",
    page_icon="🌐",
    layout="wide",
)

BASE_URL = "https://weather.uwyo.edu/wsgi/sounding"

# Common WMO radiosonde stations in/near India.
# You can add more stations to this dictionary.
STATIONS = {
    "Trivandrum, India": "43371",
    "Kochi/Cochin, India": "43353",
    "Chennai, India": "43279",
    "Bengaluru, India": "43295",
    "Hyderabad, India": "43128",
    "Mumbai/Santacruz, India": "43003",
    "Pune, India": "43063",
    "Ahmedabad, India": "42647",
    "New Delhi, India": "42182",
    "Patna, India": "42369",
    "Kolkata, India": "42809",
    "Bhubaneswar, India": "42909",
    "Nagpur, India": "42867",
    "Guwahati, India": "42410",
    "Srinagar, India": "42027",
    "Jaipur, India": "42348",
    "Thiruvananthapuram, India": "43371",
}


# ============================================================
# UWYO DOWNLOAD
# ============================================================

@st.cache_data(show_spinner=False)
def download_sounding(station, sounding_date, hour):
    """
    Download one sounding from the current UWyo WSGI server.

    requests' params= argument is deliberately used so that
    datetime and type values are URL encoded correctly.
    """
    dt = (
        pd.Timestamp(sounding_date).strftime("%Y-%m-%d")
        + f" {hour}:00:00"
    )

    params = {
        "datetime": dt,
        "id": str(station),
        "src": "FM35",
        "type": "TEXT:LIST",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=45,
        headers={"User-Agent": "UWyo-Radiosonde-Portal/1.0"},
    )

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    text = response.text

    if not text.strip():
        return None, "Empty response"

    return text, None


def parse_sounding(text, station, sounding_date, hour):
    """Parse UWyo TEXT:LIST sounding into a DataFrame."""
    lines = text.splitlines()

    rows = []
    reading = False

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("PRES"):
            reading = True
            continue

        if not reading or not line:
            continue

        if "----" in line:
            continue

        parts = line.split()

        # Standard UWyo sounding table has 11 values.
        if len(parts) < 11:
            continue

        try:
            values = [float(x) for x in parts[:11]]
        except ValueError:
            continue

        # Basic sanity check: pressure must be positive.
        if values[0] <= 0:
            continue

        rows.append(values)

    if not rows:
        return None

    columns = [
        "PRES_hPa",
        "HGHT_m",
        "TEMP_C",
        "DWPT_C",
        "RELH_percent",
        "MIXR_gkg",
        "DRCT_deg",
        "SKNT_ms",
        "THTA_K",
        "THTE_K",
        "THTV_K",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df.insert(0, "station", str(station))
    df.insert(
        1,
        "datetime_utc",
        f"{pd.Timestamp(sounding_date).strftime('%Y-%m-%d')} {hour}:00:00",
    )

    return df


def get_date_list(start_date, end_date):
    return [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]


# ============================================================
# ATMOSPHERIC CALCULATIONS
# ============================================================

def _prepare_metpy(df):
    """Return a clean, pressure-descending sounding with MetPy units."""
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["PRES_hPa", "TEMP_C", "DWPT_C"])
    work = work[work["PRES_hPa"] > 0]
    work = work.sort_values("PRES_hPa", ascending=False)
    work = work.drop_duplicates(subset="PRES_hPa")

    p = work["PRES_hPa"].to_numpy() * units.hPa
    t = work["TEMP_C"].to_numpy() * units.degC
    td = work["DWPT_C"].to_numpy() * units.degC

    return work, p, t, td


def calculate_freezing_level(pressure, temperature):
    """
    Find the first 0°C crossing by linear interpolation.
    Works across MetPy versions without mpcalc.freezing_level.
    """
    try:
        p = np.asarray(pressure.to("hPa").magnitude, dtype=float)
        t = np.asarray(temperature.to("degC").magnitude, dtype=float)

        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]

        if len(p) < 2:
            return None, None

        order = np.argsort(p)[::-1]
        p, t = p[order], t[order]

        for i in range(len(t) - 1):
            if (t[i] >= 0 and t[i + 1] <= 0) or (t[i] <= 0 and t[i + 1] >= 0):
                if np.isclose(t[i + 1], t[i]):
                    pf = p[i]
                else:
                    pf = p[i] + (0.0 - t[i]) * (p[i + 1] - p[i]) / (t[i + 1] - t[i])
                return pf * units.hPa, 0.0 * units.degC

        return None, None
    except Exception:
        return None, None


def calculate_wind_shear(df, bottom_hpa=1000, top_hpa=600):
    """Calculate vector wind shear between two pressure levels."""
    if not METPY_AVAILABLE:
        return {}

    try:
        work = df.copy().dropna(
            subset=["PRES_hPa", "DRCT_deg", "SKNT_ms"]
        )
        work = work.sort_values("PRES_hPa", ascending=False)

        p = work["PRES_hPa"].to_numpy() * units.hPa
        direction = work["DRCT_deg"].to_numpy() * units.deg
        speed = work["SKNT_ms"].to_numpy() * units("m/s")

        u, v = wind_components(speed, direction)

        def interp_component(target_hpa, component):
            target = float(target_hpa)
            order = np.argsort(work["PRES_hPa"].to_numpy())
            pressures = work["PRES_hPa"].to_numpy()[order]
            values = component.magnitude[order]
            if target < pressures.min() or target > pressures.max():
                return None
            return np.interp(target, pressures, values) * component.units

        u_bottom = interp_component(bottom_hpa, u)
        v_bottom = interp_component(bottom_hpa, v)
        u_top = interp_component(top_hpa, u)
        v_top = interp_component(top_hpa, v)

        if any(
            x is None for x in (u_bottom, v_bottom, u_top, v_top)
        ):
            return {}

        du = u_top - u_bottom
        dv = v_top - v_bottom
        shear = wind_speed(du, dv)

        return {
            f"{bottom_hpa:.0f}-{top_hpa:.0f} hPa wind shear (m/s)": float(
                shear.to("m/s").magnitude
            ),
            f"{bottom_hpa:.0f}-{top_hpa:.0f} hPa Δu (m/s)": float(
                du.to("m/s").magnitude
            ),
            f"{bottom_hpa:.0f}-{top_hpa:.0f} hPa Δv (m/s)": float(
                dv.to("m/s").magnitude
            ),
        }
    except Exception:
        return {}


def calculate_summary(df):
    """Calculate thermodynamic and kinematic sounding diagnostics."""
    result = {}

    surface = df.sort_values("PRES_hPa", ascending=False).iloc[0]

    result["Surface pressure (hPa)"] = float(surface["PRES_hPa"])
    result["Surface temperature (°C)"] = float(surface["TEMP_C"])
    result["Surface dew point (°C)"] = float(surface["DWPT_C"])
    result["Surface RH (%)"] = float(surface["RELH_percent"])
    result["Maximum height (m)"] = float(df["HGHT_m"].max())

    if not METPY_AVAILABLE:
        return result

    try:
        work, p, t, td = _prepare_metpy(df)

        if len(work) < 5:
            return result

        # Surface-based parcel.
        parcel = parcel_profile(p, t[0], td[0])

        cape, cin = cape_cin(
            p,
            t,
            td,
        )

        lcl_p, lcl_t = lcl(p[0], t[0], td[0])

        result["SBCAPE (J/kg)"] = float(cape.to("J/kg").magnitude)
        result["SBCIN (J/kg)"] = float(cin.to("J/kg").magnitude)
        result["LCL pressure (hPa)"] = float(lcl_p.to("hPa").magnitude)
        result["LCL temperature (°C)"] = float(
            lcl_t.to("degC").magnitude
        )

        # Lifted Index at 500 hPa.
        try:
            li = lifted_index(
                p,
                t,
                parcel,
            )
            result["Lifted Index (°C)"] = float(
                np.atleast_1d(li.magnitude)[0]
            )
        except Exception:
            pass

        # K-index.
        try:
            ki = k_index(p, t, td)
            result["K-Index (°C)"] = float(
                np.atleast_1d(ki.magnitude)[0]
            )
        except Exception:
            pass

        # Precipitable water through the available sounding.
        try:
            pw = precipitable_water(p, td)
            result["Precipitable Water (mm)"] = float(
                pw.to("mm").magnitude
            )
        except Exception:
            pass

        # Freezing level: first 0°C crossing.
        try:
            frz_p, frz_t = calculate_freezing_level(p, t)

            if frz_p is not None:
                result["Freezing level pressure (hPa)"] = float(
                    frz_p.to("hPa").magnitude
                )
                result["Freezing level temperature (°C)"] = float(
                    frz_t.to("degC").magnitude
                )

                frz_h = pressure_to_height_std(frz_p)
                result["Freezing level (m, std atm)"] = float(
                    frz_h.to("m").magnitude
                )

                # Prefer actual sounding height when available.
                z = work["HGHT_m"].to_numpy(dtype=float)
                pp = work["PRES_hPa"].to_numpy(dtype=float)
                tt = work["TEMP_C"].to_numpy(dtype=float)

                valid_z = np.isfinite(pp) & np.isfinite(tt) & np.isfinite(z)
                pp, tt, z = pp[valid_z], tt[valid_z], z[valid_z]

                if len(pp) >= 2:
                    order_z = np.argsort(pp)[::-1]
                    pp, tt, z = pp[order_z], tt[order_z], z[order_z]

                    for j in range(len(tt) - 1):
                        if ((tt[j] >= 0 and tt[j + 1] <= 0) or
                                (tt[j] <= 0 and tt[j + 1] >= 0)):
                            if not np.isclose(tt[j + 1], tt[j]):
                                zf = z[j] + (0 - tt[j]) * (
                                    z[j + 1] - z[j]
                                ) / (tt[j + 1] - tt[j])
                            else:
                                zf = z[j]
                            result["Freezing level (m, sounding height)"] = float(zf)
                            break
        except Exception:
            pass

        # Wind shear at standard pressure layers. These are pressure-layer
        # diagnostics, useful for comparing soundings consistently.
        result.update(calculate_wind_shear(df, 1000, 900))
        result.update(calculate_wind_shear(df, 1000, 700))
        result.update(calculate_wind_shear(df, 1000, 500))

    except Exception:
        # Keep the downloader usable even when a particular sounding
        # contains insufficient/irregular levels for one diagnostic.
        pass

    return result


# ============================================================
# PLOTS
# ============================================================

def temperature_profile(df):
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.plot(
        df["TEMP_C"],
        df["HGHT_m"],
        label="Temperature",
    )
    ax.plot(
        df["DWPT_C"],
        df["HGHT_m"],
        label="Dew point",
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Temperature / Dew-Point Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def rh_profile(df):
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.plot(
        df["RELH_percent"],
        df["HGHT_m"],
    )

    ax.set_xlabel("Relative Humidity (%)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Relative Humidity Profile")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def wind_profile(df):
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.plot(
        df["SKNT_ms"],
        df["HGHT_m"],
    )

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Wind-Speed Profile")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def skewt_plot(df):
    """Create a Skew-T/log-P diagram with parcel, CAPE/CIN and wind barbs."""
    work = df.copy()
    work = work.dropna(
        subset=["PRES_hPa", "TEMP_C", "DWPT_C"]
    )
    work = work.sort_values("PRES_hPa", ascending=False)
    work = work.drop_duplicates(subset="PRES_hPa")

    if not METPY_AVAILABLE or len(work) < 5:
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.plot(work["TEMP_C"], work["PRES_hPa"], label="Temperature")
        ax.plot(work["DWPT_C"], work["PRES_hPa"], label="Dew point")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Pressure (hPa)")
        ax.set_title("Radiosonde Thermodynamic Profile")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    p = work["PRES_hPa"].to_numpy() * units.hPa
    t = work["TEMP_C"].to_numpy() * units.degC
    td = work["DWPT_C"].to_numpy() * units.degC

    fig = plt.figure(figsize=(9, 10))
    skew = SkewT(fig, rotation=45)

    skew.plot(p, t, "r", linewidth=2, label="Temperature")
    skew.plot(p, td, "g", linewidth=2, label="Dew point")

    # Parcel ascent.
    try:
        parcel = parcel_profile(p, t[0], td[0])
        skew.plot(p, parcel, "k", linewidth=1.5, label="Surface parcel")
        skew.shade_cape(p, t, parcel)
        skew.shade_cin(p, t, parcel)
    except Exception:
        pass

    # Wind barbs, using available wind observations.
    try:
        wind_df = work.dropna(
            subset=["DRCT_deg", "SKNT_ms"]
        )
        if not wind_df.empty:
            wp = wind_df["PRES_hPa"].to_numpy() * units.hPa
            wd = wind_df["DRCT_deg"].to_numpy() * units.deg
            ws = wind_df["SKNT_ms"].to_numpy() * units("m/s")
            u, v = wind_components(ws, wd)

            # Reduce dense barbs while retaining vertical structure.
            step = max(1, len(wind_df) // 25)
            skew.plot_barbs(
                wp[::step],
                u[::step],
                v[::step],
            )
    except Exception:
        pass

    skew.ax.set_ylim(
        max(100, float(np.nanmax(p.magnitude))),
        max(10, float(np.nanmin(p.magnitude))),
    )
    skew.ax.set_xlim(
        np.nanmin(t.magnitude) - 10,
        np.nanmax(t.magnitude) + 15,
    )
    skew.ax.set_title("UWyo Radiosonde Skew-T / Log-P Diagram")
    skew.ax.grid(True, alpha=0.3)
    skew.ax.legend(loc="best")

    fig.tight_layout()
    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌐 Intelligent Radiosonde Analytics System")
st.markdown(
    """
    <div style="
        text-align: center;
        color: red;
        font-weight: bold;
        font-size: 16px;
    ">
        Developed by: Dr. Sumith Satheendran S. PhD
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(
    "Data & Credits | "
    "Original Project Implementation,Analysis,Coding and Visualizations © Sumith Satheendran S. Third-party datasets remain subject to their respective licenses and attribution requirements."
)
st.markdown(
    """
Download and analyse **University of Wyoming radiosonde soundings**.
Select a station, date range and observation time, then export the
data as CSV or ZIP.
"""
)

if METPY_AVAILABLE:
    st.success(
        f"✅ MetPy {METPY_VERSION} is enabled — "
        "CAPE, CIN, Lifted Index, K-Index, PWAT, freezing level, "
        "wind shear and Skew-T analysis are available."
    )
else:
    st.error(
        "MetPy could not be loaded, so advanced thermodynamic "
        "calculations are disabled."
    )
    with st.expander("Show MetPy import error"):
        st.code(METPY_IMPORT_ERROR or "Unknown MetPy import error")
        st.markdown(
            """
Run these commands in the **same Python environment** used to start this Streamlit app:

`python -m pip install --upgrade metpy`

`python -c "import metpy; print(metpy.__version__)"`

`python -m streamlit run app.py`
"""
        )

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Data Selection")

    station_name = st.selectbox(
        "Station",
        list(STATIONS.keys()),
    )

    station = STATIONS[station_name]

    custom_station = st.text_input(
        "Or enter WMO station ID",
        value="",
        help="Enter a 5-digit WMO ID to override the selected station.",
    ).strip()

    if custom_station:
        station = custom_station
        station_display = f"Custom station: {station}"
    else:
        station_display = f"{station_name} ({station})"

    start_date = st.date_input(
        "Start date",
        value=date.today() - timedelta(days=7),
    )

    end_date = st.date_input(
        "End date",
        value=date.today(),
    )

    time_selection = st.radio(
        "Observation time",
        ["00 UTC", "12 UTC", "Both"],
    )

    st.caption(
        f"Selected station: {station_display}"
    )

    download_clicked = st.button(
        "⬇️ Download Soundings",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "all_data" not in st.session_state:
    st.session_state.all_data = []

if "combined" not in st.session_state:
    st.session_state.combined = None

if "download_log" not in st.session_state:
    st.session_state.download_log = []


# ============================================================
# VALIDATION
# ============================================================

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

if not str(station).isdigit() or len(str(station)) != 5:
    st.error("WMO station ID must contain exactly 5 digits.")
    st.stop()


# ============================================================
# DOWNLOAD
# ============================================================

if download_clicked:

    if time_selection == "00 UTC":
        hours = ["00"]
    elif time_selection == "12 UTC":
        hours = ["12"]
    else:
        hours = ["00", "12"]

    dates = get_date_list(start_date, end_date)
    total = len(dates) * len(hours)

    progress = st.progress(0)
    status = st.empty()

    all_data = []
    download_log = []

    count = 0

    for sounding_date in dates:

        for hour in hours:

            count += 1

            status.info(
                f"Downloading {sounding_date} {hour} UTC "
                f"({count}/{total})"
            )

            try:
                text, error = download_sounding(
                    station,
                    sounding_date,
                    hour,
                )

                if text:
                    df = parse_sounding(
                        text,
                        station,
                        sounding_date,
                        hour,
                    )

                    if df is not None:
                        all_data.append(df)

                        download_log.append({
                            "date": str(sounding_date),
                            "time": f"{hour} UTC",
                            "status": "Downloaded",
                            "levels": len(df),
                        })
                    else:
                        download_log.append({
                            "date": str(sounding_date),
                            "time": f"{hour} UTC",
                            "status": "Could not parse",
                            "levels": 0,
                        })
                else:
                    download_log.append({
                        "date": str(sounding_date),
                        "time": f"{hour} UTC",
                        "status": error or "No data",
                        "levels": 0,
                    })

            except Exception as exc:
                download_log.append({
                    "date": str(sounding_date),
                    "time": f"{hour} UTC",
                    "status": f"Error: {exc}",
                    "levels": 0,
                })

            progress.progress(count / total)
            time.sleep(0.15)

    st.session_state.all_data = all_data
    st.session_state.download_log = download_log

    if all_data:
        st.session_state.combined = pd.concat(
            all_data,
            ignore_index=True,
        )
        status.success(
            f"Completed: {len(all_data)} soundings downloaded."
        )
    else:
        st.session_state.combined = None
        status.error("No soundings were found.")


# ============================================================
# RESULTS
# ============================================================

combined = st.session_state.combined
all_data = st.session_state.all_data

if combined is not None and not combined.empty:

    st.success(
        f"Station {station}: "
        f"{len(all_data)} soundings, "
        f"{len(combined):,} atmospheric levels."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Summary",
            "📈 Profiles",
            "🌡️ Sounding Analysis",
            "📋 Data",
            "💾 Downloads",
        ]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    with tab1:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Station", station)

        with c2:
            st.metric("Soundings", len(all_data))

        with c3:
            st.metric(
                "Atmospheric levels",
                f"{len(combined):,}",
            )

        with c4:
            st.metric(
                "Maximum height",
                f"{combined['HGHT_m'].max():,.0f} m",
            )

        st.subheader("Download Status")

        st.dataframe(
            pd.DataFrame(st.session_state.download_log),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Available Sounding Diagnostics")

        if METPY_AVAILABLE:
            diagnostic_rows = []
            for df in all_data:
                summary = calculate_summary(df)
                row = {
                    "datetime_utc": df["datetime_utc"].iloc[0],
                    "CAPE (J/kg)": summary.get("SBCAPE (J/kg)", np.nan),
                    "CIN (J/kg)": summary.get("SBCIN (J/kg)", np.nan),
                    "Lifted Index (°C)": summary.get("Lifted Index (°C)", np.nan),
                    "K-Index (°C)": summary.get("K-Index (°C)", np.nan),
                    "PWAT (mm)": summary.get("Precipitable Water (mm)", np.nan),
                    "Freezing level (m)": summary.get("Freezing level (m, std atm)", np.nan),
                    "0-1 km* shear (m/s)": summary.get("1000-900 hPa wind shear (m/s)", np.nan),
                    "0-3 km* shear (m/s)": summary.get("1000-700 hPa wind shear (m/s)", np.nan),
                    "0-5 km* shear (m/s)": summary.get("1000-500 hPa wind shear (m/s)", np.nan),
                }
                diagnostic_rows.append(row)

            st.dataframe(
                pd.DataFrame(diagnostic_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "*Wind-shear rows use pressure-layer approximations (1000–900, "
                "1000–700 and 1000–500 hPa), not fixed geometric heights."
            )
        else:
            st.info("Install MetPy to generate the diagnostic table.")

    # ========================================================
    # PROFILES
    # ========================================================

    with tab2:

        sounding_options = []

        for df in all_data:
            sounding_options.append(
                df["datetime_utc"].iloc[0]
            )

        selected_datetime = st.selectbox(
            "Select sounding",
            sounding_options,
        )

        selected_df = next(
            df for df in all_data
            if df["datetime_utc"].iloc[0] == selected_datetime
        )

        p1, p2 = st.columns(2)

        with p1:
            st.pyplot(
                temperature_profile(selected_df),
                clear_figure=True,
            )

        with p2:
            st.pyplot(
                rh_profile(selected_df),
                clear_figure=True,
            )

        st.pyplot(
            wind_profile(selected_df),
            clear_figure=True,
        )

    # ========================================================
    # SOUNDING ANALYSIS
    # ========================================================

    with tab3:

        selected_datetime = st.selectbox(
            "Sounding for analysis",
            [
                df["datetime_utc"].iloc[0]
                for df in all_data
            ],
            key="analysis_datetime",
        )

        selected_df = next(
            df for df in all_data
            if df["datetime_utc"].iloc[0] == selected_datetime
        )

        st.subheader("Thermodynamic Summary")

        summary = calculate_summary(selected_df)

        cols = st.columns(4)

        for i, (key, value) in enumerate(summary.items()):
            with cols[i % 4]:
                if isinstance(value, (float, np.floating)):
                    st.metric(key, f"{value:.2f}")
                else:
                    st.metric(key, value)

        st.subheader("Pressure–Temperature Profile")

        st.pyplot(
            skewt_plot(selected_df),
            clear_figure=True,
        )

        st.subheader("Wind Shear")

        if METPY_AVAILABLE:
            shear = calculate_wind_shear(selected_df, 1000, 900)
            shear.update(calculate_wind_shear(selected_df, 1000, 700))
            shear.update(calculate_wind_shear(selected_df, 1000, 500))

            shear_cols = st.columns(3)
            for i, (key, value) in enumerate(shear.items()):
                if "wind shear" in key:
                    with shear_cols[i % 3]:
                        st.metric(key, f"{value:.2f} m/s")

            st.caption(
                "Wind shear is reported for standard pressure layers. "
                "The 1000–900, 1000–700 and 1000–500 hPa layers are "
                "approximations to commonly used low/mid-level shear layers."
            )
        else:
            st.info("Install MetPy to calculate wind shear.")

    # ========================================================
    # DATA TABLE
    # ========================================================

    with tab4:

        st.subheader("Combined Dataset")

        st.dataframe(
            combined,
            use_container_width=True,
            height=550,
        )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    with tab5:

        csv_bytes = combined.to_csv(
            index=False
        ).encode("utf-8")

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as z:

            for df in all_data:

                dt = pd.to_datetime(
                    df["datetime_utc"].iloc[0]
                )

                filename = (
                    f"{station}_"
                    f"{dt.strftime('%Y%m%d_%H')}Z.csv"
                )

                z.writestr(
                    filename,
                    df.to_csv(index=False),
                )

        st.download_button(
            "📄 Download Combined CSV",
            data=csv_bytes,
            file_name=(
                f"UWyo_{station}_"
                f"{start_date}_{end_date}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "📦 Download Individual CSVs (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=(
                f"UWyo_{station}_"
                f"{start_date}_{end_date}.zip"
            ),
            mime="application/zip",
            use_container_width=True,
        )

        st.download_button(
            "📋 Download Download Log",
            data=pd.DataFrame(
                st.session_state.download_log
            ).to_csv(index=False),
            file_name=f"UWyo_{station}_download_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Radiosonde Analytics System | "
    "Data source: University of Wyoming Upper-Air Soundings"
)
