from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_backend import (
    CONFIDENCE_LEVEL,
    DEFAULT_END_YEAR,
    DEFAULT_PDS_GAP_HOURS,
    DEFAULT_PDS_PERCENTILE,
    DEFAULT_LOCATION,
    DISTRIBUTIONS,
    RAINFALL_VARIABLES,
    RETURN_PERIODS,
    SERIES_METHODS,
    calculate_analysis,
    format_candidate,
    get_open_meteo_config,
    search_best_location,
)

st.set_page_config(
    page_title="IDF Yagis Analizi",
    page_icon="IDF",
    layout="wide",
)

try:
    secret_api_key = str(st.secrets.get("OPEN_METEO_API_KEY", "")).strip()
except Exception:
    secret_api_key = ""
if secret_api_key and not os.getenv("OPEN_METEO_API_KEY"):
    os.environ["OPEN_METEO_API_KEY"] = secret_api_key

try:
    secret_customer_base = str(st.secrets.get("OPEN_METEO_CUSTOMER_BASE", "")).strip()
except Exception:
    secret_customer_base = ""
if secret_customer_base and not os.getenv("OPEN_METEO_CUSTOMER_BASE"):
    os.environ["OPEN_METEO_CUSTOMER_BASE"] = secret_customer_base

if "selected_location" not in st.session_state:
    st.session_state.selected_location = DEFAULT_LOCATION.copy()
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "search_input" not in st.session_state:
    st.session_state.search_input = format_candidate(st.session_state.selected_location)
if "latitude_input" not in st.session_state:
    st.session_state.latitude_input = f"{float(st.session_state.selected_location['latitude']):.4f}"
if "longitude_input" not in st.session_state:
    st.session_state.longitude_input = f"{float(st.session_state.selected_location['longitude']):.4f}"

pending_location_sync = st.session_state.pop("pending_location_sync", None)
if pending_location_sync is not None:
    st.session_state.search_input = format_candidate(pending_location_sync)
    st.session_state.latitude_input = f"{float(pending_location_sync['latitude']):.4f}"
    st.session_state.longitude_input = f"{float(pending_location_sync['longitude']):.4f}"


def queue_location_sync(location: dict) -> None:
    st.session_state.pending_location_sync = location.copy()


def search_and_select() -> None:
    query = st.session_state.search_input.strip()
    if not query:
        return
    with st.spinner("Konum araniyor..."):
        try:
            location = search_best_location(query)
        except Exception as exc:
            st.session_state.analysis_result = None
            st.error(f"Konum aramasi sirasinda hata olustu: {exc}")
            return
    if location is None:
        st.session_state.analysis_result = None
        st.error("Bu isimle konum bulunamadi.")
        return
    st.session_state.selected_location = location
    st.session_state.analysis_result = None
    queue_location_sync(location)
    st.rerun()


def update_manual_coordinates() -> None:
    try:
        latitude = float(st.session_state.latitude_input)
        longitude = float(st.session_state.longitude_input)
    except (TypeError, ValueError):
        st.error("Enlem ve boylam sayisal olmali.")
        return
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        st.error("Enlem -90 ile 90, boylam -180 ile 180 arasinda olmali.")
        return
    st.session_state.selected_location = {
        "name": "Ozel koordinat",
        "admin1": "",
        "country": "",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
    }
    st.session_state.analysis_result = None
    queue_location_sync(st.session_state.selected_location)
    st.rerun()


def analysis_to_frames(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    idf_rows = []
    for duration in result["durations"]:
        row = {
            "Sure (saat)": duration["duration"],
            "Seri ort. (mm/saat)": duration["sampleMean"] / duration["duration"],
            "Ornek sayisi": duration["sampleSize"],
            "Secili parametre": duration["parameterText"],
        }
        for period, intensity, low, high in zip(
            result["returnPeriods"],
            duration["intensities"],
            duration["intensityLower"],
            duration["intensityUpper"],
        ):
            row[f"{period} yil"] = intensity
            row[f"{period} yil alt"] = low
            row[f"{period} yil ust"] = high
        idf_rows.append(row)
        for distribution_key, values in duration["fitDiagnostics"].items():
            row[f"diag_{distribution_key}_rank"] = values["rank"]
            row[f"diag_{distribution_key}_aic"] = values["aic"]
            row[f"diag_{distribution_key}_ks"] = values["ks"]
            row[f"diag_{distribution_key}_ad"] = values["ad"]
    diagnostics_rows = []
    for duration in result["durations"]:
        selected = duration["fitDiagnostics"][result["distribution"]]
        diagnostics_rows.append(
            {
                "Sure (saat)": duration["duration"],
                "Secili dagilim": DISTRIBUTIONS[result["distribution"]],
                "Dagilim sirasi": selected["rank"],
                "AIC": selected["aic"],
                "KS": selected["ks"],
                "KS p": selected["ksPValue"],
                "AD": selected["ad"],
                "Olay sayisi": duration["sampleSize"],
                "Olay/yil": duration["seriesMeta"]["eventRate"],
            }
        )
    return pd.DataFrame(idf_rows), pd.DataFrame(diagnostics_rows)


def result_to_csv_bytes(result: dict) -> bytes:
    idf_df, diagnostics_df = analysis_to_frames(result)
    buffer = io.StringIO()
    meta = pd.DataFrame(
        [
            ["konum", format_candidate(result["location"])],
            ["enlem", result["location"]["latitude"]],
            ["boylam", result["location"]["longitude"]],
            ["dagilim", result["distribution"]],
            ["seri_turu", result["seriesMethod"]],
            ["yagis_temeli", result["rainfallVariable"]],
            ["guven_duzeyi", result["confidenceLevel"]],
            ["baslangic_yili", result["startYear"]],
            ["bitis_yili", result["endYear"]],
            ["olusturma_zamani", result["generatedAt"]],
        ],
        columns=["alan", "deger"],
    )
    meta.to_csv(buffer, index=False)
    buffer.write("\nIDF_TABLOSU\n")
    idf_df.to_csv(buffer, index=False)
    buffer.write("\nDAGILIM_TANILARI\n")
    diagnostics_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_result_comment(result: dict) -> str:
    durations = result["durations"]
    ranks = [duration["fitDiagnostics"][result["distribution"]]["rank"] for duration in durations]
    rank_one_count = sum(1 for rank in ranks if rank == 1)
    sample_sizes = [duration["sampleSize"] for duration in durations]
    widest_ratio = 0.0
    for duration in durations:
        for central, low, high in zip(duration["intensities"], duration["intensityLower"], duration["intensityUpper"]):
            if central > 0:
                widest_ratio = max(widest_ratio, (high - low) / central)
    confidence_text = "dar" if widest_ratio < 0.35 else "orta" if widest_ratio < 0.75 else "genis"
    rank_text = (
        "Secili dagilim tum surelerde en iyi uyumlu aday."
        if rank_one_count == len(durations)
        else f"Secili dagilim {len(durations)} surenin {rank_one_count} tanesinde en iyi uyumu verdi."
    )
    return (
        f"Bu tablo {result['seriesMethodLabel']} ve {result['rainfallVariableLabel'].lower()} uzerinden uretildi. "
        f"Ornek buyuklugu {min(sample_sizes)}-{max(sample_sizes)} araliginda. "
        f"{int(CONFIDENCE_LEVEL * 100)}% guven bandi genel olarak {confidence_text}. "
        f"{rank_text}"
    )


def render_google_map(location: dict) -> None:
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    zoom = 12 if abs(latitude) < 55 else 10
    src = f"https://maps.google.com/maps?q={latitude},{longitude}&z={zoom}&t=k&output=embed"
    components.html(
        f"""
        <div style="position:relative;height:280px;border:1px solid #d6ddd5;border-radius:8px;overflow:hidden;background:#dfe6de;">
          <iframe
            src="{src}"
            style="width:100%;height:100%;border:0;"
            loading="lazy"
            referrerpolicy="no-referrer-when-downgrade">
          </iframe>
          <div style="position:absolute;left:50%;top:50%;transform:translate(-50%, -80%);pointer-events:none;">
            <div style="width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;border-top:18px solid #c62828;filter:drop-shadow(0 2px 3px rgba(0,0,0,0.4));"></div>
          </div>
        </div>
        """,
        height=280,
    )


st.markdown(
    """
    <style>
      .main { background: linear-gradient(180deg, rgba(221,233,235,0.55), rgba(246,247,242,0) 360px), #f6f7f2; }
      .stApp { background: transparent; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.8); border: 1px solid #d6ddd5; border-radius: 8px; padding: 10px; }
      .block-card { background: rgba(255,255,255,0.86); border: 1px solid #d6ddd5; border-radius: 8px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Intensity-Duration-Frequency tablosu")
st.caption("Streamlit backend + disk cache ile Open-Meteo destekli IDF analizi")

open_meteo_config = get_open_meteo_config()

left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Girdi")
    if open_meteo_config["use_customer_api"]:
        st.success("Open-Meteo ticari API aktif")
    else:
        st.warning("Open-Meteo ucretsiz API aktif")
    st.caption("API anahtari yalnizca Secrets/env uzerinden okunur.")

    search_col, button_col = st.columns([4, 1])
    with search_col:
        st.text_input("Konum veya mekan adi", key="search_input", placeholder="Anitkabir, Galata Kulesi, Rize...")
    with button_col:
        st.write("")
        st.write("")
        if st.button("Ara", use_container_width=True):
            search_and_select()

    selected = st.session_state.selected_location
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        st.text_input("Enlem", key="latitude_input")
    with coord_col2:
        st.text_input("Boylam", key="longitude_input")
    if st.button("Koordinati uygula", use_container_width=True):
        update_manual_coordinates()

    st.caption(f"Secili nokta: {format_candidate(st.session_state.selected_location)}")
    st.caption(
        f"Koordinat: {float(st.session_state.selected_location['latitude']):.4f}, "
        f"{float(st.session_state.selected_location['longitude']):.4f}"
    )
    render_google_map(st.session_state.selected_location)

    year_col1, year_col2 = st.columns(2)
    with year_col1:
        start_year = st.number_input("Baslangic yili", min_value=1940, max_value=DEFAULT_END_YEAR, value=1995, step=1)
    with year_col2:
        end_year = st.number_input("Bitis yili", min_value=1940, max_value=DEFAULT_END_YEAR, value=DEFAULT_END_YEAR, step=1)

    distribution = st.selectbox(
        "Dagilim",
        options=list(DISTRIBUTIONS.keys()),
        format_func=lambda key: DISTRIBUTIONS[key],
    )
    rainfall_variable = st.selectbox(
        "Yagis temeli",
        options=list(RAINFALL_VARIABLES.keys()),
        format_func=lambda key: RAINFALL_VARIABLES[key],
    )
    series_method = st.selectbox(
        "Seri tipi",
        options=list(SERIES_METHODS.keys()),
        format_func=lambda key: SERIES_METHODS[key],
    )
    pds_percentile = DEFAULT_PDS_PERCENTILE
    pds_gap_hours = DEFAULT_PDS_GAP_HOURS
    if series_method == "pds":
        pds_col1, pds_col2 = st.columns(2)
        with pds_col1:
            pds_percentile = st.number_input(
                "PDS esik yuzdesi",
                min_value=90.0,
                max_value=99.9,
                value=DEFAULT_PDS_PERCENTILE,
                step=0.5,
            )
        with pds_col2:
            pds_gap_hours = st.number_input(
                "Bagimsiz olay araligi",
                min_value=6,
                max_value=240,
                value=DEFAULT_PDS_GAP_HOURS,
                step=6,
            )
    st.caption(
        "Cache anahtari: konum + yil araligi + dagilim + seri tipi. "
        "Ham Open-Meteo parcaciklari da diskte tutulur."
    )

    calculate_clicked = st.button("Tabloyu hesapla", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    status_box = st.empty()

if calculate_clicked:
    minimum_years = 8 if series_method == "ams" else 5
    if start_year > end_year:
        st.error("Baslangic yili bitis yilindan buyuk olamaz.")
    elif end_year - start_year + 1 < minimum_years:
        st.error(f"{SERIES_METHODS[series_method]} icin en az {minimum_years} yillik seri secin.")
    else:
        progress = status_box.progress(0, text="Analiz baslatiliyor...")

        def report_progress(message: str, ratio: float) -> None:
            progress.progress(min(max(ratio, 0.0), 1.0), text=message)

        try:
            with st.spinner("Open-Meteo verileri okunuyor..."):
                result = calculate_analysis(
                    st.session_state.selected_location,
                    int(start_year),
                    int(end_year),
                    distribution,
                    rainfall_variable=rainfall_variable,
                    series_method=series_method,
                    pds_percentile=float(pds_percentile),
                    pds_gap_hours=int(pds_gap_hours),
                    progress_callback=report_progress,
                )
        except Exception as exc:
            st.session_state.analysis_result = None
            status_box.empty()
            st.error(str(exc))
        else:
            st.session_state.analysis_result = result
            progress.progress(1.0, text="Hazir")

result = st.session_state.analysis_result

with right:
    if result is None:
        st.info("Sonuc burada gosterilecek. Konumu secip hesaplamayi baslatin.")
    else:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Veri kapsami", f"{result['yearsUsed']} yil")
        metric2.metric("Eksik saat", f"{result['missingHours']:,}".replace(",", "."))
        metric3.metric("Yinelenme", f"{RETURN_PERIODS[0]}-{RETURN_PERIODS[-1]} yil")
        metric4.metric("Seri tipi", result["seriesMethod"].upper())

        idf_df, diagnostics_df = analysis_to_frames(result)

        st.caption(
            f"Secili dagilim: {DISTRIBUTIONS[result['distribution']]} | "
            f"Yagis temeli: {result['rainfallVariableLabel']} | "
            f"{int(CONFIDENCE_LEVEL * 100)}% bootstrap guven araligi"
        )
        st.info(build_result_comment(result))
        st.dataframe(idf_df.round(3), use_container_width=True, hide_index=True)

        st.download_button(
            "CSV indir",
            data=result_to_csv_bytes(result),
            file_name=f"idf-{result['distribution']}-{result['startYear']}-{result['endYear']}.csv",
            mime="text/csv",
            use_container_width=False,
        )
