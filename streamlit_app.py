from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_backend import (
    CONFIDENCE_LEVEL,
    DEFAULT_END_YEAR,
    DEFAULT_LOCATION,
    DEFAULT_PDS_GAP_HOURS,
    DEFAULT_PDS_PERCENTILE,
    DISTRIBUTIONS,
    RAINFALL_VARIABLES,
    RETURN_PERIODS,
    SERIES_METHODS,
    calculate_analysis,
    format_candidate,
    search_best_location,
)

ANALYSIS_METHODS = {
    "mgm_compatible": "MGM uyumlu frekans analizi",
    "extended": "Genişletilmiş istatistiksel analiz",
}
MGM_DISTRIBUTIONS = {
    "ln2": "Log-Normal 2P",
    "ln3": "Log-Normal 3P",
    "gamma2": "Gama 2P",
    "lp3": "Log-Pearson III",
    "gumbel": "Gumbel",
}


st.set_page_config(page_title="Tasarım Yağışı ve IDF Analizi", page_icon="🌧️", layout="wide")

try:
    api_key = str(st.secrets.get("OPEN_METEO_API_KEY", "")).strip()
    customer_base = str(st.secrets.get("OPEN_METEO_CUSTOMER_BASE", "")).strip()
except Exception:
    api_key = ""
    customer_base = ""
if api_key and not os.getenv("OPEN_METEO_API_KEY"):
    os.environ["OPEN_METEO_API_KEY"] = api_key
if customer_base and not os.getenv("OPEN_METEO_CUSTOMER_BASE"):
    os.environ["OPEN_METEO_CUSTOMER_BASE"] = customer_base

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

pending = st.session_state.pop("pending_location_sync", None)
if pending:
    st.session_state.search_input = format_candidate(pending)
    st.session_state.latitude_input = f"{float(pending['latitude']):.4f}"
    st.session_state.longitude_input = f"{float(pending['longitude']):.4f}"


def sync_location(location: dict) -> None:
    st.session_state.pending_location_sync = location.copy()


def search_and_select() -> None:
    query = st.session_state.search_input.strip()
    if not query:
        st.warning("Lütfen bir konum adı yazın.")
        return
    try:
        with st.spinner("Konum aranıyor..."):
            location = search_best_location(query)
    except Exception as exc:
        st.error(f"Konum aranırken hata oluştu: {exc}")
        return
    if location is None:
        st.error("Bu adla bir konum bulunamadı.")
        return
    st.session_state.selected_location = location
    st.session_state.analysis_result = None
    sync_location(location)
    st.rerun()


def update_coordinates() -> None:
    try:
        latitude = float(st.session_state.latitude_input)
        longitude = float(st.session_state.longitude_input)
    except (TypeError, ValueError):
        st.error("Enlem ve boylam sayısal olmalıdır.")
        return
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        st.error("Enlem −90–90, boylam −180–180 aralığında olmalıdır.")
        return
    location = {
        "name": "Özel koordinat",
        "admin1": "",
        "country": "",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
    }
    st.session_state.selected_location = location
    st.session_state.analysis_result = None
    sync_location(location)
    st.rerun()


def render_map(location: dict) -> None:
    lat, lon = float(location["latitude"]), float(location["longitude"])
    src = f"https://maps.google.com/maps?q={lat},{lon}&z=12&t=k&output=embed"
    components.html(
        f"""
        <div style="height:245px;border:1px solid #d7e0df;border-radius:14px;overflow:hidden">
          <iframe src="{src}" style="width:100%;height:100%;border:0" loading="lazy"></iframe>
        </div>
        """,
        height=245,
    )


def analysis_frames(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diagnostics = [], []
    for duration in result["durations"]:
        row = {
            "Süre (saat)": duration["duration"],
            "Örnek sayısı": duration["sampleSize"],
        }
        for period, intensity, low, high in zip(
            result["returnPeriods"],
            duration["intensities"],
            duration["intensityLower"],
            duration["intensityUpper"],
        ):
            row[f"{period} yıl (mm/saat)"] = intensity
            row[f"{period} yıl alt"] = low
            row[f"{period} yıl üst"] = high
        rows.append(row)
        selected_key = duration.get("selectedDistribution", result["distribution"])
        selected = duration["fitDiagnostics"][selected_key]
        diagnostics.append(
            {
                "Süre (saat)": duration["duration"],
                "Dağılım": duration.get(
                    "selectedDistributionLabel",
                    DISTRIBUTIONS.get(selected_key, MGM_DISTRIBUTIONS.get(selected_key, selected_key)),
                ),
                "Uyum sırası": selected["rank"],
                "AIC": selected["aic"],
                "KS": selected["ks"],
                "KS p": selected["ksPValue"],
                "AD": selected["ad"],
                "Khi-kare": selected.get("chiSquare"),
                "Khi-kare sd": selected.get("chiSquareDf"),
                "Khi-kare p": selected.get("chiSquarePValue"),
                "Olay sayısı": duration["sampleSize"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def chart_frame(result: dict) -> pd.DataFrame:
    records = []
    for duration in result["durations"]:
        for period, central, low, high in zip(
            result["returnPeriods"],
            duration["intensities"],
            duration["intensityLower"],
            duration["intensityUpper"],
        ):
            records.append(
                {
                    "Süre (saat)": float(duration["duration"]),
                    "Tekerrür dönemi": f"{period} yıl",
                    "Şiddet (mm/saat)": central,
                    "Alt sınır": low,
                    "Üst sınır": high,
                }
            )
    return pd.DataFrame(records)


def chart_spec(data: pd.DataFrame) -> dict:
    values = data.to_dict(orient="records")
    return {
        "data": {"values": values},
        "width": "container",
        "height": 410,
        "layer": [
            {
                "mark": {"type": "area", "opacity": 0.10},
                "encoding": {
                    "x": {"field": "Süre (saat)", "type": "quantitative", "scale": {"type": "log"}},
                    "y": {"field": "Alt sınır", "type": "quantitative", "title": "Şiddet (mm/saat)"},
                    "y2": {"field": "Üst sınır"},
                    "color": {"field": "Tekerrür dönemi", "type": "nominal"},
                    "detail": {"field": "Tekerrür dönemi", "type": "nominal"},
                    "tooltip": [
                        {"field": "Tekerrür dönemi", "type": "nominal"},
                        {"field": "Süre (saat)", "type": "quantitative"},
                        {"field": "Alt sınır", "type": "quantitative", "format": ".2f"},
                        {"field": "Üst sınır", "type": "quantitative", "format": ".2f"},
                    ],
                },
            },
            {
                "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "Süre (saat)", "type": "quantitative", "scale": {"type": "log"}},
                    "y": {"field": "Şiddet (mm/saat)", "type": "quantitative"},
                    "color": {"field": "Tekerrür dönemi", "type": "nominal"},
                    "tooltip": [
                        {"field": "Tekerrür dönemi", "type": "nominal"},
                        {"field": "Süre (saat)", "type": "quantitative"},
                        {"field": "Şiddet (mm/saat)", "type": "quantitative", "format": ".2f"},
                        {"field": "Alt sınır", "type": "quantitative", "format": ".2f"},
                        {"field": "Üst sınır", "type": "quantitative", "format": ".2f"},
                    ],
                },
            }
        ],
    }


def result_csv(result: dict) -> bytes:
    idf_df, diagnostics_df = analysis_frames(result)
    buffer = io.StringIO()
    pd.DataFrame(
        [
            ["konum", format_candidate(result["location"])],
            ["enlem", result["location"]["latitude"]],
            ["boylam", result["location"]["longitude"]],
            ["dağılım", DISTRIBUTIONS[result["distribution"]]],
            ["seri_türü", result["seriesMethodLabel"]],
            ["yağış_temeli", result["rainfallVariableLabel"]],
            ["güven_düzeyi", result["confidenceLevel"]],
            ["başlangıç_yılı", result["startYear"]],
            ["bitiş_yılı", result["endYear"]],
        ],
        columns=["alan", "değer"],
    ).to_csv(buffer, index=False)
    buffer.write("\nIDF_TABLOSU\n")
    idf_df.to_csv(buffer, index=False)
    buffer.write("\nDAĞILIM_TANILARI\n")
    diagnostics_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def quality_message(result: dict) -> tuple[str, str]:
    years = int(result["yearsUsed"])
    missing = int(result["missingHours"])
    longest_period = int(result["returnPeriods"][-1])
    if years < 20:
        return "warning", "Kayıt süresi 20 yıldan kısa; uzun tekerrür dönemleri yüksek belirsizlik taşır."
    if longest_period > years * 5:
        return "warning", "En uzun tekerrür dönemi gözlem süresinin çok üzerindedir; sonuç ekstrapolasyondur."
    if missing > years * 24 * 10:
        return "warning", "Eksik saat sayısı yüksektir; veri kalite bölümünü kontrol edin."
    return "success", "Kayıt uzunluğu temel ön değerlendirme için yeterli görünüyor."


st.markdown(
    """
    <style>
    .stApp {background:linear-gradient(180deg,#edf5f4 0,#f8faf8 440px,#fff 100%)}
    .block-container {padding-top:2.2rem;max-width:1500px}
    div[data-testid="stMetric"] {background:#fff;border:1px solid #d9e4e2;border-radius:14px;padding:13px}
    div[data-testid="stForm"] {background:rgba(255,255,255,.88);border:1px solid #d9e4e2;border-radius:16px;padding:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌧️ Tasarım Yağışı ve IDF Analizi")
st.caption("Seçilen konum için farklı süre ve tekerrür dönemlerine ait tasarım yağışı şiddetlerini hesaplayın.")

with st.expander("Veri kaynağı ve kullanım sınırı", expanded=False):
    st.info(
        "Bu uygulama Open-Meteo üzerinden yeniden analiz/model tabanlı yağış verisi kullanır. "
        "Sonuçlar ön değerlendirme niteliğindedir; resmi istasyon IDF değerlerinin veya yerel kurum "
        "tasarım kriterlerinin yerine geçmez."
    )

left, right = st.columns([1, 1.75], gap="large")

with left:
    st.subheader("1 · Konumu seçin")
    search_col, search_button = st.columns([4, 1])
    with search_col:
        st.text_input("Konum veya mekân adı", key="search_input", placeholder="Anıtkabir, Rize, Samsun...")
    with search_button:
        st.write("")
        st.write("")
        if st.button("Ara", use_container_width=True):
            search_and_select()

    c1, c2, c3 = st.columns([1, 1, 0.8])
    with c1:
        st.text_input("Enlem", key="latitude_input")
    with c2:
        st.text_input("Boylam", key="longitude_input")
    with c3:
        st.write("")
        st.write("")
        if st.button("Uygula", use_container_width=True):
            update_coordinates()

    selected = st.session_state.selected_location
    st.markdown(f"**Seçili nokta:** {format_candidate(selected)}")
    st.caption(f"{float(selected['latitude']):.4f}, {float(selected['longitude']):.4f}")
    render_map(selected)

    st.subheader("2 · Analiz ayarları")
    analysis_method = st.radio(
        "Hesap yöntemi",
        options=list(ANALYSIS_METHODS),
        format_func=ANALYSIS_METHODS.get,
        horizontal=True,
        help=(
            "MGM uyumlu yöntem, Open-Meteo ERA5 saatlik verisine MGM'nin yıllık maksimum, "
            "standart saatlik süre, aday dağılım ve uygunluk seçimi yaklaşımını uygular."
        ),
    )
    y1, y2 = st.columns(2)
    with y1:
        start_year = st.number_input("Başlangıç yılı", 1940, DEFAULT_END_YEAR, 1995, 1)
    with y2:
        end_year = st.number_input("Bitiş yılı", 1940, DEFAULT_END_YEAR, DEFAULT_END_YEAR, 1)

    distribution = list(DISTRIBUTIONS.keys())[0]
    rainfall_variable = list(RAINFALL_VARIABLES.keys())[0]
    series_method = list(SERIES_METHODS.keys())[0]
    pds_percentile, pds_gap_hours = DEFAULT_PDS_PERCENTILE, DEFAULT_PDS_GAP_HOURS

    with st.expander("Gelişmiş ayarlar"):
        rainfall_variable = st.selectbox("Yağış temeli", list(RAINFALL_VARIABLES), format_func=RAINFALL_VARIABLES.get)
        if analysis_method == "extended":
            distribution = st.selectbox("İstatistiksel dağılım", list(DISTRIBUTIONS), format_func=DISTRIBUTIONS.get)
            series_method = st.selectbox("Seri tipi", list(SERIES_METHODS), format_func=SERIES_METHODS.get)
        else:
            series_method = "ams"
            st.caption(
                "MGM uyumlu mod: AMS; 1, 2, 3, 4, 5, 6, 8, 12, 18 ve 24 saat; "
                "LN2, LN3, Gama 2P, LP3 ve Gumbel arasından KS ve khi-kare ile otomatik seçim."
            )
        if analysis_method == "extended" and series_method == "pds":
            p1, p2 = st.columns(2)
            with p1:
                pds_percentile = st.number_input("PDS eşik yüzdesi", 90.0, 99.9, DEFAULT_PDS_PERCENTILE, 0.5)
            with p2:
                pds_gap_hours = st.number_input("Bağımsız olay aralığı (saat)", 6, 240, DEFAULT_PDS_GAP_HOURS, 6)

    calculate_clicked = st.button("Analizi hesapla", type="primary", use_container_width=True)

with right:
    status_box = st.empty()

if calculate_clicked:
    minimum_years = 8 if series_method == "ams" else 5
    if start_year > end_year:
        st.error("Başlangıç yılı bitiş yılından büyük olamaz.")
    elif end_year - start_year + 1 < minimum_years:
        st.error(f"{SERIES_METHODS[series_method]} için en az {minimum_years} yıllık seri seçin.")
    else:
        progress = status_box.progress(0, text="Analiz başlatılıyor...")

        def report_progress(message: str, ratio: float) -> None:
            progress.progress(min(max(ratio, 0.0), 1.0), text=message)

        try:
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
                analysis_method=analysis_method,
            )
        except Exception as exc:
            st.session_state.analysis_result = None
            status_box.empty()
            st.error(str(exc))
        else:
            st.session_state.analysis_result = result
            progress.progress(1.0, text="Analiz hazır")

result = st.session_state.analysis_result
with right:
    if result is None:
        st.info("Konumu ve analiz dönemini seçip **Analizi hesapla** düğmesine basın.")
    else:
        st.subheader("3 · Sonuçlar")
        if result.get("analysisMethod") == "mgm_compatible":
            st.info(
                "**MGM uyumlu hesap:** Open-Meteo ERA5 saatlik yeniden analiz verisi kullanılmıştır. "
                "Bu sonuç resmî MGM istasyon IDF değeri değildir. Saatlik çözünürlük nedeniyle "
                "5–30 dakikalık MGM süreleri hesaplanmamıştır."
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Veri dönemi", f"{result['startYear']}–{result['endYear']}")
        m2.metric("Kullanılan kayıt", f"{result['yearsUsed']} yıl")
        m3.metric("Tekerrür aralığı", f"{RETURN_PERIODS[0]}–{RETURN_PERIODS[-1]} yıl")
        m4.metric("Hesap yöntemi", result.get("analysisMethodLabel", result["seriesMethod"].upper()))

        level, message = quality_message(result)
        getattr(st, level)(message)
        tabs = st.tabs(["IDF eğrileri", "Yağış tablosu", "İstatistiksel uygunluk", "Veri kalitesi", "İndir"])
        idf_df, diagnostics_df = analysis_frames(result)

        with tabs[0]:
            st.markdown("#### Yağış şiddeti–süre–tekerrür eğrileri")
            st.caption(
                f"Çizgiler merkezi tahmini, yarı saydam alanlar %{int(CONFIDENCE_LEVEL * 100)} "
                "bootstrap güven aralığını gösterir."
            )
            st.vega_lite_chart(chart_spec(chart_frame(result)), use_container_width=True)

        with tabs[1]:
            st.dataframe(idf_df.round(3), use_container_width=True, hide_index=True)

        with tabs[2]:
            if result.get("analysisMethod") == "mgm_compatible":
                st.caption("Her süre için dağılım, KS ve khi-kare uygunluk sonuçlarına göre otomatik seçilmiştir.")
            else:
                st.caption("Daha düşük AIC ve yüksek uyum sırası, aday dağılımın göreli başarısını gösterir.")
            st.dataframe(diagnostics_df.round(4), use_container_width=True, hide_index=True)

        with tabs[3]:
            q1, q2, q3 = st.columns(3)
            q1.metric("Kullanılan yıl", result["yearsUsed"])
            q2.metric("Eksik saat", f"{result['missingHours']:,}".replace(",", "."))
            q3.metric("Güven bandı", f"%{int(CONFIDENCE_LEVEL * 100)}")
            st.warning(
                "Uzun tekerrür dönemleri gözlem süresinin ötesine istatistiksel ekstrapolasyondur. "
                "Kritik altyapı tasarımında resmi istasyon kayıtları ve kurum kriterleriyle doğrulayın."
            )

        with tabs[4]:
            st.download_button(
                "CSV raporunu indir",
                data=result_csv(result),
                file_name=f"idf-{result['distribution']}-{result['startYear']}-{result['endYear']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
