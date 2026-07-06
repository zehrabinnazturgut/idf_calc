from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import requests

DISTRIBUTIONS = {
    "lp3": "Log Pearson III",
    "gumbel": "Gumbel",
    "gev": "GEV",
}
SERIES_METHODS = {
    "ams": "Annual Maximum Series",
    "pds": "Partial Duration Series",
}
RUNOFF_VARIABLES = [
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]
DURATIONS = [1, 2, 3, 6, 12, 24]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
ARCHIVE_CHUNK_YEARS = 12
DEFAULT_END_YEAR = 2025
DEFAULT_PDS_PERCENTILE = 99.5
DEFAULT_PDS_GAP_HOURS = 72
DEFAULT_BOOTSTRAP_SAMPLES = 200
CONFIDENCE_LEVEL = 0.90
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_CUSTOMER_BASE = "https://customer-api.open-meteo.com"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
REQUEST_TIMEOUT = 90
CACHE_ROOT = Path("work/streamlit_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVE_CACHE_TTL = 60 * 60 * 24 * 14
ANALYSIS_CACHE_TTL = 60 * 60 * 24 * 30
GEOCODE_CACHE_TTL = 60 * 60 * 24 * 30

DEFAULT_LOCATION = {
    "name": "Ankara",
    "admin1": "Ankara",
    "country": "Turkiye",
    "latitude": 39.9334,
    "longitude": 32.8597,
    "timezone": "Europe/Istanbul",
}


def format_candidate(candidate: dict[str, Any]) -> str:
    return ", ".join(
        [part for part in [candidate.get("name"), candidate.get("admin1"), candidate.get("country")] if part]
    )


def get_open_meteo_config() -> dict[str, Any]:
    api_key = (os.getenv("OPEN_METEO_API_KEY") or "").strip()
    customer_base = (os.getenv("OPEN_METEO_CUSTOMER_BASE") or OPEN_METEO_CUSTOMER_BASE).rstrip("/")
    use_customer_api = bool(api_key)
    return {
        "api_key": api_key,
        "use_customer_api": use_customer_api,
        "archive_url": OPEN_METEO_ARCHIVE,
        "geocoding_url": OPEN_METEO_GEOCODING,
        "forecast_url": f"{customer_base}/v1/forecast" if use_customer_api else "https://api.open-meteo.com/v1/forecast",
        "archive_host": "archive-api.open-meteo.com" if use_customer_api else "free",
    }


def _cache_path(namespace: str, key: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    folder = CACHE_ROOT / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def _cache_read(namespace: str, key: dict[str, Any], ttl_seconds: int) -> Any | None:
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    saved_at = payload.get("saved_at", 0)
    if time.time() - saved_at > ttl_seconds:
      return None
    return payload.get("value")


def _cache_write(namespace: str, key: dict[str, Any], value: Any) -> None:
    path = _cache_path(namespace, key)
    envelope = {"saved_at": time.time(), "value": value}
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")


def _session_get_json(url: str, params: dict[str, Any], include_api_key: bool = False) -> tuple[int, dict[str, Any]]:
    request_params = dict(params)
    if include_api_key:
        config = get_open_meteo_config()
        if config["api_key"]:
            request_params["apikey"] = config["api_key"]
    response = requests.get(
        url,
        params=request_params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Codex-IDF-Streamlit/1.0"},
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return response.status_code, payload


def search_best_location(query: str) -> dict[str, Any] | None:
    query = query.strip()
    if not query:
        return None
    cache_key = {"query": query.lower()}
    cached = _cache_read("geocode", cache_key, GEOCODE_CACHE_TTL)
    if isinstance(cached, dict) and "location" in cached:
        return cached["location"]
    location = _search_nominatim(query) or _search_open_meteo(query)
    _cache_write("geocode", cache_key, {"location": location})
    return location


def _search_nominatim(query: str) -> dict[str, Any] | None:
    status, payload = _session_get_json(
        NOMINATIM_SEARCH,
        {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "accept-language": "tr",
        },
    )
    if status != 200 or not isinstance(payload, list) or not payload:
        return None
    match = payload[0]
    address = match.get("address", {})
    return {
        "name": match.get("name")
        or address.get("attraction")
        or address.get("tourism")
        or address.get("building")
        or address.get("amenity")
        or _first_display_segment(match.get("display_name"))
        or query,
        "admin1": address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or address.get("state")
        or "",
        "country": address.get("country") or "",
        "latitude": float(match["lat"]),
        "longitude": float(match["lon"]),
        "timezone": "auto",
    }


def _search_open_meteo(query: str) -> dict[str, Any] | None:
    config = get_open_meteo_config()
    status, payload = _session_get_json(
        config["geocoding_url"],
        {"name": query, "count": 1, "language": "tr", "format": "json"},
        include_api_key=False,
    )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if status != 200 or not results:
        return None
    return results[0]


def calculate_analysis(
    location: dict[str, Any],
    start_year: int,
    end_year: int,
    distribution: str,
    series_method: str = "ams",
    pds_percentile: float = DEFAULT_PDS_PERCENTILE,
    pds_gap_hours: int = DEFAULT_PDS_GAP_HOURS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    query_key = {
        "lat": round(float(location["latitude"]), 4),
        "lon": round(float(location["longitude"]), 4),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "distribution": distribution,
        "series_method": series_method,
        "pds_percentile": round(float(pds_percentile), 3),
        "pds_gap_hours": int(pds_gap_hours),
        "bootstrap_samples": int(bootstrap_samples),
        "variables": RUNOFF_VARIABLES,
        "chunk_years": ARCHIVE_CHUNK_YEARS,
        "commercial_api": get_open_meteo_config()["use_customer_api"],
    }
    cached = _cache_read("analysis", query_key, ANALYSIS_CACHE_TTL)
    if cached is not None:
        if progress_callback:
            progress_callback("Önbellekten sonuç alındı", 1.0)
        return cached

    series = fetch_archive_by_year(location, start_year, end_year, progress_callback)
    result = build_analysis(
        yearly_series=series,
        location=location,
        distribution=distribution,
        start_year=start_year,
        end_year=end_year,
        series_method=series_method,
        pds_percentile=pds_percentile,
        pds_gap_hours=pds_gap_hours,
        bootstrap_samples=bootstrap_samples,
        progress_callback=progress_callback,
    )
    _cache_write("analysis", query_key, result)
    return result


def fetch_archive_by_year(
    location: dict[str, Any],
    start_year: int,
    end_year: int,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[dict[str, Any]]:
    config = get_open_meteo_config()
    total = end_year - start_year + 1
    rows: list[dict[str, Any]] = []
    processed = 0

    for chunk_start in range(start_year, end_year + 1, ARCHIVE_CHUNK_YEARS):
        chunk_end = min(end_year, chunk_start + ARCHIVE_CHUNK_YEARS - 1)
        if progress_callback:
            progress_callback(f"{chunk_start}-{chunk_end} çekiliyor", min(processed / max(total, 1), 0.92))
        cache_key = {
            "lat": round(float(location["latitude"]), 4),
            "lon": round(float(location["longitude"]), 4),
            "start_year": chunk_start,
            "end_year": chunk_end,
            "variables": RUNOFF_VARIABLES,
            "model": "era5",
            "commercial_api": config["use_customer_api"],
        }
        payload = _cache_read("archive", cache_key, ARCHIVE_CACHE_TTL)
        if payload is None:
            status, payload = _session_get_json(
                config["archive_url"],
                {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "start_date": f"{chunk_start}-01-01",
                    "end_date": f"{chunk_end}-12-31",
                    "hourly": ",".join(RUNOFF_VARIABLES),
                    "models": "era5",
                    "timezone": location.get("timezone", "auto"),
                },
                include_api_key=config["use_customer_api"],
            )
            if status != 200 or payload.get("reason"):
                raise RuntimeError(_build_archive_error_message(status, payload, chunk_start, chunk_end))
            _cache_write("archive", cache_key, payload)
        chunk_rows = _split_chunk_into_years(payload.get("hourly", {}), chunk_start, chunk_end)
        rows.extend(chunk_rows)
        processed += len(chunk_rows)

    if progress_callback:
        progress_callback("Yıllık maksimumlar hazırlanıyor", 0.96)
    return rows


def _build_archive_error_message(status: int, payload: dict[str, Any], chunk_start: int, chunk_end: int) -> str:
    reason = payload.get("reason")
    if status == 429 or reason == "Hourly API request limit exceeded. Please try again in the next hour.":
        return (
            "Open-Meteo saatlik istek kotası bu saat için dolu. "
            "Yaklaşık bir saat sonra yeniden deneyin veya daha dar bir yıl aralığı seçin."
        )
    if reason:
        return str(reason)
    return f"{chunk_start}-{chunk_end} dönemi için Open-Meteo isteği başarısız oldu."


def _split_chunk_into_years(hourly: dict[str, Any], start_year: int, end_year: int) -> list[dict[str, Any]]:
    time_values = hourly.get("time", [])
    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        prefix = f"{year}-"
        indexes = [idx for idx, value in enumerate(time_values) if isinstance(value, str) and value.startswith(prefix)]
        yearly_hourly: dict[str, list[Any]] = {}
        for key, values in hourly.items():
            if key == "time" or not isinstance(values, list):
                continue
            yearly_hourly[key] = [values[index] if index < len(values) else None for index in indexes]
        precipitation = yearly_hourly.get("precipitation", [])
        missing = sum(1 for value in precipitation if value is None or not _is_finite(value))
        rows.append(
            {
                "year": year,
                "values": precipitation,
                "hourly": yearly_hourly,
                "hours": len(precipitation),
                "missing": missing,
            }
        )
    return rows


def build_analysis(
    yearly_series: list[dict[str, Any]],
    location: dict[str, Any],
    distribution: str,
    start_year: int,
    end_year: int,
    series_method: str = "ams",
    pds_percentile: float = DEFAULT_PDS_PERCENTILE,
    pds_gap_hours: int = DEFAULT_PDS_GAP_HOURS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    durations = []
    for duration_index, duration in enumerate(DURATIONS):
        if progress_callback:
            progress_callback(
                f"{duration} saat icin frekans analizi",
                0.96 + 0.04 * ((duration_index + 1) / max(len(DURATIONS), 1)),
            )
        events, extraction_meta = extract_duration_records(
            yearly_series=yearly_series,
            duration=duration,
            series_method=series_method,
            pds_percentile=pds_percentile,
            pds_gap_hours=pds_gap_hours,
        )
        maxima = [record["event"]["value"] for record in events]
        minimum_samples = 8 if series_method == "ams" else 12
        if len(maxima) < minimum_samples:
            raise RuntimeError(f"{duration} saat suresi icin yeterli bagimsiz olay olusmadi.")
        fit_diagnostics = compare_distributions(maxima)
        selected_fit = fit_diagnostics[distribution]
        nonexceedance_probabilities = [
            probability_for_return_period(return_period, extraction_meta["event_rate"], series_method)
            for return_period in RETURN_PERIODS
        ]
        depths = [quantile_from_fit(selected_fit, probability) for probability in nonexceedance_probabilities]
        lower_depths, upper_depths = bootstrap_confidence_intervals(
            samples=maxima,
            distribution=distribution,
            probabilities=nonexceedance_probabilities,
            bootstrap_samples=bootstrap_samples,
        )
        durations.append(
            {
                "duration": duration,
                "maxima": maxima,
                "depths": depths,
                "depthLower": lower_depths,
                "depthUpper": upper_depths,
                "intensities": [depth / duration for depth in depths],
                "intensityLower": [depth / duration for depth in lower_depths],
                "intensityUpper": [depth / duration for depth in upper_depths],
                "sampleMean": mean(maxima),
                "sampleMax": max(maxima),
                "sampleSize": len(maxima),
                "parameterText": selected_fit["parameterText"],
                "fitDiagnostics": fit_diagnostics,
                "seriesMeta": extraction_meta,
                "runoffSummary": summarize_runoff_drivers(events, duration),
            }
        )

    return {
        "location": location,
        "distribution": distribution,
        "seriesMethod": series_method,
        "seriesMethodLabel": SERIES_METHODS.get(series_method, series_method),
        "pdsPercentile": pds_percentile,
        "pdsGapHours": pds_gap_hours,
        "confidenceLevel": CONFIDENCE_LEVEL,
        "startYear": start_year,
        "endYear": end_year,
        "returnPeriods": RETURN_PERIODS,
        "durations": durations,
        "totalHours": sum(series["hours"] for series in yearly_series),
        "missingHours": sum(series["missing"] for series in yearly_series),
        "yearsUsed": len(yearly_series),
        "mgmCrosscheck": {
            "status": "placeholder",
            "message": "MGM saatlik veri baglantisi daha sonra eklenecek. Beklenen alanlar: zaman damgasi ve saatlik yagis.",
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def extract_duration_records(
    yearly_series: list[dict[str, Any]],
    duration: int,
    series_method: str,
    pds_percentile: float,
    pds_gap_hours: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if series_method == "pds":
        return extract_partial_duration_records(yearly_series, duration, pds_percentile, pds_gap_hours)
    records = []
    years_with_event = 0
    for series in yearly_series:
        event = rolling_max_event(series["values"], duration)
        if event["value"] is None or event["value"] <= 0:
            continue
        years_with_event += 1
        records.append({"series": series, "event": event, "year": series["year"]})
    event_rate = len(records) / max(len(yearly_series), 1)
    return records, {
        "method": "ams",
        "threshold": None,
        "independentGapHours": duration,
        "eventRate": event_rate,
        "event_rate": event_rate,
        "eventsPerYear": event_rate,
        "yearsWithEvents": years_with_event,
        "totalEvents": len(records),
    }


def extract_partial_duration_records(
    yearly_series: list[dict[str, Any]],
    duration: int,
    percentile: float,
    gap_hours: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_records = []
    candidate_values = []
    for series in yearly_series:
        for event in rolling_window_events(series["values"], duration):
            if event["value"] is None or event["value"] <= 0:
                continue
            record = {"series": series, "event": event, "year": series["year"]}
            candidate_records.append(record)
            candidate_values.append(event["value"])
    if not candidate_values:
        return [], {
            "method": "pds",
            "threshold": None,
            "independentGapHours": max(duration, gap_hours),
            "eventRate": 0.0,
            "event_rate": 0.0,
            "eventsPerYear": 0.0,
            "yearsWithEvents": 0,
            "totalEvents": 0,
        }
    threshold = percentile_value(candidate_values, percentile)
    separation = max(duration, gap_hours)
    filtered = [record for record in candidate_records if record["event"]["value"] >= threshold]
    filtered.sort(key=lambda item: item["event"]["value"], reverse=True)
    selected = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for record in filtered:
        year = int(record["year"])
        end_index = int(record["event"]["endIndex"])
        start_index = end_index - duration + 1
        blocked = False
        for existing_start, existing_end in occupied.get(year, []):
            if start_index <= existing_end + separation and end_index >= existing_start - separation:
                blocked = True
                break
        if blocked:
            continue
        occupied.setdefault(year, []).append((start_index, end_index))
        selected.append(record)
    selected.sort(key=lambda item: (int(item["year"]), int(item["event"]["endIndex"])))
    years_with_event = len({record["year"] for record in selected})
    event_rate = len(selected) / max(len(yearly_series), 1)
    return selected, {
        "method": "pds",
        "threshold": threshold,
        "thresholdPercentile": percentile,
        "independentGapHours": separation,
        "eventRate": event_rate,
        "event_rate": event_rate,
        "eventsPerYear": event_rate,
        "yearsWithEvents": years_with_event,
        "totalEvents": len(selected),
    }


def rolling_window_events(values: list[Any], duration: int) -> list[dict[str, Any]]:
    events = []
    total = 0.0
    missing = 0
    for index, raw in enumerate(values):
        if raw is None or not _is_finite(raw):
            missing += 1
        else:
            total += max(0.0, float(raw))
        if index >= duration:
            previous = values[index - duration]
            if previous is None or not _is_finite(previous):
                missing -= 1
            else:
                total -= max(0.0, float(previous))
        if index >= duration - 1 and missing == 0 and total > 0:
            events.append({"value": total, "endIndex": index})
    return events


def percentile_value(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = clamp((percentile / 100.0) * (len(ordered) - 1), 0.0, float(len(ordered) - 1))
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def probability_for_return_period(return_period: int, event_rate: float, series_method: str) -> float:
    if series_method == "pds":
        annual_exceedance = 1 / max(return_period, 1)
        safe_rate = max(event_rate, 1e-6)
        probability = 1 + math.log(max(1e-12, 1 - annual_exceedance)) / safe_rate
        return clamp(probability, 1e-6, 1 - 1e-6)
    return clamp(1 - 1 / max(return_period, 1), 1e-6, 1 - 1e-6)


def rolling_max_event(values: list[Any], duration: int) -> dict[str, Any]:
    total = 0.0
    missing = 0
    max_sum = float("-inf")
    end_index = -1
    for index, raw in enumerate(values):
        if raw is None or not _is_finite(raw):
            missing += 1
        else:
            total += max(0.0, float(raw))
        if index >= duration:
            previous = values[index - duration]
            if previous is None or not _is_finite(previous):
                missing -= 1
            else:
                total -= max(0.0, float(previous))
        if index >= duration - 1 and missing == 0 and total > max_sum:
            max_sum = total
            end_index = index
    if math.isfinite(max_sum):
        return {"value": max_sum, "endIndex": end_index}
    return {"value": None, "endIndex": -1}


def summarize_runoff_drivers(records: list[dict[str, Any]], duration: int) -> dict[str, Any]:
    rows = []
    for record in records:
        event_end = record["event"]["endIndex"]
        event_start = event_end - duration + 1
        before_event = event_start - 1
        top_soil = value_at(record["series"], "soil_moisture_0_to_7cm", event_end)
        mid_soil = value_at(record["series"], "soil_moisture_7_to_28cm", event_end)
        deep_soil = value_at(record["series"], "soil_moisture_28_to_100cm", event_end)
        lower_soil = value_at(record["series"], "soil_moisture_100_to_255cm", event_end)
        rows.append(
            {
                "eventRain": sum_window(record["series"], "rain", event_start, event_end),
                "eventSnowfall": sum_window(record["series"], "snowfall", event_start, event_end),
                "antecedentPrecip72h": sum_window(record["series"], "precipitation", event_start - 72, before_event),
                "soilMoistureTop": top_soil,
                "soilMoistureRoot": average_numbers([mid_soil, deep_soil, lower_soil]),
                "snowDepth": value_at(record["series"], "snow_depth", event_end),
            }
        )
    return {
        "eventRain": average_numbers([row["eventRain"] for row in rows]),
        "eventSnowfall": average_numbers([row["eventSnowfall"] for row in rows]),
        "antecedentPrecip72h": average_numbers([row["antecedentPrecip72h"] for row in rows]),
        "soilMoistureTop": average_numbers([row["soilMoistureTop"] for row in rows]),
        "soilMoistureRoot": average_numbers([row["soilMoistureRoot"] for row in rows]),
        "snowDepth": average_numbers([row["snowDepth"] for row in rows]),
    }


def value_at(series: dict[str, Any], key: str, index: int) -> float | None:
    values = series.get("hourly", {}).get(key)
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None
    value = values[index]
    return float(value) if _is_finite(value) else None


def sum_window(series: dict[str, Any], key: str, start: int, end: int) -> float | None:
    values = series.get("hourly", {}).get(key)
    if not isinstance(values, list) or end < 0:
        return None
    first = max(0, start)
    last = min(len(values) - 1, end)
    total = 0.0
    count = 0
    for index in range(first, last + 1):
        value = values[index]
        if _is_finite(value):
            total += float(value)
            count += 1
    return total if count else None


def average_numbers(values: list[float | None]) -> float | None:
    samples = [float(value) for value in values if value is not None and _is_finite(value)]
    return sum(samples) / len(samples) if samples else None


def compare_distributions(samples: list[float]) -> dict[str, dict[str, Any]]:
    diagnostics = {}
    for distribution in DISTRIBUTIONS:
        fit = fit_distribution(samples, distribution)
        diagnostics[distribution] = {
            **fit,
            "aic": calculate_aic(samples, fit),
            "ks": calculate_ks_statistic(samples, fit),
            "ksPValue": ks_p_value(len(samples), calculate_ks_statistic(samples, fit)),
            "ad": calculate_ad_statistic(samples, fit),
        }
    ranked = sorted(diagnostics.items(), key=lambda item: (item[1]["aic"], item[1]["ks"], item[1]["ad"]))
    for index, (distribution, values) in enumerate(ranked, start=1):
        values["rank"] = index
        values["distribution"] = distribution
    return diagnostics


def fit_distribution(samples: list[float], distribution: str) -> dict[str, Any]:
    if distribution == "gumbel":
        parameters = fit_gumbel(samples)
        return {
            "distribution": distribution,
            "parameters": parameters,
            "parameterText": f"mu {parameters['location']:.2f}, beta {parameters['scale']:.2f}",
            "parameterCount": 2,
        }
    if distribution == "gev":
        parameters = fit_gev_by_lmoments(samples)
        return {
            "distribution": distribution,
            "parameters": parameters,
            "parameterText": f"mu {parameters['location']:.2f}, sigma {parameters['scale']:.2f}, xi {-parameters['shape']:.3f}",
            "parameterCount": 3,
        }
    logs = [math.log10(value) for value in samples if value > 0]
    avg = mean(logs)
    sd = standard_deviation(logs)
    skew = skewness(logs)
    return {
        "distribution": distribution,
        "parameters": build_lp3_parameters(avg, sd, skew),
        "parameterText": f"log ort. {avg:.3f}, Cs {skew:.3f}",
        "parameterCount": 3,
    }


def estimate_quantile(samples: list[float], return_period: int, distribution: str) -> float:
    probability = 1 - 1 / return_period
    return quantile_from_fit(fit_distribution(samples, distribution), probability)


def describe_fit(samples: list[float], distribution: str) -> str:
    return fit_distribution(samples, distribution)["parameterText"]


def quantile_from_fit(fit: dict[str, Any], probability: float) -> float:
    distribution = fit["distribution"]
    parameters = fit["parameters"]
    probability = clamp(probability, 1e-6, 1 - 1e-6)
    if distribution == "gumbel":
        return gumbel_quantile(probability, parameters["location"], parameters["scale"])
    if distribution == "gev":
        return gev_quantile(probability, parameters["location"], parameters["scale"], parameters["shape"])
    return lp3_quantile_from_parameters(parameters, probability)


def calculate_aic(samples: list[float], fit: dict[str, Any]) -> float:
    log_likelihood = sum(log_pdf(fit, sample) for sample in samples if sample > 0)
    return 2 * fit["parameterCount"] - 2 * log_likelihood


def calculate_ks_statistic(samples: list[float], fit: dict[str, Any]) -> float:
    ordered = sorted(samples)
    n = len(ordered)
    max_diff = 0.0
    for index, sample in enumerate(ordered, start=1):
        cdf = clamp(cdf_value(fit, sample), 1e-12, 1 - 1e-12)
        max_diff = max(max_diff, abs(cdf - index / n), abs(cdf - (index - 1) / n))
    return max_diff


def ks_p_value(sample_size: int, ks_statistic: float) -> float:
    if sample_size <= 0:
        return 0.0
    z_value = (math.sqrt(sample_size) + 0.12 + 0.11 / math.sqrt(sample_size)) * ks_statistic
    total = 0.0
    for n_value in range(1, 8):
        total += ((-1) ** (n_value - 1)) * math.exp(-2 * (z_value ** 2) * (n_value ** 2))
    return clamp(2 * total, 0.0, 1.0)


def calculate_ad_statistic(samples: list[float], fit: dict[str, Any]) -> float:
    ordered = sorted(samples)
    n = len(ordered)
    if n == 0:
        return 0.0
    total = 0.0
    for index, sample in enumerate(ordered, start=1):
        cdf_low = clamp(cdf_value(fit, sample), 1e-12, 1 - 1e-12)
        cdf_high = clamp(cdf_value(fit, ordered[-index]), 1e-12, 1 - 1e-12)
        total += (2 * index - 1) * (math.log(cdf_low) + math.log(1 - cdf_high))
    return -n - total / n


def bootstrap_confidence_intervals(
    samples: list[float],
    distribution: str,
    probabilities: list[float],
    bootstrap_samples: int,
) -> tuple[list[float], list[float]]:
    rng = random.Random(42)
    boot_quantiles = [[] for _ in probabilities]
    resample_count = max(int(bootstrap_samples), 20)
    for _ in range(resample_count):
        resample = [samples[rng.randrange(len(samples))] for _ in range(len(samples))]
        fit = fit_distribution(resample, distribution)
        for index, probability in enumerate(probabilities):
            boot_quantiles[index].append(quantile_from_fit(fit, probability))
    lower_rank = (1 - CONFIDENCE_LEVEL) / 2 * 100
    upper_rank = (1 + CONFIDENCE_LEVEL) / 2 * 100
    lower = [percentile_value(values, lower_rank) for values in boot_quantiles]
    upper = [percentile_value(values, upper_rank) for values in boot_quantiles]
    return lower, upper


def cdf_value(fit: dict[str, Any], sample: float) -> float:
    distribution = fit["distribution"]
    parameters = fit["parameters"]
    if distribution == "gumbel":
        z_value = (sample - parameters["location"]) / max(parameters["scale"], 1e-12)
        return math.exp(-math.exp(-z_value))
    if distribution == "gev":
        return gev_cdf(sample, parameters["location"], parameters["scale"], parameters["shape"])
    return lp3_cdf(sample, parameters)


def log_pdf(fit: dict[str, Any], sample: float) -> float:
    distribution = fit["distribution"]
    parameters = fit["parameters"]
    sample = max(sample, 1e-12)
    if distribution == "gumbel":
        z_value = (sample - parameters["location"]) / max(parameters["scale"], 1e-12)
        return -math.log(max(parameters["scale"], 1e-12)) - z_value - math.exp(-z_value)
    if distribution == "gev":
        return gev_log_pdf(sample, parameters["location"], parameters["scale"], parameters["shape"])
    return lp3_log_pdf(sample, parameters)


def fit_gumbel(samples: list[float]) -> dict[str, float]:
    avg = mean(samples)
    sd = standard_deviation(samples)
    scale = max((sd * math.sqrt(6)) / math.pi, 0.000001)
    location = avg - 0.5772156649015329 * scale
    return {"location": location, "scale": scale}


def gumbel_quantile(probability: float, location: float, scale: float) -> float:
    return location - scale * math.log(-math.log(probability))


def fit_gev_by_lmoments(samples: list[float]) -> dict[str, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n < 4:
        fallback = fit_gumbel(samples)
        return {**fallback, "shape": 0.0}
    b0 = mean(sorted_samples)
    b1 = 0.0
    b2 = 0.0
    for i, sample in enumerate(sorted_samples):
        b1 += (i / (n - 1)) * sample
        b2 += ((i * (i - 1)) / ((n - 1) * (n - 2))) * sample
    b1 /= n
    b2 /= n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    if l2 <= 0:
        fallback = fit_gumbel(samples)
        return {**fallback, "shape": 0.0}
    tau3 = clamp(l3 / l2, -0.95, 0.95)
    c_value = 2 / (3 + tau3) - math.log(2) / math.log(3)
    shape = clamp(7.859 * c_value + 2.9554 * c_value * c_value, -0.95, 0.95)
    if abs(shape) < 0.000001:
        scale = l2 / math.log(2)
        location = l1 - 0.5772156649015329 * scale
        return {"location": location, "scale": scale, "shape": 0.0}
    gamma_term = gamma(1 + shape)
    scale = (l2 * shape) / ((1 - math.pow(2, -shape)) * gamma_term)
    location = l1 - (scale * (1 - gamma_term)) / shape
    return {"location": location, "scale": max(scale, 0.000001), "shape": shape}


def gev_quantile(probability: float, location: float, scale: float, shape: float) -> float:
    y_value = -math.log(probability)
    if abs(shape) < 0.000001:
        return location - scale * math.log(y_value)
    return location + (scale * (1 - math.pow(y_value, shape))) / shape


def gev_cdf(sample: float, location: float, scale: float, shape: float) -> float:
    if scale <= 0:
        return 0.0
    z_value = (sample - location) / scale
    if abs(shape) < 0.000001:
        return math.exp(-math.exp(-z_value))
    t_value = 1 - shape * z_value
    if t_value <= 0:
        return 0.0 if shape < 0 else 1.0
    return math.exp(-math.pow(t_value, 1 / shape))


def gev_log_pdf(sample: float, location: float, scale: float, shape: float) -> float:
    if scale <= 0:
        return float("-inf")
    z_value = (sample - location) / scale
    if abs(shape) < 0.000001:
        return -math.log(scale) - z_value - math.exp(-z_value)
    t_value = 1 - shape * z_value
    if t_value <= 0:
        return float("-inf")
    return -math.log(scale) + (1 / shape - 1) * math.log(t_value) - math.pow(t_value, 1 / shape)


def log_pearson_3_quantile(samples: list[float], probability: float) -> float:
    logs = [math.log10(value) for value in samples if value > 0]
    avg = mean(logs)
    sd = standard_deviation(logs)
    skew = skewness(logs)
    return lp3_quantile_from_parameters(build_lp3_parameters(avg, sd, skew), probability)


def build_lp3_parameters(avg: float, sd: float, skew: float) -> dict[str, float]:
    parameters = {"mean_log": avg, "sd_log": sd, "skew_log": skew}
    if sd == 0:
        return {**parameters, "mode": "degenerate", "alpha": 0.0, "beta": 0.0, "location": avg}
    if abs(skew) < 0.0001:
        return {**parameters, "mode": "normal", "alpha": 0.0, "beta": 0.0, "location": avg}
    alpha = 4 / (skew * skew)
    beta = (sd * skew) / 2
    location = avg - beta * alpha
    return {**parameters, "mode": "gamma", "alpha": alpha, "beta": beta, "location": location}


def lp3_quantile_from_parameters(parameters: dict[str, float], probability: float) -> float:
    probability = clamp(probability, 1e-6, 1 - 1e-6)
    if parameters["mode"] == "degenerate":
        return math.pow(10, parameters["mean_log"])
    if parameters["mode"] == "normal":
        return math.pow(10, parameters["mean_log"] + parameters["sd_log"] * inverse_normal(probability))
    beta = parameters["beta"]
    gamma_probability = probability if beta > 0 else 1 - probability
    return math.pow(
        10,
        parameters["location"] + beta * inverse_regularized_gamma_p(parameters["alpha"], gamma_probability),
    )


def lp3_cdf(sample: float, parameters: dict[str, float]) -> float:
    if sample <= 0:
        return 0.0
    y_value = math.log10(sample)
    if parameters["mode"] == "degenerate":
        return 1.0 if y_value >= parameters["mean_log"] else 0.0
    if parameters["mode"] == "normal":
        z_value = (y_value - parameters["mean_log"]) / max(parameters["sd_log"], 1e-12)
        return 0.5 * (1 + math.erf(z_value / math.sqrt(2)))
    beta = parameters["beta"]
    z_value = (y_value - parameters["location"]) / beta
    if z_value <= 0:
        return 0.0 if beta > 0 else 1.0
    gamma_cdf = regularized_gamma_p(parameters["alpha"], z_value)
    return gamma_cdf if beta > 0 else 1 - gamma_cdf


def lp3_log_pdf(sample: float, parameters: dict[str, float]) -> float:
    if sample <= 0:
        return float("-inf")
    y_value = math.log10(sample)
    if parameters["mode"] == "degenerate":
        return 0.0 if abs(y_value - parameters["mean_log"]) < 1e-12 else float("-inf")
    if parameters["mode"] == "normal":
        sd = max(parameters["sd_log"], 1e-12)
        z_value = (y_value - parameters["mean_log"]) / sd
        return -math.log(sample * math.log(10) * sd * math.sqrt(2 * math.pi)) - 0.5 * z_value * z_value
    beta = parameters["beta"]
    z_value = (y_value - parameters["location"]) / beta
    if z_value <= 0:
        return float("-inf")
    return (
        -log_gamma(parameters["alpha"])
        + (parameters["alpha"] - 1) * math.log(z_value)
        - z_value
        - math.log(abs(beta))
        - math.log(sample * math.log(10))
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    avg = mean(values)
    sd = standard_deviation(values)
    if sd == 0:
        return 0.0
    n = len(values)
    third_moment = sum(((value - avg) / sd) ** 3 for value in values)
    return (n / ((n - 1) * (n - 2))) * third_moment


def log_gamma(value: float) -> float:
    coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ]
    if value < 0.5:
        return math.log(math.pi) - math.log(math.sin(math.pi * value)) - log_gamma(1 - value)
    x_value = 0.9999999999998099
    shifted = value - 1
    for index, coefficient in enumerate(coefficients):
        x_value += coefficient / (shifted + index + 1)
    t_value = shifted + len(coefficients) - 0.5
    return 0.5 * math.log(2 * math.pi) + (shifted + 0.5) * math.log(t_value) - t_value + math.log(x_value)


def gamma(value: float) -> float:
    return math.exp(log_gamma(value))


def regularized_gamma_p(a_value: float, x_value: float) -> float:
    if x_value <= 0:
        return 0.0
    if x_value < a_value + 1:
        ap_value = a_value
        total = 1 / a_value
        delta = total
        for n_value in range(1, 101):
            ap_value += 1
            delta *= x_value / ap_value
            total += delta
            if abs(delta) < abs(total) * 1e-12:
                break
        return total * math.exp(-x_value + a_value * math.log(x_value) - log_gamma(a_value))
    b_value = x_value + 1 - a_value
    c_value = 1 / 1e-30
    d_value = 1 / b_value
    h_value = d_value
    for i_value in range(1, 101):
        an_value = -i_value * (i_value - a_value)
        b_value += 2
        d_value = an_value * d_value + b_value
        if abs(d_value) < 1e-30:
            d_value = 1e-30
        c_value = b_value + an_value / c_value
        if abs(c_value) < 1e-30:
            c_value = 1e-30
        d_value = 1 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1) < 1e-12:
            break
    return 1 - math.exp(-x_value + a_value * math.log(x_value) - log_gamma(a_value)) * h_value


def inverse_regularized_gamma_p(a_value: float, probability: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return float("inf")
    z_value = inverse_normal(probability)
    guess = a_value * math.pow(max(0.05, 1 - 1 / (9 * a_value) + z_value / (3 * math.sqrt(a_value))), 3)
    low = 0.0
    high = max(a_value + 12 * math.sqrt(a_value), guess * 2, 1)
    while regularized_gamma_p(a_value, high) < probability:
        high *= 2
        if high > 1e8:
            break
    guess = clamp(guess, low + 1e-10, high - 1e-10)
    for _ in range(32):
        cdf = regularized_gamma_p(a_value, guess)
        error = cdf - probability
        if abs(error) < 1e-10:
            return guess
        if error < 0:
            low = guess
        else:
            high = guess
        pdf = math.exp((a_value - 1) * math.log(guess) - guess - log_gamma(a_value))
        newton = guess - error / max(pdf, 1e-300)
        guess = newton if math.isfinite(newton) and low < newton < high else (low + high) / 2
    return guess


def inverse_normal(probability: float) -> float:
    a_values = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
    b_values = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c_values = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_values = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1.")
    p_low = 0.02425
    p_high = 1 - p_low
    if probability < p_low:
        q_value = math.sqrt(-2 * math.log(probability))
        return (
            (((((c_values[0] * q_value + c_values[1]) * q_value + c_values[2]) * q_value + c_values[3]) * q_value + c_values[4]) * q_value + c_values[5])
            / ((((d_values[0] * q_value + d_values[1]) * q_value + d_values[2]) * q_value + d_values[3]) * q_value + 1)
        )
    if probability <= p_high:
        q_value = probability - 0.5
        r_value = q_value * q_value
        return (
            (((((a_values[0] * r_value + a_values[1]) * r_value + a_values[2]) * r_value + a_values[3]) * r_value + a_values[4]) * r_value + a_values[5])
            * q_value
            / (((((b_values[0] * r_value + b_values[1]) * r_value + b_values[2]) * r_value + b_values[3]) * r_value + b_values[4]) * r_value + 1)
        )
    q_value = math.sqrt(-2 * math.log(1 - probability))
    return -(
        (((((c_values[0] * q_value + c_values[1]) * q_value + c_values[2]) * q_value + c_values[3]) * q_value + c_values[4]) * q_value + c_values[5])
        / ((((d_values[0] * q_value + d_values[1]) * q_value + d_values[2]) * q_value + d_values[3]) * q_value + 1)
    )


def _first_display_segment(display_name: Any) -> str:
    return str(display_name).split(",")[0].strip() if isinstance(display_name, str) else ""


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
