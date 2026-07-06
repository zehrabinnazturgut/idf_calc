from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

DISTRIBUTIONS = {
    "lp3": "Log Pearson III",
    "gumbel": "Gumbel",
    "gev": "GEV",
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
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    query_key = {
        "lat": round(float(location["latitude"]), 4),
        "lon": round(float(location["longitude"]), 4),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "distribution": distribution,
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
    result = build_analysis(series, location, distribution, start_year, end_year)
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
) -> dict[str, Any]:
    durations = []
    for duration in DURATIONS:
        events = []
        for series in yearly_series:
            event = rolling_max_event(series["values"], duration)
            if event["value"] is not None and event["value"] > 0:
                events.append({"series": series, "event": event})
        maxima = [record["event"]["value"] for record in events]
        if len(maxima) < 8:
            raise RuntimeError(f"{duration} saat süresi için yeterli yıllık maksimum seri oluşmadı.")
        depths = [estimate_quantile(maxima, period, distribution) for period in RETURN_PERIODS]
        durations.append(
            {
                "duration": duration,
                "maxima": maxima,
                "depths": depths,
                "intensities": [depth / duration for depth in depths],
                "sampleMean": mean(maxima),
                "sampleMax": max(maxima),
                "parameterText": describe_fit(maxima, distribution),
                "runoffSummary": summarize_runoff_drivers(events, duration),
            }
        )

    return {
        "location": location,
        "distribution": distribution,
        "startYear": start_year,
        "endYear": end_year,
        "returnPeriods": RETURN_PERIODS,
        "durations": durations,
        "totalHours": sum(series["hours"] for series in yearly_series),
        "missingHours": sum(series["missing"] for series in yearly_series),
        "yearsUsed": len(yearly_series),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


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


def estimate_quantile(samples: list[float], return_period: int, distribution: str) -> float:
    probability = 1 - 1 / return_period
    if distribution == "gumbel":
        stats = fit_gumbel(samples)
        return gumbel_quantile(probability, stats["location"], stats["scale"])
    if distribution == "gev":
        stats = fit_gev_by_lmoments(samples)
        return gev_quantile(probability, stats["location"], stats["scale"], stats["shape"])
    return log_pearson_3_quantile(samples, probability)


def describe_fit(samples: list[float], distribution: str) -> str:
    if distribution == "gumbel":
        fit = fit_gumbel(samples)
        return f"mu {fit['location']:.2f}, beta {fit['scale']:.2f}"
    if distribution == "gev":
        fit = fit_gev_by_lmoments(samples)
        return f"mu {fit['location']:.2f}, sigma {fit['scale']:.2f}, xi {-fit['shape']:.3f}"
    logs = [math.log10(value) for value in samples]
    return f"log ort. {mean(logs):.3f}, Cs {skewness(logs):.3f}"


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


def log_pearson_3_quantile(samples: list[float], probability: float) -> float:
    logs = [math.log10(value) for value in samples if value > 0]
    avg = mean(logs)
    sd = standard_deviation(logs)
    skew = skewness(logs)
    if sd == 0:
        return math.pow(10, avg)
    if abs(skew) < 0.0001:
        return math.pow(10, avg + sd * inverse_normal(probability))
    alpha = 4 / (skew * skew)
    beta = (sd * skew) / 2
    location = avg - beta * alpha
    gamma_probability = probability if beta > 0 else 1 - probability
    return math.pow(10, location + beta * inverse_regularized_gamma_p(alpha, gamma_probability))


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
