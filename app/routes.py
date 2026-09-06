import csv
import io
import json
import logging
import math
import os
import random
import time
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from cachetools import TTLCache

from flask import (
    Blueprint,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .services.auth_service import get_user_by_id
from .models import ReadinessQuizResult
from .services.history_service import list_analysis_for_user, save_analysis_for_user


bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

READINESS_PASS_SCORE = 7
READINESS_TOTAL_QUESTIONS = 10
READINESS_QUIZ_SESSION_KEY = "readiness_quiz_questions"
HORIZON_MAX_ASSETS = 5
HORIZON_DEFAULT_HORIZON_YEARS = 3
HORIZON_MAX_HORIZON_YEARS = 5
HORIZON_WARNING_KEY = "horizon_disclaimer_ack"
HORIZON_DISCLAIMER_TEXT = (
    "Esta simulación no predice el futuro. Se trata de una proyección experimental generada a partir de patrones históricos. "
    "En inversión, los resultados pasados no garantizan resultados futuros. Esta herramienta tiene finalidad educativa y demostrativa, "
    "no debe usarse para tomar decisiones financieras reales."
)
HORIZON_METHOD_DESCRIPTION = (
    "Este modo remezcla patrones de rentabilidad histórica para construir una trayectoria futura hipotética. "
    "No calcula lo que va a ocurrir, sino un escenario experimental posible dentro de una simulación educativa. "
    "Para construir el escenario se utiliza una muestra de rentabilidades históricas suficientemente amplia cuando está disponible. "
    "Aun así, el resultado es solo una trayectoria hipotética y no una estimación fiable del futuro."
)
HORIZON_MIN_HISTORY_YEARS = 5
HORIZON_DEFAULT_HISTORY_YEARS = 10
HORIZON_MAX_HISTORY_YEARS = 15
HORIZON_MAX_HISTORY_POINTS = 220
HORIZON_MAX_MONTHLY_RETURN = 0.35
HORIZON_HISTORY_CACHE_TTL_SECONDS = 120
HORIZON_HISTORY_RETRY_DELAYS_SECONDS = (0.0, 0.8, 1.6)
HORIZON_HISTORY_CACHE: TTLCache = TTLCache(
    maxsize=128, ttl=HORIZON_HISTORY_CACHE_TTL_SECONDS
)
AI_TUTOR_DISCLAIMER = (
    "Este análisis tiene finalidad educativa y se basa únicamente en los datos de la simulación. "
    "No constituye asesoramiento financiero ni una recomendación de inversión real."
)
AI_TUTOR_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
AI_TUTOR_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS") or 18)
AI_TUTOR_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS") or 800)
AI_TUTOR_MAX_WARNINGS = 6
READINESS_QUIZ_QUESTIONS = [
    {
        "id": "risk_return",
        "prompt": "¿Qué suele ocurrir cuando una inversión ofrece potencial de rentabilidad más alto?",
        "options": [
            "Normalmente también implica más riesgo.",
            "Garantiza beneficios sin caídas.",
            "Siempre bate al benchmark.",
            "Reduce automáticamente la volatilidad.",
        ],
        "correctIndex": 0,
        "explanation": "Mayor rentabilidad esperada suele venir acompañada de mayor incertidumbre y oscilación.",
        "topic": "riesgo-rentabilidad",
    },
    {
        "id": "diversification",
        "prompt": "¿Cuál es el principal objetivo de diversificar una cartera?",
        "options": [
            "Reducir el impacto de un único activo o sector.",
            "Eliminar por completo el riesgo.",
            "Duplicar siempre la rentabilidad.",
            "Evitar comparar con un benchmark.",
        ],
        "correctIndex": 0,
        "explanation": "Diversificar ayuda a no depender demasiado de una sola posición, aunque no elimina todo el riesgo.",
        "topic": "diversificación",
    },
    {
        "id": "benchmark",
        "prompt": "En esta aplicación, ¿para qué sirve el benchmark?",
        "options": [
            "Para comparar el comportamiento de tu cartera frente a una referencia.",
            "Para fijar automáticamente el precio de compra.",
            "Para ocultar la volatilidad del portfolio.",
            "Para guardar sesiones en el historial.",
        ],
        "correctIndex": 0,
        "explanation": "El benchmark permite ver si tu cartera lo hace mejor, peor o parecido a una referencia de mercado.",
        "topic": "benchmark",
    },
    {
        "id": "volatility",
        "prompt": "¿Qué describe mejor la volatilidad?",
        "options": [
            "La intensidad con la que el valor de una inversión sube y baja en el tiempo.",
            "El capital inicial invertido.",
            "La rentabilidad acumulada garantizada.",
            "La cantidad de turnos del modo carrera.",
        ],
        "correctIndex": 0,
        "explanation": "La volatilidad mide la variabilidad de los precios o rendimientos, no si algo es bueno o malo por sí solo.",
        "topic": "volatilidad",
    },
    {
        "id": "dca",
        "prompt": "¿Qué representa DCA o inversión periódica en la app?",
        "options": [
            "Aportar cantidades periódicas para repartir el punto de entrada en el tiempo.",
            "Comprar solo cuando el benchmark cae.",
            "Una técnica para eliminar drawdowns.",
            "Un modo de exportar el informe final.",
        ],
        "correctIndex": 0,
        "explanation": "DCA reparte las compras en el tiempo y puede suavizar el riesgo de entrar todo en un solo punto.",
        "topic": "dca",
    },
    {
        "id": "drawdown",
        "prompt": "¿Qué indica un drawdown en el informe final?",
        "options": [
            "La caída desde un máximo previo hasta un mínimo posterior.",
            "La rentabilidad anual compuesta exacta.",
            "El número de operaciones realizadas.",
            "El peso del benchmark en la cartera.",
        ],
        "correctIndex": 0,
        "explanation": "El drawdown ayuda a entender cuánto llegó a retroceder una estrategia desde su mejor punto anterior.",
        "topic": "drawdown",
    },
    {
        "id": "simulation_vs_real",
        "prompt": "¿Qué diferencia clave existe entre esta app y una inversión real?",
        "options": [
            "La app simula escenarios con datos históricos y no ejecuta operaciones reales.",
            "La app garantiza resultados futuros.",
            "La app elimina los riesgos de mercado.",
            "La app obliga a comprar acciones reales al cerrar un turno.",
        ],
        "correctIndex": 0,
        "explanation": "La herramienta es educativa: compara escenarios y decisiones, pero no invierte dinero real.",
        "topic": "simulación",
    },
    {
        "id": "career_turns",
        "prompt": "¿Qué implica tomar decisiones por turnos en el Modo Carrera?",
        "options": [
            "Ajustar la cartera en distintos tramos históricos y observar cómo evoluciona.",
            "Repetir siempre la misma asignación sin contexto.",
            "Ignorar los eventos y el benchmark.",
            "Bloquear el historial del usuario.",
        ],
        "correctIndex": 0,
        "explanation": "El Modo Carrera divide el periodo en fases para que tomes decisiones y veas su impacto acumulado.",
        "topic": "modo-carrera",
    },
    {
        "id": "final_report",
        "prompt": "En el informe final, ¿qué comparan Portfolio, Benchmark y Tracking?",
        "options": [
            "El resultado de tu cartera, la referencia de mercado y la diferencia entre ambos.",
            "Tres formas distintas de guardar la sesión.",
            "El capital inicial, el capital final y el correo del usuario.",
            "La teoría, el historial y el login.",
        ],
        "correctIndex": 0,
        "explanation": "Portfolio resume tu estrategia, Benchmark la referencia y Tracking cómo te separas de ella.",
        "topic": "informe-final",
    },
    {
        "id": "auth_history",
        "prompt": "¿Qué ventaja principal tiene usar una cuenta autenticada frente al modo invitado?",
        "options": [
            "Conservar historial y progreso, incluido el acceso al Modo Carrera, entre sesiones.",
            "Eliminar automáticamente la volatilidad.",
            "Obtener una rentabilidad mejor en el informe.",
            "Acceder a precios futuros reales.",
        ],
        "correctIndex": 0,
        "explanation": "La autenticación permite persistir historial, sesiones de carrera y el aprobado del test entre accesos.",
        "topic": "usuarios-autenticados",
    },
]


# ----------------------
#   Vistas HTML
# ----------------------
@bp.get("/")
def home():
    if not _current_user_id() and not session.get("guest"):
        return redirect(url_for("auth.login_page"))
    current_user = get_user_by_id(_current_user_id()) if _current_user_id() else None
    return render_template(
        "home.html", active="home", nav_mode="landing", current_user=current_user
    )


@bp.get("/inicio")
def inicio_alias():
    return render_template("inicio.html", active="inicio", nav_mode="practice")


@bp.get("/empresas")
def empresas_page():
    accept = request.accept_mimetypes
    wants_json = request.args.get("format") == "json" or (
        accept.best == "application/json"
        or accept["application/json"] >= accept["text/html"]
    )
    if wants_json and request.args.get("format") != "html":
        return listar_empresas()
    return render_template("empresas.html", active="empresas", nav_mode="practice")


@bp.get("/nuevo-analisis")
def analisis_page():
    return render_template("analisis.html", active="analisis", nav_mode="practice")


@bp.get("/historial")
def historial_page():
    if _is_guest_user():
        return redirect(url_for("main.home"))
    if not _current_user_id():
        return redirect(url_for("auth.login_page"))
    return render_template("historial.html", active="historial", nav_mode="practice")


def _current_user_id() -> int | None:
    raw = session.get("user_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _is_guest_user() -> bool:
    return bool(session.get("guest")) and not bool(session.get("user_id"))


def _build_readiness_question_set() -> list[dict]:
    prepared = []
    for item in READINESS_QUIZ_QUESTIONS:
        options = []
        for index, label in enumerate(item["options"]):
            options.append(
                {
                    "id": f"{item['id']}:opt:{index}",
                    "label": label,
                    "correct": index == int(item["correctIndex"]),
                }
            )
        random.shuffle(options)
        prepared.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "options": options,
                "explanation": item["explanation"],
                "topic": item["topic"],
            }
        )
    random.shuffle(prepared)
    return prepared


def _get_or_create_readiness_question_set(force_new: bool = False) -> list[dict]:
    stored = session.get(READINESS_QUIZ_SESSION_KEY)
    if force_new or not stored:
        stored = _build_readiness_question_set()
        session[READINESS_QUIZ_SESSION_KEY] = stored
        session.modified = True
    return stored


def _clear_readiness_question_set() -> None:
    session.pop(READINESS_QUIZ_SESSION_KEY, None)
    session.modified = True


def _readiness_status_payload() -> dict:
    current_user_id = _current_user_id()
    if current_user_id:
        record = (
            ReadinessQuizResult.select()
            .where(ReadinessQuizResult.user == current_user_id)
            .first()
        )
        passed = bool(record.passed) if record else False
        return {
            "passed": passed,
            "score": record.score if record else 0,
            "total_questions": (
                record.total_questions if record else READINESS_TOTAL_QUESTIONS
            ),
            "pass_score": READINESS_PASS_SCORE,
            "storage": "server",
            "user_authenticated": True,
            "guest": False,
            "passed_at": (
                record.passed_at.isoformat() + "Z"
                if record and record.passed_at
                else None
            ),
        }

    guest_payload = session.get("readiness_guest") or {}
    passed = bool(guest_payload.get("passed"))
    return {
        "passed": passed,
        "score": int(guest_payload.get("score") or 0),
        "total_questions": int(
            guest_payload.get("total_questions") or READINESS_TOTAL_QUESTIONS
        ),
        "pass_score": READINESS_PASS_SCORE,
        "storage": "session",
        "user_authenticated": False,
        "guest": _is_guest_user(),
        "passed_at": guest_payload.get("passed_at"),
    }


@bp.get("/aprende")
def aprende_page():
    return render_template(
        "aprende.html",
        active="aprende",
        nav_mode="practice",
        readiness_status=_readiness_status_payload(),
        readiness_pass_score=READINESS_PASS_SCORE,
        readiness_total_questions=READINESS_TOTAL_QUESTIONS,
    )


@bp.get("/manual")
def manual_page():
    return render_template("manual.html", active="manual", nav_mode="manual")


def _horizon_ack_key() -> str:
    user_id = _current_user_id()
    if user_id:
        return f"user:{user_id}"
    if _is_guest_user():
        return "guest"
    return "anon"


def _horizon_acknowledged() -> bool:
    payload = session.get(HORIZON_WARNING_KEY) or {}
    if not isinstance(payload, dict):
        return False
    return bool(payload.get(_horizon_ack_key()))


def _set_horizon_acknowledged() -> None:
    payload = session.get(HORIZON_WARNING_KEY) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload[_horizon_ack_key()] = True
    session[HORIZON_WARNING_KEY] = payload
    session.modified = True


def _normalize_horizon_weights(assets: list[dict]) -> list[dict]:
    cleaned = []
    total_weight = 0.0
    for asset in assets:
        ticker = str(asset.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            weight = float(asset.get("weight") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight < 0:
            weight = 0.0
        cleaned.append({"ticker": ticker, "weight": weight})
        total_weight += weight

    if not cleaned:
        return []

    if total_weight <= 0:
        even_weight = round(1 / len(cleaned), 6)
        return [{**asset, "weight": even_weight} for asset in cleaned]

    return [{**asset, "weight": asset["weight"] / total_weight} for asset in cleaned]


def _horizon_identity_payload() -> dict:
    return {
        "acknowledged": _horizon_acknowledged(),
        "disclaimer": HORIZON_DISCLAIMER_TEXT,
        "method_description": HORIZON_METHOD_DESCRIPTION,
        "max_assets": HORIZON_MAX_ASSETS,
        "default_horizon_years": HORIZON_DEFAULT_HORIZON_YEARS,
        "max_horizon_years": HORIZON_MAX_HORIZON_YEARS,
    }


@bp.get("/modo-horizonte")
def horizon_page():
    return render_template(
        "horizon.html",
        active="horizon",
        nav_mode="practice",
        horizon_config=_horizon_identity_payload(),
    )


@bp.post("/api/horizon/disclaimer/accept")
def horizon_accept_disclaimer_api():
    _set_horizon_acknowledged()
    return jsonify({"ok": True, **_horizon_identity_payload()})


@bp.get("/api/horizon/from-career/<session_id>")
def horizon_from_career_api(session_id: str):
    from .career import _resolve_session_for_request, _session_analysis_range

    session_payload = _resolve_session_for_request(session_id)
    if not session_payload:
        return (
            jsonify({"error": "No se pudo acceder a la sesión de carrera indicada."}),
            404,
        )

    warnings: list[str] = []

    latest_alloc = []
    completed_turns = session_payload.get("completed_turns") or []
    if completed_turns:
        last_snapshot = completed_turns[-1] or {}
        latest_alloc = last_snapshot.get("alloc") or []
    if not latest_alloc:
        decisions = session_payload.get("decisions") or []
        if decisions:
            latest_alloc = (decisions[-1] or {}).get("alloc") or []

    assets = []
    for position in latest_alloc:
        ticker = str(position.get("ticker") or "").strip().upper()
        if not ticker or ticker == "CASH":
            continue
        try:
            weight = float(position.get("weight") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        assets.append({"ticker": ticker, "weight": weight})

    if not assets:
        portfolio = session_payload.get("portfolio") or {}
        positions = portfolio.get("positions") or []
        for position in positions:
            ticker = str(position.get("ticker") or "").strip().upper()
            if not ticker or ticker == "CASH":
                continue
            try:
                weight = float(position.get("weight") or 0)
            except (TypeError, ValueError):
                weight = 0.0
            assets.append({"ticker": ticker, "weight": weight})

    had_explicit_weights = any(float(item.get("weight") or 0) > 0 for item in assets)
    normalized_assets = _normalize_horizon_weights(assets)[:HORIZON_MAX_ASSETS]
    if normalized_assets and not had_explicit_weights:
        warnings.append(
            "No se encontraron pesos finales exactos; se han usado pesos equivalentes."
        )

    tickers = [item["ticker"] for item in normalized_assets]
    weights = [round(float(item["weight"]), 6) for item in normalized_assets]

    final_value = (
        session_payload.get("capital_current")
        or session_payload.get("capital")
        or session_payload.get("capital_initial")
    )
    if completed_turns:
        final_value = (completed_turns[-1] or {}).get("portfolio_value") or final_value

    try:
        initial_value = float(final_value)
    except (TypeError, ValueError):
        initial_value = 10000.0
        warnings.append(
            "No se encontró valor final de cartera; se usa valor inicial por defecto."
        )
    else:
        if initial_value <= 0:
            initial_value = 10000.0
            warnings.append(
                "No se encontró valor final de cartera; se usa valor inicial por defecto."
            )

    try:
        _start_d, projection_end_d, career_period_start, career_period_end = (
            _session_analysis_range(session_payload)
        )
    except Exception:
        period = session_payload.get("period") or {}
        career_period_start = str(period.get("start") or "")
        career_period_end = str(period.get("end") or career_period_start or "")
        projection_end_d = None

    projection_start = (
        projection_end_d.isoformat()
        if projection_end_d is not None
        else (career_period_end or None)
    )

    return jsonify(
        {
            "ok": True,
            "source": "career",
            "session_id": session_id,
            "display_name": "Continuación desde Modo Carrera",
            "tickers": tickers,
            "weights": weights,
            "assets": normalized_assets,
            "initial_value": max(initial_value, 1000.0),
            "career_period_start": career_period_start or None,
            "career_period_end": career_period_end or None,
            "projection_start": projection_start,
            "warnings": warnings,
            "disclaimer": HORIZON_DISCLAIMER_TEXT,
            "method_description": HORIZON_METHOD_DESCRIPTION,
        }
    )


@bp.post("/api/horizon/simulate")
def horizon_simulate_api():
    payload = request.get_json(silent=True) or {}
    tickers_raw = payload.get("tickers")
    weights_raw = payload.get("weights")
    assets_raw = payload.get("assets")
    source = str(payload.get("source") or "manual").strip().lower() or "manual"
    session_id = str(payload.get("session_id") or "").strip()
    projection_start_raw = str(payload.get("projection_start") or "").strip()

    assets_input = []
    if isinstance(assets_raw, list) and assets_raw:
        assets_input = assets_raw
    elif isinstance(tickers_raw, list):
        if isinstance(weights_raw, list) and len(weights_raw) == len(tickers_raw):
            assets_input = [
                {"ticker": ticker, "weight": weight}
                for ticker, weight in zip(tickers_raw, weights_raw)
            ]
        else:
            assets_input = [{"ticker": item} for item in tickers_raw]

    assets = _normalize_horizon_weights(assets_input)
    if not assets:
        return (
            jsonify(
                {
                    "error": "Debes seleccionar al menos un activo válido para generar el escenario experimental."
                }
            ),
            400,
        )
    if len(assets) > HORIZON_MAX_ASSETS:
        return (
            jsonify(
                {
                    "error": f"El modo Horizonte admite como máximo {HORIZON_MAX_ASSETS} activos en esta versión."
                }
            ),
            400,
        )

    try:
        horizon_years = int(payload.get("horizon") or HORIZON_DEFAULT_HORIZON_YEARS)
    except (TypeError, ValueError):
        horizon_years = 0
    if horizon_years < 1 or horizon_years > HORIZON_MAX_HORIZON_YEARS:
        return (
            jsonify(
                {
                    "error": f"Selecciona un horizonte válido entre 1 y {HORIZON_MAX_HORIZON_YEARS} años."
                }
            ),
            400,
        )

    try:
        initial_value = float(payload.get("initial_value") or 10000)
    except (TypeError, ValueError):
        initial_value = 10000.0
    if initial_value <= 0:
        return jsonify({"error": "El valor inicial debe ser mayor que cero."}), 400

    if source == "career" and session_id:
        from .career import _resolve_session_for_request

        if not _resolve_session_for_request(session_id):
            return (
                jsonify(
                    {
                        "error": "No puedes usar una sesión de carrera ajena o inexistente como origen."
                    }
                ),
                404,
            )

    projection_start = None
    if projection_start_raw:
        try:
            projection_start = date.fromisoformat(projection_start_raw)
        except ValueError:
            projection_start = None

    end_d = projection_start or date.today()
    history_years = _get_horizon_history_years(horizon_years)
    start_d = end_d - timedelta(days=365 * history_years)
    warnings = []
    valid_assets = []
    monthly_returns = []
    provider_temporarily_limited = False
    had_clamped_outliers = False

    for asset in assets:
        ticker = asset["ticker"]
        try:
            df = _download_history_df(ticker, start_d, end_d, include_actions=False)
            price_series = _extract_market_price_series(df, ticker)
            monthly, return_meta = _compute_horizon_monthly_returns(
                price_series, ticker
            )
        except BacktestError as exc:
            if exc.status_code >= 500:
                provider_temporarily_limited = True
            warnings.append(str(exc))
            continue
        except HorizonSimulationError as exc:
            warnings.append(str(exc))
            continue
        except Exception:
            warnings.append(
                f"{ticker} se ha excluido porque no se pudo normalizar su histórico de mercado."
            )
            continue

        min_required_points = max(36, horizon_years * 12)
        if (
            price_series.empty
            or len(price_series) < 252
            or len(monthly) < min_required_points
        ):
            warnings.append(
                f"{ticker} se ha excluido porque no dispone de una muestra histórica suficientemente amplia para este horizonte experimental."
            )
            continue

        if return_meta.get("had_outliers"):
            had_clamped_outliers = True
            warnings.append(
                f"{ticker} contiene retornos mensuales extremos en el histórico reciente. Se ha limitado su impacto para evitar una proyección experimental absurda."
            )

        valid_assets.append(
            {
                "ticker": ticker,
                "weight": asset["weight"],
                "series": price_series,
                "monthly_points": len(monthly),
            }
        )
        monthly_returns.append(monthly.rename(ticker))

    if not valid_assets:
        status_code = 503 if provider_temporarily_limited else 400
        error_message = (
            "No se pudieron obtener datos del activo en este momento. La fuente de mercado ha limitado temporalmente las peticiones. Prueba de nuevo dentro de unos segundos o utiliza otro activo."
            if provider_temporarily_limited
            else "No hay datos históricos suficientes para construir el escenario experimental con los activos seleccionados."
        )
        return jsonify({"error": error_message, "warnings": warnings}), status_code

    total_valid_weight = sum(item["weight"] for item in valid_assets) or 1.0
    valid_assets = [
        {**item, "weight": item["weight"] / total_valid_weight} for item in valid_assets
    ]

    hist_frames = []
    for item in valid_assets:
        series = item["series"]
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue
        normalized = values / float(values.iloc[0]) * 100
        normalized = _downsample_horizon_series(
            normalized, max_points=HORIZON_MAX_HISTORY_POINTS
        )
        hist_frames.append(normalized.rename(item["ticker"]))
    hist_df = (
        pd.concat(hist_frames, axis=1).dropna(how="all")
        if hist_frames
        else pd.DataFrame()
    )
    if hist_df.empty:
        return (
            jsonify(
                {
                    "error": "No se pudo construir la serie histórica base para el escenario experimental.",
                    "warnings": warnings,
                }
            ),
            400,
        )
    hist_df = hist_df.ffill().dropna(how="any")
    hist_weights = pd.Series(
        {
            item["ticker"]: item["weight"]
            for item in valid_assets
            if item["ticker"] in hist_df.columns
        }
    )
    hist_weights = hist_weights / hist_weights.sum()
    blended_hist = hist_df.mul(hist_weights, axis=1).sum(axis=1)

    monthly_df = pd.concat(monthly_returns, axis=1).dropna(how="all")
    monthly_df = monthly_df.ffill().dropna(how="any")
    if monthly_df.empty:
        return (
            jsonify(
                {
                    "error": "No se pudieron combinar patrones mensuales suficientes para generar el escenario experimental.",
                    "warnings": warnings,
                }
            ),
            400,
        )

    weights = pd.Series({item["ticker"]: item["weight"] for item in valid_assets})
    available_cols = [col for col in monthly_df.columns if col in weights.index]
    monthly_df = monthly_df[available_cols]
    weights = weights[available_cols]
    weights = weights / weights.sum()
    blended_monthly = monthly_df.mul(weights, axis=1).sum(axis=1)
    if blended_monthly.empty:
        return (
            jsonify(
                {
                    "error": "No hay patrones mensuales suficientes para proyectar el escenario experimental.",
                    "warnings": warnings,
                }
            ),
            400,
        )

    future_months = horizon_years * 12
    rng_seed = (
        sum(sum(ord(ch) for ch in item["ticker"]) for item in valid_assets)
        + future_months
        + int(initial_value)
    )
    rng = random.Random(rng_seed)
    sample_pool = blended_monthly.tolist()
    simulated_returns = [
        float(sample_pool[rng.randrange(len(sample_pool))])
        for _ in range(future_months)
    ]

    future_anchor = end_d + timedelta(days=30)
    future_dates = pd.date_range(start=future_anchor, periods=future_months, freq="ME")
    last_hist_value = float(blended_hist.iloc[-1]) if not blended_hist.empty else 100.0
    projected_base = [last_hist_value]
    projected_value = [float(initial_value)]
    for ret in simulated_returns:
        projected_base.append(projected_base[-1] * (1 + float(ret)))
        projected_value.append(projected_value[-1] * (1 + float(ret)))

    historical_series = [
        [idx.isoformat(), round(float(value), 4)] for idx, value in blended_hist.items()
    ]
    projected_series = (
        [[blended_hist.index[-1].date().isoformat(), round(float(last_hist_value), 4)]]
        if not blended_hist.empty
        else []
    )
    projected_series.extend(
        [
            [
                future_dates[idx].date().isoformat(),
                round(float(projected_base[idx + 1]), 4),
            ]
            for idx in range(future_months)
        ]
    )

    final_value = projected_value[-1]
    scenario_total_return = (final_value / initial_value) - 1 if initial_value else 0.0
    scenario_annualized_return = (
        (final_value / initial_value) ** (1 / horizon_years) - 1
        if initial_value and horizon_years > 0
        else 0.0
    )
    scenario_volatility = (
        pd.Series(simulated_returns).std() * math.sqrt(12)
        if len(simulated_returns) > 1
        else 0.0
    )

    return jsonify(
        {
            "disclaimer": HORIZON_DISCLAIMER_TEXT,
            "historical_series": historical_series,
            "projected_series": projected_series,
            "metrics": {
                "initial_value": round(float(initial_value), 2),
                "projected_final_value": round(float(final_value), 2),
                "scenario_total_return": round(float(scenario_total_return), 6),
                "scenario_annualized_return": round(
                    float(scenario_annualized_return), 6
                ),
                "scenario_volatility": round(float(scenario_volatility), 6),
                "assets_used": [item["ticker"] for item in valid_assets],
                "horizon_years": horizon_years,
                "history_years_used": history_years,
                "history_points_displayed": len(blended_hist),
                "monthly_samples_used": len(blended_monthly),
                "extreme_returns_limited": had_clamped_outliers,
            },
            "scenario_note": "Este resultado corresponde a una trayectoria generada aleatoriamente a partir de retornos históricos. No representa una expectativa ni una previsión.",
            "warnings": warnings,
            "method_description": HORIZON_METHOD_DESCRIPTION,
            "source": source,
            "session_id": session_id or None,
            "projection_start": (
                projection_start.isoformat() if projection_start else None
            ),
            "assets": [
                {"ticker": item["ticker"], "weight": round(float(item["weight"]), 6)}
                for item in valid_assets
            ],
        }
    )


@bp.get("/modo-carrera")
def career_page():
    readiness_status = _readiness_status_payload()
    return render_template(
        "career.html",
        active="career",
        nav_mode="career",
        readiness_status=readiness_status,
        readiness_gate_blocked=not readiness_status.get("passed"),
    )


def _ai_tutor_is_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and OpenAI is not None


@bp.get("/api/ai/status")
def ai_status_api():
    return jsonify({"configured": _ai_tutor_is_configured()})


def _sanitize_ai_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _sanitize_ai_value(item)
            for key, item in value.items()
            if _sanitize_ai_value(item) is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [
            item
            for item in (_sanitize_ai_value(item) for item in value)
            if item is not None
        ]
    return str(value)


def _safe_round_number(value, digits: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _extract_json_object_from_text(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _truncate_text(value, max_len: int = 240):
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _summarize_career_events(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    summary: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        entry = {
            "kind": item.get("kind"),
            "scope": item.get("scope"),
            "label": item.get("label") or item.get("title") or item.get("name"),
            "direction": item.get("direction"),
            "ticker": item.get("ticker"),
            "sector": item.get("sector"),
            "impact": _safe_round_number(item.get("impact"), 6),
            "remaining_turns": item.get("remaining_turns"),
        }
        summary.append({k: v for k, v in entry.items() if v is not None})
    return summary


def _compute_turn_contributions(turn: dict[str, Any]) -> dict[str, float]:
    alloc = turn.get("alloc") or []
    returns_map = turn.get("ret_by_ticker_final") or {}
    if not isinstance(alloc, list) or not isinstance(returns_map, dict):
        return {}

    normalized_returns: dict[str, float] = {}
    for key, value in returns_map.items():
        ticker_key = str(key or "").strip().upper()
        if not ticker_key:
            continue
        try:
            normalized_returns[ticker_key] = float(value)
        except (TypeError, ValueError):
            continue

    contributions: dict[str, float] = {}
    for item in alloc:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            weight = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            continue
        if ticker not in normalized_returns:
            continue
        ticker_return = normalized_returns[ticker]
        if not math.isfinite(ticker_return):
            continue
        contributions[ticker] = round(weight * ticker_return, 6)
    return contributions


def _alloc_to_weight_map(alloc_list: Any) -> dict[str, float]:
    if not isinstance(alloc_list, list):
        return {}
    result: dict[str, float] = {}
    for item in alloc_list:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            result[ticker] = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            result[ticker] = 0.0
    return result


def _weights_match_with_tolerance(
    left: dict[str, float], right: dict[str, float], tolerance: float = 1e-6
) -> bool:
    tickers = set(left) | set(right)
    return all(abs(left.get(ticker, 0.0) - right.get(ticker, 0.0)) <= tolerance for ticker in tickers)


def _compute_strategy_changes(
    completed_turn_snapshots: list[dict[str, Any]], tolerance: float = 1e-6
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    ordered_turns = [turn for turn in completed_turn_snapshots if isinstance(turn, dict)]
    for previous, current in zip(ordered_turns, ordered_turns[1:]):
        current_alloc = _alloc_to_weight_map(current.get("alloc") or [])
        baseline_alloc = _alloc_to_weight_map(previous.get("alloc_next_suggested") or [])
        if not baseline_alloc:
            changes.append(
                {
                    "from_turn": previous.get("turn_n") or previous.get("n"),
                    "to_turn": current.get("turn_n") or current.get("n"),
                    "rebalanced": None,
                    "turnover": None,
                    "changes": [],
                    "baseline": "unknown",
                }
            )
            continue

        if _weights_match_with_tolerance(baseline_alloc, current_alloc, tolerance=tolerance):
            changes.append(
                {
                    "from_turn": previous.get("turn_n") or previous.get("n"),
                    "to_turn": current.get("turn_n") or current.get("n"),
                    "rebalanced": False,
                    "turnover": 0.0,
                    "changes": [],
                    "baseline": "previous_alloc_next_suggested",
                }
            )
            continue

        tickers = sorted(set(baseline_alloc) | set(current_alloc))
        transition_changes = []
        diff_sum = 0.0
        for ticker in tickers:
            old_weight = baseline_alloc.get(ticker, 0.0)
            new_weight = current_alloc.get(ticker, 0.0)
            delta = new_weight - old_weight
            diff_sum += abs(delta)
            transition_changes.append(
                {
                    "ticker": ticker,
                    "previous_weight": round(old_weight, 6),
                    "new_weight": round(new_weight, 6),
                    "delta": round(delta, 6),
                }
            )
        changes.append(
            {
                "from_turn": previous.get("turn_n") or previous.get("n"),
                "to_turn": current.get("turn_n") or current.get("n"),
                "rebalanced": True,
                "turnover": round(0.5 * diff_sum, 6),
                "changes": transition_changes,
                "baseline": "previous_alloc_next_suggested",
            }
        )
    return changes


def _build_career_ai_payload_v2(session: dict, report: dict) -> dict:
    report = report or {}
    meta = report.get("meta") or {}
    portfolio_metrics = (report.get("portfolio_equity") or {}).get("metrics") or {}
    benchmark = report.get("benchmark") or {}
    benchmark_metrics = benchmark.get("metrics") or {}
    turns = report.get("turns") or []
    if not isinstance(turns, list):
        turns = []

    turns_payload = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_payload = {
            "turn": turn.get("n") or turn.get("turn_n"),
            "start": (turn.get("range") or {}).get("start"),
            "end": (turn.get("range") or {}).get("end"),
            "allocation_start": [
                {
                    "ticker": str(item.get("ticker") or "").strip().upper(),
                    "weight": _safe_round_number(item.get("weight"), 6),
                }
                for item in (turn.get("alloc") or [])
                if isinstance(item, dict) and item.get("ticker")
            ],
            "use_dca": bool(turn.get("use_dca")),
            "dca_in_turn": turn.get("dca_in_turn"),
            "turn_return": turn.get("turn_return"),
            "turn_return_market": turn.get("turn_return_market"),
            "ret_by_ticker_final": turn.get("ret_by_ticker_final") or {},
            "ticker_contributions": _compute_turn_contributions(turn),
            "ret_portfolio_shift": turn.get("ret_portfolio_shift"),
            "events_applied": _summarize_career_events(turn.get("events_applied") or []),
            "portfolio_value_end": turn.get("portfolio_value"),
        }
        if turn.get("ret_ticker_shift"):
            turn_payload["ret_ticker_shift"] = turn.get("ret_ticker_shift")
        turns_payload.append(turn_payload)

    strategy_source = [
        snap
        for snap in (session.get("completed_turns") or [])
        if isinstance(snap, dict)
    ]

    strategy_changes = _compute_strategy_changes(strategy_source)
    best_turn = None
    worst_turn = None
    if turns_payload:
        ranked_turns = [
            turn
            for turn in turns_payload
            if isinstance(turn.get("turn_return"), (int, float))
        ]
        if ranked_turns:
            best = max(ranked_turns, key=lambda turn: float(turn.get("turn_return") or 0.0))
            worst = min(ranked_turns, key=lambda turn: float(turn.get("turn_return") or 0.0))
            best_turn = {"turn": best.get("turn"), "turn_return": best.get("turn_return")}
            worst_turn = {"turn": worst.get("turn"), "turn_return": worst.get("turn_return")}

    payload = {
        "simulation_summary": {
            "period": report.get("range") or {},
            "difficulty": meta.get("difficulty"),
            "turns_total": meta.get("turns_total"),
            "turns_closed": meta.get("turns_closed"),
            "capital_initial": meta.get("capital_initial"),
            "capital_final": meta.get("capital_current"),
            "capital_invested_total": meta.get("invested_so_far"),
            "pnl_abs": meta.get("pnl_abs"),
            "pnl_pct_on_invested_capital": meta.get("pnl_pct"),
            "portfolio_total_return": portfolio_metrics.get("total_return"),
            "portfolio_cagr": portfolio_metrics.get("CAGR"),
            "portfolio_max_drawdown": {
                "value": portfolio_metrics.get("max_drawdown"),
                "basis": "turn_closures",
            },
            "benchmark_ticker": benchmark.get("ticker"),
            "benchmark_total_return": benchmark_metrics.get("total_return"),
            "benchmark_cagr": benchmark_metrics.get("CAGR"),
        },
        "turn_highlights": {
            "best_turn": best_turn,
            "worst_turn": worst_turn,
            "active_rebalances": sum(1 for item in strategy_changes if item.get("rebalanced") is True),
            "unknown_rebalance_transitions": sum(1 for item in strategy_changes if item.get("rebalanced") is None),
        },
        "turns": turns_payload,
        "strategy_changes": strategy_changes,
    }
    return _sanitize_ai_value(payload) or {}


def _build_career_ai_payload(session: dict, report: dict) -> dict:
    report = report or {}
    meta = report.get("meta") or {}
    portfolio_metrics = (report.get("portfolio_equity") or {}).get("metrics") or {}
    benchmark = report.get("benchmark") or {}
    benchmark_metrics = benchmark.get("metrics") or {}
    tracking = report.get("tracking") or {}
    score = report.get("score") or {}
    metric_quality = report.get("metric_quality") or {}
    turns = report.get("turns") or []
    if not isinstance(turns, list):
        turns = []
    theoretical = report.get("theoretical") or {}
    warnings = []

    tickers = []
    seen = set()
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for item in turn.get("alloc") or []:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)

    latest_alloc = []
    if turns:
        latest_turn = turns[-1] if isinstance(turns[-1], dict) else {}
        latest_alloc = [
            {
                "ticker": str(item.get("ticker") or "").strip().upper(),
                "weight": _safe_round_number(item.get("weight"), 4),
            }
            for item in (latest_turn.get("alloc") or [])
            if isinstance(item, dict) and item.get("ticker")
        ]

    if not latest_alloc:
        warnings.append(
            "Hay datos limitados en esta sesión, por lo que el análisis se centra en las métricas disponibles."
        )

    payload = {
        "simulation_type": "career_mode",
        "session_id": meta.get("session_id"),
        "difficulty": meta.get("difficulty"),
        "historical_period": report.get("range") or {},
        "player_alias": meta.get("player") or None,
        "tickers_used": tickers[:12],
        "latest_allocation": latest_alloc[:10],
        "turns_total": meta.get("turns_total"),
        "turns_closed": meta.get("turns_closed"),
        "initial_value": meta.get("capital_initial"),
        "final_value": meta.get("capital_current"),
        "invested_so_far": meta.get("invested_so_far"),
        "pnl_abs": meta.get("pnl_abs"),
        "pnl_pct": meta.get("pnl_pct"),
        "portfolio_metrics": {
            "cagr": portfolio_metrics.get("CAGR"),
            "volatility": portfolio_metrics.get("vol_annual"),
            "max_drawdown": portfolio_metrics.get("max_drawdown"),
            "total_return": portfolio_metrics.get("total_return"),
        },
        "benchmark": {
            "ticker": benchmark.get("ticker"),
            "cagr": benchmark_metrics.get("CAGR"),
            "volatility": benchmark_metrics.get("vol_annual"),
            "max_drawdown": benchmark_metrics.get("max_drawdown"),
            "total_return": benchmark_metrics.get("total_return"),
        },
        "tracking": {
            "active_return": tracking.get("active_return"),
            "tracking_error": tracking.get("tracking_error"),
            "information_ratio": tracking.get("information_ratio"),
        },
        "turnover_avg": report.get("turnover_avg"),
        "metric_quality": {
            "portfolio_months_count": metric_quality.get("portfolio_months_count"),
            "benchmark_months_count": metric_quality.get("benchmark_months_count"),
            "tracking_joined_months_count": metric_quality.get("tracking_joined_months_count"),
            "volatility_reliable": metric_quality.get("volatility_reliable"),
            "benchmark_volatility_reliable": metric_quality.get("benchmark_volatility_reliable"),
            "tracking_error_reliable": metric_quality.get("tracking_error_reliable"),
        },
        "score": {
            "stars": score.get("stars"),
            "value": score.get("value"),
            "notes": score.get("notes"),
        },
        "warnings": [
            _truncate_text(item, 220)
            for item in ((report.get("warnings") or [])[:AI_TUTOR_MAX_WARNINGS])
            + warnings
            if item
        ],
        "theoretical_summary": {
            key: theoretical.get(key)
            for key in ("k1", "k2", "k3", "method")
            if theoretical.get(key) is not None
        },
        "event_counts": {
            "turns_with_events": sum(
                1
                for turn in turns
                if isinstance(turn, dict)
                and (turn.get("events_applied") or turn.get("events_new"))
            ),
            "events_applied_total": sum(
                len(turn.get("events_applied") or [])
                for turn in turns
                if isinstance(turn, dict)
            ),
        },
    }
    return _sanitize_ai_value(payload) or {}


def _generate_career_ai_analysis(ai_payload: dict) -> dict:
    if not _ai_tutor_is_configured():
        raise RuntimeError("El Tutor IA no está configurado en este entorno.")

    model_name = os.environ.get("OPENAI_MODEL") or AI_TUTOR_DEFAULT_MODEL
    timeout_seconds = AI_TUTOR_TIMEOUT_SECONDS
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout_seconds)
    system_prompt = (
        "Eres un tutor educativo de inversión simulada dentro de una aplicación académica. "
        "Tu función es actuar como un profesor que revisa cómo el usuario ha gestionado una partida concreta del Modo Carrera. "
        "No eres un auditor técnico del sistema ni un asesor financiero. "
        "Analizas únicamente el payload proporcionado por la aplicación. "
        "No utilices conocimiento macroeconómico, histórico o de mercado externo para rellenar contexto ausente. "
        "No inventes datos, turnos, precios, causas, contrafactuales ni métricas no incluidas en el payload. "
        "No recomiendes comprar o vender activos reales, no hagas predicciones y no presentes una decisión como universalmente buena o mala. "
        "Describe solo el efecto observado en esta simulación. "
        "Cuando ayude, cita turnos, pesos, retornos y contribuciones concretas. "
        "Si falta información, dilo con honestidad y no la completes con suposiciones. "
        "No caracterices la gestión como conservadora, agresiva, defensiva o con etiquetas similares basándote solo en la frecuencia de rebalanceo. "
        "Describe hechos observables, como pocos rebalanceos, alta continuidad de posiciones o baja actividad de rebalanceo, solo si los datos lo respaldan. "
        "No menciones diversificación ni la importancia de diversificar si el payload no contiene una métrica específica de diversificación. "
        "En ese caso, habla únicamente de pesos, reparto entre activos, concentración observada o contribuciones cuando esos datos existan. "
        "No conviertas lo ocurrido en esta partida en recomendaciones generales. "
        "Evita fórmulas como mantener la cartera puede ser efectivo, conviene rebalancear, considera rebalanceos oportunos, aprovechar oportunidades o deberías ajustar riesgos. "
        "Formula siempre los aprendizajes sobre esta simulación concreta, usando expresiones como En esta simulación, En los turnos observados, La partida muestra o Como ejercicio educativo, puede revisarse. "
        "final_advice no debe contener instrucciones de actuación futura sobre la cartera. Debe resumir qué merece la pena comprender o revisar de la partida ya realizada. "
        "Cuando compares una contribución con el retorno total de un turno, evita aproximaciones verbales incorrectas; si no es necesario calcular la proporción, da directamente ambos números. "
        "Un cambio de pesos por drift no es una decisión del usuario. "
        "Solo existe rebalanceo cuando strategy_changes[].rebalanced == true. "
        "Si strategy_changes[].rebalanced == false, puedes decir que el usuario mantuvo la cartera sin modificar activamente esa asignación. "
        "Si strategy_changes[].rebalanced == null, no infieras si hubo rebalanceo. "
        "ticker_contributions representa contribución dentro de ese turno. "
        "Puedes explicar contribuciones por turno, pero no sumes automáticamente contribuciones de distintos turnos para atribuir el resultado final global a un ticker. "
        "La presencia de DCA solo indica aportaciones adicionales durante la simulación. No afirmes que el DCA mejoró o empeoró rentabilidad, riesgo, precio medio o resultado salvo que exista un contrafactual calculado, que actualmente no existe. "
        "Sí puedes usar DCA para explicar la diferencia entre pnl_pct_on_invested_capital y portfolio_total_return. "
        "ret_portfolio_shift se suma directamente al retorno total del turno y puede expresarse como efecto en puntos porcentuales sobre el turno. "
        "ret_ticker_shift[ticker] modifica el retorno de ese ticker. No equivale directamente a su efecto en puntos porcentuales sobre toda la cartera; para eso debes utilizar ticker_contributions, que ya incorpora el peso del activo. "
        "La comparación con benchmark es solo global. "
        "Solo compara la estrategia con el benchmark si benchmark_total_return y/o benchmark_cagr contienen valores numéricos disponibles. Si faltan, son null o no están presentes, indica brevemente que no hay datos suficientes del benchmark y no inventes ninguna comparación. "
        "Puedes comparar portfolio_total_return, portfolio_cagr, benchmark_total_return y benchmark_cagr. "
        "No existe benchmark por turno, así que no atribuyas la diferencia frente al benchmark a un turno concreto como hecho cerrado. "
        "portfolio_max_drawdown con basis=turn_closures significa el mayor retroceso observado en la curva agregada por cierres de turno. "
        "Nunca lo describas como drawdown diario o intraturno. "
        "Distingue correctamente pnl_pct_on_invested_capital, portfolio_total_return y portfolio_cagr. "
        "No los mezcles como si fueran la misma métrica. "
        'Debes incluir literalmente este disclaimer en la respuesta final: "'
        + AI_TUTOR_DISCLAIMER
        + '". '
        "Devuelve exclusivamente JSON válido con estas claves exactas: "
        "summary, management_review, impact_factors, event_effects, benchmark_comparison, learning_points, final_advice, disclaimer. "
        "summary, management_review, benchmark_comparison y final_advice deben ser strings breves. "
        "impact_factors, event_effects y learning_points deben ser arrays de 2 a 5 strings cortos y concretos cuando haya datos suficientes. "
        "Mantén la respuesta compacta, clara y útil."
    )
    user_prompt = (
        "Analiza esta partida del Modo Carrera como un profesor que revisa cómo se ha gestionado la cartera. "
        "Prioriza lo que ocurrió en los turnos, la evolución de asignaciones, los rebalanceos activos reales, las contribuciones dentro de cada turno, el efecto de los eventos y la comparación global con el benchmark. "
        "Elimina del discurso volatilidad, tracking error, information ratio, calidad de métricas, warnings técnicos y contexto macro externo. "
        "Usa turn_highlights para identificar mejor turno, peor turno y conteos mínimos de rebalanceo. "
        "No confundas drift con rebalanceo. "
        "Si strategy_changes marca rebalanced=false, trátalo como continuidad pasiva de la cartera. "
        "Usa solo los datos del payload.\n\n"
        f"Payload de la partida:\n{json.dumps(ai_payload, ensure_ascii=False, indent=2)}"
    )
    openai_started_at = time.monotonic()
    logger.info(
        "ai.tutor.openai_start",
        extra={
            "session_id": ai_payload.get("session_id"),
            "turns_closed": ai_payload.get("turns_closed"),
            "model": model_name,
            "timeout_seconds": timeout_seconds,
            "payload_chars": len(user_prompt),
        },
    )
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_output_tokens=AI_TUTOR_MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        error_name = exc.__class__.__name__
        elapsed = round(time.monotonic() - openai_started_at, 3)
        if "Timeout" in error_name or "timeout" in str(exc).lower():
            logger.warning(
                "ai.tutor.openai_timeout",
                extra={
                    "session_id": ai_payload.get("session_id"),
                    "model": model_name,
                    "elapsed_seconds": elapsed,
                },
            )
            raise TimeoutError(
                "El Tutor IA ha tardado demasiado en responder. Inténtalo de nuevo en unos segundos."
            ) from exc
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": ai_payload.get("session_id"),
                "phase": "openai_call",
                "model": model_name,
                "elapsed_seconds": elapsed,
                "error_name": error_name,
            },
        )
        raise

    content = getattr(response, "output_text", "") or ""
    logger.info(
        "ai.tutor.openai_done",
        extra={
            "session_id": ai_payload.get("session_id"),
            "model": model_name,
            "elapsed_seconds": round(time.monotonic() - openai_started_at, 3),
            "content_length": len(content),
        },
    )

    parsed = _extract_json_object_from_text(content)
    if not parsed:
        fallback_text = content.strip()
        if fallback_text:
            logger.warning(
                "ai.tutor.parse_fallback",
                extra={
                    "session_id": ai_payload.get("session_id"),
                    "content_length": len(fallback_text),
                },
            )
            parsed = {
                "summary": fallback_text,
                "strengths": [],
                "improvements": [],
                "benchmark_analysis": "La respuesta del proveedor no llegó en formato estructurado completo. Se muestra un resumen textual con los datos disponibles.",
                "risk_notes": [],
                "historical_context": "Los datos de la sesión se han interpretado de forma limitada debido al formato de respuesta recibido.",
                "learning_recommendations": [],
                "final_advice": "Puedes volver a intentarlo si quieres obtener una estructura más completa del análisis educativo.",
                "disclaimer": AI_TUTOR_DISCLAIMER,
            }
        else:
            raise ValueError(
                "La respuesta del proveedor de IA llegó vacía o no contenía JSON utilizable."
            )

    logger.info(
        "ai.tutor.parse_done",
        extra={
            "session_id": ai_payload.get("session_id"),
            "keys": sorted(parsed.keys())[:12],
        },
    )
    parsed["disclaimer"] = AI_TUTOR_DISCLAIMER
    return _sanitize_ai_value(parsed) or {"disclaimer": AI_TUTOR_DISCLAIMER}


@bp.get("/api/readiness/status")
def readiness_status_api():
    payload = _readiness_status_payload()
    payload["required_score"] = READINESS_PASS_SCORE
    payload["total_questions_default"] = READINESS_TOTAL_QUESTIONS
    return jsonify(payload)


@bp.post("/api/ai/career-analysis/<session_id>")
def ai_career_analysis_api(session_id: str):
    from .career import _resolve_session_for_request, get_career_report_for_session

    endpoint_started_at = time.monotonic()
    logger.info("ai.tutor.start", extra={"session_id": session_id})

    if not _ai_tutor_is_configured():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "El Tutor IA no está configurado en este entorno.",
                    "error_type": "ai_not_configured",
                    "retryable": False,
                    "details": "Falta configuración del proveedor IA en este despliegue.",
                }
            ),
            503,
        )

    try:
        report = get_career_report_for_session(session_id, include_series=False)
        logger.info(
            "ai.tutor.report_ready",
            extra={"session_id": session_id, "has_report": bool(report)},
        )
    except Exception as exc:
        message = getattr(exc, "description", str(exc))
        status_code = getattr(exc, "code", 400)
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "report",
                "status_code": status_code,
                "message": message[:200],
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": message
                    or "Primero genera el informe final de la carrera antes de usar el Tutor IA.",
                    "error_type": "career_report_unavailable",
                    "retryable": False,
                    "details": "No se pudo preparar el informe base de la sesión para el Tutor IA.",
                }
            ),
            status_code,
        )

    if not report:
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "session_resolve",
                "status_code": 404,
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "No puedes analizar una sesión ajena o inexistente.",
                    "error_type": "session_not_found",
                    "retryable": False,
                    "details": "La sesión indicada no existe o no es accesible para este usuario.",
                }
            ),
            404,
        )

    turns = report.get("turns") or []
    if not isinstance(turns, list) or not turns:
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "report_validation",
                "status_code": 422,
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Esta sesión no contiene datos suficientes para generar un análisis IA.",
                    "error_type": "insufficient_session_data",
                    "retryable": False,
                    "details": "La sesión no tiene turnos o métricas suficientes para un análisis educativo estable.",
                }
            ),
            422,
        )

    session_payload = _resolve_session_for_request(session_id) or {}
    ai_payload = _build_career_ai_payload_v2(session_payload, report)
    logger.info(
        "ai.tutor.summary_ready",
        extra={
            "session_id": session_id,
            "turns_closed": (ai_payload.get("simulation_summary") or {}).get("turns_closed"),
            "payload_chars": len(json.dumps(ai_payload, ensure_ascii=False)),
        },
    )
    try:
        analysis = _generate_career_ai_analysis(ai_payload)
    except TimeoutError as exc:
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "timeout",
                "status_code": 504,
                "elapsed_seconds": round(time.monotonic() - endpoint_started_at, 3),
                "message": str(exc)[:200],
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "El Tutor IA ha tardado demasiado en responder. Inténtalo de nuevo en unos segundos.",
                    "error_type": "ai_timeout",
                    "retryable": True,
                    "details": "La llamada al proveedor de IA superó el tiempo máximo configurado para esta app.",
                }
            ),
            504,
        )
    except RuntimeError as exc:
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "config",
                "status_code": 503,
                "message": str(exc)[:200],
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": "ai_not_configured",
                    "retryable": False,
                    "details": "El Tutor IA no está disponible en este entorno.",
                }
            ),
            503,
        )
    except ValueError as exc:
        logger.warning(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "parse",
                "status_code": 502,
                "message": str(exc)[:200],
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "El proveedor de IA no respondió correctamente. Puedes intentarlo de nuevo más tarde.",
                    "error_type": "ai_parse_error",
                    "retryable": True,
                    "details": "La respuesta del proveedor no se pudo interpretar de forma segura para esta sesión.",
                }
            ),
            502,
        )
    except Exception:
        logger.exception(
            "ai.tutor.error",
            extra={
                "session_id": session_id,
                "phase": "openai_or_analysis",
                "elapsed_seconds": round(time.monotonic() - endpoint_started_at, 3),
            },
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "No se pudo generar el análisis IA para esta sesión.",
                    "error_type": "ai_analysis_error",
                    "retryable": True,
                    "details": "El proveedor de IA no respondió correctamente. Puedes intentarlo de nuevo más tarde.",
                    "warnings": [
                        "La generación depende de un servicio externo y puede fallar temporalmente."
                    ],
                }
            ),
            502,
        )

    sections = [
        {
            "title": "Resumen de la partida",
            "type": "text",
            "content": analysis.get("summary"),
        },
        {
            "title": "Cómo gestionaste la cartera",
            "type": "text",
            "content": analysis.get("management_review"),
        },
        {
            "title": "Qué factores tuvieron más impacto",
            "type": "list",
            "content": analysis.get("impact_factors") or [],
        },
        {
            "title": "Efecto de los eventos",
            "type": "list",
            "content": analysis.get("event_effects") or [],
        },
        {
            "title": "Comparación con el benchmark",
            "type": "text",
            "content": analysis.get("benchmark_comparison"),
        },
        {
            "title": "Qué puedes aprender de esta partida",
            "type": "list",
            "content": analysis.get("learning_points") or [],
        },
        {
            "title": "Conclusión educativa",
            "type": "text",
            "content": analysis.get("final_advice"),
        },
    ]
    logger.info(
        "ai.tutor.session_resolved",
        extra={
            "session_id": session_id,
            "elapsed_seconds": round(time.monotonic() - endpoint_started_at, 3),
        },
    )
    return jsonify(
        {
            "ok": True,
            "analysis": analysis,
            "sections": sections,
            "disclaimer": AI_TUTOR_DISCLAIMER,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "warnings": ai_payload.get("warnings") or [],
        }
    )


@bp.get("/api/readiness/questions")
def readiness_questions_api():
    restart = request.args.get("restart") in {"1", "true", "yes"}
    questions = _get_or_create_readiness_question_set(force_new=restart)
    public_questions = []
    for index, item in enumerate(questions, start=1):
        public_questions.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "options": [
                    {"id": option["id"], "label": option["label"]}
                    for option in item["options"]
                ],
                "explanation": item["explanation"],
                "topic": item["topic"],
                "step": index,
                "contextTitle": (
                    "Conceptos básicos" if index <= 5 else "Cómo leer la simulación"
                ),
                "contextHint": (
                    "Piensa en riesgo, diversificación, benchmark y horizonte temporal."
                    if index <= 5
                    else "Relaciona cada respuesta con las pantallas, métricas y decisiones de la app."
                ),
            }
        )
    return jsonify(
        {
            "questions": public_questions,
            "pass_score": READINESS_PASS_SCORE,
            "total_questions": len(public_questions),
        }
    )


@bp.post("/api/readiness/submit")
def readiness_submit_api():
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers") or []
    if not isinstance(answers, list):
        return jsonify({"error": "El formato de respuestas no es válido."}), 400

    question_set = _get_or_create_readiness_question_set()
    if len(answers) != len(question_set):
        return (
            jsonify(
                {"error": "Debes responder todas las preguntas del recorrido final."}
            ),
            400,
        )

    score = 0
    result_items = []
    question_map = {item["id"]: item for item in question_set}
    for answer in answers:
        if not isinstance(answer, dict):
            return (
                jsonify({"error": "Cada respuesta debe indicar pregunta y opción."}),
                400,
            )
        question_id = answer.get("questionId")
        option_id = answer.get("optionId")
        question = question_map.get(question_id)
        if not question:
            return (
                jsonify(
                    {"error": "Se ha detectado una pregunta no válida en el intento."}
                ),
                400,
            )
        selected_option = next(
            (option for option in question["options"] if option["id"] == option_id),
            None,
        )
        correct_option = next(
            (option for option in question["options"] if option["correct"]), None
        )
        is_correct = bool(
            selected_option
            and correct_option
            and selected_option["id"] == correct_option["id"]
        )
        if is_correct:
            score += 1
        result_items.append(
            {
                "id": question["id"],
                "prompt": question["prompt"],
                "selectedOptionId": selected_option["id"] if selected_option else None,
                "selectedLabel": selected_option["label"] if selected_option else None,
                "correctOptionId": correct_option["id"] if correct_option else None,
                "correctLabel": correct_option["label"] if correct_option else None,
                "correct": is_correct,
                "explanation": question["explanation"],
                "topic": question["topic"],
            }
        )

    passed = score >= READINESS_PASS_SCORE
    current_user_id = _current_user_id()
    passed_at = datetime.utcnow() if passed else None

    if current_user_id:
        record, created = ReadinessQuizResult.get_or_create(
            user=current_user_id,
            defaults={
                "passed": passed,
                "score": score,
                "total_questions": len(question_set),
                "passed_at": passed_at,
                "answers_json": json.dumps(result_items, ensure_ascii=False),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )
        if not created:
            record.passed = passed
            record.score = score
            record.total_questions = len(question_set)
            record.passed_at = passed_at
            record.answers_json = json.dumps(result_items, ensure_ascii=False)
            record.updated_at = datetime.utcnow()
            record.save()
        storage = "server"
    else:
        session["readiness_guest"] = {
            "passed": passed,
            "score": score,
            "total_questions": len(question_set),
            "passed_at": passed_at.isoformat() + "Z" if passed_at else None,
        }
        session.modified = True
        storage = "session"

    _clear_readiness_question_set()

    return jsonify(
        {
            "passed": passed,
            "score": score,
            "total_questions": len(question_set),
            "pass_score": READINESS_PASS_SCORE,
            "results": result_items,
            "storage": storage,
            "can_access_career": passed,
            "user_authenticated": bool(current_user_id),
            "guest": _is_guest_user(),
        }
    )


# ----------------------
#   Health
# ----------------------
@bp.get("/health")
def health():
    return jsonify(status="ok")


@bp.get("/favicon.ico")
def favicon():
    static_dir = Path(__file__).resolve().parent / "static" / "img"
    return send_from_directory(static_dir, "favicon.png")


# ----------------------
#   Datos de empresas
# ----------------------
DATA_PATH = Path(__file__).resolve().parent / "data" / "empresas.json"


def _cargar_empresas():
    with DATA_PATH.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    for e in data:
        assert {"ticker", "nombre", "sector"} <= set(e.keys())
    return data


EMPRESAS = _cargar_empresas()


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


@bp.get("/empresas/sectores")
def listar_sectores():
    sectores = sorted(
        {(e.get("sector") or "").strip() for e in EMPRESAS if e.get("sector")}
    )
    return jsonify(sectores)


@bp.get("/empresas-data")
def listar_empresas():
    """
    Devuelve lista de empresas.
    Filtros opcionales:
      - ?sector=... -> igualdad exacta normalizada
      - ?q=... -> búsqueda parcial en ticker o nombre
      - ?page&per_page ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ (opcional) si se envÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­an, responde paginado
    """
    sector = request.args.get("sector")
    q = request.args.get("q")

    resultado = EMPRESAS

    if sector:
        target = _norm(sector)
        resultado = [e for e in resultado if _norm(e.get("sector", "")) == target]

    if q:
        needle = _norm(q)

        def coincide(e):
            return needle in _norm(e.get("ticker", "")) or needle in _norm(
                e.get("nombre", "")
            )

        resultado = [e for e in resultado if coincide(e)]

    # Solo paginar si el cliente lo pide
    page = request.args.get("page")
    per_page = request.args.get("per_page")
    if page or per_page:
        return jsonify(_paginate(resultado, page, per_page))
    return jsonify(resultado)


# ----------------------
#   Motor de análisis
# ----------------------
def _validar_payload(p):
    errores = []

    ticker = p.get("ticker")
    if not ticker or not isinstance(ticker, str):
        errores.append("Falta 'ticker' (string).")

    importe = p.get("importe_inicial")
    if not isinstance(importe, (int, float)) or importe <= 0:
        errores.append("'importe_inicial' debe ser numerico > 0.")

    horizonte = p.get("horizonte_anios")
    if horizonte is None:
        errores.append("Falta el campo 'horizonte_anios'.")
    elif not isinstance(horizonte, int):
        errores.append("El campo 'horizonte_anios' debe ser un entero.")
    elif horizonte < 1:
        errores.append(
            "Horizonte mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­nimo: 1 aÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±o (horizonte_anios >= 1)."
        )

    sup = p.get("supuestos") or {}
    if not isinstance(sup, dict):
        errores.append("'supuestos' debe ser un objeto con porcentajes.")
        sup = {}
    else:
        p["supuestos"] = sup

    def pct_ok(clave, minimo, maximo):
        if clave not in sup:
            return None
        valor = sup.get(clave)
        if valor is None:
            return None
        if not isinstance(valor, (int, float)):
            errores.append(f"'{clave}' debe ser numerico.")
            return None
        if valor < minimo or valor > maximo:
            errores.append(f"'{clave}' debe estar entre {minimo} y {maximo}.")
        return valor

    pct_ok("crecimiento_anual_pct", 0, 100)
    pct_ok("margen_seguridad_pct", 0, 100)
    pct_ok("roe_pct", 0, 100)
    pct_ok("deuda_sobre_activos_pct", 0, 100)

    just = p.get("justificacion")
    if just is not None and not isinstance(just, str):
        errores.append("'justificacion' debe ser texto.")
    elif isinstance(just, str) and len(just.strip()) < 5:
        errores.append("La 'justificacion' debe tener al menos 5 caracteres.")

    modo = p.get("modo") or "SIN_DCA"
    if modo not in {"DCA", "SIN_DCA"}:
        errores.append("'modo' debe ser 'DCA' o 'SIN_DCA'.")
    elif modo == "DCA":
        dca = p.get("dca") or {}
        if not isinstance(dca, dict):
            errores.append("'dca' debe ser un objeto con aporte y frecuencia.")
            dca = {}
        aporte = dca.get("aporte")
        if aporte is None or not isinstance(aporte, (int, float)) or aporte < 0:
            errores.append("'dca.aporte' no puede ser negativo.")
        frecuencia = (dca.get("frecuencia") or "").upper()
        if frecuencia not in {"WEEKLY", "MONTHLY", "QUARTERLY", "ANNUAL"}:
            errores.append("'dca.frecuencia' no es valida.")
    else:
        p["dca"] = None

    crec = sup.get("crecimiento_anual_pct")
    if isinstance(crec, (int, float)) and crec > 25:
        errores.append("Crecimiento anual > 25% sostenido es probablemente irrealista.")

    return errores


def _normalizar_payload(datos):
    datos = dict(datos or {})

    ticker = datos.get("ticker")
    if isinstance(ticker, str):
        datos["ticker"] = ticker.strip().upper()

    def to_float(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if "importe_inicial" in datos:
        imp = to_float(datos.get("importe_inicial"))
        if imp is not None:
            datos["importe_inicial"] = imp if not imp.is_integer() else int(imp)

    if "horizonte_anios" in datos:
        try:
            datos["horizonte_anios"] = int(round(float(datos["horizonte_anios"])))
        except (TypeError, ValueError):
            pass

    sup_in = datos.get("supuestos")
    sup = dict(sup_in) if isinstance(sup_in, dict) else {}

    mapping = {
        "crecimiento_anual_estimado": "crecimiento_anual_pct",
        "margen_seguridad_pct": "margen_seguridad_pct",
        "roe_pct": "roe_pct",
        "deuda_sobre_activos_pct": "deuda_sobre_activos_pct",
    }
    for origen, destino in mapping.items():
        if origen in datos and destino not in sup:
            sup[destino] = datos[origen]

    for clave in (
        "crecimiento_anual_pct",
        "margen_seguridad_pct",
        "roe_pct",
        "deuda_sobre_activos_pct",
    ):
        val = to_float(sup.get(clave))
        sup[clave] = 0.0 if val is None else val

    datos["supuestos"] = sup

    modo = datos.get("modo")
    if modo not in {"DCA", "SIN_DCA"}:
        modo = "SIN_DCA"
    datos["modo"] = modo

    if modo == "DCA":
        dca = datos.get("dca")
        if not isinstance(dca, dict):
            dca = {}
        aporte = to_float(dca.get("aporte"))
        aporte_norm = 0.0 if aporte is None else aporte
        if isinstance(aporte_norm, float) and aporte_norm.is_integer():
            aporte_norm = int(aporte_norm)
        frecuencia = (dca.get("frecuencia") or "MONTHLY").upper()
        datos["dca"] = {"aporte": aporte_norm, "frecuencia": frecuencia}
    else:
        datos["dca"] = None

    just = datos.get("justificacion")
    if just is not None and not isinstance(just, str):
        datos["justificacion"] = str(just)

    if "crecimiento_anual_estimado" in datos:
        ce = to_float(datos["crecimiento_anual_estimado"])
        datos["crecimiento_anual_estimado"] = (
            ce if ce is not None else sup.get("crecimiento_anual_pct")
        )
    else:
        datos["crecimiento_anual_estimado"] = sup.get("crecimiento_anual_pct")

    if "margen_seguridad_pct" in datos:
        ms = to_float(datos["margen_seguridad_pct"])
        datos["margen_seguridad_pct"] = (
            ms if ms is not None else sup.get("margen_seguridad_pct")
        )
    else:
        datos["margen_seguridad_pct"] = sup.get("margen_seguridad_pct")

    for key in ("inicio", "fin"):
        val = datos.get(key)
        if isinstance(val, str):
            val = val.strip()
            datos[key] = val or None
        elif val not in (None,):
            datos[key] = str(val)

    return datos


def _puntuar_y_observar(p):
    """Heurística muy simple para MVP: 0–100."""
    sup = p["supuestos"]
    horizon = p["horizonte_anios"]

    score = 50
    obs = []

    # Horizonte
    if horizon >= 10:
        score += 10
        obs.append({"tipo": "ok", "msg": "Horizonte largo (≥10 años)."})
    elif horizon >= 5:
        score += 5
        obs.append({"tipo": "ok", "msg": "Horizonte adecuado (≥5 años)."})

    # ROE
    roe = sup.get("roe_pct", 0)
    if roe >= 15:
        score += 10
    elif roe >= 8:
        score += 5

    # Deuda
    deuda = sup.get("deuda_sobre_activos_pct", 0)
    if deuda <= 30:
        score += 10
    elif deuda <= 60:
        score += 3
    else:
        score -= 5

    # Margen de seguridad
    margen = sup.get("margen_seguridad_pct", 0)
    if margen >= 20:
        score += 10
        obs.append({"tipo": "ok", "msg": "Margen de seguridad sólido (≥20%)."})
    elif margen >= 10:
        score += 3
        obs.append(
            {"tipo": "mejora", "msg": "Margen de seguridad algo justo (10–20%)."}
        )
    else:
        score -= 5
        obs.append({"tipo": "alerta", "msg": "Margen de seguridad bajo (<10%)."})

    # Crecimiento
    crec = sup.get("crecimiento_anual_pct", 0)
    if crec > 25:
        score -= 10
        obs.append(
            {
                "tipo": "alerta",
                "msg": "Supuesto de crecimiento >25% parece optimista/irrealista.",
            }
        )
    elif crec >= 5:
        score += 5
        obs.append({"tipo": "ok", "msg": "Crecimiento razonable (5–25%)."})
    else:
        obs.append(
            {"tipo": "mejora", "msg": "Crecimiento bajo: compénsalo con precio/margen."}
        )

    # Justificación
    if len((p.get("justificacion") or "").strip()) >= 60:
        score += 5
        obs.append({"tipo": "ok", "msg": "Buena justificación (detallada)."})
    else:
        obs.append(
            {
                "tipo": "mejora",
                "msg": "Amplía la justificación: riesgos, sensibilidad, comparables.",
            }
        )

    score = max(0, min(100, int(round(score))))

    if score >= 80:
        resumen = "Análisis sólido."
    elif score >= 60:
        resumen = "Análisis razonable con áreas de mejora."
    else:
        resumen = "Análisis débil: revisa supuestos, riesgos y valoración."

    return score, obs, resumen


# ----------------------
#   Persistencia análisis


def _fix_mojibake(s):
    if not isinstance(s, str):
        return s
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def _sanear_registro(r: dict) -> dict:
    if not isinstance(r, dict):
        return r
    if "resumen" in r:
        r["resumen"] = _fix_mojibake(r["resumen"])
    if isinstance(r.get("observaciones"), list):
        out = []
        for o in r["observaciones"]:
            if not isinstance(o, dict):
                continue
            msg = _fix_mojibake(o.get("msg", ""))
            if any(k in msg.lower() for k in ("roe", "deuda")):
                continue
            out.append({**o, "msg": msg})
        r["observaciones"] = out
    return r


# ----------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
ANALISIS_PATH = DATA_DIR / "analisis.json"


def _cargar_lista(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        data = [_sanear_registro(x) for x in data]
    return data


def _guardar_lista(path: Path, lista):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lista, list):
        lista = [_sanear_registro(x) for x in lista]
    with path.open("w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=2)


def _current_user_id():
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    return user.id if user else None


def _is_guest_user():
    return bool(session.get("guest"))


def _registrar_analisis(datos):
    errores = _validar_payload(datos)
    if errores:
        return jsonify({"valido": False, "errores": errores}), 400

    puntuacion, observaciones, resumen = _puntuar_y_observar(datos)

    registro = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ticker": datos.get("ticker"),
        "importe_inicial": datos.get("importe_inicial"),
        "horizonte_anios": datos.get("horizonte_anios"),
        "supuestos": datos.get("supuestos", {}),
        "justificacion": datos.get("justificacion", ""),
        "modo": datos.get("modo"),
        "dca": datos.get("dca"),
        "crecimiento_anual_estimado": datos.get("crecimiento_anual_estimado"),
        "margen_seguridad_pct": datos.get("margen_seguridad_pct"),
        "puntuacion": puntuacion,
        "observaciones": observaciones,
        "resumen": resumen,
    }

    registro["inicio"] = datos.get("inicio")
    registro["fin"] = datos.get("fin")

    backtest_payload = {
        "ticker": registro.get("ticker"),
        "importe_inicial": registro.get("importe_inicial"),
        "horizonte_anios": registro.get("horizonte_anios"),
        "modo": registro.get("modo"),
        "dca": registro.get("dca"),
        "inicio": registro.get("inicio"),
        "fin": registro.get("fin"),
    }
    backtest_snapshot = None
    try:
        backtest_snapshot = _market_backtest_core(backtest_payload)
    except BacktestError:
        backtest_snapshot = None
    except Exception:
        backtest_snapshot = None

    registro["backtest"] = backtest_snapshot

    registro = _sanear_registro(registro)

    user_id = _current_user_id()
    if user_id and not _is_guest_user():
        save_analysis_for_user(
            user_id=user_id,
            ticker=registro.get("ticker"),
            payload={
                "importe_inicial": registro.get("importe_inicial"),
                "horizonte_anios": registro.get("horizonte_anios"),
                "supuestos": registro.get("supuestos", {}),
                "justificacion": registro.get("justificacion", ""),
                "modo": registro.get("modo"),
                "dca": registro.get("dca"),
                "crecimiento_anual_estimado": registro.get(
                    "crecimiento_anual_estimado"
                ),
                "margen_seguridad_pct": registro.get("margen_seguridad_pct"),
                "inicio": registro.get("inicio"),
                "fin": registro.get("fin"),
            },
            result={
                "puntuacion": registro.get("puntuacion"),
                "observaciones": registro.get("observaciones"),
                "resumen": registro.get("resumen"),
                "backtest": registro.get("backtest"),
            },
        )

    return jsonify(
        {
            "valido": True,
            "puntuacion": puntuacion,
            "observaciones": observaciones,
            "resumen": resumen,
            "registro": {
                "id": registro["id"],
                "timestamp": registro["timestamp"],
                "ticker": registro["ticker"],
                "importe_inicial": registro["importe_inicial"],
                "horizonte_anios": registro["horizonte_anios"],
                "modo": registro["modo"],
                "dca": registro["dca"],
            },
        }
    )


@bp.post("/analisis")
def crear_analisis():
    datos_brutos = request.get_json(silent=True) or {}
    datos = _normalizar_payload(datos_brutos)
    return _registrar_analisis(datos)


@bp.post("/api/propuestas")
def crear_propuesta_api():
    datos_brutos = request.get_json(silent=True) or {}
    datos = _normalizar_payload(datos_brutos)
    return _registrar_analisis(datos)


@bp.get("/analisis")
def listar_analisis():
    """
    Devuelve el historial de análisis, mostrando primero los más recientes.
    Filtros opcionales (se aplican ANTES del paginado):
      - ?ticker=MSFT      (case-insensitive, igualdad exacta)
      - ?desde=YYYY-MM-DD (inclusive por fecha de timestamp)
      - ?hasta=YYYY-MM-DD (exclusivo del dÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a siguiente; simplificamos usando prefijos)
    Paginado opcional:
      - ?page, ?per_page
    """
    user_id = _current_user_id()
    if _is_guest_user():
        return (
            jsonify({"error": "El modo invitado no dispone de historial guardado."}),
            403,
        )
    if not user_id:
        return (
            jsonify({"error": "Debes iniciar sesión para consultar tu historial."}),
            401,
        )

    ticker = request.args.get("ticker")
    desde = request.args.get("desde")
    hasta = request.args.get("hasta")
    historial = list_analysis_for_user(user_id, ticker=ticker, desde=desde, hasta=hasta)

    page = request.args.get("page")
    per_page = request.args.get("per_page")
    if page or per_page:
        return jsonify(_paginate(historial, page, per_page))
    return jsonify(historial)


@bp.get("/analisis.csv")
def exportar_analisis_csv():
    """
    Exporta el historial de análisis en CSV (UTF-8 con BOM para Excel).
    Acepta los mismos filtros que GET /analisis: ?ticker, ?desde, ?hasta
    """
    user_id = _current_user_id()
    if _is_guest_user():
        return (
            jsonify({"error": "El modo invitado no permite exportar historial."}),
            403,
        )
    if not user_id:
        return (
            jsonify({"error": "Debes iniciar sesión para exportar tu historial."}),
            401,
        )

    ticker = request.args.get("ticker")
    desde = request.args.get("desde")
    hasta = request.args.get("hasta")
    historial = list_analysis_for_user(user_id, ticker=ticker, desde=desde, hasta=hasta)

    headers = [
        "id",
        "timestamp",
        "ticker",
        "importe_inicial",
        "horizonte_anios",
        "puntuacion",
        "resumen",
    ]

    has_backtest = any(h.get("backtest") for h in historial)
    if has_backtest:
        headers += ["bt_start", "bt_end", "bt_invested", "bt_final", "bt_pnl_pct"]

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(headers)
    for h in historial:
        row = [
            h.get("id", ""),
            h.get("timestamp", ""),
            h.get("ticker", ""),
            h.get("importe_inicial", ""),
            h.get("horizonte_anios", ""),
            h.get("puntuacion", ""),
            (h.get("resumen", "") or "").replace("\n", " ").strip(),
        ]
        if has_backtest:
            bt = h.get("backtest") or {}
            row.extend(
                [
                    bt.get("start") or "",
                    bt.get("end") or "",
                    bt.get("invested") if bt.get("invested") is not None else "",
                    bt.get("final_value") if bt.get("final_value") is not None else "",
                    bt.get("pnl_pct") if bt.get("pnl_pct") is not None else "",
                ]
            )
        w.writerow(row)

    # Añadimos BOM para que Excel detecte UTF-8 automáticamente
    csv_text = "\ufeff" + out.getvalue()

    return Response(
        csv_text,
        headers={
            "Content-Disposition": 'attachment; filename="analisis.csv"',
            "Content-Type": "text/csv; charset=utf-8",
        },
        status=200,
    )


# ----------------------
#   Helpers generales
# ----------------------


def _paginate(lista, page: str | None, per_page: str | None):
    p = int(page) if page and page.isdigit() and int(page) > 0 else 1
    pp = int(per_page) if per_page and per_page.isdigit() and int(per_page) > 0 else 10
    total = len(lista)
    start = (p - 1) * pp
    end = start + pp
    items = lista[start:end]
    has_next = end < total
    return {
        "items": items,
        "page": p,
        "per_page": pp,
        "total": total,
        "has_next": has_next,
    }


def _parse_date_yyyy_mm_dd(s: str | None):
    if not s:
        return None
    try:
        from datetime import datetime

        # interpretamos fecha en UTC a medianoche
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _filtrar_analisis(
    historial, ticker: str | None, desde: str | None, hasta: str | None
):
    if ticker:
        tnorm = _norm(ticker)
        historial = [h for h in historial if _norm(h.get("ticker", "")) == tnorm]
    if desde:
        historial = [h for h in historial if h.get("timestamp", "")[:10] >= desde]
    if hasta:
        historial = [h for h in historial if h.get("timestamp", "")[:10] < hasta]
    return historial


# --- Yahoo Finance: Datos de mercado y backtest ---
def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _extract_series(df: pd.DataFrame, column: str, ticker: str) -> pd.Series:
    if df is None or df.empty or column not in df:
        return pd.Series(dtype=float)
    series = df[column].copy()
    if isinstance(series, pd.DataFrame):
        if ticker in series.columns:
            series = series[ticker]
        else:
            series = series.iloc[:, 0]
    return series


def _series_with_date_index(series: pd.Series) -> pd.Series:
    series = series.copy()
    if not series.empty:
        series.index = pd.to_datetime(series.index).date
    return series


def _series_to_map(series: pd.Series) -> dict[str, float | None]:
    return {
        str(idx): (None if pd.isna(val) else float(val)) for idx, val in series.items()
    }


def _first_price_on_or_after(series: pd.Series, target: date) -> float | None:
    if series.empty:
        return None
    for idx, value in series.sort_index().items():
        if idx >= target and not pd.isna(value):
            return float(value)
    return None


def _last_price_on_or_before(series: pd.Series, target: date) -> float | None:
    if series.empty:
        return None
    for idx in series.sort_index().index[::-1]:
        value = series.loc[idx]
        if idx <= target and not pd.isna(value):
            return float(value)
    return None


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


class BacktestError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class HorizonSimulationError(Exception):
    def __init__(
        self, message: str, status_code: int = 400, warnings: list[str] | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.warnings = warnings or []


def _download_history_df(
    ticker: str, start_d: date, end_d: date, include_actions: bool = True
) -> pd.DataFrame:
    ticker_clean = (ticker or "").strip().upper()
    not_found_msg = (
        f"No se encontraron datos para el ticker '{ticker_clean}'. "
        "Asegúrate de usar el símbolo bursátil correcto."
    )
    provider_limited_msg = (
        "La fuente de datos de mercado ha limitado temporalmente la petición. "
        "Hemos reintentado varias veces, pero no hemos podido obtener datos ahora mismo. "
        "Espera unos segundos y vuelve a intentarlo."
    )
    generic_provider_msg = (
        f"No se pudieron obtener datos de mercado para '{ticker_clean}' en este momento. "
        "Hemos reintentado automáticamente la descarga, pero la fuente sigue sin responder de forma estable."
    )
    if not ticker_clean:
        raise BacktestError(not_found_msg, 404)

    cache_key = (ticker_clean, str(start_d), str(end_d), bool(include_actions))
    cached = HORIZON_HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    saw_rate_limit = False
    saw_transient_empty = False
    last_exception = None

    for attempt, delay_seconds in enumerate(
        HORIZON_HISTORY_RETRY_DELAYS_SECONDS, start=1
    ):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            df = yf.download(
                ticker_clean,
                start=str(start_d),
                end=str(end_d + timedelta(days=1)),
                interval="1d",
                auto_adjust=False,
                actions=include_actions,
                progress=False,
            )
            df = _normalize_price_df(df)
            if df is None or df.empty:
                saw_transient_empty = True
                last_exception = BacktestError(
                    "Descarga vacía o sin datos normalizables", 503
                )
                continue
            extracted_series = _extract_market_price_series(df, ticker_clean)
            if extracted_series.empty:
                saw_transient_empty = True
                last_exception = BacktestError(
                    "Serie histórica vacía tras normalización", 503
                )
                continue
            HORIZON_HISTORY_CACHE[cache_key] = df.copy()
            return df
        except YFRateLimitError as exc:
            saw_rate_limit = True
            last_exception = exc
            continue
        except BacktestError as exc:
            last_exception = exc
            if exc.status_code >= 500:
                continue
            raise
        except Exception as exc:
            last_exception = exc
            continue

    if saw_rate_limit:
        raise BacktestError(provider_limited_msg, 503) from last_exception
    if saw_transient_empty:
        raise BacktestError(generic_provider_msg, 503) from last_exception
    raise BacktestError(not_found_msg, 404) from last_exception


def _extract_market_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    normalized_df = _normalize_price_df(df)
    series = _extract_series(normalized_df, "Adj Close", ticker)
    if series.empty:
        series = _extract_series(normalized_df, "Close", ticker)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if not series.empty else pd.Series(dtype=float)
    if series is None:
        return pd.Series(dtype=float)

    series = pd.to_numeric(pd.Series(series).copy(), errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype=float)

    datetime_index = pd.to_datetime(series.index, errors="coerce")
    valid_mask = ~pd.isna(datetime_index)
    series = series.loc[valid_mask]
    datetime_index = datetime_index[valid_mask]
    if series.empty:
        return pd.Series(dtype=float)

    series.index = pd.DatetimeIndex(datetime_index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    return series


def _get_horizon_history_years(horizon_years: int) -> int:
    if horizon_years <= 1:
        return max(HORIZON_MIN_HISTORY_YEARS, 5)
    if horizon_years >= 5:
        return HORIZON_MAX_HISTORY_YEARS
    if horizon_years >= 3:
        return HORIZON_DEFAULT_HISTORY_YEARS
    return max(HORIZON_MIN_HISTORY_YEARS, 7)


def _downsample_horizon_series(
    series: pd.Series, max_points: int = HORIZON_MAX_HISTORY_POINTS
) -> pd.Series:
    if series.empty or len(series) <= max_points:
        return series
    step = max(1, math.ceil(len(series) / max_points))
    sampled = series.iloc[::step].copy()
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
        sampled = sampled[~sampled.index.duplicated(keep="last")]
    return sampled


def _compute_horizon_monthly_returns(
    series: pd.Series, ticker: str
) -> tuple[pd.Series, dict[str, bool]]:
    if series.empty:
        raise HorizonSimulationError(
            f"{ticker} no dispone de una serie temporal válida para construir la simulación experimental.",
            400,
        )
    monthly = series.resample("ME").last().pct_change()
    monthly = monthly.replace([math.inf, -math.inf], pd.NA).dropna()
    if monthly.empty:
        raise HorizonSimulationError(
            f"{ticker} no genera retornos mensuales suficientes para esta simulación experimental.",
            400,
        )
    outlier_mask = monthly.abs() > HORIZON_MAX_MONTHLY_RETURN
    had_outliers = bool(outlier_mask.any())
    monthly = monthly.clip(
        lower=-HORIZON_MAX_MONTHLY_RETURN, upper=HORIZON_MAX_MONTHLY_RETURN
    )
    if monthly.empty:
        raise HorizonSimulationError(
            f"{ticker} no genera retornos mensuales suficientes para esta simulación experimental.",
            400,
        )
    return monthly, {"had_outliers": had_outliers}


def _compute_price_summary(
    ticker: str, start_d: date, end_d: date, df: pd.DataFrame
) -> dict:
    adj = _series_with_date_index(_extract_series(df, "Adj Close", ticker))
    close = _series_with_date_index(_extract_series(df, "Close", ticker))
    dividends = _series_with_date_index(_extract_series(df, "Dividends", ticker))

    if adj.empty and close.empty:
        raise BacktestError("Sin datos de precios para el rango solicitado", 404)

    notes: list[str] = []

    start_price_adj = _first_price_on_or_after(adj, start_d)
    if start_price_adj is None:
        notes.append("Precio inicial ajustado no disponible en el rango")

    end_price_adj = _last_price_on_or_before(adj, end_d)
    if end_price_adj is None:
        notes.append("Precio final ajustado no disponible en el rango")

    start_price = _first_price_on_or_after(close, start_d)
    if start_price is None:
        notes.append("Precio inicial sin ajustar no disponible en el rango")

    end_price = _last_price_on_or_before(close, end_d)
    if end_price is None:
        notes.append("Precio final sin ajustar no disponible en el rango")

    variation_adj_pct = None
    if start_price_adj and end_price_adj and start_price_adj != 0:
        variation_adj_pct = (end_price_adj / start_price_adj - 1) * 100

    variation_raw_pct = None
    if start_price and end_price and start_price != 0:
        variation_raw_pct = (end_price / start_price - 1) * 100

    has_dividends = (
        bool(dividends.fillna(0).ne(0).any()) if not dividends.empty else False
    )

    now_price = None
    try:
        fast_info = yf.Ticker(ticker).fast_info
        candidate = getattr(fast_info, "last_price", None)
        if candidate is not None and not (
            isinstance(candidate, float) and math.isnan(candidate)
        ):
            now_price = float(candidate)
    except Exception:
        now_price = None

    if now_price is None:
        fallback = end_price_adj if end_price_adj is not None else None
        if fallback is None and not adj.dropna().empty:
            fallback = float(adj.dropna().iloc[-1])
        if fallback is not None:
            now_price = fallback
            notes.append("Tiempo real no disponible; se usa ultimo cierre ajustado")
        else:
            notes.append("Tiempo real no disponible")

    return {
        "start_price_adj": _round_or_none(start_price_adj, 4),
        "end_price_adj": _round_or_none(end_price_adj, 4),
        "variation_adj_pct": _round_or_none(variation_adj_pct, 2),
        "start_price": _round_or_none(start_price, 4),
        "end_price": _round_or_none(end_price, 4),
        "variation_raw_pct": _round_or_none(variation_raw_pct, 2),
        "now_price": _round_or_none(now_price, 4) if now_price is not None else None,
        "has_dividends": has_dividends,
        "notes": notes,
        "adj_series": adj,
    }


def _iso(d):
    if pd.isna(d):
        return None
    return str(pd.to_datetime(d).date())


@bp.get("/market/ohlc/<ticker>")
def market_ohlc(ticker):
    """
    Devuelve OHLCV + Adj Close para un ticker.
    Query params:
      - start=YYYY-MM-DD
      - end=YYYY-MM-DD
      - interval=1d|1wk|1mo (default 1d)
    Respuesta: lista de objetos {date, open, high, low, close, adj_close, volume}
    """
    t = (ticker or "").strip()
    if not t:
        return jsonify({"error": "Ticker requerido"}), 400

    start = request.args.get("start")
    end = request.args.get("end")
    interval = request.args.get("interval", "1d")

    try:
        df = yf.download(
            t,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return jsonify([])

        df = _normalize_price_df(df)

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        ).reset_index()

        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "date": _iso(r.get("Date")),
                    "open": None if pd.isna(r.get("open")) else float(r.get("open")),
                    "high": None if pd.isna(r.get("high")) else float(r.get("high")),
                    "low": None if pd.isna(r.get("low")) else float(r.get("low")),
                    "close": None if pd.isna(r.get("close")) else float(r.get("close")),
                    "adj_close": (
                        None
                        if pd.isna(r.get("adj_close"))
                        else float(r.get("adj_close"))
                    ),
                    "volume": (
                        None if pd.isna(r.get("volume")) else int(r.get("volume"))
                    ),
                }
            )
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _gen_schedule(start_date: date, end_date: date, freq: str):
    step = {"WEEKLY": 7, "MONTHLY": 30, "QUARTERLY": 91, "ANNUAL": 365}.get(
        (freq or "").upper(), 30
    )
    d = start_date
    while d <= end_date:
        yield d
        d = d + timedelta(days=step)


def _nearest_trading_close(adj_close_by_day: dict, d: date):
    # busca el primer dÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a con dato >= fecha objetivo (forward fill hacia adelante)
    for i in range(0, 14):
        k = str(d + timedelta(days=i))
        v = adj_close_by_day.get(k)
        if v is not None:
            return float(v)
    return None


def _market_backtest_core(payload: dict) -> dict:
    t = (payload.get("ticker") or "").strip().upper()
    if not t:
        raise BacktestError("Ticker requerido", 400)

    try:
        horizon = int(payload.get("horizonte_anios") or 0)
    except (TypeError, ValueError):
        raise BacktestError("Horizonte invalido", 400)
    if horizon < 1:
        raise BacktestError("Horizonte >= 1 año", 400)

    try:
        invested_initial = float(payload.get("importe_inicial") or 0)
    except (TypeError, ValueError):
        invested_initial = 0.0
    if invested_initial <= 0:
        raise BacktestError("Importe inicial debe ser mayor a 0", 400)

    modo = (payload.get("modo") or "SIN_DCA").upper()
    if modo not in {"DCA", "SIN_DCA"}:
        modo = "SIN_DCA"

    dca = payload.get("dca") or {}
    aporte = 0.0
    freq = "MONTHLY"
    if modo == "DCA" and isinstance(dca, dict):
        try:
            aporte = float(dca.get("aporte") or 0.0)
        except (TypeError, ValueError):
            aporte = 0.0
        if aporte < 0:
            raise BacktestError("Aporte DCA no puede ser negativo", 400)
        freq = (dca.get("frecuencia") or "MONTHLY").upper()

    today = date.today()
    start_raw = payload.get("inicio")
    if start_raw:
        start_d = _parse_iso_date(start_raw)
        if not start_d:
            raise BacktestError("Fecha de inicio invalida", 400)
    else:
        try:
            start_d = today.replace(year=today.year - horizon)
        except ValueError:
            start_d = today - timedelta(days=365 * horizon)

    end_raw = payload.get("fin")
    if end_raw:
        end_d = _parse_iso_date(end_raw)
        if not end_d:
            raise BacktestError("Fecha de fin invalida", 400)
    else:
        end_d = today

    if end_d < start_d:
        raise BacktestError(
            "La fecha de fin debe ser posterior o igual a la inicial", 400
        )

    df = _download_history_df(t, start_d, end_d, include_actions=True)
    if df is None or df.empty:
        raise BacktestError("Sin datos para el rango solicitado", 404)

    metrics = _compute_price_summary(t, start_d, end_d, df)
    adj = metrics.get("adj_series")
    if adj is None or adj.empty:
        raise BacktestError("Sin datos de precios ajustados", 404)

    adj_map = _series_to_map(adj)

    first_px = _nearest_trading_close(adj_map, start_d)
    if first_px is None:
        raise BacktestError("No hay precio inicial cercano", 404)

    invested = invested_initial
    shares = invested / first_px if first_px else 0.0

    if modo == "DCA" and aporte > 0:
        for d in _gen_schedule(start_d + timedelta(days=1), end_d, freq):
            px = _nearest_trading_close(adj_map, d)
            if px:
                invested += aporte
                shares += aporte / px

    last_px = _last_price_on_or_before(adj, end_d)
    if last_px is None and not adj.dropna().empty:
        last_px = float(adj.dropna().iloc[-1])
    if last_px is None:
        raise BacktestError("No hay precio final disponible", 404)

    final_value = shares * last_px
    pnl_abs = final_value - invested
    pnl_pct = (pnl_abs / invested) * 100 if invested > 0 else 0.0

    result = {
        "ticker": t,
        "start": str(start_d),
        "end": str(end_d),
        "desde": str(start_d),
        "hasta": str(end_d),
        "invested": _round_or_none(invested, 2),
        "shares": float(shares),
        "last_price": _round_or_none(last_px, 4),
        "final_value": _round_or_none(final_value, 2),
        "pnl_abs": _round_or_none(pnl_abs, 2),
        "pnl_pct": _round_or_none(pnl_pct, 2),
        "modo": modo,
        "start_price_adj": metrics["start_price_adj"],
        "end_price_adj": metrics["end_price_adj"],
        "variation_adj_pct": metrics["variation_adj_pct"],
        "start_price": metrics["start_price"],
        "end_price": metrics["end_price"],
        "variation_raw_pct": metrics["variation_raw_pct"],
        "now_price": metrics["now_price"],
        "has_dividends": metrics["has_dividends"],
        "notes": metrics["notes"],
    }

    metrics.pop("adj_series", None)
    return result


@bp.post("/market/backtest")
def market_backtest():
    """
    Calcula el resultado real de una inversión usando precios ajustados (Adj Close).
    Body JSON esperado:
    {
      "ticker": "AAPL",
      "importe_inicial": 1000,
      "horizonte_anios": 3,
      "modo": "DCA"|"SIN_DCA",
      "dca": {"aporte": 100, "frecuencia": "MONTHLY"} | null,
      "inicio": "YYYY-MM-DD" (opcional; por defecto hoy - horizonte_anios),
      "fin": "YYYY-MM-DD" (opcional; por defecto hoy)
    }
    """
    payload = request.get_json(silent=True) or {}
    try:
        result = _market_backtest_core(payload)
        return jsonify(result)
    except BacktestError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@bp.get("/market/summary")
def market_summary():
    t = (request.args.get("ticker") or "").strip().upper()
    if not t:
        return jsonify({"error": "Ticker requerido"}), 400

    start_raw = request.args.get("start")
    if not start_raw:
        return jsonify({"error": "Parametro start requerido"}), 400
    start_d = _parse_iso_date(start_raw)
    if not start_d:
        return jsonify({"error": "Fecha de inicio invalida"}), 400

    end_raw = request.args.get("end")
    if end_raw:
        end_d = _parse_iso_date(end_raw)
        if not end_d:
            return jsonify({"error": "Fecha de fin invalida"}), 400
    else:
        end_d = date.today()

    if end_d < start_d:
        return (
            jsonify(
                {"error": "La fecha de fin debe ser posterior o igual a la inicial"}
            ),
            400,
        )

    # adjusted = _as_bool(request.args.get("adjusted"), True) # no usado

    df = _download_history_df(t, start_d, end_d, include_actions=True)
    if df is None or df.empty:
        return jsonify({"error": "Sin datos para el rango solicitado"}), 404

    try:
        metrics = _compute_price_summary(t, start_d, end_d, df)
    except BacktestError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    metrics.pop("adj_series", None)

    response = {
        "ticker": t,
        "start": str(start_d),
        "end": str(end_d),
        "start_price_adj": metrics["start_price_adj"],
        "end_price_adj": metrics["end_price_adj"],
        "variation_adj_pct": metrics["variation_adj_pct"],
        "start_price": metrics["start_price"],
        "end_price": metrics["end_price"],
        "variation_raw_pct": metrics["variation_raw_pct"],
        "now_price": metrics["now_price"],
        "has_dividends": metrics["has_dividends"],
        "notes": metrics["notes"],
    }
    return jsonify(response)


@bp.get("/market/ohlc_csv")
def market_ohlc_csv():
    t = (request.args.get("ticker") or "").strip().upper()
    if not t:
        return jsonify({"error": "Ticker requerido"}), 400

    start_raw = request.args.get("start")
    if not start_raw:
        return jsonify({"error": "Parametro start requerido"}), 400
    start_d = _parse_iso_date(start_raw)
    if not start_d:
        return jsonify({"error": "Fecha de inicio invalida"}), 400

    end_raw = request.args.get("end")
    if end_raw:
        end_d = _parse_iso_date(end_raw)
        if not end_d:
            return jsonify({"error": "Fecha de fin invalida"}), 400
    else:
        end_d = date.today()

    if end_d < start_d:
        return (
            jsonify(
                {"error": "La fecha de fin debe ser posterior o igual a la inicial"}
            ),
            400,
        )

    adjusted = _as_bool(request.args.get("adjusted"), True)

    df = yf.download(
        t,
        start=str(start_d),
        end=str(end_d + timedelta(days=1)),
        interval="1d",
        auto_adjust=False if not adjusted else False,
        actions=True,
        progress=False,
    )
    df = _normalize_price_df(df)
    if df is None or df.empty:
        return jsonify({"error": "Sin datos para el rango solicitado"}), 404

    df_out = df.reset_index().copy()
    columns_order = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    for column in columns_order:
        if column not in df_out.columns:
            df_out[column] = None
    df_out = df_out[columns_order]
    df_out = df_out.sort_values("Date")

    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    filename = f"{t}_{start_d}_{end_d}{'_adj' if adjusted else ''}.csv"
    return Response(
        buffer.getvalue(),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8",
        },
        status=200,
    )
