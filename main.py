"""AgriSmart backend.

Run:  uvicorn main:app --reload
Open: http://127.0.0.1:8000
"""
import io
import logging
import os
import textwrap
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

log = logging.getLogger("agrismart")
BASE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "soil_crop_model.pkl"))

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_TTL = 3600  # seconds
# SoilGrids reports TOTAL nitrogen (g/kg); the model expects a plant-available index (0-140).
# This factor is a rough proxy, not a lab value.
N_SCALE = 25

# Illustrative planning figures only. Replace with crop-specific economics.
ECON = {
    "Stable": ("Minimal (< 4%)", "$0 / acre"),
    "Watch": ("10% - 20%", "$450 / acre"),
    "High risk": ("38% - 52%", "$1,620 / acre"),
}

_soil_cache: dict[tuple, tuple[float, dict]] = {}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = joblib.load(MODEL_PATH)
        log.info("Loaded model from %s", MODEL_PATH)
    except FileNotFoundError:
        app.state.model_error = f"Model file not found at {MODEL_PATH}"
    except Exception as exc:  # corrupt file, sklearn version mismatch, ...
        app.state.model_error = f"Could not load model: {exc}"
    if app.state.model_error:
        log.error(app.state.model_error)
    app.state.http = httpx.AsyncClient(timeout=12)
    yield
    await app.state.http.aclose()


app = FastAPI(title="AgriSmart", lifespan=lifespan)


def get_model(request: Request):
    if request.app.state.model is None:
        raise HTTPException(503, request.app.state.model_error or "Model unavailable")
    return request.app.state.model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Params(BaseModel):
    N: float = Field(ge=0, le=140)
    P: float = Field(ge=5, le=145)
    K: float = Field(ge=5, le=205)
    temperature: float = Field(ge=8, le=44)
    humidity: float = Field(ge=14, le=100)
    ph: float = Field(ge=3.5, le=10)
    rainfall: float = Field(ge=20, le=300)


class ReportRequest(Params):
    location: str = "No point selected"
    source: str = "Manual inputs"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def assess(p: Params):
    infertile = (p.ph < 5.0 or p.ph > 8.0) or (p.N < 20 and p.P < 15)
    gap = 0 if 6.0 <= p.ph <= 7.5 else (6.0 - p.ph if p.ph < 6.0 else p.ph - 7.5)
    ph_score = max(0.0, 1 - gap / 2.5)
    nutrient = (min(p.N / 50, 1) + min(p.P / 40, 1) + min(p.K / 40, 1)) / 3
    score = round(100 * (0.4 * ph_score + 0.6 * nutrient))  # heuristic index, not agronomically validated
    tier = "High risk" if infertile else ("Stable" if score >= 65 else "Watch")
    return score, tier


def advice(p: Params):
    tips = []
    if p.ph < 5.0:
        tips.append("Soil is strongly acidic: apply agricultural lime.")
    elif p.ph < 6.0:
        tips.append("Soil is mildly acidic: a light lime application will help most crops.")
    elif p.ph > 8.0:
        tips.append("Soil is alkaline: use organic compost or elemental sulfur.")
    elif p.ph > 7.5:
        tips.append("Soil is mildly alkaline: add compost and watch micronutrient levels.")
    if p.N < 30:
        tips.append("Nitrogen is low: add compost or urea, or rotate with legumes.")
    if p.P < 25:
        tips.append("Phosphorus is low: apply a phosphate fertiliser such as SSP or DAP.")
    if p.K < 25:
        tips.append("Potassium is low: apply muriate of potash.")
    return tips or ["Parameters sit in a good range for high yield."]


def analyze(model, p: Params) -> dict:
    cols = list(getattr(model, "feature_names_in_", FEATURES))  # match training column order
    df = pd.DataFrame([p.model_dump()])[cols]
    crop = str(model.predict(df)[0])
    top = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        for i in probs.argsort()[::-1][:3]:
            top.append({"crop": str(model.classes_[i]), "probability": round(float(probs[i]), 4)})
    score, tier = assess(p)
    yield_drop, loss = ECON[tier]
    return {
        "crop": crop,
        "top": top,
        "score": score,
        "tier": tier,
        "yield_impact": yield_drop,
        "financial_risk": loss,
        "tips": advice(p),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health(request: Request):
    return {"ok": request.app.state.model is not None, "error": request.app.state.model_error}


@app.post("/api/predict")
def predict(params: Params, request: Request):
    return analyze(get_model(request), params)


@app.get("/api/soil")
async def soil(request: Request, lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-540, le=540)):
    lon = ((lon + 180) % 360) - 180  # wrap longitudes from a panned map
    key = (round(lat, 3), round(lon, 3))
    hit = _soil_cache.get(key)
    if hit and time.time() - hit[0] < SOIL_TTL:
        return hit[1]

    params = [("lat", key[0]), ("lon", key[1]), ("property", "phh2o"), ("property", "nitrogen"),
              ("property", "soc"), ("depth", "0-5cm"), ("value", "mean")]
    try:
        resp = await request.app.state.http.get(SOILGRIDS_URL, params=params)
    except httpx.TimeoutException:
        raise HTTPException(504, "SoilGrids timed out. Try again in a moment.")
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach SoilGrids.")
    if resp.status_code == 429:
        raise HTTPException(429, "SoilGrids rate limit reached (about 5 requests/min). Wait a minute and click again.")
    if resp.status_code != 200:
        raise HTTPException(502, f"SoilGrids returned status {resp.status_code}.")

    raw = {}
    try:
        for layer in resp.json()["properties"]["layers"]:
            factor = layer.get("unit_measure", {}).get("d_factor") or 1  # mapped -> conventional units
            val = layer["depths"][0]["values"].get("mean")
            raw[layer["name"]] = None if val is None else val / factor
    except (KeyError, IndexError, ValueError):
        raise HTTPException(502, "Unexpected response from SoilGrids.")

    if raw.get("phh2o") is None and raw.get("nitrogen") is None:
        raise HTTPException(404, "No soil data at this point (water, ice or built-up area). Try nearby land.")

    ph, n_g, soc = raw.get("phh2o"), raw.get("nitrogen"), raw.get("soc")
    result = {
        "lat": key[0],
        "lon": key[1],
        "ph": round(min(max(ph, 3.5), 10.0), 1) if ph is not None else None,
        "N": int(min(max(n_g * N_SCALE, 0), 140)) if n_g is not None else None,
        "soc": round(soc, 1) if soc is not None else None,
    }
    _soil_cache[key] = (time.time(), result)
    return result


@app.post("/api/report")
def report(req: ReportRequest, request: Request):
    model = get_model(request)
    params = Params(**req.model_dump(include=set(FEATURES)))
    res = analyze(model, params)

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    y = 750

    def line(text, font="Helvetica", size=11, gap=20, x=50, wrap=90):
        nonlocal y
        pdf.setFont(font, size)
        for i, part in enumerate(textwrap.wrap(text, wrap) or [""]):
            pdf.drawString(x + (12 if i else 0), y, part)
            y -= gap if i == len(textwrap.wrap(text, wrap)) - 1 else 15

    line("AgriSmart Live Intelligence Lab Report", "Helvetica-Bold", 18, 30)
    line(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    line(f"Location (lat, lng): {req.location}")
    line(f"Data source: {req.source}", gap=26)
    line(f"Recommended crop: {res['crop'].upper()}", "Helvetica-Bold")
    line(f"Fertility score: {res['score']}/100    Risk level: {res['tier'].upper()}")
    line(f"Projected yield impact: {res['yield_impact']}")
    line(f"Estimated financial exposure: {res['financial_risk']} (illustrative)", gap=28)
    if res["top"]:
        line("Top crop matches:", "Helvetica-Bold")
        for t in res["top"]:
            line(f"- {t['crop'].title()}: {t['probability']:.0%}", x=70)
        y -= 8
    line("Parameters:", "Helvetica-Bold")
    for col in FEATURES:
        line(f"- {col}: {getattr(params, col)}", x=70)
    y -= 8
    line("Recommendations:", "Helvetica-Bold")
    for tip in res["tips"]:
        line(f"- {tip}", x=70, wrap=80)
    y -= 8
    line("Sources: ISRIC SoilGrids REST API and the trained crop-matching model.", size=9)
    pdf.showPage()
    pdf.save()

    return Response(
        buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="AgriSmart_Report.pdf"'},
    )


# Static frontend last, so /api/* routes win.
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")