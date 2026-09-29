import inspect
import io
from datetime import datetime

import folium
import joblib
import pandas as pd
import requests
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_folium import st_folium

st.set_page_config(page_title="Soil Infertility Rate Prediction Model", page_icon="🌍", layout="wide")

# Streamlit replaced `use_container_width=True` with `width="stretch"` in newer versions.
STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DEFAULTS = {"N": 50, "P": 45, "K": 40, "temperature": 26.5, "humidity": 65.0, "ph": 6.5, "rainfall": 110.0}
PRESETS = {
    "Tropical": {"temperature": 28.0, "humidity": 85.0, "rainfall": 220.0},
    "Arid": {"temperature": 34.0, "humidity": 35.0, "rainfall": 45.0},
    "Temperate": {"temperature": 18.0, "humidity": 65.0, "rainfall": 100.0},
}
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
# SoilGrids gives TOTAL nitrogen (g/kg). The model expects a plant-available N index (0-140).
# This factor is a rough proxy (same scaling as the original code: cg/kg / 4), not a lab value.
N_SCALE = 25

# Illustrative planning figures only; replace with crop-specific economics.
ECON = {
    "Stable": ("Minimal (< 4%)", "$0 / acre"),
    "Watch": ("10% - 20%", "$450 / acre"),
    "High risk": ("38% - 52%", "$1,620 / acre"),
}
TIER_COLOR = {"Stable": "#3f7d3a", "Watch": "#b7791f", "High risk": "#b3382c"}

st.markdown(
    """
<style>
.ph-track{position:relative;height:14px;border-radius:7px;margin:.4rem 0 1.1rem;
  background:linear-gradient(90deg,#b3382c 0%,#d98e2b 20%,#c9b83a 32%,#4c7a34 38%,#4c7a34 62%,#2f6f9f 80%,#5b4a8a 100%)}
.ph-marker{position:absolute;top:-6px;width:5px;height:26px;border-radius:3px;background:#fff;
  border:2px solid #2b2420;transition:left .5s ease}
.ph-scale{display:flex;justify-content:space-between;font-size:.75rem;opacity:.7;margin-top:-.7rem}
.badge{display:inline-block;padding:.2rem .8rem;border-radius:999px;color:#fff;font-weight:600}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("soil_crop_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("`soil_crop_model.pkl` was not found next to this script. Add the trained model and reload.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ----------------------------------------------------------------------------
# Soil data (ISRIC SoilGrids)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _query_soilgrids(lat: float, lon: float) -> dict:
    """Cached per coordinate. Exceptions are NOT cached, so failures retry on the next click."""
    resp = requests.get(
        SOILGRIDS_URL,
        params=[("lat", lat), ("lon", lon), ("property", "phh2o"), ("property", "nitrogen"),
                ("property", "soc"), ("depth", "0-5cm"), ("value", "mean")],
        timeout=10,
    )
    resp.raise_for_status()
    out = {}
    for layer in resp.json()["properties"]["layers"]:
        factor = layer.get("unit_measure", {}).get("d_factor") or 1  # convert mapped -> conventional units
        val = layer["depths"][0]["values"].get("mean")
        out[layer["name"]] = None if val is None else val / factor
    return out


def sync_soil(lat: float, lon: float) -> dict:
    lon = ((lon + 180) % 360) - 180  # the map can return wrapped longitudes
    try:
        raw = _query_soilgrids(round(lat, 3), round(lon, 3))
    except requests.Timeout:
        return {"ok": False, "msg": "SoilGrids timed out. Its public API is rate limited (about 5 calls/min); wait a moment and click again."}
    except Exception as exc:
        return {"ok": False, "msg": f"SoilGrids request failed ({exc}). Sliders keep their current values."}

    if raw.get("phh2o") is None and raw.get("nitrogen") is None:
        return {"ok": False, "msg": "No soil data at this point (water, ice or built-up area). Try a nearby land location."}

    ph = raw.get("phh2o")
    n_g = raw.get("nitrogen")
    return {
        "ok": True,
        "ph": round(min(max(ph, 3.5), 10.0), 1) if ph is not None else None,
        "N": int(min(max(n_g * N_SCALE, 0), 140)) if n_g is not None else None,
        "soc": round(raw["soc"], 1) if raw.get("soc") is not None else None,
    }


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)
st.session_state.setdefault("soil", None)
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("location", "No point selected yet")
st.session_state.setdefault("sync_msg", None)
st.session_state.setdefault("prev_crop", None)
st.session_state.setdefault("map_view", {"center": [20.5937, 78.9629], "zoom": 4})


def apply_preset(name):
    st.session_state.update(PRESETS[name])


def resync_from_map():
    soil = st.session_state.soil
    if soil:
        if soil.get("ph") is not None:
            st.session_state.ph = soil["ph"]
        if soil.get("N") is not None:
            st.session_state.N = soil["N"]


# ----------------------------------------------------------------------------
# Header + tabs
# ----------------------------------------------------------------------------
st.title("🌍 AgriSmart: Live Soil Intelligence")
st.caption("Click the map to pull topsoil pH and nitrogen from ISRIC SoilGrids. Every slider re-runs the crop match and risk score instantly.")

tab_map, tab_analysis, tab_report = st.tabs(["🗺️ Locate field", "📊 Live analysis", "📄 Report"])

# ---- Map tab (runs before the sidebar so a click can update slider values) ----
with tab_map:
    style = st.radio("Map style", ["Street", "Satellite"], horizontal=True, label_visibility="collapsed")
    view = st.session_state.map_view
    fmap = folium.Map(location=view["center"], zoom_start=view["zoom"], tiles=None)
    if style == "Satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
        ).add_to(fmap)
    else:
        folium.TileLayer("OpenStreetMap").add_to(fmap)
    if st.session_state.last_click:
        folium.Marker(list(st.session_state.last_click), tooltip=st.session_state.location).add_to(fmap)

    map_data = st_folium(
        fmap, height=400, key="farm_map", use_container_width=True,
        returned_objects=["last_clicked", "center", "zoom"],
    )

    if map_data:
        if map_data.get("center") and map_data.get("zoom"):
            c = map_data["center"]
            st.session_state.map_view = {"center": [c["lat"], c["lng"]], "zoom": map_data["zoom"]}

        click = map_data.get("last_clicked")
        if click:
            point = (round(click["lat"], 5), round(click["lng"], 5))
            if point != st.session_state.last_click:  # only fetch for a NEW click
                with st.spinner("Fetching topsoil data from SoilGrids..."):
                    res = sync_soil(*point)
                st.session_state.last_click = point
                st.session_state.location = f"{point[0]:.4f}, {point[1]:.4f}"
                if res["ok"]:
                    st.session_state.soil = res
                    resync_from_map()
                    st.session_state.sync_msg = ("success", f"Loaded live baseline for {st.session_state.location}.")
                else:
                    st.session_state.soil = None
                    st.session_state.sync_msg = ("warning", res["msg"])
                st.rerun()  # redraw with the marker in place

    if st.session_state.sync_msg:
        kind, text = st.session_state.sync_msg
        getattr(st, kind)(text)

    soil = st.session_state.soil
    if soil:
        m1, m2, m3 = st.columns(3)
        m1.metric("Topsoil pH (0-5 cm)", soil["ph"] if soil["ph"] is not None else "n/a")
        m2.metric("Nitrogen index", soil["N"] if soil["N"] is not None else "n/a", help="Scaled from total N; a proxy, not a lab value.")
        m3.metric("Organic carbon (g/kg)", soil["soc"] if soil["soc"] is not None else "n/a")

# ---- Sidebar ----
sb = st.sidebar
sb.header("🔬 Field parameters")
sb.caption(f"📍 {st.session_state.location}")

sb.caption("Climate presets")
for col, name in zip(sb.columns(len(PRESETS)), PRESETS):
    col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), **STRETCH)

sb.slider("Nitrogen (N) - kg/ha", 0, 140, key="N")
sb.slider("Phosphorus (P) - kg/ha", 5, 145, key="P")
sb.slider("Potassium (K) - kg/ha", 5, 205, key="K")
sb.slider("Temperature (°C)", 8.0, 44.0, step=0.5, key="temperature")
sb.slider("Humidity (%)", 14.0, 100.0, step=1.0, key="humidity")
sb.slider("Soil pH", 3.5, 10.0, step=0.1, key="ph")
sb.slider("Rainfall (mm)", 20.0, 300.0, step=5.0, key="rainfall")

soil = st.session_state.soil
sb.button("↺ Reset N and pH to map values", on_click=resync_from_map, disabled=soil is None, **STRETCH)

if soil is None:
    source = "Manual inputs"
else:
    edited = (soil.get("ph") is not None and abs(st.session_state.ph - soil["ph"]) > 0.05) or (
        soil.get("N") is not None and st.session_state.N != soil["N"]
    )
    source = "Live SoilGrids + manual edits" if edited else "Live SoilGrids"
sb.info(f"Data source: {source}")


# ----------------------------------------------------------------------------
# Analysis logic
# ----------------------------------------------------------------------------
def assess(n, p, k, ph):
    infertile = (ph < 5.0 or ph > 8.0) or (n < 20 and p < 15)
    ph_gap = 0 if 6.0 <= ph <= 7.5 else (6.0 - ph if ph < 6.0 else ph - 7.5)
    ph_score = max(0.0, 1 - ph_gap / 2.5)
    nutrient_score = (min(n / 50, 1) + min(p / 40, 1) + min(k / 40, 1)) / 3
    score = round(100 * (0.4 * ph_score + 0.6 * nutrient_score))
    tier = "High risk" if infertile else ("Stable" if score >= 65 else "Watch")
    return score, tier


def advice(n, p, k, ph):
    tips = []
    if ph < 5.0:
        tips.append("Soil is strongly acidic: apply agricultural lime.")
    elif ph < 6.0:
        tips.append("Soil is mildly acidic: a light lime application will help most crops.")
    elif ph > 8.0:
        tips.append("Soil is alkaline: use organic compost or elemental sulfur.")
    elif ph > 7.5:
        tips.append("Soil is mildly alkaline: add compost and watch micronutrient levels.")
    if n < 30:
        tips.append("Nitrogen is low: add compost or urea, or rotate with legumes.")
    if p < 25:
        tips.append("Phosphorus is low: apply a phosphate fertiliser such as SSP or DAP.")
    if k < 25:
        tips.append("Potassium is low: apply muriate of potash.")
    return tips or ["Parameters sit in a good range for high yield."]


def ph_strip(ph):
    pos = (ph - 3.5) / (10.0 - 3.5) * 100
    return (
        f'<div class="ph-track"><div class="ph-marker" style="left:calc({pos:.1f}% - 2px)"></div></div>'
        '<div class="ph-scale"><span>3.5 acidic</span><span>neutral</span><span>10 alkaline</span></div>'
    )


input_df = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])
input_df = input_df[list(getattr(model, "feature_names_in_", FEATURES))]  # match training column order

try:
    crop = str(model.predict(input_df)[0])
    top = None
    if hasattr(model, "predict_proba"):
        top = pd.Series(model.predict_proba(input_df)[0], index=model.classes_).sort_values(ascending=False).head(3)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

n, p, k, ph = (st.session_state[x] for x in ("N", "P", "K", "ph"))
score, tier = assess(n, p, k, ph)
yield_drop, loss = ECON[tier]
tips = advice(n, p, k, ph)

if st.session_state.prev_crop and st.session_state.prev_crop != crop:
    st.toast(f"Best crop changed: {st.session_state.prev_crop.title()} → {crop.title()}", icon="🌱")
st.session_state.prev_crop = crop

# ---- Analysis tab ----
with tab_analysis:
    c1, c2, c3 = st.columns(3)
    c1.metric("Best crop match", crop.title(), f"{top.iloc[0]:.0%} confidence" if top is not None else None, delta_color="off")
    c2.metric("Soil fertility score", f"{score} / 100")
    c3.markdown("Risk level")
    c3.markdown(f'<span class="badge" style="background:{TIER_COLOR[tier]}">{tier}</span>', unsafe_allow_html=True)
    st.progress(score / 100, text=f"Fertility index: {score}/100 (heuristic, based on pH and N-P-K)")

    st.markdown(f"**Soil pH {ph:.1f}**")
    st.markdown(ph_strip(ph), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Top crop matches")
        if top is not None:
            st.bar_chart(top.rename("Probability"), horizontal=True)
        else:
            st.info("This model does not expose class probabilities.")
    with right:
        st.subheader("Economic risk")
        e1, e2 = st.columns(2)
        e1.metric("Projected yield impact", yield_drop)
        e2.metric("Estimated financial risk", loss)
        st.caption("Illustrative planning figures, not a forecast.")
        st.subheader("Recommendations")
        for tip in tips:
            (st.warning if tier != "Stable" else st.info)(tip)

    with st.expander("Active parameter profile"):
        st.dataframe(input_df, **STRETCH)


# ---- Report tab ----
def create_pdf() -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    y = 750

    def line(text, font="Helvetica", size=11, gap=20, x=50):
        nonlocal y
        pdf.setFont(font, size)
        pdf.drawString(x, y, text)
        y -= gap

    line("AgriSmart Live Intelligence Lab Report", "Helvetica-Bold", 18, 30)
    line(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    line(f"Location (lat, lng): {st.session_state.location}")
    line(f"Data source: {source}")
    line(f"Recommended crop: {crop.upper()}")
    line(f"Fertility score: {score}/100   Risk level: {tier.upper()}", gap=20)
    line(f"Projected yield impact: {yield_drop}")
    line(f"Estimated financial exposure: {loss} (illustrative)", gap=30)
    if top is not None:
        line("Top crop matches:", "Helvetica-Bold")
        for name, prob in top.items():
            line(f"- {str(name).title()}: {prob:.0%}", x=70)
        y -= 10
    line("Parameters:", "Helvetica-Bold")
    for col in input_df.columns:
        line(f"- {col}: {input_df[col].values[0]}", x=70)
    y -= 10
    line("Recommendations:", "Helvetica-Bold")
    for tip in tips:
        line(f"- {tip}"[:95], x=70)
    y -= 10
    line("Sources: ISRIC SoilGrids REST API and the trained crop-matching model.", size=9)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


with tab_report:
    st.subheader("Soil health card")
    st.write(f"**{crop.title()}** on a **{tier.lower()}** field, fertility score **{score}/100**, at {st.session_state.location}.")
    st.download_button(
        "📄 Download soil health card (PDF)",
        data=create_pdf(),
        file_name="AgriSmart_Live_Report.pdf",
        mime="application/pdf",
        **STRETCH,
    )