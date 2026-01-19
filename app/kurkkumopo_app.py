"""
Streamlit application for interacting with the trained Kickstarter success model.

The app allows users to input project details, view model predictions,
inspect feature contributions, and receive qualitative feedback generated
by a language model (Perplexity).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import dotenv
import base64
from perplexity import Perplexity
from models.logreg_optimized import DATA_PATH, MODEL_PATH


# -----------------------------
# Model loading
# -----------------------------


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Mallia ei löydy polusta: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_data
def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataa ei löydy polusta: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


# -----------------------------
# Streamlit features
# -----------------------------


@st.cache_data
def get_category_options(df: pd.DataFrame):
    df = df.copy()

    def _extract_months(series: pd.Series) -> list[int]:
        """ Extract a sorted list of month values present in the given date column."""
        months = pd.to_datetime(series, errors="coerce").dt.month.dropna().unique()
        month_values = sorted(int(m) for m in months)
        return month_values or list(range(1, 13))

    parent_cats = sorted(df["category_parent_name"].dropna().unique())

    parent_to_children: dict[str, list[str]] = {}
    for parent in parent_cats:
        children = (
            df.loc[df["category_parent_name"] == parent, "category_name_reduced"]
            .dropna()
            .unique()
        )
        parent_to_children[parent] = sorted(children)

    options = {
        "category_parent_name": parent_cats,
        "parent_to_children": parent_to_children,
        "country": sorted(df["country"].dropna().unique()),
        "currency": sorted(df["currency"].dropna().unique()),
        "launched_date_month": _extract_months(df["launched_date"]),
        "deadline_month": _extract_months(df["deadline"]),
    }
    return options


# -----------------------------
# Feature contributions
# -----------------------------


def get_feature_contributions(
    pipeline,
    df_row: pd.DataFrame,
    preprocessor_name: str = "preprocess",
    clf_name: str = "clf",
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute feature contributions for a single input row.

    The contribution of each feature is calculated as x_j * coef_j
    on the logit scale. Returns the top positive and negative features.
    """

    preprocessor = pipeline.named_steps[preprocessor_name]
    clf = pipeline.named_steps[clf_name]

    X = preprocessor.transform(df_row)
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    if hasattr(X, "toarray"):
        x_vec = X.toarray()[0]
    else:
        x_vec = np.array(X[0]).ravel()

    contribs = x_vec * coefs

    idx_sorted = np.argsort(np.abs(contribs))[::-1]

    top_pos: List[Tuple[str, float]] = []
    top_neg: List[Tuple[str, float]] = []

    for idx in idx_sorted:
        val = float(contribs[idx])
        feat = feature_names[idx]
        if val > 0 and len(top_pos) < top_n:
            top_pos.append((feat, val))
        elif val < 0 and len(top_neg) < top_n:
            top_neg.append((feat, val))
        if len(top_pos) >= top_n and len(top_neg) >= top_n:
            break

    return {
        "top_positive": top_pos,
        "top_negative": top_neg,
    }


# -----------------------------
# 4. Perplexity AI call
# -----------------------------


def call_perplexity_feedback(payload: Dict[str, Any]) -> str:
    """
    Call the Perplexity chat completions API to generate qualitative feedback.

    The feedback is based on the model prediction and feature contribution
    signals provided in the payload.
    """

    dotenv.load_dotenv()
    client = Perplexity()

    system_prompt = """
You are a professional Kickstarter consultant and campaign copy strategist.

Rules:
- Use ONLY the provided DATA as your source of truth.
- Do NOT invent details that are not in the DATA.
- Do NOT mention models, coefficients, ML, statistics, probabilities, or transformations (including logs).
- Do not greet. No disclaimers. Be direct and helpful.
- Avoid generic advice unless it is clearly supported by the DATA signals.
- Use proportional language: stronger claims only when the DATA signals are strong.
"""

    user_prompt = f"""
DATA (only source):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Task (max 200 words). Format exactly like this:
Strengths:
- (3–5 bullets)

Risks:
- (3–5 bullets)

Suggestions:
- (2–3 bullets)
- Each suggestion must directly address one of the listed risks
- Keep suggestions practical and specific

Constraints:
- Do not quote or repeat exact numbers from DATA (no log values).
- Do not comment on currency or country unless non-US/USD AND clearly impactful.
- Do not comment on language if it is English.
- Write directly to the project creator.
"""

    completion = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    try:
        # OpenAI-compatible format
        return completion.choices[0].message.content
    except Exception as e:
        return f"Perplexity API -kutsu epäonnistui: {e}"


# -----------------------------
# Payload for the LLM model
# -----------------------------


def build_llm_payload(
    user_features: Dict[str, Any],
    pipeline,
) -> Dict[str, Any]:
    """
    Build a structured payload for a language model.

    The payload includes:
    - the original user input features
    - the model's predicted probability of success
    - the most important positive and negative feature contributions
    """

    df_row = pd.DataFrame([user_features])

    proba = float(pipeline.predict_proba(df_row)[0, 1])
    contribs = get_feature_contributions(pipeline, df_row)

    payload = {
        "project_features": user_features,
        "prediction": {"p_success": proba},
        "feature_contributions": contribs,
        "feature_legend": {
            "num__*": "numeric project metrics (goal, duration, lengths, creator history)",
            "cat__*": "categorical selections (category, country, currency, months, language flags, missing flags)",
            "blurb__*": "keywords detected from the blurb text",
            "name__*": "keywords detected from the title",
        },
        "data_notes": [
            "Missing flags: value 1 means the field is missing/empty; value 0 means it is provided.",
            "Never mention transformations like logs. If a feature name contains '_log', describe it as 'higher goal' vs 'lower goal'.",
            "For small creator history (1–3 prior projects), avoid intent-based judgments; suggest adding proof of execution instead."
        ],
    }
    return payload


# -----------------------------
# 6. Streamlit UI
# -----------------------------


APP_DIR = Path(__file__).resolve().parent
IMG = APP_DIR / "assets" / "kurkkumopo.png"

def main():
    model = load_model()
    df_train = load_training_data()
    options = get_category_options(df_train)
    st.set_page_config(page_title="Projektin onnistumisennuste", layout="wide")

    if IMG.exists():
        img_bytes = IMG.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        st.markdown(
            f"""
            <div style="
                width: 100%;
                height: 400px;
                overflow: hidden;
                border-radius: 14px;
                background: #f5f5f5;
                margin-bottom: 1rem;
            ">
                <img src="data:image/png;base64,{b64}"
                    style="
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        object-position: center;
                    ">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("Projektin onnistumisennuste (LogReg + Perplexity)")

    st.sidebar.header("Kurkkumopo App")

    st.subheader("Tekstikentät")

    blurb = st.text_area(
        "Projektikuvaus (blurb)",
        height=200,
        help="Kuvaa lyhyesti projektin idea, mitä olet jo tehnyt ja mihin rahat käytetään.",
    )
    name = st.text_input(
        "Projektin otsikko (name)", help="Lyhyt projektin nimi / otsikko."
    )

    st.subheader("Tavoite ja kesto")

    col1, col2 = st.columns(2)
    with col1:
        usd_goal_fx = st.number_input(
            "Tavoitesumma (USD, valuuttamuunnoksen jälkeen)",
            min_value=1.0,
            value=10000.0,
            step=100.0,
            help="Käytä samaa mittayksikköä kuin koulutusdatassa (usd_goal_fx).",
        )
        project_duration_days = st.number_input(
            "Kampanjan kesto (päiviä)",
            min_value=1,
            value=30,
            step=1,
        )
    with col2:
        creator_prev_projects = st.number_input(
            "Tekijän aiempien projektien määrä",
            min_value=0,
            value=0,
            step=1,
        )
        creator_prev_projects_successful = st.number_input(
            "Tekijän aiempien onnistuneiden projektien määrä",
            min_value=0,
            value=0,
            step=1,
        )


    PLACEHOLDER = "-- Valitse --"

    st.subheader("Kategoria ja maa")

    col3, col4 = st.columns(2)

    with col3:
        category_parent_name = st.selectbox(
            "Pääkategoria",
            options=[PLACEHOLDER] + options["category_parent_name"],
            index=0,
            help="Valitse Others- pääkategoria, jos et löydä sopivaa.",
        )

        if category_parent_name == PLACEHOLDER:
            st.info("Valitse ensin pääkategoria nähdäksesi alakategoriat.")
            subcat_options = [PLACEHOLDER]
        else:
            subcats = options["parent_to_children"].get(category_parent_name, [])
            if subcats:
                subcat_options = [PLACEHOLDER] + subcats
            else:
                st.warning("Tälle pääkategorialle ei löytynyt alakategorioita datassa.")
                subcat_options = [PLACEHOLDER]

        category_name_reduced = st.selectbox(
            "Alakategoria (category_name_reduced)",
            options=subcat_options,
            index=0,
            help="Valitse Others- alakategoria, jos et löydä sopivaa.",
        )

        country = st.selectbox(
            "Maa",
            options=[PLACEHOLDER] + options["country"],
            index=0,
            help="Maatunnus esim. US, GB, FI...",
        )

    with col4:
        currency = "USD"
        launched_date_month = st.selectbox(
            "Aloituskuukausi (1-12)",
            options=[PLACEHOLDER] + options["launched_date_month"],
            index=0,
            help="Kuukausi on numeraalisena 1-12 (esim. Tammikuu 1, Helmikuu 2)",
        )

        deadline_month = st.selectbox(
            "Deadlinen kuukausi (1-12)",
            options=[PLACEHOLDER] + options["deadline_month"],
            index=0,
            help="Kuukausi on numeraalisena 1-12 (esim. Tammikuu 1, Helmikuu 2)",
        )

    st.subheader("Kielen tiedot")

    blurb_is_english = st.checkbox(
        "Kuvaus on pääosin kirjoitettu englanniksi", value=True
    )
    name_is_english = st.checkbox(
        "Otsikko on pääosin kirjoitettu englanniksi", value=True
    )

    submitted = st.button("Laske ennuste ja hae palaute")

    if not submitted:
        st.info("Täytä projektin tiedot ja paina **Laske ennuste ja hae palaute**.")
        return

    errors = []
    if category_parent_name == PLACEHOLDER:
        errors.append("Valitse pääkategoria.")
    if category_name_reduced == PLACEHOLDER:
        errors.append("Valitse alakategoria.")
    if country == PLACEHOLDER:
        errors.append("Valitse maa.")
    if currency == PLACEHOLDER:
        errors.append("Valitse valuutta.")
    if launched_date_month == PLACEHOLDER:
        errors.append("Valitse aloituskuukausi.")
    if deadline_month == PLACEHOLDER:
        errors.append("Valitse deadline-kuukausi.")

    if errors:
        st.error("Korjaa nämä ennen jatkamista:\n- " + "\n- ".join(errors))
        return


    blurb_missing = int(len(blurb.strip()) == 0)
    name_missing = int(len(name.strip()) == 0)

    usd_goal_fx_log = float(np.log1p(usd_goal_fx))

    blurb_len = len(blurb.split()) if blurb.strip() else 0
    name_len = len(name.split()) if name.strip() else 0

    user_features: Dict[str, Any] = {
        # text
        "blurb": blurb,
        "name": name,
        # numeric
        "usd_goal_fx_log": usd_goal_fx_log,
        "creator_prev_projects_successful": int(creator_prev_projects_successful),
        "creator_prev_projects": int(creator_prev_projects),
        "project_duration_days": int(project_duration_days),
        "blurb_len": int(blurb_len),
        "name_len": int(name_len),
        # categorical
        "category_name_reduced": category_name_reduced,
        "category_parent_name": category_parent_name,
        "country": country,
        "currency": currency,
        "launched_date_month": int(launched_date_month),
        "deadline_month": int(deadline_month),
        "blurb_missing": int(blurb_missing),
        "blurb_is_english": int(blurb_is_english),
        "name_missing": int(name_missing),
        "name_is_english": int(name_is_english),
    }

    # Prediction + contributions + payload for LLM
    payload = build_llm_payload(user_features, model)
    p_success = payload["prediction"]["p_success"]
    p_percent = int(round(p_success * 100))


    # ------------------------
    # ML prediction
    # ------------------------


    st.subheader("Logistic Regression -mallin ennuste")

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.metric(
            label="Arvioitu onnistumistodennäköisyys",
            value=f"{p_percent} %",
        )
        st.write(
            "Tämä ennuste perustuu historiallisiin projekteihin, jotka ovat "
            "samankaltaisia tavoitesumman, keston, kategorian ja tekstin suhteen."
        )

    with col_side:
        st.markdown("**Syötteiden yhteenveto**")
        st.write(f"- Tavoitesumma (usd_goal_fx): {usd_goal_fx:,.0f}")
        st.write(f"- Kampanjan kesto: {project_duration_days} päivää")
        st.write(f"- Blurb-pituus: {blurb_len} sanaa")
        st.write(f"- Otsikon pituus: {name_len} sanaa")
        st.write(
            f"- Aiemmat projektit (onnistuneet / kaikki): "
            f"{creator_prev_projects_successful} / {creator_prev_projects}"
        )


    # ------------------------
    # Feature contributions
    # ------------------------


    st.subheader("Tärkeimmät tekijät tälle projektille (LogReg-kontribuutiot)")

    contribs = payload["feature_contributions"]
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**Featuret, jotka nostivat ennustetta**")
        if contribs["top_positive"]:
            for feat, val in contribs["top_positive"]:
                st.write(f"- `{feat}` (kontribuutio: +{val:.3f})")
        else:
            st.write("_Ei selkeitä positiivisia tekijöitä tälle riville._")

    with col_neg:
        st.markdown("**Featuret, jotka laskivat ennustetta**")
        if contribs["top_negative"]:
            for feat, val in contribs["top_negative"]:
                st.write(f"- `{feat}` (kontribuutio: {val:.3f})")
        else:
            st.write("_Ei selkeitä negatiivisia tekijöitä tälle riville._")


    # ------------------------
    # LLM feedback (Perplexity)
    # ------------------------

    
    st.subheader("Perplexityn generoima palaute")

    with st.spinner("Haetaan palautetta Perplexityltä..."):
        feedback = call_perplexity_feedback(payload)

    st.write(feedback)
    st.caption(
        "Huom: Palaute perustuu Logistic Regression -mallin tilastolliseen arvioon "
        "sekä Perplexity-kielimallin tulkintaan. Se ei ole takuu lopputuloksesta."
    )


if __name__ == "__main__":
    main()
