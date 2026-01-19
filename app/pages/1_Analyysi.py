"""
Streamlit analysis page for inspecting model performance and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import re
import joblib
import matplotlib.pyplot as plt

from models.logreg_optimized import DATA_PATH, MODEL_PATH
from utils.data_preparation import prepare_data
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix


# ============================================================
# CACHING
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def evaluate_model_on_testset():
    """
    Evaluoi pipeline testisetillä.

    Palauttaa:
    - acc, f1, auc, cm, report
    - y_true (0/1), y_proba (P(y=1)), classes
    """
    model = load_model()
    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        num_features=[
            "usd_goal_fx_log",
            "creator_prev_projects_successful",
            "creator_prev_projects",
            "project_duration_days",
            "blurb_len",
            "name_len",
        ],
        cat_features=[
            "category_name_reduced",
            "category_parent_name",
            "country",
            "currency",
            "launched_date_month",
            "deadline_month",
            "blurb_missing",
            "blurb_is_english",
            "name_missing",
            "name_is_english",
        ],
        text_blurb="blurb",
        text_name="name",
        rand=42,
)


    clf = model.named_steps["clf"]
    classes = list(clf.classes_)

    y_true = pd.Series(y_test).astype(int).values

    if hasattr(model, "predict_proba"):
        pos_idx = classes.index(1)
        y_proba = model.predict_proba(X_test)[:, pos_idx]
        auc = roc_auc_score(y_true, y_proba)
    else:
        y_scores = model.decision_function(X_test)
        auc = roc_auc_score(y_true, y_scores)
        y_proba = y_scores

    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return acc, f1, auc, cm, report, y_true, y_proba, classes


@st.cache_data
def get_feature_table():
    model = load_model()
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    feature_names = pre.get_feature_names_out()
    coefs = clf.coef_[0]

    df_feat = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    })

    df_feat["group"] = np.select(
        [
            df_feat["feature"].str.startswith("blurb__"),
            df_feat["feature"].str.startswith("name__"),
            df_feat["feature"].str.startswith("cat__"),
            df_feat["feature"].str.startswith("num__"),
        ],
        ["blurb", "name", "categorical", "numeric"],
        default="other",
    )

    return df_feat


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Mallin analyysi", layout="wide")
st.title("Mallin analyysi")
st.divider()


# ============================================================
# DATA & MODEL LOAD
# ============================================================

model = load_model()
df = load_data()
df_feat = get_feature_table()

pre = model.named_steps["preprocess"]
clf = model.named_steps["clf"]


# ============================================================
# BASIC INFO
# ============================================================

st.header("Yleiskuva mallin käyttämästä datasta")

total_cells = df.shape[0] * df.shape[1]
nan_total = int(df.isna().sum().sum())
nan_pct = (nan_total / total_cells) * 100 if total_cells else 0.0

try:
    n_features = len(pre.get_feature_names_out())
except Exception:
    n_features = None

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Rivien lkm", f"{len(df):,}".replace(",", " "))
with colB:
    st.metric("Puuttuvia arvoja (NaN)", f"{nan_total:,}".replace(",", " "))
    st.caption(f"{nan_pct:.2f} % kaikista soluista")
with colC:
    st.metric("Mallin käyttämien piirteiden lkm", n_features if n_features is not None else "N/A")


# ============================================================
# SUCCESSFUL PROJECTS
# ============================================================

if "state" in df.columns:
    state = df["state"].astype(str).str.lower()
    mask = state.isin(["successful", "failed"])
    df2 = df.loc[mask].copy()

    if len(df2) > 0:
        success_rate = (df2["state"].astype(str).str.lower() == "successful").mean() * 100
        
        st.subheader("Onnistuneet projektit (%)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Onnistumisaste (successful)", f"{success_rate:.1f} %")
        with c2:
            st.metric("Successful (kpl)", int((df2["state"].astype(str).str.lower() == "successful").sum()))
        with c3:
            st.metric("Failed (kpl)", int((df2["state"].astype(str).str.lower() == "failed").sum()))
    else:
        st.info("Sarake 'state' löytyi, mutta siinä ei ollut arvoja 'successful'/'failed'.")
else:
    st.info("Datasta ei löytynyt saraketta 'state', joten onnistumisastetta ei voi näyttää.")


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Datan esikatselu"):
    st.markdown("#### Sarakkeet")
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:8px'>" +
        "".join(
            f"<span style='padding:4px 8px;border:1px solid #ddd;border-radius:12px;font-size:12px'>{c}</span>"
            for c in df.columns
        ) +
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### Ensimmäiset rivit")
    st.dataframe(df.head(20), width='stretch')


# ============================================================
# MODEL SPECS
# ============================================================

st.divider()
st.header("Mallin perustiedot")

params = clf.get_params()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Malli**")
    st.markdown(type(clf).__name__)
    st.caption("Pipeline: preprocess → clf")

with col2:
    st.markdown("**Asetukset**")
    st.write(f"- **Optimointi (solver):** `{params.get('solver')}`")
    st.write(f"- **Regularisointi:** `{params.get('penalty')}`")
    st.write(f"- **Regularisoinnin voimakkuus (C):** `{params.get('C')}`")
    st.write(f"- **Luokkien painotus:** `{params.get('class_weight')}`")


# ============================================================
# METRICS
# ============================================================

st.subheader("Mallin perusmetriikat (testidata)")

try:
    acc, f1, auc, cm, report, y_true, y_proba, classes = evaluate_model_on_testset()

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("F1", f"{f1:.3f}")
    m3.metric("ROC-AUC", f"{auc:.3f}")

    st.subheader("Confusion matrix")

    tn, fp, fn, tp = cm.ravel()

    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    pred_pos_rate = (tp + fp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) else 0.0

    colL, colR = st.columns(2)

    with colL:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Ennuste: Failed", "Ennuste: Successful"])
        ax.set_yticklabels(["Todellinen: Failed", "Todellinen: Successful"])

        ax.tick_params(rotation=15)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        st.pyplot(fig, width='stretch')

    with colR:
        st.metric("Precision (Successful)", f"{precision_pos:.2f}")
        st.metric("Recall (Successful)", f"{recall_pos:.2f}")
        st.metric("Ennusteita luokassa 'Successful'", f"{pred_pos_rate*100:.1f} %")

        with st.expander("Classification report"):
            rep_df = pd.DataFrame(report).T
            st.dataframe(rep_df, width='stretch')

    st.subheader("Ennuste vs. toteuma")

    st.info("""
    - **Väärä positiivinen (False Positive)**: malli ennustaa onnistumista vaikka projekti epäonnistui
    - **Väärä negatiivinen (False Negative)**: malli ennustaa epäonnistumista vaikka projekti onnistui
    """)

    total_errors = fp + fn

    if total_errors == 0:
        st.success("Mallilla ei ole virheitä testisetillä (harvinainen tapaus).")
    else:
        fp_pct = fp / total_errors * 100
        fn_pct = fn / total_errors * 100

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Väärät positiiviset (FP)", fp)
            st.caption(f"{fp_pct:.1f} % kaikista virheistä")
        with c2:
            st.metric("Väärät negatiiviset (FN)", fn)
            st.caption(f"{fn_pct:.1f} % kaikista virheistä")

        if fn > fp:
            st.success("🔒 **Malli on konservatiivinen.**")
        elif fp > fn:
            st.warning("⚠️ **Malli on optimistinen.**")
        else:
            st.info("⚖️ **Malli on tasapainoinen.**")

except Exception as e:
    st.warning("Mallin evaluointi testisetillä epäonnistui.")
    st.exception(e)

# ============================================================
# TOP 10 BLURB
# ============================================================

st.divider()
st.header("Tekstipiirteet: blurb")

st.markdown("##### TOP-10 merkittävintä ennusteeseen vaikuttavaa sanaa")


blurb_df = df_feat[df_feat["group"] == "blurb"].copy()
blurb_df["word"] = blurb_df["feature"].str.replace("blurb__", "", regex=False)

top_pos = blurb_df.sort_values("coef", ascending=False).head(10)
top_neg = blurb_df.sort_values("coef", ascending=True).head(10)

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("##### 🟢 Onnistumista ennustavat sanat")
    for _, row in top_pos.iterrows():
        st.write(f"**{row['word']}**  (+{row['coef']:.3f})")

with col_neg:
    st.markdown("##### 🔴 Epäonnistumista ennustavat sanat ")
    for _, row in top_neg.iterrows():
        st.write(f"**{row['word']}**  ({row['coef']:.3f})")

with st.expander("Tulkinta"):
    st.write(
        """
- **positiivinen kerroin** = sana liittyy useammin onnistuneisiin projekteihin  
- **negatiivinen kerroin** = sana liittyy useammin epäonnistuneisiin projekteihin  

Kertoimet kuvaavat tilastollista yhteyttä datassa, eivät kausaalista vaikutusta.
        """
    )

# ============================================================
# TOP 10 NAME
# ============================================================

st.header("Tekstipiirteet: name")

st.markdown("##### TOP-10 merkittävintä ennusteeseen vaikuttavaa sanaa")

name_df = df_feat[df_feat["group"] == "name"].copy()
name_df["word"] = name_df["feature"].str.replace("name__", "", regex=False)

name_df = name_df[name_df["word"].astype(str).str.len() > 0]

top_pos_name = name_df.sort_values("coef", ascending=False).head(10)
top_neg_name = name_df.sort_values("coef", ascending=True).head(10)

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("##### 🟢 Onnistumista ennustavat sanat")
    for _, row in top_pos_name.iterrows():
        st.write(f"**{row['word']}**  (+{row['coef']:.3f})")

with col_neg:
    st.markdown("##### 🔴 Epäonnistumista ennustavat sanat")
    for _, row in top_neg_name.iterrows():
        st.write(f"**{row['word']}**  ({row['coef']:.3f})")

# ============================================================
# TOP PARENT CATEGORIES
# ============================================================

st.divider()
st.header("Kategoriset piirteet: pääkategoriat")

st.markdown("##### TOP-10 merkittävintä ennusteeseen vaikuttavaa pääkategoriaa")

pref = "cat__category_parent_name_"
cat_parent_df = df_feat[df_feat["feature"].str.startswith(pref)].copy()

if cat_parent_df.empty:
    st.info("Ei löytynyt featureita muodossa cat__category_parent_name_*. Tarkista feature-nimet.")
else:
    cat_parent_df["category_parent"] = cat_parent_df["feature"].str.replace(pref, "", regex=False)

    top_pos_cat = cat_parent_df.sort_values("coef", ascending=False).head(10)
    top_neg_cat = cat_parent_df.sort_values("coef", ascending=True).head(10)

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("##### 🟢 Onnistumista ennustavat pääkategoriat")
        for _, row in top_pos_cat.iterrows():
            st.write(f"**{row['category_parent']}**  (+{row['coef']:.3f})")

    with col_neg:
        st.markdown("##### 🔴 Epäonnistumista ennustavat pääkategoriat")
        for _, row in top_neg_cat.iterrows():
            st.write(f"**{row['category_parent']}**  ({row['coef']:.3f})")

    with st.expander("Tulkinta"):
        st.write(
        """
Kertoimet kuvaavat mallin oppimaa keskimääräistä vaikutusta pääkategorialle
(one-hot-koodattuna).

Tulokset heijastavat tilastollista yhteyttä datassa ja voivat osin selittyä
muilla tekijöillä (esim. tavoitesumma, projektin tyyli).
        """
    )

# ============================================================
# TOP SUBCATEGORIES
# ============================================================

st.header("Kategoriset piirteet: alakategoriat")

st.markdown("##### TOP-10 merkittävintä ennusteeseen vaikuttavaa alakategoriaa")

pref = "cat__category_name_reduced_"
subcat_df = df_feat[df_feat["feature"].str.startswith(pref)].copy()

if subcat_df.empty:
    st.info("Ei löytynyt featureita muodossa cat__category_name_reduced_*. Tarkista feature-nimet.")
else:
    subcat_df["subcategory"] = subcat_df["feature"].str.replace(pref, "", regex=False)

    top_pos_sub = subcat_df.sort_values("coef", ascending=False).head(10)
    top_neg_sub = subcat_df.sort_values("coef", ascending=True).head(10)

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("##### 🟢 Onnistumista ennustavat alakategoriat")
    for _, row in top_pos_sub.iterrows():
        st.write(f"**{row['subcategory']}**  (+{row['coef']:.3f})")

with col_neg:
    st.markdown("##### 🔴 Epäonnistumista ennustavat alakategoriat")
    for _, row in top_neg_sub.iterrows():
        st.write(f"**{row['subcategory']}**  ({row['coef']:.3f})")

# ============================================================
# NUMERIC FEATURES
# ============================================================

st.divider()
st.header("Numeeriset piirteet")

st.markdown("##### Tärkeimmät numeeriset piirteet vaikutuksen voimakkuuden mukaan (|coef|)")

num_df = df_feat[df_feat["feature"].str.startswith("num__")].copy()

if num_df.empty:
    st.info("Ei löytynyt numeerisia featureita (num__*).")
else:
    num_df["name"] = num_df["feature"].str.replace("num__", "", regex=False)
    top_num = num_df.sort_values("abs_coef", ascending=False).head(6).copy()

    pretty_name = {
        "usd_goal_fx_log": "Tavoitesumma",
        "project_duration_days": "Kampanjan kesto (päiviä)",
        "creator_prev_projects_successful": "Aiemmat onnistumiset",
        "creator_prev_projects": "Aiemmat projektit (yht.)",
        "blurb_len": "Blurbin pituus (sanoja)",
        "name_len": "Otsikon pituus (sanoja)",
    }

    explanation = {
        "usd_goal_fx_log": "Korkeampi tavoitesumma on datassa yhteydessä matalampaan onnistumistodennäköisyyteen, "
        "mikä heijastaa rahoitustavoitteen haastavuutta.",
        "project_duration_days": "Rakenteellinen signaali: hyvin lyhyet tai pitkät kampanjat "
        "poikkeavat tyypillisistä onnistuneista projekteista.",
        "creator_prev_projects_successful": "Aiempien onnistumisten määrä kasvattaa luottamussignaalia.",
        "creator_prev_projects": "Projektien kokonaismäärä kuvaa tekijän kokemusta, mutta ilman onnistumisia signaali "
        "on heikompi tai jopa negatiivinen.",
        "blurb_len": "Liian lyhyt tai liian pitkä kuvaus voi vaikuttaa; malli tunnistaa tyylitrendejä.",
        "name_len": "Nimen (name) pituus on heikko mutta positiivinen signaali, joka liittyy otsikon informatiivisuuteen.",
    }

    def render_card(row):
        name = row["name"]
        coef = float(row["coef"])
        abs_coef = float(row["abs_coef"])

        ball = "🟢" if coef > 0 else "🔴" if coef < 0 else "⚪"
        title = pretty_name.get(name, name)
        desc = explanation.get(name, "Tälle muuttujalle ei ole vielä määritelty selitettä.")

        st.markdown(
            f"""
            <div style="
                border: 1px solid #e6e6e6;
                border-radius: 14px;
                padding: 12px 14px;
                margin: 6px 0;
                background: rgba(0,0,0,0.02);
            ">
              <div style="font-size: 16px; font-weight: 600; margin-bottom: 4px;">
                {ball} {title}
              </div>
              <div style="font-size: 13px; line-height: 1.35; opacity: 0.9;">
                {desc}
              </div>
              <div style="font-size: 12px; margin-top: 8px; opacity: 0.7;">
                coef: {coef:+.3f} &nbsp;&nbsp;|coef|: {abs_coef:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    left, right = st.columns(2)
    rows = top_num.to_dict("records")

    for i, row in enumerate(rows):
        with (left if i % 2 == 0 else right):
            render_card(row)

# ============================================================
# FEATURE GROUPS
# ============================================================

st.divider()
st.header("Piirreryhmät")

st.info("""
- Piirreryhmien vertailu mallin oppimien painojen perusteella: mitä suurempi keskimääräinen |coef|-arvo, 
        sitä voimakkaampi vaikutus piirreryhmällä on mallin ennusteeseen.  
- Tekstipiirteet on jaettu projektin nimeen (name) ja kuvaukseen (blurb), jotta niiden vaikutusta voidaan tarkastella erikseen.
""")

group_order = ["blurb", "name", "categorical", "numeric", "other"]

group_stats = (
    df_feat.groupby("group")
    .agg(
        n_features=("feature", "count"),
        mean_abs_coef=("abs_coef", "mean"),
        sum_abs_coef=("abs_coef", "sum"),
    )
    .reset_index()
)

group_stats["group"] = pd.Categorical(group_stats["group"], categories=group_order, ordered=True)
group_stats = group_stats.sort_values("group")
group_stats = group_stats[group_stats["n_features"] > 0].copy()

name_map = {
    "blurb": "Blurb (teksti)",
    "name": "Name (otsikko)",
    "categorical": "Kategoriat ym.",
    "numeric": "Numeeriset",
    "other": "Muut",
}
group_stats["group_label"] = group_stats["group"].map(name_map).fillna(group_stats["group"].astype(str))

colL, colR = st.columns([2, 1])

with colL:
    plot_df = group_stats[["group_label", "mean_abs_coef"]].set_index("group_label")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(plot_df.index, plot_df["mean_abs_coef"].values)
    ax.set_ylabel("Keskiarvo |coef|")
    ax.set_title("Keskimääräinen vaikutus per feature (|coef|)")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig, width='stretch')

with colR:
    show = group_stats[["group_label", "n_features", "mean_abs_coef", "sum_abs_coef"]].copy()
    show = show.rename(columns={
        "group_label": "Ryhmä",
        "n_features": "Featureitä",
        "mean_abs_coef": "Keski |coef|",
        "sum_abs_coef": "Summa |coef|",
    })
    show["Keski |coef|"] = show["Keski |coef|"].astype(float).round(4)
    show["Summa |coef|"] = show["Summa |coef|"].astype(float).round(2)

    st.write("**Yhteenveto**")
    st.dataframe(show, width='stretch', hide_index=True)

if not group_stats.empty:
    winner = group_stats.sort_values("mean_abs_coef", ascending=False).iloc[0]
    st.success(
        f"Eniten painoa per feature: **{winner['group_label']}** "
        f"(keski |coef| = {winner['mean_abs_coef']:.4f})."
    )


# ============================================================
# AVERAGE PROJECT GOAL
# ============================================================

st.divider()
st.header("Tavoitesumma (goal)")

goal_col = "usd_goal_fx"
goal_log_col = "usd_goal_fx_log"

if goal_col in df.columns:
    goal_vals = pd.to_numeric(df[goal_col], errors="coerce").dropna()
    if len(goal_vals) > 0:
        p50 = float(np.percentile(goal_vals, 50))
        p75 = float(np.percentile(goal_vals, 75))
        p90 = float(np.percentile(goal_vals, 90))

        c1, c2, c3 = st.columns(3)
        c1.metric("Median goal (usd_goal_fx)", f"{p50:,.0f}".replace(",", " "))
        c2.metric("75% persentiili", f"{p75:,.0f}".replace(",", " "))
        c3.metric("90% persentiili", f"{p90:,.0f}".replace(",", " "))
    else:
        st.info("Goal-sarakkeessa ei ollut kelvollisia arvoja.")
elif goal_log_col in df.columns:
    st.info("Datassa on vain log-muoto (usd_goal_fx_log). Voidaan silti näyttää mallin paino, mutta ei USD-jakaumaa.")
else:
    st.info("Datasta ei löytynyt goal-saraketta (usd_goal_fx / usd_goal_fx_log).")

goal_feat = df_feat[df_feat["feature"] == "num__usd_goal_fx_log"]
if not goal_feat.empty:
    coef_goal = float(goal_feat["coef"].iloc[0])
    st.metric("Goal-featuren paino (mallissa)", f"{coef_goal:+.3f}")


# ============================================================
# TOKENIZATION ARTIFACTS
# ============================================================

st.divider()
st.header("Tokenisoinnin artefaktit")

st.info("""
TF-IDF-tokenisoinnista muodostuneet ei-informatiiviset tekstipiirteet/artefaktit:
- hyvin lyhyet tokenit (esim. 1–2 merkkiä)
- apostrofijäämät (esim. we're --> re)
- numerot / yksiköt
""")

tok_df = df_feat[df_feat["group"].isin(["blurb", "name"])].copy()
tok_df["token"] = tok_df["feature"].str.replace(r"^(blurb__|name__)", "", regex=True)

apostrophe_fragments = {"ve", "ll", "re", "d", "t", "m", "s"}
unit_like = {"sec", "secs", "second", "seconds", "min", "mins", "minute", "minutes", "hour", "hours"}

def token_flags(t: str) -> dict:
    t0 = str(t).strip().lower()

    flags = {}
    flags["short_1_2"] = len(t0) <= 2
    flags["short_3"] = len(t0) == 3
    flags["apostrophe_fragment"] = t0 in apostrophe_fragments
    flags["contains_digit"] = any(ch.isdigit() for ch in t0)
    flags["numeric_only"] = t0.isdigit()
    flags["unit_like"] = t0 in unit_like or bool(re.search(r"(sec|secs|second|seconds|min|mins|minute|minutes|hour|hours)\b", t0))
    flags["non_alpha"] = bool(re.search(r"[^a-z]", t0))
    flags["very_common_word"] = t0 in {"the", "and", "or", "to", "of", "in", "for", "with", "on", "at"}

    flags["suspicious"] = (
        flags["short_1_2"]
        or flags["apostrophe_fragment"]
        or flags["contains_digit"]
        or flags["unit_like"]
        or flags["very_common_word"]
    )
    return flags

flag_rows = tok_df["token"].apply(token_flags).apply(pd.Series)
tok_df = pd.concat([tok_df.reset_index(drop=True), flag_rows.reset_index(drop=True)], axis=1)

top_k = st.slider("Kuinka monesta tärkeimmästä tokenista etsitään artefakteja?", 50, 1000, 200, step=50)

top_tokens = tok_df.sort_values("abs_coef", ascending=False).head(top_k).copy()
sus = top_tokens[top_tokens["suspicious"]].copy()

if sus.empty:
    st.success("Ei löytynyt epäilyttäviä tokeneita valitun top-k joukon sisältä. 👍")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tutkittuja tokeneita", len(top_tokens))
    c2.metric("Artefakteja", len(sus))
    c3.metric("Osuus", f"{(len(sus)/len(top_tokens))*100:.1f} %")

    st.caption("Epäilyttävät = lyhyet tokenit / apostrofijäämät / numerot / yksiköt / hyvin yleiset sanat")

    pos = sus.sort_values("coef", ascending=False).head(15)
    neg = sus.sort_values("coef", ascending=True).head(15)

    colL, colR = st.columns(2)

    with colL:
        st.markdown("##### 🟢 Ennustetta nostavat artefaktit")
        for _, r in pos.iterrows():
            st.write(f"**{r['token']}**  (+{float(r['coef']):.3f})")
            reasons = []
            if r["short_1_2"]:
                reasons.append("lyhyt")
            if r["apostrophe_fragment"]:
                reasons.append("apostrofijäämä")
            if r["contains_digit"]:
                reasons.append("sis. numero")
            if r["unit_like"]:
                reasons.append("yksikkö/aika")
            if r["very_common_word"]:
                reasons.append("yleinen sana")
            if reasons:
                st.caption(" • ".join(reasons))

    with colR:
        st.markdown("##### 🔴 Ennustetta laskevat artefaktit")
        for _, r in neg.iterrows():
            st.write(f"**{r['token']}**  ({float(r['coef']):.3f})")
            reasons = []
            if r["short_1_2"]:
                reasons.append("lyhyt")
            if r["apostrophe_fragment"]:
                reasons.append("apostrofijäämä")
            if r["contains_digit"]:
                reasons.append("sis. numero")
            if r["unit_like"]:
                reasons.append("yksikkö/aika")
            if r["very_common_word"]:
                reasons.append("yleinen sana")
            if reasons:
                st.caption(" • ".join(reasons))

st.info("""
Artefaktien välttämiseksi tekstiesikäsittelyä voidaan vielä parantaa:

- stopword-listan tarkentaminen
- token_patternin säätäminen
- harvinaisten tokenien karsinta (min_df)
- numeeristen ja yksikköilmausten siivous
""")

# ============================================================
# CALIBRATION
# ============================================================

st.divider()
st.header("Ennusteiden luotettavuus")

st.subheader("Kalibrointi ja päätöskynnys")

st.info("""
- **Calibration bins (ennustetodennäköisyyksien luokat)** kertoo, kuinka hyvin mallin ennustamat todennäköisyydet vastaavat toteutunutta onnistumisastetta.
- **Threshold-arvoa säätämällä** nähdään miten Confusion Matrix sekä virhetyypit (FP/FN) muuttuvat päätöskynnyksen mukaan.
""")

y_true = pd.Series(y_true).astype(int)
p = pd.Series(y_proba).astype(float)

y_true = y_true.astype(float).fillna(0).astype(int)
p = pd.Series(y_proba).astype(float)

bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

cal_df = pd.DataFrame({"p": p, "y": y_true})
cal_df["bin"] = pd.cut(cal_df["p"], bins=bins, labels=labels, include_lowest=True, right=True)

summary = (
    cal_df.groupby("bin", observed=True)
    .agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        success_rate=("y", "mean"),
    )
    .reset_index()
)

summary["mean_pred_pct"] = (summary["mean_pred"] * 100).round(1)
summary["success_rate_pct"] = (summary["success_rate"] * 100).round(1)

colL, colR = st.columns([2, 1])

with colL:
    st.markdown("### Calibration bins")

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["mean_pred_pct"], width=0.4, label="Ennuste (keskim.) %")
    ax.bar(x + 0.2, summary["success_rate_pct"], width=0.4, label="Outcome %")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["bin"].astype(str))
    ax.set_ylabel("%")
    ax.set_title("Calibration: ennuste vs toteuma (bins)")
    ax.legend()
    st.pyplot(fig, width='stretch')

    with st.expander("Näytä calibration-taulukko"):
        show = summary[["bin", "n", "mean_pred_pct", "success_rate_pct"]].rename(columns={
            "bin": "Binni",
            "n": "kpl",
            "mean_pred_pct": "Ennuste (keski) %",
            "success_rate_pct": "Toteuma %"
        })
        st.dataframe(show, width='stretch', hide_index=True)

    summary["gap_pct"] = (summary["success_rate_pct"] - summary["mean_pred_pct"]).abs()
    if summary["n"].sum() > 0 and summary["gap_pct"].notna().any():
        worst = summary.sort_values("gap_pct", ascending=False).iloc[0]
        st.success(
            f"Suurin ero luokassa (bin) **{worst['bin']}**: "
            f"ennuste {worst['mean_pred_pct']:.1f}% vs toteuma {worst['success_rate_pct']:.1f}% "
            f"(n={int(worst['n'])})."
        )

with colR:
    st.markdown("### Päätöskynnys (Threshold)")

    thr = st.slider("Threshold", 0.10, 0.90, 0.50, 0.05)

    y_pred_thr = (p.values >= thr).astype(int)
    cm_thr = confusion_matrix(y_true.values, y_pred_thr)
    tn, fp, fn, tp = cm_thr.ravel()

    total_errors = fp + fn
    fp_pct = (fp / total_errors * 100) if total_errors else 0.0
    fn_pct = (fn / total_errors * 100) if total_errors else 0.0

    st.metric("FP", int(fp))
    st.caption(f"{fp_pct:.1f} % kaikista virheistä")
    st.metric("FN", int(fn))
    st.caption(f"{fn_pct:.1f} % kaikista virheistä")

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.imshow(cm_thr)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Failed", "Pred Success"])
    ax2.set_yticklabels(["True Failed", "True Success"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_thr[i, j]), ha="center", va="center")
    ax2.set_title(f"Confusion matrix @ thr={thr:.2f}")
    st.pyplot(fig2, width='stretch')

    if fp > fn:
        st.warning("Tällä kynnyksellä malli on **optimistinen** (enemmän FP kuin FN).")
    elif fn > fp:
        st.success("Tällä kynnyksellä malli on **konservatiivinen** (enemmän FN kuin FP).")
    else:
        st.info("Tällä kynnyksellä FP ja FN ovat **tasapainossa**.")
