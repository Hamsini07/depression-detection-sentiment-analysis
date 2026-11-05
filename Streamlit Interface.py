"""
app2UI.py - Streamlit front-end for the models dumped by project2.ipynb (Decision Tree / KNN / ANN)

Run:
    pip install streamlit scikit-learn nltk pandas joblib
    streamlit run app2UI.py

Expected folder layout (created by the last cells of project2.ipynb):
    app2UI.py
    models2/
        results.csv
        reddit/   meta.json + model.joblib   (TF-IDF + classifier pipeline)
        twitter/  meta.json + model.joblib
"""
import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "models2"

DATASETS = {
    "reddit": {
        "title": "Reddit model - Depression detection",
        "note": "Predicts whether a post resembles Reddit posts labelled as *depression*.",
        "examples": [
            "is annoyed with the amount of glass on dublin road and the number of puncture i am getting",
            "my friend had a party this weekend and it seemed to have been awesome",
            "for about a week now i ve been experiencing extreme anxiety and panic, restlessness, and it keeps waking me up at night",
            "i feel empty and hopeless and nothing i do matters anymore",
        ],
    },
    "twitter": {
        "title": "Twitter model - Sentiment analysis",
        "note": "Predicts general **sentiment** (positive / negative). This is *not* a depression detector.",
        "examples": [
            "I absolutely love this, best day ever!",
            "This is the worst service I have ever had, so disappointed.",
            "Can't wait for the weekend, heading to the beach with my friends.",
            "My phone broke again and I'm so annoyed.",
        ],
    },
}


# ----------------------------------------------------------------------------- preprocessing
# MUST stay identical to the functions in project2.ipynb
@st.cache_resource(show_spinner="Preparing language resources...")
def load_nlp():
    for res, pkg in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_twitter(text: str) -> str:
    stop_words, lemmatizer = load_nlp()
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return " ".join(lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words)


def normalize_reddit(text: str) -> str:
    _, lemmatizer = load_nlp()
    text = re.sub(r"[^a-z]+", " ", str(text).lower())
    return " ".join(lemmatizer.lemmatize(t) for t in text.split())


PREPROCESSORS = {"reddit": normalize_reddit, "twitter": preprocess_twitter}


# ----------------------------------------------------------------------------- model loading
def available_datasets():
    return [d for d in DATASETS if (MODEL_DIR / d / "meta.json").exists()]


@st.cache_resource(show_spinner="Loading model...")
def load_model(dataset: str):
    """Returns (meta, proba_fn). proba_fn(list[str]) -> np.ndarray of P(positive class)."""
    folder = MODEL_DIR / dataset
    meta = json.loads((folder / "meta.json").read_text())
    pipeline = joblib.load(folder / meta["files"]["model"])   # TF-IDF + classifier (scikit-learn)
    return meta, lambda texts: pipeline.predict_proba(list(texts))[:, 1]


# ----------------------------------------------------------------------------- explanation
def explain_words(prepared_text: str, proba_fn, max_words: int = 60) -> pd.DataFrame:
    """Occlusion: remove each distinct word and see how P(positive) changes."""
    tokens = prepared_text.split()
    if not tokens:
        return pd.DataFrame(columns=["word", "effect"])
    base = float(proba_fn([prepared_text])[0])
    words = list(dict.fromkeys(tokens))[:max_words]
    variants = [" ".join(t for t in tokens if t != w) or " " for w in words]
    effect = base - np.asarray(proba_fn(variants), dtype=float)   # >0: word pushes towards the positive class
    df = pd.DataFrame({"word": words, "effect": effect})
    return df.reindex(df["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ----------------------------------------------------------------------------- UI helpers
def verdict(prob: float, threshold: float, labels: list):
    positive = prob >= threshold
    return labels[1] if positive else labels[0], positive


def single_prediction_tab(dataset, meta, proba_fn, threshold):
    labels = meta["labels"]
    cfg = DATASETS[dataset]
    st.caption(cfg["note"])

    example = st.selectbox("Try an example (optional)", ["- write my own -"] + cfg["examples"])
    text = st.text_area("Text to analyse", value="" if example.startswith("- write") else example,
                        height=150, placeholder="Paste a post or tweet here...")

    if not st.button("Analyse", type="primary"):
        return
    if not text.strip():
        st.warning("Please enter some text first.")
        return

    prepared = PREPROCESSORS[meta["preprocess"]](text)
    if not prepared.strip():
        st.warning("After cleaning, no usable words are left (only numbers, symbols or stop-words). Try a longer text.")
        return

    prob = float(proba_fn([prepared])[0])
    label, positive = verdict(prob, threshold, labels)

    c1, c2 = st.columns(2)
    c1.metric("Prediction", label)
    c2.metric(f"P({labels[1]})", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    (st.error if (positive and dataset == "reddit") else st.info)(
        f"The model leans towards **{label}** (threshold {threshold:.2f})."
    )
    if dataset == "reddit" and positive:
        st.markdown(
            "**This is not a diagnosis.** If you or someone you know is struggling, please talk to a "
            "mental-health professional, a trusted person, or your local emergency / crisis line."
        )

    with st.expander("Why? Words that influenced this prediction", expanded=True):
        expl = explain_words(prepared, proba_fn)
        if expl.empty:
            st.write("Nothing to explain.")
        else:
            top = expl.head(12).copy()
            top["pushes towards"] = np.where(top["effect"] > 0, labels[1], labels[0])
            st.bar_chart(top.set_index("word")["effect"], horizontal=True)
            st.caption(f"Positive bars push towards **{labels[1]}**, negative bars towards **{labels[0]}**. "
                       "Computed by removing each word and re-running the model (approximate).")
            st.dataframe(top[["word", "effect", "pushes towards"]].round(4), hide_index=True, use_container_width=True)
    with st.expander("Text as seen by the model"):
        st.code(prepared)


def batch_tab(dataset, meta, proba_fn, threshold):
    labels = meta["labels"]
    st.write("Upload a CSV file with one text per row.")
    file = st.file_uploader("CSV file", type="csv", key=f"upload_{dataset}")
    if file is None:
        return
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file: {exc}")
        return
    if df.empty:
        st.warning("The file has no rows.")
        return

    text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
    col = st.selectbox("Column containing the text", text_cols, key=f"col_{dataset}")
    if not st.button("Run on file", key=f"run_{dataset}"):
        return

    prep = PREPROCESSORS[meta["preprocess"]]
    prepared = df[col].fillna("").astype(str).map(prep)
    valid = prepared.str.strip() != ""
    probs = np.full(len(df), np.nan)
    if valid.any():
        probs[valid.to_numpy()] = proba_fn(prepared[valid].tolist())

    out = df.copy()
    out[f"P({labels[1]})"] = np.round(probs, 4)
    out["prediction"] = np.where(np.isnan(probs), "n/a (empty after cleaning)",
                                 np.where(probs >= threshold, labels[1], labels[0]))
    st.success(f"Done - {int(valid.sum())} of {len(df)} rows analysed.")
    st.dataframe(out, use_container_width=True)
    st.bar_chart(out["prediction"].value_counts())
    st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")


def comparison_tab():
    path = MODEL_DIR / "results.csv"
    if not path.exists():
        st.info("`models2/results.csv` not found - re-run the last cells of project2.ipynb.")
        return
    res = pd.read_csv(path)
    st.write("Test-set results of every model trained in the notebook:")
    st.dataframe(res.sort_values(["dataset", "accuracy"], ascending=[True, False]),
                 hide_index=True, use_container_width=True)
    st.bar_chart(res.pivot(index="model", columns="dataset", values="accuracy"))


# ----------------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Depression & Sentiment Detector", page_icon="🧠", layout="wide")
    st.title("🧠 Depression & Sentiment Detector")
    st.warning("Research prototype - **not** a medical or diagnostic tool. "
               "Predictions can be wrong and must never replace professional judgement.")

    options = available_datasets()
    if not options:
        st.error(f"No saved models found in `{MODEL_DIR}`.\n\n"
                 "Run **project2.ipynb** to the end (it creates the `models2/` folder), "
                 "then put that folder next to `app2UI.py`.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        dataset = st.radio("Model", options, format_func=lambda d: DATASETS[d]["title"])
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Lower = flag more texts as the positive class (higher recall, more false alarms).")

    try:
        meta, proba_fn = load_model(dataset)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the model: {exc}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.subheader("Loaded model")
        st.write(f"**{meta['model_name']}** ({meta['model_type']})")
        m = meta["metrics"]
        a, b = st.columns(2)
        a.metric("Accuracy", f"{m['accuracy']:.3f}")
        b.metric("F1", f"{m['f1']:.3f}")
        a.metric("Precision", f"{m['precision']:.3f}")
        b.metric("Recall", f"{m['recall']:.3f}")
        st.caption(f"Metrics on the held-out test set; positive class = '{meta['labels'][1]}'.")

    tab1, tab2, tab3 = st.tabs(["Single text", "Batch (CSV)", "Model comparison"])
    with tab1:
        single_prediction_tab(dataset, meta, proba_fn, threshold)
    with tab2:
        batch_tab(dataset, meta, proba_fn, threshold)
    with tab3:
        comparison_tab()


if __name__ == "__main__":
    main()
