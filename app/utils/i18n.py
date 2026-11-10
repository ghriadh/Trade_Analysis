"""
Lightweight i18n helper for the Streamlit app.

Language choice lives in st.session_state (shared across all pages in a
multipage app) — this module only manages that state and renders the
toggle widget. Each page owns its own STRINGS dict and looks words up
through t(), e.g.:

    STRINGS = {
        "en": {"title": "Global Trade Dynamics"},
        "fr": {"title": "Dynamique commerciale mondiale"},
    }
    st.title(t(STRINGS, "title"))
"""

import streamlit as st

DEFAULT_LANG = "en"


def get_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def set_lang(lang: str) -> None:
    st.session_state["lang"] = lang


def t(strings: dict, key: str) -> str:
    """Look up `key` for the current language, falling back to English, then the key itself."""
    lang = get_lang()
    en = strings.get("en", {})
    return strings.get(lang, en).get(key, en.get(key, key))


def render_lang_toggle() -> None:
    """Render the FR/EN sidebar toggle. Call once, from the main entry script
    (Home.py), before building st.navigation — so it appears on every page."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANG

    with st.sidebar:
        choice = st.radio(
            "Language",
            ["EN", "FR"],
            index=0 if get_lang() == "en" else 1,
            horizontal=True,
            key="lang_toggle_radio",
            label_visibility="collapsed",
        )
        set_lang("en" if choice == "EN" else "fr")
