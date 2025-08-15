import streamlit as st
import sys, platform

st.set_page_config(page_title="Hello", layout="wide")
st.title("✅ Minimal app rendered")
st.write({
    "python": sys.version.split()[0],
    "platform": platform.platform()
})
