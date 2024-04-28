import streamlit as st

st.set_page_config(page_title="Rescue Lotteries")

st.title("Rescue Lotteries")

st.subheader("Introduction")
st.write(
    """
This is a small app that is supposed to compute useful quantities for different lottery procedures,
which are commonplace in ethical debates surrounding recues dilemmas. So far this app is very
minimalistic. Given a set of possibly overlapping groups it computes the probabilities that any particular
group and any particular claimant will win the lottery. So far only Vong\'s exclusive and equal composition-sensitive
lotteries and Timmermann's individualist lottery are implemented.

Further plans involve:
- implementing metrics to evaluate how well these lotteries fulfill certain potential ethical criteria,
e.g. absolute fairness, comparative fairness (both in Vong\'s Paper) or procedural fairness (as described in Rasmussen)
"""
)

st.subheader("Glossary")
st.write("EXCS - Vong's exclusive composition sensitive lottery")
st.write("EQCS - Vong's equal composition sensitive lottery")
st.write("TI - Timmermann's individualist lottery")
st.write(
    "Pruning - Refers to the procedure of deleting groups which are proper subgroups of other groups before performing the lottery. Some lotteries are sensitive to this (e.g. EXCS, EQCS), some are not (e.g. TI)"
)
