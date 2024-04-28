import numpy as np
import pandas as pd
import streamlit as st
from lotteries.probability_utils import claimant_probabilities
from lotteries.symbolic_lotteries import EXCSLottery, EQCSLottery, TILottery

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

st.subheader("Definition of the groups")

st.number_input("Number of groups", key="numGroups1", min_value=2)
st.number_input("Number of claimants", key="numClaimants1", min_value=2)
df = pd.DataFrame(
    np.zeros((st.session_state["numClaimants1"], st.session_state["numGroups1"]))
)
edited_df = st.data_editor(
    df,
    column_config={
        i: st.column_config.NumberColumn(min_value=0, max_value=1) for i in df.columns
    },
)

if st.button("Compute"):
    claimant_mat = edited_df.to_numpy()
    group_stats = [pd.Series(data=claimant_mat.sum(axis=0), name="size")]
    claimant_stats = []
    for LotteryCLass in [EXCSLottery, EQCSLottery, TILottery]:
        for remove_subgroups in [False, True]:
            name_suffix = "_pruned" if remove_subgroups else ""
            lottery = LotteryCLass(claimant_mat, remove_subgroups=remove_subgroups)
            group_probabilities_temp = lottery.compute_on_orbits()
            claimant_probabilities_temp = claimant_probabilities(
                claimant_mat=claimant_mat, group_probabilities=group_probabilities_temp
            )
            group_stats.append(
                pd.Series(
                    group_probabilities_temp,
                    name=f"{lottery.lottery_name}{name_suffix}",
                )
            )
            claimant_stats.append(
                pd.Series(
                    claimant_probabilities_temp,
                    name=f"{lottery.lottery_name}{name_suffix}",
                )
            )
    group_df = pd.concat(group_stats, axis=1)
    expected_number_lives_saved_EXCS = (group_df["size"] * group_df["EXCS"]).sum()
    expected_number_lives_saved_EQCS = (group_df["size"] * group_df["EQCS"]).sum()
    expected_number_lives_saved_TI = (group_df["size"] * group_df["TI"]).sum()
    claimant_df = pd.concat(claimant_stats, axis=1)
    st.subheader("Fairness metrics")
    st.write("Coming soon")

    st.subheader("Expected number of lives saved")
    st.write(expected_number_lives_saved_EXCS)
    st.write(expected_number_lives_saved_EQCS)
    st.write(expected_number_lives_saved_TI)

    st.subheader("Probability of benefiting a particular group")
    st.write(group_df)

    st.subheader("Probability of benefiting a particular claimant")
    st.write(claimant_df)
