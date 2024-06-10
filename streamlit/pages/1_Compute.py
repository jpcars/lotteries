import numpy as np
import pandas as pd
import streamlit as st
from lotteries.probability_utils import claimant_probabilities
from lotteries.symbolic_lotteries import (
    EXCSLottery,
    EQCSLottery,
    TILottery,
    TaurekLottery,
)

st.set_page_config(page_title="Compute")

st.title("Compute")

st.subheader("Definition of the groups")

col1, col2 = st.columns(2)

with col1:
    st.number_input("Number of groups", key="numGroups", min_value=2)

with col2:
    st.number_input("Number of claimants", key="numClaimants", min_value=2)

st.write(
    "In the matrix below each row represents a claimant, each column represents an outcome group. "
    "The number in cell (i,j) indicates the probability that claimant i will be saved, "
    "given that outcome group j was selected."
)
df = pd.DataFrame(
    np.zeros((st.session_state["numClaimants"], st.session_state["numGroups"]))
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
    for LotteryCLass in [EXCSLottery, EQCSLottery, TILottery, TaurekLottery]:
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
                    name=f"{lottery.lottery_code}{name_suffix}",
                )
            )
            claimant_stats.append(
                pd.Series(
                    claimant_probabilities_temp,
                    name=f"{lottery.lottery_code}{name_suffix}",
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
