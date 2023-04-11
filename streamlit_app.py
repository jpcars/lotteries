import pandas as pd
import streamlit as st
from streamlit_tags import st_tags

from lotteries.app_utils import validate_group_input
from lotteries.examples import vong_1, vong_2
from lotteries.group_constructions import small_large_example
from lotteries.lottery import EXCSLottery, EQCSLottery, TILottery

st.title("Distribution Lotteries")

st.subheader("Introduction")
st.write(
    """
This is a small app that is supposed to compute useful quantities for different lottery procedures,
which are commonplace in ethical debates surrounding the distribution of goods. So far this app is very
minimalistic. Given a set of possibly overlapping groups it computes the probabilities that any particular
group and any particular claimant will win the lottery. So far only Vong\'s exclusive and equal composition-sensitive
lotteries are implemented.

Further plans involve:
- implementing other lotteries, such as Timmermann\'s individualist lottery (if you want others, please say so)
- implementing metrics to evaluate how well these lotteries fulfill certain potential ethical criteria,
e.g. absolute fairness, comparative fairness (both in Vong\'s Paper) or procedural fairness (as described in Rasmussen)
"""
)

st.subheader("Glossary")
st.write("EXCS - Vong's exclusive composition sensitive lottery")
st.write("EQCS - Vong's equal composition sensitive lottery")
st.write("TI - Timmermann's individualist lottery")

st.subheader("Definition of the groups")
input_method = st.selectbox(
    label="Select input method",
    options=["Choose an option", "Manually", "Predefined Examples"],
)
groups = []
validated = True
display_results = False
if input_method == "Manually":
    with st.form("Number of groups"):
        st.number_input(
            "How many groups of claimants are there?", key="numGroups", min_value=2
        )
        st.form_submit_button("Submit")

    if "numGroups" in st.session_state.keys():
        with st.form("Claimants in groups"):
            for i in range(1, st.session_state["numGroups"] + 1):
                groups.append(
                    st_tags(
                        label=f"Names of claimants in group {i}:",
                        text="Type name and press Enter to add",
                        value=[],
                        maxtags=-1,
                        key=f"{i}",
                    )
                )
            display_results = st.form_submit_button("Submit")

        validated = validate_group_input(groups)
        if display_results and validated:
            group_series = []
            claimant_series = []
            for LotteryCLass in [EXCSLottery, EQCSLottery, TILottery]:
                lottery = LotteryCLass(groups)
                lottery.compute()
                group_series_temp, claimant_series_temp = lottery.probabilities()
                group_series_temp.index = (
                        group_series_temp.index + 1
                )  # adding 1 in order to make the enumeration of groups in the app start at 1 instead of 0 # noqa: E501
                group_series.append(group_series_temp)
                claimant_series.append(claimant_series_temp)
            group_df = pd.concat(group_series, axis=1)
            claimant_df = pd.concat(claimant_series, axis=1)
            st.subheader("Fairness metrics")

            st.subheader("Probability of benefiting a particular group")
            st.write(group_df)

            st.subheader("Probability of benefiting a particular claimant")
            st.write(claimant_df)

elif input_method == 'Predefined Examples':
    example = input_method = st.selectbox(
        label="Select an example",
        options=["Choose an option", "Vong 1", "Vong 2"],
    )
    if example == "Vong 1":
        st.write(
            """
            This example is taken from Vong. The total number of claimants is N and there are two different group sizes p and q. N must be divisible by q.
            The set of benefitable group is constructed as follows:
            - Any group of size p is benefitable
            - The benefitable groups of size q form a disjoint cover of the set of claimants, i.e. every claimant is in one and only one of the groups of size q.
            """
        )
        with st.form("Number of claimants"):
            number_claimants = st.number_input(
                "How many claimants are there?", key="number_claimants", min_value=9
            )
            size_1 = st.number_input(
                "What is p?", key="size_1", min_value=2
            )
            size_2 = st.number_input(
                "What is q?", key="size_2", min_value=3
            )
            display_results = st.form_submit_button("Submit")
            if number_claimants % size_2 != 0:
                st.error('N is not divisible by q.')
            else:
                probabilities = {}
                for lottery in ['EXCS', 'EQCS', 'IL']:
                    probabilities[lottery] = vong_1(number_claimants, size_2, group_size_fine=size_1, lottery=lottery)
                st.subheader('Prbability of selecting one of the groups of size q')
                st.write(probabilities)

    elif example == "Vong 2":
        st.write(
            """
            This example is taken from Vong. The total number of claimants is N, where N is even. In the paper Vong uses N=1000. There are three groups:
            - Group 1: Claimants 1 through N/2
            - Group 2: Claimants N/2+1 through N
            - Group 3: Claimants 2 through N-1
            
            
            """
        )
        with st.form("Number of claimants"):
            number_claimants = st.number_input(
                "How many claimants are there?", key="number_claimants", min_value=10
            )
            display_results = st.form_submit_button("Submit")
            if number_claimants % 2 != 0:
                st.error("Please make sure that the number of claimants is even.")
            else:
                probabilities = {}
                for lottery in ['EXCS', 'EQCS', 'IL']:
                    probabilities[lottery] = vong_2(number_claimants, lottery=lottery)
                st.subheader("Probability of benefiting one of the larger groups")
                st.write(probabilities)
