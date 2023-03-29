import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
from lotteries.exclusive_composition_sensitive_lottery import EXCSLottery

st.title('Distribution Lotteries')

st.subheader('Introduction')
st.write("""
This is a small app that is supposed to compute useful quantities for different lottery procedures,
which are commonplace in ethical debates surrounding the distribution of goods. So far this app is very
minimalistic. Given a set of possibly overlapping groups it computes the probabilities that any particular
group and any particular claimant will win the lottery. So far only Vong\'s exclusive composition sensitive
lottery is implemented.

Further plans involve:
- implementing other lotteries, such as Timmermann\'s individualist lottery and Vong\'s equal composition sensitive
lottery, (if you want others, please say so)
- implementing metrics to evaluate how well these lotteries fulfill certain potential ethical criteria,
e.g. absolute fairness, comparative fairness (both in Vong\'s Paper) or procedural fairness (as described in Rasmussen)
""")

st.subheader('Glossary')
st.write('EXCS - Vong\'s exclusive composition sensitive lottery')
st.write('EQCS - Vong\'s equal composition sensitive lottery')
st.write('IL - Timmermann\'s individualist lottery')

st.subheader('Definition of the groups')
with st.form("Number of groups"):
    numProducts = st.number_input('How many groups of claimants are there?', key='numGroups', min_value=2)
    st.form_submit_button("Submit")

groups = []
if 'numGroups' in st.session_state.keys():
    with st.form("Claimants in groups"):
        for i in range(1, st.session_state['numGroups'] + 1):
            groups.append(st_tags(
                label=f'Names of claimants in group {i}:',
                text='Type name and press Enter to add',
                value=[],
                maxtags=-1,
                key=f'{i}'))
        submitted = st.form_submit_button("Submit")

if any([len(group) == 0 for group in groups]):
    raise Exception('Please make sure that all groups are non-empty.')

removed_duplicates = []
for group in groups:
    if set(group) not in removed_duplicates:
        removed_duplicates.append(set(group))

if len(removed_duplicates) < len(groups):
    raise Exception('Please make sure that all groups are unique.')

if submitted:
    lottery = EXCSLottery(groups)
    lottery.initialize()
    lottery.iterate()
    group_probabilities = {}
    claimant_probabilities = {}
    for group in lottery.groups['inactive'].values():
        group_probabilities[group.name] = group.claim
        if group.claim > 0:
            for claimant in group.claimants:
                claimant_probabilities[claimant.name] = claimant_probabilities.get(claimant.name, 0) + group.claim
    group_probabilities_series = pd.Series(data=group_probabilities, name='EXCS')
    group_probabilities_series.index = group_probabilities_series.index + 1  # adding 1 in order to make the enumeration of groups in the app start at 1 instead of 0 # noqa: E501
    group_probabilities_series.sort_index(inplace=True)
    group_probabilities_series.index.name = 'group'
    claimant_probabilities_series = pd.Series(data=claimant_probabilities, name='EXCS')
    claimant_probabilities_series.sort_index(inplace=True)
    claimant_probabilities_series.index.name = 'claimant'

    st.subheader('Fairness metrics')

    st.subheader('Probability of benefitting a particular group')
    st.write(group_probabilities_series)

    st.subheader('Probability of benefitting a particular claimant')
    st.write(claimant_probabilities_series)
