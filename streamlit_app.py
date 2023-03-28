import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
from lotteries.exclusive_composition_sensitive_lottery import EXCSLottery

st.title('Distribution Lotteries')

st.subheader('Introduction')
st.write('This is a small app that is supposed to compute useful quantities for different lottery procedures,'
         'which are commonplace in ethical debates surrounding the distribution of goods.'
         'So far this app is very minimalistic. Given a set of possibly overlapping groups it computes the'
         'probabilities that any particular group and any particular claimant will win the lottery. So far only Vong\'s'
         'exclusive composition sensitive lottery is implemented.')

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
        for i in range(st.session_state['numGroups']):
            groups.append(st_tags(
                label=f'Names of claimants in group {i}:',
                text='Type name and press Enter to add',
                value=[],
                maxtags=-1,
                key=f'{i}'))
        submitted = st.form_submit_button("Submit")

if submitted and all([len(group) > 0 for group in groups]):
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
    group_probabilities_s = pd.Series(data=group_probabilities, name='EXCS')
    claimant_probabilities_s = pd.Series(data=claimant_probabilities, name='EXCS')
    group_probabilities_s.index.name = 'group'
    claimant_probabilities_s.index.name = 'claimant'

    st.subheader('Fairness metrics')
    st.write(group_probabilities_s)

    st.subheader('Probability of benefitting a particular group')
    st.write(group_probabilities_s)

    st.subheader('Probability of benefitting a particular claimant')
    st.write(claimant_probabilities_s)
