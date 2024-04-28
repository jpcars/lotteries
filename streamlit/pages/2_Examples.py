import streamlit as st

from lotteries.examples import vong_1, vong_2

st.set_page_config(page_title="Examples")

st.title("Examples")

example = st.sidebar.selectbox(
    label="Select an example",
    options=["Choose an option", "Vong 1", "Vong 2"],
)

if example == "Vong 1":
    st.write(
        """
        This example is taken from Vong 2020, p.342ff. The total number of claimants is N and there are two different group sizes p and q. N must be divisible by q.
        The set of benefitable group is constructed as follows:
        - Any group of size p is benefitable
        - The benefitable groups of size q form a disjoint cover of the set of claimants, i.e. every claimant is in one and only one of the groups of size q.

        Vong uses the values N=1000, p=2, q=500.
        """
    )
    with st.form("Number of claimants"):
        number_claimants = st.number_input(
            "How many claimants are there?",
            key="number_claimants",
            min_value=2,
            value=1000,
        )
        size_1 = st.number_input("What is p?", key="size_1", min_value=2, value=2)
        size_2 = st.number_input("What is q?", key="size_2", min_value=2, value=500)
        display_results = st.form_submit_button("Submit")
        if number_claimants % size_2 != 0:
            st.error("N is not divisible by q.")
        elif size_1 == size_2:
            st.error("Please make sure that p!=q.")
        else:
            probabilities = {}
            for lottery in ["EXCS", "EQCS", "TI"]:
                probabilities[lottery] = vong_1(
                    number_claimants,
                    size_1=size_1,
                    size_2=size_2,
                    lottery=lottery,
                )
            st.subheader("Prbability of selecting one of the groups of size q")
            st.write(probabilities)

elif example == "Vong 2":
    st.write(
        """
        This example is taken from Vong 2020 p.339ff. The total number of claimants is N, where N is even. There are three groups:
        - Group 1: Claimants 1 through N/2 (size: N/2)
        - Group 2: Claimants N/2+1 through N (size: N/2)
        - Group 3: Claimants 2 through N-1 (size: N-2)

        Group 3 is the largest group for N>4.

        In the paper Vong uses N=1000.
        """
    )
    with st.form("Number of claimants"):
        number_claimants = st.number_input(
            "How many claimants are there?",
            key="number_claimants",
            min_value=2,
            value=1000,
        )
        display_results = st.form_submit_button("Submit")
        if number_claimants % 2 != 0:
            st.error("Please make sure that the number of claimants is even.")
        else:
            probabilities = {}
            for lottery in ["EXCS", "EQCS", "TI"]:
                probabilities[lottery] = vong_2(number_claimants, lottery=lottery)
            st.subheader("Probability of selecting group 3")
            st.write(probabilities)

else:
    st.write("Please choose an option in the sidebar.")
