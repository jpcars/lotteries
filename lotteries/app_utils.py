import streamlit as st


def validate_group_input(groups):
    empty_groups = any([len(group) == 0 for group in groups])
    if empty_groups:
        st.error("Please make sure that all groups are non-empty.")

    removed_duplicates = []
    for group in groups:
        if set(group) not in removed_duplicates:
            removed_duplicates.append(set(group))

    duplicates = len(removed_duplicates) < len(groups)
    if duplicates:
        st.error("Please make sure that all groups are unique.")

    return (not empty_groups) and (not duplicates)
