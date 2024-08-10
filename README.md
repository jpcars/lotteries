# Rescue lotteries

To do:
 - Write documentation
 - Adjust probability reporting
 - Implement actual symbolic lottery
 - get rid of self.lottery_name and use str of class instead
 - Implement entire-claim-of-claimant-to-largest-group-of-claimant lottery

## Lotteries

| Lottery                                 | without uncertainty | with uncertainty	 |
|-----------------------------------------|----------------|-----------------|
| Exclusive composition sensitive lottery |&check;| &cross;*        |
| Equal composition sensitive lottery     |&check;| &cross;*        |
| Timmermann's individualist lottery      |&check;| &check;         |
| Taurek's coin toss                      |&check;| &check;         |

*EXCS and EQCS both do not have straightforward generalizations to cases with uncertainty. Several alternatives
could be implemented.


## Run streamlit app locally

To run the streamlit app execute

    streamlit run streamlit_app.py

in the command line.
