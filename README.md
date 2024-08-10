# Rescue lotteries

## The ethical debate around rescue dilemmas

Rescue dilemmas are a class of thought experiments used by ethicists to test their intuition and devise fairer
decision procedures on who to save if it is not possible to save everyone. A common setup is the following:

```
A ship captain receives two calls for rescue from two different sinking boats at the same time. One boat has one
person on board, the other one has two persons on board. The boats are too far apart to save both of them. The
people on the boat that is not chosen to be rescued will drown. Without knowing anything else, what should the
captain do to decide, which of the two groups to rescue?
```

While simple on the surface, problems of this sort lead to deep philosophical questions about fairness and the debate
around these dilemmas has a long history. Some philosophers participating in this debate see a lot of value in
designing specific dilemmas to test their own decision procedures. One category of decision procedures of particular
interest are so-called lotteries. These procedures don't result in saving one group outright but instead give every
group some probability to be saved.

Depending on the complexity of the considered cases the mathematical analysis of these procedures quickly gets
intricate. The aim of this repo is to provide tools to practitioners to ease the analysis effort by automating
computations.


## Run streamlit app locally

To run the streamlit app execute

    streamlit run streamlit/Rescue_Lotteries.py

in the command line.


## To do
 - Adjust probability reporting
 - Implement actual symbolic lottery
 - Implement entire-claim-of-claimant-to-largest-group-of-claimant lottery
