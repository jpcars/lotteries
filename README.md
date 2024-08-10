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


## Currently implemented lotteries

For references on these lotteries please check [[1]](#1), [[2]](#2) and [[3]](#3).



| Lottery                                 | without uncertainty | with uncertainty	 |
|-----------------------------------------|----------------|-----------------|
| Exclusive composition sensitive lottery |&check;| &cross;*        |
| Equal composition sensitive lottery     |&check;| &cross;*        |
| Timmermann's individualist lottery      |&check;| &cross;         |
| Taurek's coin toss                      |&check;| &cross;         |

*EXCS and EQCS both do not have straightforward generalizations to cases with uncertainty. Several alternatives
could be implemented.


## Local setup

### Python
It is encouraged to create a virtual environment and use poetry to handle the dependency management.

We are using pynauty for graph isomorphism computations. This depends on the C library nauty and requires a C
compiler to be installed.

When all requirements are met please run

    poetry install

inside the virtual environment to install all dependencies (including dev dependencies).

### Postgres

The library contains the functionality to persist results in a postgres database. The library and the streamlit app
**do not** use this functionality by default.

There are two main use cases:
1. Since the computation of large rescue dilemmas is recursive database lookups can increase performance
2. The database will serve as foundation for a framework for the structured analysis of rescue cases.

If you want to use this feature, please make sure that you have a connection to a postgres server. Please create a
database called `lotteries` and create the tables listed in `lottery/database/sql_scripts/create_tables.sql`.

A file called database.ini in the directory `lottery/database` is expected. The minimum content of this file is

    [postgresql]
    host=<host>
    database=lotteries
    user=<username>
    password=<password>

The provided user needs read and write access for the above-mentioned tables.

## Setup with Docker

Coming soon

## Run streamlit app locally

To run the streamlit app execute

    streamlit run streamlit/Rescue_Lotteries.py

in the command line.


## To do
 - Write tests
 - Adjust probability reporting
 - Implement actual symbolic lottery
 - Implement entire-claim-of-claimant-to-largest-group-of-claimant lottery

## References
<a id="1">[1]</a>
Vong, Gerard (2020):
Weighing Up Weighted Lotteries: Scarcity, Overlap Cases, and Fair Inequalities of Chance,
Ethics 130 (3), pp. 320-348.

<a id="2">[2]</a>
Timmermann, Jens (2004):
The Individualist Lottery: How People Count, but Not Their Numbers,
Analysis 64 (2), pp. 106-112.

<a id="3">[3]</a>
Taurek, John (1977):
Should the Numbers Count?,
Philosophy & Public Affairs 6 (4), pp. 293-316.
