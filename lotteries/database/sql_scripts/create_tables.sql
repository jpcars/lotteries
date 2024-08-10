CREATE TABLE fact_dilemmas (
            dilemma_key SERIAL PRIMARY KEY,
            graph_certificate BYTEA NOT NULL UNIQUE,
            matrix_rep FLOAT[][] NOT NULL,
            number_claimants INTEGER NOT NULL,
            number_groups INTEGER NOT NULL
        );
CREATE TABLE fact_dilemmas_uncertainty (
            dilemma_key SERIAL PRIMARY KEY,
            graph_certificate BYTEA NOT NULL UNIQUE,
            matrix_rep VARCHAR(10)[][] NOT NULL,
            number_claimants INTEGER NOT NULL,
            number_groups INTEGER NOT NULL
        );
CREATE TABLE dim_lotteries (
            lottery_code VARCHAR(4) PRIMARY KEY,
            lottery_name VARCHAR(255) NOT NULL UNIQUE
        );
CREATE TABLE fact_probabilities (
            dilemma_key INTEGER NOT NULL,
            lottery_code VARCHAR(4) NOT NULL,
            PRIMARY KEY (dilemma_key , lottery_code),
            FOREIGN KEY (dilemma_key)
                REFERENCES fact_dilemmas (dilemma_key)
                ON UPDATE CASCADE ON DELETE CASCADE,
            FOREIGN KEY (lottery_code)
                REFERENCES dim_lotteries (lottery_code)
                ON UPDATE CASCADE ON DELETE CASCADE,
            group_probabilities FLOAT[] NOT NULL
        );
CREATE TABLE fact_probabilities_uncertainty (
            dilemma_key INTEGER NOT NULL,
            lottery_code VARCHAR(4) NOT NULL,
            PRIMARY KEY (dilemma_key , lottery_code),
            FOREIGN KEY (dilemma_key)
                REFERENCES fact_dilemmas_uncertainty (dilemma_key)
                ON UPDATE CASCADE ON DELETE CASCADE,
            FOREIGN KEY (lottery_code)
                REFERENCES dim_lotteries (lottery_code)
                ON UPDATE CASCADE ON DELETE CASCADE,
            group_probabilities TEXT[] NOT NULL
        );
