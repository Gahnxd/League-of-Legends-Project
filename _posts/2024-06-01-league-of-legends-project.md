---
title: "League of Legends Side and Ban Picks Analysis"
layout: post
---

# League of Legends Side and Picks Analysis

Authors: Nathaphat Taleongpong, Gahn Wuwong

# Introduction

In this project, we perform data analysis and utilize machine learning techniques on a League of Legends dataset, with a particular emphasis on champion bans, the starting side on the map, and their impact on game outcomes.

League of Legends (LoL) is a strategy game with two teams battling against one another to destroy their bases. The dataset in this study is from Oracle's Exilir containing data of over 10,000 LoL competitive matches. The main focus will be between the years 2023-2024.

In LoL before the match begins, a team is given a side that they start on, blue or red. In the game, the blue side corresponds to your botlane having the blue buff monster, whilst the red side the botlane has the red buff monster. This can change or shift the team composition. The central question we pose is *The prediction of the result of the match depending on the side and numerical data within the first 15 minutes of the game*. This allows coaches to determine which team has the current advantage.

### General Data Information

The data consists of 326592 rows and 131 columns as we are only using the dataset of 2023 and 2024. The relevant columns we use in this study are: 

The columns of interest to me are `'gameid, game, side, result, league, teamname, teamid, date, champion, ban1, ban2, ban3, ban4, ban5, goldat15, xpat15, csat15, golddiffat15, xpdiffat15, csdiffat15, killsat15, assistsat15, deathsat15'`

**Description of Columns:**

- `gameid`: Identifier for a game.
- `game`: Game number of a match.
- `side`: The team's side in the game.
- `result`: Outcome of the game.
- `league`: The league that the game was played.
- `teamname`: Name of the team.
- `teamid`: Unique identifier for team.
- `date`: Date when the game happened.
- `champion`: Champion chosen by player
- `ban1, ban2, ban3, ban4, ban5`: Columns ban1, ban2, ban3, ban4, ban5 is what each team picked to ban.
- `goldat15`: Gold of player at 15 minutes.
- `xpat15`: XP of player at 15 minutes.
- `csat15`: Minions killed at 15 minutes.
- `golddiffat15`: Difference in gold between the positions in team1 and team2 at the 15 minutes.
- `xpdiffat15`: Difference in experience points between positions in team1 and team2 at the 15 minutes.
- `csdiffat15`: Difference in creep score between the team1 and team2 at the 15 minutes.
- `killsat15`: Number of kills achieved by the player at the 15 minutes.
- `assistsat15`: Number of assists achieved by the player at the 15 minutes.
- `deathsat15`: Number of deaths suffered by the player at the 15 minutes.



# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

The data cleaning process began with removing incomplete data, we took out rows in `datacompleteness` that weren't labeled as 'complete'. This ensured that most of the columns will have all data in them, however, we also checked to ensure that all data was actually complete as well. Following this, only the relevant columns specified in our columns of interest were kept.

The date column was converted into datetime format, which allow us to perform any date analysis if required. Subsequently, the index was reset to ensure it was continuous and started from zero. Then we filtered out games without a game ID, as the game ID serves as a fundamental identifier for tracking individual games throughout the analysis.

We also found that there was 24 rows (same game ID) had missing data in `golddiffat15', xpat15` were removed. Additionally, games without complete ban picks were dropped, as this information is essential for understanding team strategies and dynamics during matches.

Overall, these cleaning steps were essential for preparing the dataset for analysis, ensuring that only high-quality, complete data was used. By removing incomplete or erroneous entries and standardizing the data format, the cleaned dataset provides a solid foundation for meaningful analysis and insights into the dynamics of the games.

|    | gameid                |   game | side   |   result | league   | teamname                 | teamid                                  | playername   | playerid                                  |   participantid | date                | champion   | ban1   | ban2    | ban3   | ban4   | ban5   |   goldat15 |   xpat15 |   csat15 |   golddiffat15 |   xpdiffat15 |   csdiffat15 |   killsat15 |   assistsat15 |   deathsat15 |
|---:|:----------------------|-------:|:-------|---------:|:---------|:-------------------------|:----------------------------------------|:-------------|:------------------------------------------|----------------:|:--------------------|:-----------|:-------|:--------|:-------|:-------|:-------|-----------:|---------:|---------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|
|  0 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Soboro       | oe:player:38e0af7278d6769d0c81d7c4b47ac1e |               1 | 2022-01-10 07:44:08 | Renekton   | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5025 |     7560 |      135 |            391 |          345 |           14 |           0 |             1 |            0 |
|  1 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Raptor       | oe:player:637ed20b1e41be1c51bd1a4cb211357 |               2 | 2022-01-10 07:44:08 | Xin Zhao   | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5366 |     5320 |       89 |            541 |         -275 |          -11 |           2 |             3 |            2 |
|  2 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Feisty       | oe:player:d1ae0e2f9f3ac1e0e0cdcb86504ca77 |               3 | 2022-01-10 07:44:08 | LeBlanc    | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5118 |     6942 |      120 |           -475 |          153 |            1 |           0 |             3 |            0 |
|  3 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Gamin        | oe:player:998b3e49b01ecc41eacc392477a98cf |               4 | 2022-01-10 07:44:08 | Samira     | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5461 |     4591 |      115 |           -793 |        -1343 |          -34 |           2 |             1 |            2 |
|  4 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Loopy        | oe:player:e9741b3a238723ea6380ef2113fae63 |               5 | 2022-01-10 07:44:08 | Leona      | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       3836 |     3588 |       28 |            443 |         -497 |            7 |           1 |             2 |            2 |


## Univariate Analysis

The bar chart,"Distribution of Games per Match", there is a pattern which looks similar to a right-skew towards single-game matches, with over 400,000 instances. As the number of games per match increases, the number of matches sharply decreases, which means matches with 5 games are significantly less common.



The bar chart, "Distribution of Sides", was plotted to make sure that there was an equal number of sides between all the games (Red equals Blue). The bar chart confirmed that there was data which was missing for this, therefore we were able to continue with our data analysis.


<iframe
  src="../assets/Univariate_GPerM.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

<iframe
  src="../assets/Univariate_Sides.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>


## Bivariate Analysis

<iframe
  src="../assets/Bivariate_SideWinRate.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>


<iframe
  src="../assets/Bivariate_BannedWinRate.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>


## Interesting Aggregates 

| side   |   Aatrox |   Ahri |   Akali |   Akshan |   Alistar | ... |      Zoe |     Zyra |
|--------|----------|--------|---------|----------|-----------|-----|---------:|---------:|
| Blue   | 0.516355 | 0.5396 | 0.5361  | 0.5756   | 0.5612    | ... | 0.534146 | 0.473684 |
| Red    | 0.467016 | 0.4869 | 0.473   | 0.5208   | 0.4745    | ... | 0.466125 | 0.421053 |


# Assessment of Missingness

## NMAR Analysis

We believe that the columns `ban1` to `ban5` is NMAR because players can choose to ban or not to ban. However, the game format may have changed over time, so we will explore whether the missingness in the ban columns are MAR depending on the year and month.

## Missingness Dependency

**MAR Depending on Year**

**Null Hypothesis**: Missingness of `ban4` and `ban5` is not dependent on `year`.     
   
**Alternate Hypothesis**: Missingness of `ban4` and `ban5` is dependent on `year`.       

**Test Statistic**: TVD between distribution of `year` by missing and not missing `ban4` and `ban5`.

**Significance Level**: 0.05

<iframe
  src="../assets/Missingness_TVD.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

We got a p-value of 0, which is lower than the significance level, so we reject the null hypothesis. This means that `ban4` and `ban5` is dependent on `year`.

**MAR Depending on Year and Month**

**Null Hypothesis**: Missingness of `ban4` and `ban5` is not dependent on `year-month`.     
   
**Alternate Hypothesis**: Missingness of `ban4` and `ban5` is dependent on `year-month`.       

**Test Statistic**: TVD between distribution of `year` by missing and not missing `ban4` and `ban5`.

**Significance Level**: 0.05

<iframe
  src="../assets/Missingness2_TVD.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The p-value is 0, which is lower than the significance level, so we reject the null hypothesis. This means that the missingness in `ban4` and `ban5` is dependent on `year-month`.

This dependency on year and month is due to the seasonal changes in the format of the game. Moving forward, we will be using the game's latest ban format, so we will remove any games that previously only had 2 bans and 3 bans. Additionally, to make the data easier to use and analyze, we will only include games where both teams banned 5 characters.

# Hypothesis Testing

**Question**: Do both blue and red teams have the same chance of winning?       

**Null Hypothesis**: Blue and red teams have the same chances of winning.       

**Alternative Hypothesis**: Blue and red teams don't have the same changes of winning.       

**Test Statistic**: The absolute difference in win rate between blue and red teams.      

**Significance Level**: 0.05        

<iframe
  src="../assets/HypothesisTest.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The p-value is 0, which is lower than the significance level, so we reject the null hypothesis. This means that blue and red teams don't have the same chances of winning. 

We suspect that this might be because of the format of the bans pick, since the blue team gets to pick some of their bans before the red team.

# Framing a Prediction Problem

Since the blue and red teams have different chances of winning, we aim to explore whether this bias can be leveraged to predict the game's outcome. To enhance the prediction's accuracy, we will also use the bans as features. Bans are crucial to the game and are the only information we have before the game starts, given the absence of data on picks. 

**Prediction Problem**      
Our prediction problem is a classification task. Specifically, we are performing binary classification to predict the outcome of a game (win or loss) based on the side and bans.

**Response Variable**       
The response variable is the game's result, which can be either a win (1) or a loss (0).

**Model**       
We will be using a CatBoost Classifier to predict the game's outcome based on the side and bans. The reason we chose the CatBoost Classifier is because it can handle categories that appear in the test set but were not present in the training set. This capability is particularly important is our dataset where the banned characters might vary between training and testing sets.

**Evaluation Metric**       
We will use accuracy to evaluate our model. Accuracy is chosen because the proportions of wins and losses are nearly balanced (47% for red and 53% for blue), making it an appropriate metric to assess the model's overall performance in correctly predicting game outcomes. Accuracy provides a clear and direct measure of the model's effectiveness.

# Baseline Model
# Final Model
### Feature Engineering


We decided to experiment with more numerical features within the data. We added the difference in gold, xp, and cs at 15 minutes into the game in order to better determine if the team has an advantage early on in the game. We will also use the number of kills, assists, and death in the first 15 minutes to better predict the probability of the team winning. Additionally, we also included the game number of the match to takeinto account of the stamina of the players.

### Grid Search Cross Validation

**Hyperparameters**     

`n_estimators`: Number of decision trees. More iterations can capture more patterns but increase training time.    

`max_depth`: Depth of the trees. Controls the complexity of the model. Deeper trees can model more complex interactions but may overfit.  

`min_samples_leaf`: Minimum number of samples required to be at a leaf node. Helps prevent overfitting.  

`max_features`: The function used to decide how many features to consider when fitting. Helps prevent overfitting.  

<iframe
  src="../assets/Confusion.html"
  width="650"
  height="600"
  frameborder="0"
></iframe>


# Fairness Analysis
<iframe
  src="../assets/FairnessTest.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The p-value is higher than the significance level, so we fail to reject the null hypothesis. This means that our model is fair and the difference in accuracy between blue and red teams is due to random chance.