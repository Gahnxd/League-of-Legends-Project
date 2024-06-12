---
title: "League of Legends Side and Bans Picks Analysis"
layout: post
---

Authors: Nathaphat Taleongpong, Gahn Wuwong

# Introduction

In this project, we perform data analysis and utilize machine learning techniques on a League of Legends dataset, with a particular emphasis on champion bans, the starting side on the map, and their impact on game outcomes.

League of Legends (LoL) is a strategy game with two teams battling against one another to destroy their bases. The dataset in this study is from Oracle's Exilir containing data of over 10,000 LoL competitive matches.

In LoL before the match begins, a team is given a side that they start on blue or bed. In the game, the blue side corresponds to your botlane having the blue buff monster, whilst the red side the botlane has the red buff monster. This can change or shift the team composition.

### General Data Information

The data consists of 922644 rows and 131 columns.



# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

During the data cleaning process we initially filtered the dataset to include only entries with complete data by retaining rows where the 'datacompleteness' column was 'complete'. This step ensured that only the most reliable data entries were analyzed and no missing data would be used for our model.

Next, specific columns relevant to the analysis were selected: 'gameid, game, side, result, league, teamname, teamid, playername, playerid, participantid, date, champion, ban1, ban2, ban3, ban4, ban5, goldat15, xpat15, csat15, golddiffat15, xpdiffat15, csdiffat15, killsat15, assistsat15, deathsat15'. The 'date' column was then converted to datetime format to be able to perform cleaning on because different years there may have been rules in the bans you can choose.

Then we performed a reset index so that ..., and anything with missing 'gameid' values were dropped. This step ensured that only identifiable and traceable game data remained.

To further refine the dataset, any games with missing ban picks (from 'ban1' to 'ban5') were identified and removed. This was done by identifying unique game IDs with missing ban picks and then dropping all corresponding entries. We decided to drop games that don't have a gameid because we won't be able to group by games with no id. Since we can't impute missing bans, we have to drop all games that have missing bans.

The head of the cleaned DataFrame, showing the first few rows of the refined dataset, confirmed that the data was now consistent and ready for in-depth analysis. This cleaning process enhances the dataset's quality, so that we can perform better analysis and have a stronger model when it comes to predicting.

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

<style>
    .table-container {
        max-height: 200px; /* Adjust this value as needed */
        overflow: hidden; /* This will hide the vertical scrollbar */
        display: block;
        width: 100%;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
        word-wrap: break-word;
    }
    th, td {
        max-width: 100px; /* Adjust this value as needed */
    }
</style>

<div class="table-container">
<table>
  <tr>
    <th>side</th>
    <th>Aatrox</th>
    <th>Ahri</th>
    <th>Akali</th>
    <th>Akshan</th>
    <th>Alistar</th>
    <th>...</th>
    <th>Zoe</th>
    <th>Zyra</th>
  </tr>
  <tr>
    <td>Blue</td>
    <td>0.516355</td>
    <td>0.5396</td>
    <td>0.5361</td>
    <td>0.5756</td>
    <td>0.5612</td>
    <td>...</td>
    <td>0.534146</td>
    <td>0.473684</td>
  </tr>
  <tr>
    <td>Red</td>
    <td>0.467016</td>
    <td>0.4869</td>
    <td>0.473</td>
    <td>0.5208</td>
    <td>0.4745</td>
    <td>...</td>
    <td>0.466125</td>
    <td>0.421053</td>
  </tr>
</table>
</div>



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
# Framing a Prediction Problem
# Baseline Model
# Final Model
# Fairness Analysis
