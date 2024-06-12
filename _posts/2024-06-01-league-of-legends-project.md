---
title: "League of Legends Side and Picks Analysis"
layout: post
---


**Authors**: Nathaphat Taleongpong, Gahn Wuwong

# Introduction

In this project, we perform data analysis and utilize machine learning techniques on a League of Legends dataset, with a particular emphasis on champion picks, the starting side on the map, and their impact on game outcomes.

League of Legends is a strategy game where two teams compete to destroy each other's bases. The dataset used in this study is from Oracle's Elixir containing data of over 10,000 competitive matches between 2022-2024.

In League of Legends, before the match begins, each team is assigned a starting side: blue or red. The blue side's botlane has the blue buff monster, while the red side's botlane has the red buff monster. These differences can influence the team composition and the outcome of the game. Our central question is *How does the side, champion picks, and early game team stats impact the result of the game?* Answering this question will allows coaches to determine which team has the current advantage.

### General Data Information

The original dataset consists of 326,592 rows and 131 columns, as we are focusing on data from 2022 to 2024. The relevant columns used in this study are:

`gameid, game, side, result, league, teamname, teamid, date, champion, ban1, ban2, ban3, ban4, ban5, goldat15, xpat15, csat15, golddiffat15, xpdiffat15, csdiffat15, killsat15, assistsat15, deathsat15`    

After dropping irrelevant rows and columns, we are left with 268,740 rows and 23 columns.

**Description of Columns:**

- `gameid`: Unique identifier for each game.
- `game`: Game number in a match series.
- `side`: The team's starting side (blue or red).
- `result`: Outcome of the game (win or loss).
- `league`: The league in which the game was played.
- `teamname`: Name of the team.
- `teamid`: Unique identifier for the team.
- `date`: Date when the game took place.
- `champion`: Champion chosen by the player
- `ban1, ban2, ban3, ban4, ban5`: Champions banned by the team.
- `goldat15`: Gold accumulated by the player at 15 minutes.
- `xpat15`: Experience points accumulated by player at 15 minutes.
- `csat15`: Minions killed by the player 15 minutes.
- `golddiffat15`: Difference in gold between the players in team1 and team2 at the 15 minutes.
- `xpdiffat15`: Difference in experience points between players in team1 and team2 at the 15 minutes.
- `csdiffat15`: Difference in creep score between players in team1 and team2 at the 15 minutes.
- `killsat15`: Number of kills achieved by the player at the 15 minutes.
- `assistsat15`: Number of assists achieved by the player at the 15 minutes.
- `deathsat15`: Number of deaths suffered by the player at the 15 minutes.



# Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The data cleaning process began with removing incomplete data by excluding rows where `datacompleteness` wasn't labeled as `'complete'`. This step ensured that most of the columns will not have incomplete data. However, some of the data was still missing, so we proceeded by dropping irrelevant columns that were not part of our columns of interest.

Next, the `date` column was converted into datetime format, enabling us to perform any date-related analysis if required. We also filtered out games with missing `gameid`, since identifying and tracking individual games is essential for our study.

Then we removed 24 rows (from the same sets of games) that had missing data in the columns `golddiffat15`, `xpat15`, `csdiffat15`, `killsat15`, `assistsat15`,`deathsat15`. Since these columns could be significant factors in determining the game's outcome, we decided to not impute the values. Finally, we dropped games without complete ban picks to ensure that the dataset is aligned with the latest game format.

Overall, these steps provided us with clean data to begin our analysis. The first few rows of the cleaned DataFrame are shown below:

|    | gameid                |   game | side   |   result | league   | teamname                 | teamid                                  | playername   | playerid                                  |   participantid | date                | champion   | ban1   | ban2    | ban3   | ban4   | ban5   |   goldat15 |   xpat15 |   csat15 |   golddiffat15 |   xpdiffat15 |   csdiffat15 |   killsat15 |   assistsat15 |   deathsat15 |
|---:|:----------------------|-------:|:-------|---------:|:---------|:-------------------------|:----------------------------------------|:-------------|:------------------------------------------|----------------:|:--------------------|:-----------|:-------|:--------|:-------|:-------|:-------|-----------:|---------:|---------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|
|  0 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Soboro       | oe:player:38e0af7278d6769d0c81d7c4b47ac1e |               1 | 2022-01-10 07:44:08 | Renekton   | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5025 |     7560 |      135 |            391 |          345 |           14 |           0 |             1 |            0 |
|  1 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Raptor       | oe:player:637ed20b1e41be1c51bd1a4cb211357 |               2 | 2022-01-10 07:44:08 | Xin Zhao   | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5366 |     5320 |       89 |            541 |         -275 |          -11 |           2 |             3 |            2 |
|  2 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Feisty       | oe:player:d1ae0e2f9f3ac1e0e0cdcb86504ca77 |               3 | 2022-01-10 07:44:08 | LeBlanc    | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5118 |     6942 |      120 |           -475 |          153 |            1 |           0 |             3 |            0 |
|  3 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Gamin        | oe:player:998b3e49b01ecc41eacc392477a98cf |               4 | 2022-01-10 07:44:08 | Samira     | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       5461 |     4591 |      115 |           -793 |        -1343 |          -34 |           2 |             1 |            2 |
|  4 | ESPORTSTMNT01_2690210 |      1 | Blue   |        0 | LCKC     | Fredit BRION Challengers | oe:team:68911b3329146587617ab2973106e23 | Loopy        | oe:player:e9741b3a238723ea6380ef2113fae63 |               5 | 2022-01-10 07:44:08 | Leona      | Karma  | Caitlyn | Syndra | Thresh | Lulu   |       3836 |     3588 |       28 |            443 |         -497 |            7 |           1 |             2 |            2 |


### Univariate Analysis

<iframe
  src="../assets/Univariate_GPerM.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of Games per Match" histogram illustrates the frequency of matches based on the number of games played per match in the dataset. The x-axis represents the number of games per match, ranging from 1 to 5, while the y-axis represents the count of matches. The chart shows that single-game matches are the most common, with over 400,000 instances. The frequency drops significantly for two-game matches and continues to decline for matches with 3, 4, and 5 games. Overall, the distribution is strongly right-skewed, indicating that most matches consist of just one game, with fewer matches involving multiple games.

<iframe
  src="../assets/Univariate_Sides.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of Sides" histogram was plotted to ensure an equal distribution of sides between all the games. The x-axis represents the sides (Blue and Red), while the y-axis represents the count of games. The chart shows that the number of games starting on the Blue side is equal to the number of games starting on the Red side. This confirmed that there was no significant imbalance or missing data regarding the starting sides, allowing us to proceed with our data analysis.


### Bivariate Analysis

<iframe
  src="../assets/Bivariate_SideWinRate.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Win Rate by Side" pie chart shows the percentage of wins based on the starting side of a team. The chart shows that the Blue side has a higher win rate of 52.7%, while the Red side has a win rate of 47.3%. This indicates that teams starting on the Blue side tend to win more often compared to those starting on the Red side.

<iframe
  src="../assets/Bivariate_BannedWinRate.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Win Rate by Banned Character" bar chart shows how the win rate of a team varies depending on the champion they chose to ban. The x-axis lists the banned characters, while the y-axis represents the win rate. The chart shows that banning certain champions, like Yorick and Warwick, correlates with higher win rates. This analysis helps identify whether there are significant outliers in win rates based on banned champions, indicating strategic advantages in banning specific characters.

### Interesting Aggregates 

This aggregate is a pivot table comparing the win rates of different champions when played on the blue side versus the red side. The purpose of this table is to identify which champions yield the highest win rates and to determine if either side has a significant advantage. There seems to be a trend of the blue side generally having higher win rates then the red side. This pattern is consistent across most champions, indicating that the blue side tends to have a higher overall win rate.

| side   |   Aatrox |   Ahri |   Akali |   Akshan |   Alistar | ... |      Zoe |     Zyra |
|--------|----------|--------|---------|----------|-----------|-----|---------:|---------:|
| Blue   | 0.516355 | 0.5396 | 0.5361  | 0.5756   | 0.5612    | ... | 0.534146 | 0.473684 |
| Red    | 0.467016 | 0.4869 | 0.473   | 0.5208   | 0.4745    | ... | 0.466125 | 0.421053 |

The table highlights the comparative advantage of the blue side over the red side for various champions, providing insights into potential strategic decisions in champion selection and game planning.

# Assessment of Missingness

### NMAR Analysis

Many columns in the dataset have missing values, which could be atributed to the structure of the dataset where each game has 10 rows representing each player and 2 rows representing the overall team statistics. This structure results in missing values for the player-specific columns in the team rows. However, the missingness of these columns does not affect our analysis since we can manipulate the data to ensure the features are correctly represented.

Another pattern of missingness is observed in the columns `ban1` to `ban5`, which represent the champions banned by the team. We believe this missingness is Not Missing at Random (NMAR) because players have the option to ban or not to ban champions. However, the game format has changed over time, which could contribute to the missingness. To confirm this, we will explore whether the missingness in the ban columns is Missing At Random (MAR) depending on the year and month. By analyzing the missing data patterns across different time periods, we can determine if the missingness correlates with specific years and months, indicating that changes in the game format influences the missingness of the ban data. The result of this analysis will help us decide how to handle the missing data in the ban columns to appropriately use them in our models.

### Missingness Dependency

**MAR Depending on Year**

**Null Hypothesis**: Missingness of `ban4` and `ban5` is not dependent on `year`.     
   
**Alternate Hypothesis**: Missingness of `ban4` and `ban5` is dependent on `year`.       

**Test Statistic**: TVD between distribution of `year` by missing and not missing `ban4` and `ban5`.

**Significance Level**: 0.05

To test the hypothesis, we calculated the observed total variation distance (TVD) between the distribution of `year` by missing and not missing `ban4` and `ban5`. Next, we simulated the missingness of `ban4` and `ban5` under the null hypothesis, which assumes that the missingness is not dependent on `year`. This process was repeated numerous times to generate a distribution of total variation distances under the null hypothesis. Finally, we calculated the p-value by comparing the observed TVD to the simulated distribution.

<iframe
  src="../assets/Missingness_TVD.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of TVD" histogram shows the simulated distribution of TVD under the null hypothesis. The observed TVD is marked on the right side of the chart. The p-value calculated from this test is 0, which is lower than the significance level of 0.05. Therefore, we reject the null hypothesis, which assumed that the missingness in `ban4` and `ban5` is not dependent on `year`. This is likely due to changes in the game format over time, which influences the ban data. In order have a more detailed analysis, we will also explore the missingness dependency on the specific month of each year.

**MAR Depending on Year and Month**

**Null Hypothesis**: Missingness of `ban4` and `ban5` is not dependent on `year-month`.     
   
**Alternate Hypothesis**: Missingness of `ban4` and `ban5` is dependent on `year-month`.       

**Test Statistic**: TVD between distribution of `year` by missing and not missing `ban4` and `ban5`.

**Significance Level**: 0.05

For this hypothesis test, we extracted the month and year from the date column and combined them to create a new column `year-month`. This allows us to group the data by year and month and analyze the missingness pattern. Again, we calculated the observed total variation distance (TVD) betwen the distribution of `year-month` by missing and not missing `ban4` and `ban5`. We then simulated the missingness of `ban4` and `ban5` under the null hypothesis, which assumes that the missingness is not dependent on `year-month`. We repeated this process numerous times to generate a distribution of total variation distances under the null hypothesis. Finally, we calculated the p-value by comparing the observed TVD to the simulated distribution.

<iframe
  src="../assets/Missingness2_TVD.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of TVD" histogram shows the simulated distribution of TVD under the null hypothesis. The observed TVD is marked on the right side of the chart. The p-value calculated from this test is 0, which is lower than the significance level of 0.05. Therefore, we reject the null hypothesis, which assumed that the missingness in `ban4` and `ban5` is not dependent on `year-month`.

This dependency on year and month is likely due to seasonal changes in the format of the game. Moving forward, we will be using the game's latest ban format, so we will remove any games that previously only had only 2 or 3 bans. Additionally, to make the data easier to use and analyze, we will only include games where both teams banned 5 characters.

# Hypothesis Testing

In our hypothesis test, we wanted to determine whether there is a difference between the chances of winning for teams starting on the Red versus the Blue side. This analysis is crucial in ensuring fairness for both starting sides in a competitive environment. Ideally, the win rates for both sides should be relatively similar, however, we will investigate whether the difference in win rates are statisically significant or not.

**Question**: Do both blue and red teams have the same chance of winning?       

**Null Hypothesis**: Blue and red teams have the same chances of winning.       

**Alternative Hypothesis**: Blue and red teams don't have the same changes of winning.       

**Test Statistic**: The absolute difference in win rate between blue and red teams.      

**Significance Level**: 0.05        

To conduct this hypothesis test, we first calculated the absolute difference in observed win rates between blue and red teams. Then we simulated the win rates under the null hypothesis, which assumes that both sides have the same chance of winning (50/50). This process was repeated numerous times to generate a distribution of win rate differences under the null hypothesis. Finally, we calculated the p-value by comparing the observed absolute difference to this simulated distribution.

<iframe
  src="../assets/HypothesisTest.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of Win Rate Differences" histogram shows the simulated distribution of win rate differences under the null hypothesis. The observed difference is marked on the right side of the chart. The p-value calculated from this test is 0, which is lower than the significance level of 0.05. Therefore, we reject the null hypothesis, which assumed that blue and red teams have the same chances of winning. Factors such as the format of bans and picks, where the blue team gets to select some of their bans before the Red team, might contribute to this observed difference.

# Framing a Prediction Problem

Since the blue and red teams have different chances of winning, we aim to explore whether this bias can be leveraged to predict the game's outcome. To enhance the prediction's accuracy, we will also use champion picks as features. Picks are crucial to the game and are one of the few pieces of information we have at the beginning of the game. 

**Prediction Problem**      
Our prediction problem is a classification task. Specifically, we are performing binary classification to predict the outcome of a game (win or loss) based on the side and champion picks.

**Response Variable**       
The response variable is the game's result, which can be either a win (1) or a loss (0).

**Model**       
We will be using a RandomForest Classifier to predict the game's outcome based on the side and champion picks. RandomForest is a powerful ensemble learning method that can handle non-linear relationships and interactions between features. It is also robust to overfitting and can handle high-dimensional data.

**Evaluation Metric**       
We will use accuracy to evaluate our model. Accuracy is chosen because the proportions of wins and losses are nearly balanced (47% for red and 53% for blue), making it an appropriate metric to assess the model's overall performance in correctly predicting game outcomes. Accuracy provides a clear and direct measure of the model's effectiveness.

# Baseline Model

For the baseline model we decided to use RandomForestClassifier (RFC) with two nominal features: `champion` and `side`. Since the available champions are predetermined, we do not need to worry about data leakage between training and testing sets. This allowed us to manually performed one-hot encoding on the `champion` feature before fitting the model pipeline. We also one-hot encoded the `side` feature to represent the starting side as a numerical value. Then we split our dataset into training and testing sets, with 20% of the data allocated to testing. We then trained the RFC on the training data and evaluated its performance using accuracy on the testing data. We still need to improve it to be able to generalize better by improving hyperparameters and adding more features in.

|Training Accuracy|Testing Accuracy|
|---|---|
|0.6397|0.5220|

The model's training accuracy indicates that it performed reasonably well on the training data. However, the testing accuracy is around 52%, which is only slightly better than randomly guessing beased on side. This suggests that the model's performance drops on unseen data and that the side and champion picks alone are not sufficient to predict the game's outcome. To improve generalization, we plan to optimize hyperparameters and add more numerical features in the final model.

# Final Model
### Feature Engineering

In the final model, we increased our model's performance by introducing more numerical features within the data. We added the differences in gold, experience points, and minions killed at 15 minutes into the game in order to better determine if a team has an advantage early on in the game. We also included the total number of kills, assists, and death in the first 15 minutes to better predict the probability of the team winning. Additionally, we also included the game number of the match to takeinto account of the stamina of the players. 

We used the one-hot encoded `champion` and `side` feature again in this model. We then used the standard scaler to scale the numerical features. Finally we fit the model using the RandomForestClassifier with the following hyperparameters:

|Hyperparameters|Value|
|---|---|
|n_estimators|100|
|max_depth|25|
|min_samples_leaf|5|

We then evaluated the model's performance using accuracy on the testing data.

|Training Accuracy|Testing Accuracy|
|---|---|
|0.8026|0.7479|

### Grid Search Cross Validation

To further improve the model's performance, we performed Grid Search Cross Validation to optimize the hyperparameters. We used the following hyperparameters:  

`n_estimators`: Number of decision trees. More iterations can capture more patterns but increase training time.    

`max_depth`: Depth of the trees. Controls the complexity of the model. Deeper trees can model more complex interactions but may overfit.  

`min_samples_leaf`: Minimum number of samples required to be at a leaf node. Helps prevent overfitting.  

`max_features`: The function used to decide how many features to consider when fitting. Helps prevent overfitting.  

We used the same features and scaling as the model mentioned above. We then performed Grid Search Cross Validation with 5 folds with the following hyperparameters:

|Hyperparameters|Values|
|---|---|
|n_estimators| 75, 100, 125|
|max_depth| 20, 25, 30|
|min_samples_leaf| 6, 9, 12|
|max_features| sqrt, log2|

The best hyperparameters found were:

|Hyperparameters|Values|
|---|---|
|n_estimators| 125|
|max_depth| 20|
|min_samples_leaf| 9|
|max_features| log2|

The performance of the model with the optimized hyperparameters is as follows:

|Training Accuracy|Testing Accuracy|
|---|---|
|0.7493|0.7473|

Our final model achieved a training accuracy of 74.93% and a testing accuracy of 74.73%, showing a 22% increase in testing accuracy compared to the baseline model. By including additional features and optimizing hyperparameters, the model was able to generalize better to unseen data and predict the game's outcome based on side, champion picks, and early game tem statistics with acceptable accuracy. 

A potential limitation of the model is that it may not account for all factors influencing the game's outcome, such as player skill, team coordination, and in-game strategies.
However, the model provides valuable insights into the impact of side, champion picks, and early game team stats on the game's result. These insights can help optimize team strategies and improve performance in competitive play.

<iframe
  src="../assets/Confusion.html"
  width="650"
  height="600"
  frameborder="0"
></iframe>

The confusion matrix shows the model's performance on the testing data. The model correctly predicted 2,763 wins and 2,816 losses, while misclassifying 967 wins as losses and 920 losses as wins. The model's precision is 75% and recall is 74%. This indicates that the model has a good balance between precision and recall, correctly identifying the game's outcome in most cases.

# Fairness Analysis

In order to assess the fairness of our model in predicting the game's outcome, we performed a fairness test to determine if the model has similar performance for both blue and red sides. The fairness test compares the accuracy of the model for blue and red teams and determines if any differences are due to random chance.

**Question**: Does my model have a similar performance for both blue and red sides?       

**Null Hypothesis**: Our model is fair. The accuracy of the model is roughly the same for both sides and any differences are due to random chance.       

**Alternative Hypothesis**: Our model is unfair. The accuracy of the model is significantly different for both sides.       

**Test Statistic**: The absolute difference in accuracy between blue and red teams.      

**Significance Level**: 0.05        

To conduct this fairness analysis, we first calculated the absolute difference in accuracy between blue and red teams. Then we simulated the absolute difference in accuracy under the null hypothesis, which assumes that both sides have the same accuracy. This process was repeated numerous times to generate a distribution of accuracy differences under the null hypothesis. Finally, we calculated the p-value by comparing the observed absolute difference to this simulated distribution.

<iframe
  src="../assets/FairnessTest.html"
  width="650"
  height="420"
  frameborder="0"
></iframe>

The "Distribution of Absolute Difference in Accuracy" histogram shows the simulated distribution under the null hypothesis, with the observed difference marked in the middle of the distribution. The p-value calculated from this test is higher than the significance level of 0.05. Therefore, we fail to reject the null hypothesis, indicating that our model is fair and the difference in accuracy between blue and red teams is due to random chance.