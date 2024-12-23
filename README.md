# Prediciting-NFL-Outcomes-Using-ML
## Introduction
This project uses machine learning to predict NFL game winners based on the 2023 season data. I scraped public data across offensive, defensive, and special teams statistics, tested multiple models, and identified which stats most contributed to a team's success.

For this analysis, I scraped publicly available data off nfl.com, focusing on season-level statistics across offense, defense, and special teams for each team. Despite the dataset's limitations, I was eager to see if a particular statistic would emerge as a key feature in predicting outcomes and, if so, what that statistic would be and why.

## Project Approach
Using the scraped data, I trained various machine learning models to predict game outcomes. Given the dataset's high-level nature (season stats rather than game-by-game details), I knew it would be challenging for the models to achieve high accuracy. However, my primary curiosity was in feature importance—whether a specific stat would consistently stand out across different models.

Through experimentation with multiple models and optimization techniques, one statistic emerged as the top feature: Field Goals. I found this result surprising. I expected metrics like total touchdowns, rushing touchdowns, total yards, defensive points allowed, to play a more defining role. However, offensive scoring statistics did not even make it into the top 10 features in many of the models.

## Key Finding: The Importance of Field Goals
On average, an NFL team kicks 1.76 field goals per game, accounting for approximately 23% of all NFL points (about 5.28 points per game). By contrast, teams score roughly three touchdowns per game, totaling 18–21 points. Despite touchdowns contributing more to overall scoring, the machine learning model identified field goals as the most predictive feature for game outcomes.

This wasn’t a random result. Watching games closely, I noticed a pattern: field goals often determine the winner—not just in tied games or games decided by three points, but in how they influence the entire fourth quarter's strategy.

## Field Goals and Game Strategy

The significance of a three-point lead extends far beyond the scoreboard—it reshapes the game. A small lead alters coaching decisions, clock management, and defensive aggression.

#### Consider This Scenario:
- **Situation:** 7:50 left in the 4th quarter, tied game, 1st down at your 30-yard line:
  - **Win probability if tied:** 58%.
  - **Down by a field goal:** Drops to 33%.
  - **Down by a touchdown:** Drops to 10%.
  - **Down by a touchdown and field goal:** Drops to just 3%.

- These shifts highlight the **psychological and strategic weight** of a three-point difference.

#### Decision-Making: Field Goal vs. Going for It
- **Kicking a Field Goal (Within the 30-yard line):**
  - Provides a **99% chance** of scoring three points.
- **Going for It on 4th Down:**
  - **28% chance** of scoring a touchdown.
  - **51% chance** of eventually settling for a field goal (based on average 4th down conversion rates and subsequent scoring probabilities).

- While percentages vary based on specific situations (e.g., down-and-distance, team), the **guaranteed points of a field goal often outweigh the risk.**

#### Strategic Implications:
Early in the game, risking three points for a potential touchdown might seem worthwhile. However, the percentages are not in your favor, and in the fourth quarter those three points can be game-changing.


## Dataset
The dataset used for this analysis was compiled by scraping public 2023 NFL statistics off nfl.com. It included all statistics pertaining to offense, defense, and special teams for each team over the course of the entire season. (300+ features)  (NFL Stats)

### Data Preparation
1. Scraping: Season statistics for the 2023 NFL season were scraped using Python scripts that interacted with nfl.com. Libraries like requests and beautifulsoup were used to fetch and parse HTML tables containing the data. (scraping.ipynb)

(FieldGoalStatsSpecialTeams.csv, PassingStatsDefense.csv, RushingStatsDefense.csv, ReceivingStatsDefense.csv, PassingStatsOffense.csv, InterceptionStatsDefense.csv, KickoffReturnsStatsSpecialTeams.csv, PuntReturnsStatsSpecialTeams.csv, RushingStatsOffense.csv, ReceivingStatsOffense.csv, PuntingStatsSpecialTeams.csv, KickoffsSpecialTeams.csv, ScoringStatsSpecialTeams.csv, DownsStatsOffense.csv, ScoringStatsOffense.csv, FumbleStatsDefense.csv, DownsStatsDefense.csv, TackleStatsDefense.csv, and ScoringStatsDefense.csv)

2. Combining Files: After cleaning the data, I organized the files so that offensive, defensive, and special teams statistics each had their own dedicated file, along with a comprehensive file containing all statistics. I grouped each file into a relevant list and passed the list into a function that combines rows based on the 'Team' column, ensuring that statistics from different files are properly aligned by team name. This process merges the data, retaining only one 'Team' column in the final dataset to eliminate duplicates. Additionally, a prefix (e.g., 'offense', 'defense', or 'special teams') was added to the column names from each file, except for the 'Team' column, to clearly indicate the source of each statistic. The result is a single, consolidated DataFrame where all team statistics are grouped by category and labeled for easy analysis. This produced individual files for offensive, defensive, and special teams statistics, as well as a comprehensive file combining all data.(combining_files.ipynb)
 
 (offense_merged_stats.csv, defense_merged_stats.csv, specialteams_merged_stats.csv, combined_all_stats.csv)

3. Merging with Games File: The stats files and combined stat files were integrated with a games file containing details such as scores and team names for every NFL game in the 2023 season. A function processed each row in the games file, adding the corresponding team stats for both home and away teams. Each feature was clearly labeled as 'home' or 'away' to ensure accurate alignment with the respective teams. (combining_files.ipynb)
  
4. Preprocessing: A new feature, home_win, was added to each column to indicate whether the home team won the game. This was determined based on the final scores in the games file. This target variable served as the basis for training and evaluating machine learning models. The inclusion of home_win was critical for focusing the analysis on predicting game outcomes. (combining_files.ipynb)

## Models 
All code and results are in my modeling.ipynb file.

### Individual File Testing
I began by testing each individual file using a Random Forest classifier, running the model 40 times and recording the average accuracy and standard deviation for each dataset. Offensive statistics consistently produced the best performance.

<img width="595" alt="Screenshot 2024-12-23 at 12 50 56 PM" src="https://github.com/user-attachments/assets/2aba33ee-9375-4b12-9413-4a5fb79c82ea" />

### Testing Multiple Models
Next, I evaluated merged datasets using multiple models: Random Forest, Support Vector Machine (SVM), Logistic Regression, Gradient Boosting, and Neural Network. Random Forest performed best across the merged files, achieving the highest accuracy overall. Surprisingly, the Neural Network model achieved 72% accuracy with the combined data.

I also analyzed the top three features for each model using the combined stats data. Logistic Regression, Random Forest, and Gradient Boosting all identified field goal stats as one of their top three features out of more than 300 
total features.

### Optimization Techniques
The highest accuracy achieved using all features was 71% with a Random Forest model. To improve performance, I applied several optimization techniques:
1. Grid Search for Hyperparameter Tuning
Conducted a grid search to optimize the model’s parameters, such as the number of estimators, maximum depth, and minimum samples per split.

Surprisingly, the optimized Random Forest model achieved 69% accuracy—lower than the baseline model. This drop likely occurred because the grid search hyperparameters overfit the training data, making the model less generalizable to unseen data. Field goal stats remained the top feature, reinforcing their importance.

2. Feature Selection
Reduced the dataset to the top 190 features based on feature importance, which increased accuracy from 71% to 75%. By focusing on the most relevant features, the model avoided noise and redundant variables. Field goals continued to be the most important feature.

3. Cross-Validation
Introduced cross-validation to test the model’s performance on outside data. Using the top 100 features, the model achieved an accuracy of 64% with a standard deviation of 0.06. This drop is expected because cross-validation provides a more realistic measure of performance by testing the model on unseen folds, revealing its true predictive power. The results highlighted how challenging it is for a machine learning model to predict NFL outcomes using season-level data alone.

## Results & Insights
The results demonstrate the difficulty of accurately predicting NFL outcomes with total season data, as the variability in game dynamics makes it hard for machine learning models to generalize. However, the consistent importance of field goal stats across models underscores their critical role in determining game outcomes. These insights suggest that certain in-game decisions—such as prioritizing field goal opportunities—can significantly impact a team's success.

## Conclusion
This project demonstrates how machine learning can uncover insights that reshape how we view the game. Field goals, though often overlooked compared to touchdowns, emerged as the defining feature in predicting NFL game winners. Their impact on strategy, clock management, and team decisions underscores why they play a pivotal role in game outcomes.
By focusing on field goals, this project highlights the broader purpose of sports analytics: to reveal hidden patterns and challenge assumptions, ultimately enabling better decision-making for teams and coaches.
