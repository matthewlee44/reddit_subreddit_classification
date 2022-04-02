# Subreddit Post Classification
---

## Problem Statement

Subreddit posts related to dating and relationships seem to cover many common issues and situations. In many of these posts, users lament the frustrations and struggles of finding a romantic partner among a sea of people with vastly different familial, financial and other personal interests.

This project aims to build a classifier using natural language processing that can distinguish between posts from the subreddit r/dating and r/datingoverforty as accurately as possible despite their similarities.

## Executive Summary

Although many of the discussions within both of these subreddits share common themes and issues, I have determined that natural language processing tools can uncover enough differences to create an accurate classifer of their posts.

### Findings

While exploring the data, I ran a logistic regression model which calculated coefficients on each word used in the posts of these subreddits. These coefficients provide a gauge for the level of impact the presence of a word in a post has in the likelihood that post is predicted to be in one subreddit or the other. The words/topics with the most predictive impact in the logistic regression model for each subreddit is summarized below.

| **r/dating**                | **r/datingoverforty**        |
|-----------------------------|------------------------------|
| Numbers in the 20s          | Numbers in the 40s and above |
| Girl, girlfriend, boyfriend | Divorce                      |
| School/College              | Children                     |
|                             | Marriage                     |

### Results

I ran the text data through 5 different models with various pre-processing strategies and model hyper-parameters. The best model was a logistic regression model implementing a function that preprocessed the text using the insights from the analysis above. This model achieved an accuracy score of 82.02%. A full summary of the metrics achieved by model are provided below.

| **Metric**      | **Score** |
|-----------------|-----------|
| Accuracy Score  | 82.02%    |
| Precision Score | 82%       |
| Recall Score    | 82%       |
| F1 Score        | 82%       |

To further improve upon the accuracy of this model, I would like to explore taking the following next steps: (i) adding more features about the text such as (a) sentiment analysis on each post and (b) counts of the number of words and sentences used in each post, and (ii) exploring other natural language processing strategies to differentiate the subreddits further.
