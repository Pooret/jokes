**TODO:**

19th:

- prepare a dictionary of commonly mispelled words and corrected words
- use helper datasets (https://www.kaggle.com/code/kyakovlev/jigsaw-general-helper-public/script)
- finish data cleaning to prepare for data preparation

# Introduction

## Data Sources and Inital Explorations

data source - [one-million-reddit-jokes](https://query.data.world/s/htrdsouy327xqa4w457qx6k6sjtj6r)

The dataset was downloaded as a csv file, and contained 1M reddit postings from the r/Jokes subreddit, and stored in a Pandas DataFrame with dimensions (1000000, 12). The first ten rows with the text features and scores are shown here:

![Screen Shot 2022-10-19 at 12 39 01 PM](https://user-images.githubusercontent.com/64797107/196752309-70664abd-bf38-4428-9d15-4aaeb7493e9c.png)

"title" refers to the title of the post that the user creates. The "selftext" is the content of the post, and it is visible after a user clicks title. 

![image](https://user-images.githubusercontent.com/64797107/196763792-7cabc910-c445-402f-9ec6-7cc48490454d.png)


It's worth nothing that many jokes in this datatable don't  (*see NaNs*). The "score" value describes the number of upvotes, i.e. the number of positive ratings the post received. Posts can additionally be downvoted, and while reddit allows for negative values, the minimum value in the dataset is zero. When a user posts something to reddit, however, they are automatically given a single upvote, so I am making the assumption that values of zero in this dataset were downvoted. 

The scores range from 0 - 142,733, with an avg of 139.7 $\pm$ 1674.0

## [Data Wrangling](https://github.com/Pooret/jokes/blob/main/data%20wrangling.ipynb)

The first step in any data science project is take in the raw data and transform it so that it can then be readily fed into a pipeline for downstream analysis. The process by which the raw data is changed and manipulated into a more useable form is called data wrangling (*this is also know as data cleaning, data munging, or data remediation*). I prefer wrangling as it is an apt description of this process; on average, most of the hours spent working on a data science project will be alloted to this step, and it very much akin to wrestling unruly raw data into a more submissive, usable state that behaves well as it gets processed along the pipeline.

### Plan of attack
We are going to be working mainly with text data as python strings, so I will be making heavy use of regular expressions for all steps of the data wrangling process. As this project's objective is to train a model that can classify between the funny (*the positive class*) and the un-funny (*the negative class*) jokes that my father sends, an ideal dataset to train this model on would contain an equal number positive and negative samples of my father's jokes. And while I could ask my father to send me a complete list of his jokes to use as training data (*I have a partial for validation purposes*), a model trained on this wouldn't generalize well given the sample I have from him for validation purposes is already limited and highly imbalanced. To get enough representative examples of his jokes, I will be using data from the [r/Jokes subreddit](https://www.reddit.com/r/Jokes/) and filtering it for content that most resembles his own. 

For unicode characters, the emoji package contains the unicode strings compiled nicely for us by using emoji.get_emoji_regexp(). 

## Exploratory Data Analysis (Predictive Analytics)
*What story do the data tell in regards to the target? i.e. what questions can I test my predictions? What are some assumptions backed by the data? (stats and probability just about makes assumptions from the data).

__only include figures (2 best, 3 max) that show something important to story

Objectives: 
 \* n-gram frequencies
 \* P(n-gram) funny or not funny (check with sklearn naive bayes)
 \* vectorization (not sure if making my own or using pre-trained embeddings)
 \* pca visualization
 \* svd (check hands on book for implementation)
 \* cosine similarity
 \* 
 
##

possible use cases:
\* jokes filter for my dad 
\* joke generation (auto complete)
\* chatbot??
\* 
\* Document analysis for funny/not funny using KNN (this is the perfect way to analyze that "blip")
 
**##TODO**
 \* implement autocorrect for text normalization
 \* 

