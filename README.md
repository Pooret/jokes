# Act I: Setup:

## Introduction 
**see springboard checklist and handson ml checklist**

### Plot Point

# Act II: Confrontration 
**Predictive Analytics**
  I predict __ is (funny|not funny) based off of (statistic / prob) + visual
  *The data suggest that X is is (1|0) with **(likelylihood|with % prob|% certainty) as shown by _ fig:

## Rising Actions
**stakes get high**

### Plot point














## Brief Intro 
 \* state goal of project. 
  (**see springboard checklist and handson ml checklist**) 

*short blurb about the data. Sompre pretty pictures. Some conclusions.* 

# Exposition # (what is the goal of the intro: to get data ready for predictive analytics and drive the story)

## Data Sources and Inital Explorations

data source - [one-million-reddit-jokes](https://query.data.world/s/htrdsouy327xqa4w457qx6k6sjtj6r)

The dataset was downloaded as a csv file, which has 1M posts from the r/Jokes subreddit. Of the relevant features, the "title" is the title's post or the joke's setup. The "selftext" is the punchline, or what you see once a user clicks on the post's content. It's worth nothing that many jokes in this datatable don't meet this criterion (*see NaNs*). The "score" value describes the number of upvotes, i.e. the number of positive ratings the post received. Posts can additionally be downvoted, and while reddit allows for negative values, the minimum value in the dataset is zero. When a user posts something to reddit, however, they are automatically given a single upvote, so I am making the assumption that values of zero in this dataset were downvoted. 

The scores range from 0 - 142,733, with an avg of 139.7 $\pm$ 1674.0

## [Data Wrangling](https://github.com/Pooret/jokes/blob/main/data%20wrangling.ipynb)

The ultimate goal in wrangling these data is to create a dataset to classify a joke as either funny or not funny. The  and to this end, I filtered for posts that were relevant to further reduced the data to include posts that met criteria for my father (e.g. he doesn't use dirty jokes). Additionally, user edits, To keep things simple to start, I removed jokes that contain numbers and emojis, but I can include these later as not much data were lost as a result. From those filtered data I took posts that were downvoted and took an equal amount of posts that are above a min_vote threshold. 

## Exploratory Data Analysis (Predictive Analytics)
*What story do the data tell in regards to the target? i.e. what questions can I test my predictions? What are some assumptions backed by the data? (stats and probability just about makes assumptions from the data).

__only include figures (2 best, 3 max) that show something important to story (what 

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

