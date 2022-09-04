# Jokes

## Introduction 

## Data Sources and Exploration

data source - [one-million-reddit-jokes](https://query.data.world/s/htrdsouy327xqa4w457qx6k6sjtj6r)

The dataset was downloaded as a csv file, which has 1M posts from the r/Jokes subreddit. Of the relevant features, the "title" is the title's post or the joke's setup. The "selftext" is the punchline, or what you see once a user clicks on the post's content. It's worth nothing that many jokes in this datatable don't meet this criterion (*see NaNs*). The "score" value describes the number of upvotes, i.e. the number of positive ratings the post received. Posts can additionally be downvoted, and while reddit allows for negative values, the minimum value in the dataset is zero. When a user posts something to reddit, however, they are automatically given a single upvote, so I am making the assumption that values of zero in this dataset were downvoted. 

The scores range from 0 - 142,733, with an avg of 139.7 $\pm$ 1674.0

## [Data Wrangling](https://github.com/Pooret/jokes/blob/main/data%20wrangling.ipynb)

The ultimate goal in wrangling these data is to create a dataset for to classify as either funny or not funny, and to this end, I filtered for posts that were relevant to further reduced the data to include posts that met criteria for my father (e.g. he doesn't use dirty jokes). Additionally, user edits, To keep things simple to start, I removed jokes that contain numbers and emojis, but I can include these later as not much data were lost as a result. From those filtered data I took posts that were downvoted and took an equal amount of posts that are above a min_vote threshold. 

## Exploratory Data Analysis

I read in the filtered training data and preprocessed the text 
