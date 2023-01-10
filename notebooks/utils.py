# TODO normalize color palettes
# TODO add descriptions to functions
# TODO naive bayes

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import regex as re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

PROJECT_ROOT_DIR = "."
PROJECT_ID = 'jokes'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", PROJECT_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

dads_jokes = [['Police car loses wheels to thief!', 'Cops are working tirelessly to nab suspect'],
             ['My friend David had his id stolen', 'now he is just Dav.'],
             ['I have a few jokes about unemployed people', 'but none of them work'],
             ['I had a patient that talked and talked and talked','It was clear that there needed to be an organization Similar to Q-Anon for compulive talkers called Onandon-andon'],
             ['My ex-wife still misses me.', 'But her aim is starting to improve.']]


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=72):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def make_timestamp(unix_ts, tz='US/Eastern'):
    from datetime import datetime
    import pytz
    # timezone = pytz.timezone(tz)
    # convert unix utc to timestamp utc
    ts = datetime.utcfromtimestamp(unix_ts)
    # return is to that it is tz-aware
    # return timezone.localize(ts)
    return ts


def populate_pos_class(df, lower_bound=1, upper_bound=1500, class_size=10000):
    desired_class_size = class_size
    max_class_size = len(df[df['score'] == 0])
    if class_size > max_class_size:
        print(f'maximum sampling size is {max_class_size}\n proceeding with this value...')
        class_size = max_class_size
    best_score_for_binary_class = find_best_score(df, lower_bound, upper_bound, class_size)

    # create mask of best score for binary classification and use it to update the class size
    pos_class_mask = df['score'] > best_score_for_binary_class
    updated_class_size = len(df[pos_class_mask])

    if updated_class_size > max_class_size:
        updated_class_size = max_class_size

    # if number of elements is less than desired class size
    if updated_class_size < class_size:
        print('WARNING, CURRENT PARAMETERS RESULT IN CLASS SIZE REDUCTION')
        print(f'desired size {desired_class_size}')
        print(f'current size {updated_class_size}')

    return pos_class_mask, updated_class_size


def find_best_score(df, lower_bound, upper_bound, class_size):
    print(5 * '*', 'calculating', 5 * '*')

    # iterate over score range and get distance
    for score in range(lower_bound, upper_bound):

        num_samples = len(df[df['score'] > score])
        dist = num_samples - class_size

        if dist <= 0:
            return score

    return score

def remove_accents(raw_text):
    # https: // www.programcreek.com / python /?CodeExample = remove + accents
    raw_text = re.sub(u"[àáâãäå]", 'a', raw_text)
    raw_text = re.sub(u"[èéêë]", 'e', raw_text)
    raw_text = re.sub(u"[ìíîï]", 'i', raw_text)
    raw_text = re.sub(u"[òóôõö]", 'o', raw_text)
    raw_text = re.sub(u"[ùúûü]", 'u', raw_text)
    raw_text = re.sub(u"[ýÿ]", 'y', raw_text)
    raw_text = re.sub(u"[ñ]", 'n', raw_text)
    return raw_text


# https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
def preprocess_text(text, flg_stemm=False, flg_lemm=True, stop_words=None, diagnose=False):
    if diagnose:
        original_text = text

    if stop_words == "default":
        stop_words = stopwords.words('english')

    # remove html escapes
    text = re.sub(r"\S+;", ' ', str(text))
    text = re.sub(r"[^a-z0-9\s]", '', text.lower().strip())
    text = remove_accents(text)
    text_tokens = text.split()

    # stopwords
    if stop_words is not None:
        text_tokens = [word for word in text_tokens if word not in stop_words]

    # stemmings
    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        text_tokens = [ps.stem(word) for word in text_tokens]

    # lemmatization
    if flg_lemm:
        lem = nltk.stem.WordNetLemmatizer()
        text_tokens = [lem.lemmatize(word) for word in text_tokens]

    if diagnose:
        return " ".join(text_tokens), text_tokens, original_text

    return " ".join(text_tokens)

def inv_freq(text, stop_words=None, min_df=0.0, max_df=1.0, nmin=1, nmax=1):
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv_if = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, ngram_range=(nmin, nmax))
    if_doc = cv_if.fit_transform(text)
    sum_words_if = if_doc.sum(axis=0)
    words_freq_if = [(word, sum_words_if[0, i]) for word, i in cv_if.vocabulary_.items()]
    words_freq_if = sorted(words_freq_if, key=lambda x: x[1], reverse=True)
    inv_freq_df = pd.DataFrame(words_freq_if, columns=['word', 'freq'])

    return if_doc, cv_if, inv_freq_df



