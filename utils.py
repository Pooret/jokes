#TODO normalize color pallettes
#TODO stats test of differnce between good/bag bigrams trigrams
#TODO fix test size
#TODO normalize text
#TODO bar plots
#TODO time leap year

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
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

flags = ['wrong sub', 'sex', 'gay', 'black man', 'black people']

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=72):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def evaluate_model(true, pred, predicted_prob):
    classes = np.unique(true)
    y_val_array = pd.get_dummies(true, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(true, pred)
    auc = metrics.roc_auc_score(true, pred)
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(true, pred))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(true, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_val_array[:, i],
                                                 predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_val_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


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


def inv_freq(text, stop_words=None, min_df=0.0, max_df=1.0, nmin=1, nmax=1):
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv_if = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, ngram_range=(nmin, nmax))
    if_doc = cv_if.fit_transform(text)
    sum_words_if = if_doc.sum(axis=0)
    words_freq_if = [(word, sum_words_if[0, i]) for word, i in cv_if.vocabulary_.items()]
    words_freq_if = sorted(words_freq_if, key=lambda x: x[1], reverse=True)
    inv_freq_df = pd.DataFrame(words_freq_if, columns=['word', 'freq'])

    return if_doc, cv_if, inv_freq_df

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


# chi-squared test
from sklearn.feature_selection import chi2 as chi_squared


def chi_squared_text(y, X_train, trained_vectorizer, p_lim=0.95):
    X_names = trained_vectorizer.get_feature_names()

    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = chi_squared(X_train, y == cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat, "chi2": chi2}))
        dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_lim]

    X_names = dtf_features['feature'].unique().tolist()
    for cat in np.unique(y):
        print("# {}:".format(cat))
        print("  . selected features:",
              len(dtf_features[dtf_features["y"] == cat]))
        print("  . top features:", ",".join(
            dtf_features[dtf_features["y"] == cat]["feature"].values[:10]))
        print(" ")

    return X_names, dtf_features
