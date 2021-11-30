import pandas as pd
import os.path
import html
import spacy
import time

from string import punctuation
from langdetect import detect
from google.cloud import translate_v2 as translate


# data
TICKETS = pd.read_csv(f'./data/tickets/tickets.csv')
STATUS_LOG = pd.read_csv(f'./data/tickets/status_log.csv')

# google translate setup
CREDENTIALS_PATH = './translation-credentials.json'
CLIENT = (
    translate.Client.from_service_account_json(CREDENTIALS_PATH)
    if os.path.exists(CREDENTIALS_PATH)
    else None
)

# spacy setup
NLP = spacy.load('de_core_news_md')


def preprocess(force=False, test=False, translate=False):
    """Preprocess ticket data into a pandas dataframe ready for futher work."""
    s = './data/messages.parquet'
    if force or test or not os.path.exists(s):
        messages = TICKETS[[
            'ID',
            'Bearbeiter',
            'Angelegt Am',
            'Kategorie ID',
        ]].dropna()
        messages = messages.rename(columns={
            'ID': 'id',
            'Bearbeiter': 'operator',
            'Angelegt Am': 'timestamp',
            'Kategorie ID': 'category',
        })
        if test:
            messages = messages[:100]
        ops = messages.groupby('operator')['operator'].agg(['count'])
        ops = ops.reset_index().sort_values('count', ascending=False)
        ops = list(ops['operator'][:-10])
        messages = messages[messages['operator'].apply(lambda x: x in ops)]
        messages['category'] = messages['category'].apply(lambda x: x.strip())
        messages = messages[messages['category'].str.len() > 0]
        messages['timestamp'] = pd.to_datetime(
            messages['timestamp'],
            infer_datetime_format=True,
            utc=True,
        )
        # messages['year'] = messages['timestamp'].apply(lambda x: x.year)
        messages['text'] = messages['id'].apply(
            lambda x: get_first_message(fetch_ticket(x)),
        )
        messages = messages.dropna()
        messages['text'] = messages['text'].apply(lambda x: clean(x))
        messages['language'] = messages['text'].apply(
            lambda x: detect_language(x),
        )
        messages['translated-text'] = messages.apply(
            lambda row: (
                row['text']
                if not translate or row['language'] == 'de'
                else translate_to_german(row['text'])
            ),
            axis=1,
        )
        '''
        messages['text'] = messages['text'].apply(
            lambda x: ' '.join(extract_keywords(x)),
        )
        '''
        messages = messages.reset_index(drop=True)
        messages.to_parquet(s)
        return messages
    return pd.read_parquet(s)


def fetch_ticket(identifier):
    """Return data of ticket with given identifier as pandas dataframe."""
    try:
        return pd.read_csv(f'./data/tickets/{identifier}.csv')
    except:
        return None


def get_first_message(ticket_data):
    """Get first real message in ticket conversations.

    Sometimes there are weird first messages contained in the ticket CSVs,
    that start with SY-SYSID.

    """
    if ticket_data is None:
        return None
    starts = ['SY-DBSYS', 'SY-SYSID']
    for index, row in ticket_data.iterrows():
        if not any([row['Text'].startswith(start) for start in starts]):
            # filter out test messages like `2000000151`: "Es brennt"
            if (
                len(row['Text']) < 100
                or row['Nachrichtentyp'].strip() != 'Beschreibung'
            ):
                return None
            return row['Text']


def clean(text):
    """Clean text of any weird characters."""
    words = text.replace('\n', ' ').split()
    out = []
    for word in words:
        word = word.strip('/-_<>&')
        if word:
            out.append(word)
    return ' '.join(out)


def detect_language(text):
    """Detect the language of the given text."""
    return detect(text)


def translate_to_german(text):
    """Translate text from any language to german via Google Translate API."""
    assert CLIENT, 'no Google Translate credentials provided'
    time.sleep(0.2)  # rate limiting
    s = CLIENT.translate(text, target_language='de')['translatedText']
    return html.unescape(s)


def extract_keywords(text):
    """Extract and clean most important words in the text."""
    tag = ['PROPN', 'ADJ', 'NOUN', 'VERB', 'ADV', 'NUM']
    doc = NLP(text.lower())
    result = []
    for token in doc:
        if(token.text in NLP.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in tag):
            result.append(token.lemma_)
    return result
