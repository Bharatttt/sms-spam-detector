"""
Robust loader for the messy, inconsistently formatted SMS ham/spam CSVs:
some have no header, some have the label glued onto the message text,
and some have stray commas that split a message into extra columns.
"""
import re
from pathlib import Path
import pandas as pd

DATASET_FILES = [
    'AI_IN_CY_-_Sheet1.csv',
    'INDIAN_SPAM1.csv',
    'TXT16.csv',
    'TXT17.csv',
    'TXT18.csv',
    'TXT19__1_.csv',
    'USER_6_edited.csv',
    'USER_7_edited.csv',
    'USER_8_edited.csv',
    'USER_9_edited.csv',
    'align-11.csv',
    'align-13.csv',
    'us1.csv',
    'us2.csv',
    'us4.csv',
    'us5.csv',
]


def _robust_read(path):
    try:
        return pd.read_csv(path, header=None, encoding='utf-8',
                            engine='python', on_bad_lines='skip', dtype=str)
    except Exception:
        return pd.read_csv(path, header=None, encoding='latin-1',
                            engine='python', on_bad_lines='skip', dtype=str)


def load_raw_dataset(path):
    raw = _robust_read(path)
    ncols = raw.shape[1]

    if ncols >= 2:
        col0 = raw[0].astype(str)
        rest = raw.iloc[:, 1:].fillna('').astype(str).agg(' '.join, axis=1).str.strip()
        labels, messages = col0, rest
    else:
        s = raw[0].astype(str)
        labels = s.str.extract(r'^\s*(HAM|SPAM|ham|spam)\b', expand=False)
        messages = s.str.replace(r'^\s*(HAM|SPAM|ham|spam)\b', '', regex=True).str.strip()

    df = pd.DataFrame({'label_raw': labels, 'message_raw': messages})
    df['label_raw'] = df['label_raw'].astype(str).str.upper().str.strip()
    df = df[df['label_raw'].isin(['HAM', 'SPAM'])]
    df = df[df['message_raw'].astype(str).str.strip() != '']
    df = df[df['message_raw'].astype(str).str.lower() != 'nan']
    df['label'] = (df['label_raw'] == 'SPAM').astype(int)
    return df[['label', 'message_raw']].reset_index(drop=True)


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(path):
    df = load_raw_dataset(path)
    df['message_clean'] = df['message_raw'].apply(preprocess_text)
    df = df[df['message_clean'].str.split().str.len() > 0].reset_index(drop=True)
    return df


def load_combined_corpus(base_dir):
    frames = []
    for fname in DATASET_FILES:
        df = load_and_clean(Path(base_dir) / fname)
        df['source'] = fname
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['message_clean']).reset_index(drop=True)
    return combined
