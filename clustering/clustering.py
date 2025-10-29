import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re

def cluster_total_news(news_data):
    # 패턴 컴파일
    pattern = re.compile(r'\[.*?\]')

    df_news_data = pd.DataFrame(news_data, encoding="utf-8")
    for idx, title in enumerate(df_news_data['제목']):
        no_bracket = pattern.sub('', title)
        df_news_data.loc[idx, "제목"] = no_bracket.strip()
    news_title = df_news_data.copy()[['제목']]
    tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1, 5))
    tfidf_vectorizer.fit(news_title['제목'])
    vector = tfidf_vectorizer.transform(news_title['제목']).toarray()
    vector = np.array(vector)
    model = DBSCAN(eps=0.1, min_samples=1, metric="cosine")
    result = model.fit_predict(vector)
    df_news_data['cluster_num'] = result
    return df_news_data

