import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re

# 패턴 컴파일
pattern = re.compile(r'\[.*?\]')

df = pd.read_csv("./한국경제.csv", encoding="utf-8")
for idx, title in enumerate(df['제목']):
	no_bracket = pattern.sub('', title)
	df.loc[idx, "제목"] = no_bracket.strip()
df_title = df.copy()[['제목']]
tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1, 5))
tfidf_vectorizer.fit(df_title['제목'])
vector = tfidf_vectorizer.transform(df_title['제목']).toarray()
vector = np.array(vector)
model = DBSCAN(eps=0.1, min_samples=1, metric="cosine")
result = model.fit_predict(vector)
df['cluster_num'] = result

# #3 대표 기사 추출

def print_cluster_result(train):
    clusters = []
    counts = []
    top_title = []
    top_noun = []
    for cluster_num in set(result):
        # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
        if(cluster_num == -1 or cluster_num == 0): 
            continue
        else:
            print("cluster num : {}".format(cluster_num))
            temp_df = train[train['cluster_num'] == cluster_num] # cluster num 별로 조회
            clusters.append(cluster_num)
            counts.append(len(temp_df))
            top_title.append(temp_df.reset_index()['제목'][0])
            top_noun.append(temp_df.reset_index()['본문'][0]) # 군집별 첫번째 기사를 대표기사로 ; tfidf방식
            for title in temp_df['제목']:
                print(title) # 제목으로 살펴보자
            print()

    cluster_result = pd.DataFrame({'cluster_num':clusters, 'count':counts, 'top_title':top_title, 'top_noun':top_noun})
    return cluster_result

print(print_cluster_result(df))
