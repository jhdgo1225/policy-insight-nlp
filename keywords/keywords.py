from keybert import KeyBERT
from kiwipiepy import Kiwi
from transformers import BertModel
import time

"""
- 함수 설명: 클러스터링된 뉴스 본문의 키워드를 추출한 후 이들을 집합시킵니다.
- 매개변수 설명
    bodies: str | List<str>, 문장, 또는 문장들의 리스트
"""
def extract_keywords(bodies):
    # 1. 한국어 문장 임베딩을 위한 BERT 모델 로드
    model = BertModel.from_pretrained('skt/kobert-base-v1')

    # 2. 임베딩 모델을 KeyBERT에 활용
    kw_model = KeyBERT(model)

    # 3. 형태소 분석 Kiwi 인스턴스 생성
    kiwi = Kiwi()

    # 4. 명사 추출
    nouns_list = []

    analyzed_sentences = kiwi.analyze(bodies)
    for sentence in analyzed_sentences:
        # bodies는 문자열(문장) 타입이거나 문자열의 집합을 나타낸다.
        if isinstance(bodies, str):
            tokens = sentence[0]
        elif isinstance(bodies, list):
            tokens = sentence[0][0]
        # 명사 태그만 추출
        nouns = [token.form for token in tokens if token.tag == 'NNG']
        nouns_list.append(' '.join(nouns))

    # 5. 키워드 추출(N-gram: 1, 불용어 허용하지 않음, 상위 20개만 선별)
    keywords = kw_model.extract_keywords(nouns_list, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=20)

    result = set()
    for keyword in keywords:
        if isinstance(keyword, tuple):
            result.add(keyword[0])
        elif isinstance(keyword, list):
            result.update([text[0] for text in keyword])
    return result


if __name__ == "__main__":
    text = """서울 지진 대피소에 대한 데이터 분석을 위해서는 어떤 종류의 데이터가 필요할까요? 예를 들어, 서울시의 지진 대피소 위치, 수용 가능 인원, 대피소 내부 시설물, 대피소 이용 현황 등의 정보가 필요할 것입니다. 지진 대피소 위치 분석 예시: 지진 대피소 위치는 서울시 공공데이터 포털에서 제공하는 "서울시 지진대피소 안내" 데이터를 사용할 수 있습니다. 이 데이터셋에는 지진 대피소 명칭, 위치(주소), 좌표, 수용 인원, 관리 기관 등의 항목이 포함되어 있습니다. 이를 바탕으로 대피소 위치를 지도에 시각화하여 지진 발생 시 대피소가 필요한 지역을 파악할 수 있습니다. 대피소 이용 현황 분석 예시: 대피소 이용 현황은 서울시에서 제공하는 "서울시 재난정보 실시간 수집 및 제공 서비스" 데이터를 사용할 수 있습니다. 이 데이터셋에는 대피소 이용 현황(대피소 이용 가능 여부, 이용 중인 인원 수), 지진 발생 시 대피소 이용 현황 등의 정보가 포함되어 있습니다. 이를 바탕으로 대피소 이용 현황을 분석하여 인원이 많은 대피소를 파악하거나, 대피소 이용 가능 여부 등을 파악할 수 있습니다."""
    only_print = list(extract_keywords(text))
    print("--------- 키워드 추출 -------------")
    for i in only_print:
        print(i)
