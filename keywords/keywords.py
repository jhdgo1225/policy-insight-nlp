from keybert import KeyBERT
from kiwipiepy import Kiwi
from transformers import BertModel

model = BertModel.from_pretrained('skt/kobert-base-v1')
kw_model = KeyBERT(model)

text = """서울 지진 대피소에 대한 데이터 분석을 위해서는 어떤 종류의 데이터가 필요할까요? 예를 들어, 서울시의 지진 대피소 위치, 수용 가능 인원, 대피소 내부 시설물, 대피소 이용 현황 등의 정보가 필요할 것입니다. 지진 대피소 위치 분석 예시: 지진 대피소 위치는 서울시 공공데이터 포털에서 제공하는 "서울시 지진대피소 안내" 데이터를 사용할 수 있습니다. 이 데이터셋에는 지진 대피소 명칭, 위치(주소), 좌표, 수용 인원, 관리 기관 등의 항목이 포함되어 있습니다. 이를 바탕으로 대피소 위치를 지도에 시각화하여 지진 발생 시 대피소가 필요한 지역을 파악할 수 있습니다. 대피소 이용 현황 분석 예시: 대피소 이용 현황은 서울시에서 제공하는 "서울시 재난정보 실시간 수집 및 제공 서비스" 데이터를 사용할 수 있습니다. 이 데이터셋에는 대피소 이용 현황(대피소 이용 가능 여부, 이용 중인 인원 수), 지진 발생 시 대피소 이용 현황 등의 정보가 포함되어 있습니다. 이를 바탕으로 대피소 이용 현황을 분석하여 인원이 많은 대피소를 파악하거나, 대피소 이용 가능 여부 등을 파악할 수 있습니다."""

kiwi = Kiwi()
nouns_list = []
for sentence in kiwi.analyze(text):
    nouns = [token.form for token in sentence[0] if token.tag.startswith('NN')]
    if nouns:
        nouns_list.extend(nouns)
result_text = ' '.join(nouns_list)

keywords = kw_model.extract_keywords(result_text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=20)
print(keywords)