pos_neg_categories = {
    "긍정": {
        "기쁨/행복": ["기쁨", "신이 난"],
        "감사": ["감사하는"],
        "신뢰": ["신뢰하는"],
        "자신감": ["자신하는"],
        "편안/느긋": ["편안한", "느긋"],
        "만족": ["만족스러운"],
        "흥분/열정": ["흥분"],
        "안도": ["안도"]
    },
    "부정": {
        # 1) "분노/짜증"에 "구역질 나는", "혐오스러운" 추가
        "분노/짜증": [
            "분노", "툴툴대는", "짜증내는", "방어적인", "악의적인", 
            "안달하는", "노여워하는", "성가신", "질투하는", "억울한",
            "구역질 나는",      # (7)
            "혐오스러운"        # (47)
        ],
        "슬픔": [
            "슬픔", "비통한", "우울한", "마비된", "눈물이 나는", "가난한 불우한"
        ],
        "불안/두려움": [
            "불안", "두려운", "스트레스 받는", "취약한", 
            "걱정스러운", "조심스러운", "초조한"
        ],
        "당혹/충격": [
            "혼란스러운", "당혹스러운", "충격 받은", 
            "당황", "고립된(당황한)", "혼란스러운(당황한)"
        ],
        "수치심/죄책감": [
            "후회되는", "남의 시선을 의식하는", "열등감", 
            "죄책감의", "부끄러운", "한심한"
        ],
        "절망/좌절/환멸": [
            "좌절한", "실망한", "염세적인", "낙담한", 
            "환멸을 느끼는", "회의적인"
        ],
        "상처/배신": [
            "상처", "배신당한", "희생된", "괴로워하는"
        ],
        "외로움/고립": [
            "고립된", "버려진", "외로운"
        ]
    }
}

label_to_category = {}

for polarity, category_dict in pos_neg_categories.items():
    for category_name, labels in category_dict.items():
        for label in labels:
            label_to_category[label] = category_name

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")
# 2. 요약할 텍스트 준비
text = """


네이버 굿즈를 받았다는 소식으로

작년 마지막 날을 마무리했었는데…

​


어느새 또 한 해가

휙- 하고 지나가버렸어요.

​

2024년 한 해를 돌아봅니다.

​

​

​

​


1월

​

1월에는 대출금을 다 갚았어요.

대출이 0원이 되면서 신용도가 올라갔습니다.

​


대출이 없으면 성공한 거라던데…

성공한 남자가 되었어요.

​

​

​

​


2월

​

T끼리 가족 합산이 75년이래요.

할인되는 건 좋은데

한 통신사를 너무 오래 쓴건 아닌가 하는 생각이 들어요.

관성적으로 선택하는 것들이 점점 늘어납니다.

​

​

​

​


3월

​

3월엔 전국 드루이드 모임이 있었어요.

세 시간 동안 모였다가 흩어지는 행사인데

500명에 가까운 인파가 모여서 북적북적.

​


미숙한 주최자는 혼이 나가 버렸습니다.

​

​

​

​


4월

​

가까운 곳에 맥도날드가 생겼어요.

​


이제 차로 40분만 나가면

나도 맥도날드 먹을 수 있어!

​

​

​

​


5월

​

아팠어요.

4년이 넘는 폐교 생활 동안 큰 탈이 없었는데

느닷없이 몸이 말을 안 들었어요.

​

병원에 가고, 약 먹고, 쉬기만 하다 보니

인생의 노잼시기가 찾아왔어요.

​

​

​

​


6월

​

카페에서 커피 한잔하고 나와보니

누군가 멀쩡히 주차되어 있는 차를 박았습니다.

범퍼에 난 흠집이 신경쓰이긴 하지만

누구나 실수 할 수 있는 걸요.

크고 작은 기스가 이미 있습니다.

괜찮다며 보내드렸어요.

​

​

​

​


7월

​


드디어 먹태깡을 먹었어요.

이게 존재하는 과자였군요.

​

열심히 마트를 기웃거려서 샀는데

제 입맛에는 별로입니다.

​

​

​

​


8월

​

지박령의 책이 출간되었습니다.

수학특성화중학교를 함께 썼던 작가님과

협업해서 완성된

청소년 공부법 소설이랍니다.

​


고스트 티처의 밀착 과외.

만관부.

​

​

​

​


9월.

​

장인어른께서 별세하셨습니다.

살펴야 할 것과 챙겨야 할 것들이 많은 시간이었습니다.

​

​

​

​


10월

​


폐교생활백서 이야기가

책으로 엮여 나왔습니다.

​

그리고 폐교로 사람들을 초대했어요.

4일간 행사에 450여 명이 초대되었습니다.

​

​

​

​


​

11월

​

첫눈이 내렸습니다.

​


폐교 입장 팔찌(종이)를 첫눈이 올 때까지

빼지 않은 분이 계셨어요.

소원이 이루어지셨을까요?

​


그리고… 프로개의 숲 1.0 부지를 구했습니다.

​

​

​

​


12월

​

프로개의 숲 부지 근처의 

오래된 2층 집을 구했습니다.

다시 대출이 생겼어요…

​

도배장판만 하고 이사하려 했는데…

누수를 발견했습니다.

​

난방 배관을 새로 하려다… 천장을 부수고

내부 계단을 만들고 있는 나를 발견합니다.

​


새 건물을 지을 걸 그랬나…

​

​

​

​


연말연시 기분을 내야할 때인데

나라가 슬픔에 잠겨 있어요.

​

정치는 혼란하고

갑작스러운 항공기 사고 소식으로

마음에 슬픔이 깃들었습니다.

​

차가운 눈이 계속 내려

쌓이고만 있는 것 같아요.

​

어떤 회사는 곧 문을 닫을 것 같고

어떤 회사는 인원 감축 얘기가 돌아요.

​

물가는 계속 올라 생활비가 빠듯하고

대출의 이자는 계속 오르며

환율도 요동칩니다.

​

삶에 폭풍우가 계속 몰아치는데

가족 앞에서는

강한 모습을 보이고 싶습니다.

​

부모님께, 배우자에게, 자식들에게

털어놓지 못한 말들만 쌓여갑니다.

​

열심히 살아왔는데도

 아무것도 남은 게 없는 것 같고

​

 누군가의 사소한 일상조차도

나에게는 사치처럼 느껴질 때도 있습니다.

​

누구에게도 힘듬을 말하지 못 하고

혼자 눈물을 삼키는 자신이

한심하다고 여기지 않았으면 좋겠습니다.

​

오늘은 이 포스트를 읽은 누군가에게

친구가 되어볼까 봐요.

그러한 친구는 무심히 물어봅니다.

​

​

아침은 먹었어?

​

​

#프로개 #드루이드 #안녕 #2024

"""

print(len(text))


def chunk_text(text, chunk_size=600, overlap=100):
    # chunk_size: 한 번에 자를 최대 문자 수
    # overlap: 덩어리 간 겹치는 문자 수
    # 예: 첫 덩어리는 text[0:780], 다음 덩어리는 text[680:1460], ...

    step = chunk_size - overlap  # 다음 덩어리를 시작할 인덱스 간격
    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        # 실제 텍스트 범위를 넘어가는 경우 처리
        chunk = text[start:end]
        chunks.append(chunk)

        # 다음 덩어리의 시작 인덱스 계산
        start += step

        # 만약 마지막 덩어리가 chunk_size보다 짧았다면 반복 종료
        if start >= len(text):
            break

    return chunks

result_chunks = chunk_text(text, chunk_size=780, overlap=100)

sentimental_results = {}

for chunk in result_chunks:
    result = pipe(chunk)  # [{'label': '...', 'score': ...}, ...]
    for item in result:
        label = item["label"]  # 감정 라벨
        if label in label_to_category:
            category = label_to_category[label]  # 카테고리 찾기
            if category not in sentimental_results:
                sentimental_results[category] = []  # 카테고리 초기화
            sentimental_results[category].append(item)

from collections import Counter

# 카테고리별 등장 개수 세기
category_counts = Counter()
for category, items in sentimental_results.items():
    category_counts[category] += len(items)

# 가장 많은 개수를 가진 카테고리 찾기
most_common_categories = category_counts.most_common()  # [(카테고리, 개수), ...]
max_count = most_common_categories[0][1]  # 최대 등장 개수

# 같은 개수를 가진 카테고리 필터링
candidates = [cat for cat, count in most_common_categories if count == max_count]

# 만약 개수가 동일한 카테고리가 여러 개라면 score로 결정
if len(candidates) == 1:
    # 개수가 가장 큰 카테고리가 하나라면 바로 선택
    final_category = candidates[0]
else:
    # 점수가 가장 높은 카테고리를 선택
    highest_score = 0
    highest_score_category = None
    highest_score_label = None

    for category in candidates:  # 동일한 개수의 카테고리들만 순회
        for item in sentimental_results[category]:
            if item["score"] > highest_score:
                highest_score = item["score"]
                highest_score_label = item["label"]
                highest_score_category = category

    final_category = highest_score_category

# 최종 결과 출력
print(f"최종 카테고리: {final_category}")
if 'highest_score_label' in locals():  # 점수 기준으로 선정된 경우
    print(f"라벨: {highest_score_label}, score: {highest_score}")

# 전체 라벨 추출
all_labels = []
for items in sentimental_results.values():
    all_labels.extend(item["label"] for item in items)

print(f"전체 라벨: {all_labels}")

# 결과
# 최종 카테고리: 분노/짜증
# 전체 라벨: ['안달하는', '혐오스러운', '조심스러운']

# 분노/짜증: [{'label': '안달하는', 'score': 0.07493038475513458}, {'label': '혐오스러운', 'score': 0.09415222704410553}]
# 불안/두려움: [{'label': '조심스러운', 'score': 0.08116652071475983}]