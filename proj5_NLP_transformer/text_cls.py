# step1: import modules
from transformers import pipeline

# step2: create inference object
# model 부분에 user이름/모델명 이 부분만 수정해주면 됨
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# step3: prepare data
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# step4: inference
result = classifier(text)

# step5: post processing
print(result)