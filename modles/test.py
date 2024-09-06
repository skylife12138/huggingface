from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("fuck")
print(res)