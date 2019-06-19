import gensim

model = gensim.models.Word2Vec.load("vocabulary.model")

result = model.most_similar("news")

for e in result:
    print(e[0], e[1])

