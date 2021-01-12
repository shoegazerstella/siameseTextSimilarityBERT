from operator import itemgetter
from keras.models import load_model
from embedding import *

best_model_path = 'data/checkpoints_bert_lstm/1607508730/model.h5'
model = load_model(best_model_path)

sentence_bert_model = load_embedding_model()

test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]

sentences1 = [i[0] for i in test_sentence_pairs]
sentences2 = [i[1] for i in test_sentence_pairs]

emb1 = sentence_bert_model.encode(sentences1)
emb2 = sentence_bert_model.encode(sentences2)
# reshape
emb1 = emb1.reshape(-1, 768, 1)
emb2 = emb2.reshape(-1, 768, 1)

preds = list(model.predict([emb1, emb2], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
results.sort(key=itemgetter(2), reverse=True)

for i in results:
    print(i[0])
    print(i[1])
    print(i[2])
    print('')