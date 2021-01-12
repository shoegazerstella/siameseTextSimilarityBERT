import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, TFBertModel #BertTokenizer

##### SENTENCE BERT

def load_embedding_model():
    sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens') #roberta-large-nli-stsb-mean-tokens
    return sentence_bert_model

def embed_batch(df, sentence_bert_model):
    
    sentences1 = df['sentences1'].tolist()
    sentences2 = df['sentences2'].tolist()
    
    labels = np.array(df['is_similar'].tolist())
    
    emb1 = sentence_bert_model.encode(sentences1)
    emb2 = sentence_bert_model.encode(sentences2)

    # reshape for lstm layer
    emb1 = emb1.reshape(-1, 768, 1)
    emb2 = emb2.reshape(-1, 768, 1)
    
    return emb1, emb2, labels

