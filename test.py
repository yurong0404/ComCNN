import nltk
from pycocoevalcap.cider.cider import Cider

def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))
    
def TF_IDF(ngram_list, ngram, total_ngram_count):
    count = ngram_list.count(ngram)
    tf = count / total_ngram_count
    # in the case of our dataset, tf-idf is either (tf*1) or (0* every large number)
    # so idf=1 results in the same consequence
    idf = 1
    return tf * idf

def CIDEr(true, pred):
    true = nltk.word_tokenize(true)
    pred = nltk.word_tokenize(pred)
    N = 4
    CIDEr_score = 0
    for n in range(1,5):
        true_ngram = ngram(true, n)
        pred_ngram = ngram(pred, n)
        if len(true_ngram)==0 or len(pred_ngram)==0:
            break
        total_ngram = true_ngram + pred_ngram
        total_ngram_count_in_cand = 1e-10
        total_ngram_count_in_ref = 1e-10
        #print(set(total_ngram))
        for t in set(total_ngram):
            total_ngram_count_in_cand += pred_ngram.count(t)
            total_ngram_count_in_ref += true_ngram.count(t)
        g_cand = [TF_IDF(pred_ngram, t, total_ngram_count_in_cand) for t in set(total_ngram)]
        g_ref = [TF_IDF(true_ngram, t, total_ngram_count_in_ref) for t in set(total_ngram)]
        #print(g_cand)
        #print(g_ref)
        # inner product of two list
        g = sum([a*b for a,b in zip(g_cand, g_ref)])
        abs_cand = sum([a**2 for a in g_cand]) ** 0.5
        abs_ref = sum([a**2 for a in g_ref]) ** 0.5
        CIDEr_score += (g / (abs_cand * abs_ref)) / N
        break
        print("CIDEr",n,(g / (abs_cand * abs_ref)) / N)
    return CIDEr_score
            
print(CIDEr("Did I eat ?", "I ate it."))