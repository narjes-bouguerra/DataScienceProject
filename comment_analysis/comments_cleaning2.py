from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import pandas as pd
import nltk
import string
import re
import sys
from autocorrect import Speller
from nltk.stem.snowball import FrenchStemmer


lab_name = sys.argv[0]

df = pd.read_excel('../../resource/' + lab_name + '/data/raw_data.xlsx', sheet_name='Data')
comment_list = []
clean_comment_list = []
resultantList_clean = []
resultantList = []
words = {'puisquelle': 'puisque', 'puisquil': 'puisque', 'tt': 'toute', 'sasatisfait': 'satisfait',
         'satisfai': 'satisfait', 'stisfaite': 'satisfaite', 'rdtour': 'retour', 'pad': 'pas', 'post': 'poste',
         'gyneco': 'gynoclean', 'gels': 'gel', 'cicatrisant': 'cicatrices',
         'promis': 'promesse', 'disponibl': 'disponible', 'switch': 'sweetch', 'près rit': 'prerscrire',
         'cmd': 'commande'}

punctuation = string.punctuation + '’"-°'
lemmatizer = FrenchLefffLemmatizer()
spell = Speller(lang='fr')
stemmer = FrenchStemmer()


def find_similar_word(s, kw, thr=0.5):
    from difflib import SequenceMatcher
    out = []
    for i in s:
        f = False
        for j in i.split():
            for k in kw:
                if SequenceMatcher(a=j, b=k).ratio() > thr:
                    out.append(k)
                    f = True
                if f:
                    break
            if f:
                break
        else:
            out.append(i)
    return out


def preprocess_sentence(comment_list):
    preprocess_list = []
    words = {'puisquelle': 'puisque', 'puisquil': 'puisque', 'tt': 'toute', 'sasatisfait': 'satisfait',
             'satisfai': 'satisfait', 'stisfaite': 'satisfaite', 'rdtour': 'retour', 'pad': 'pas', 'post': 'poste',
             'gyneco': 'gynoclean', 'gels': 'gel', 'cicatrisant': 'cicatrices',
             'promis': 'promesse', 'disponibl': 'disponible', 'switch': 'sweetch', 'près rit': 'prerscrire',
             'cmd': 'commande', 'mai': 'mais', 'moi': 'moins', 'parce': 'parceque'}

    for sentence in comment_list:
        sentence = re.sub(' +', ' ', sentence)

        # supprimer les punctuation
        sentence_w_punct = "".join([i.lower() for i in sentence if i not in punctuation])

        # supprimer les chiffres
        sentence_w_punct = ''.join([i for i in sentence_w_punct if not i.isdigit()])
        if sentence_w_punct == 'stock':
            sentence_w_punct = str(sentence_w_punct.replace(str(sentence_w_punct), ''))
        if len(sentence_w_punct) > 2:
            if sentence_w_punct[2:7] == 'stock':
                sentence_w_punct = str(sentence_w_punct.replace(str(sentence_w_punct[:2]), ''))

        if len(sentence_w_punct) > 5:
            if sentence_w_punct[:5] == 'stock':
                sentence_w_punct = str(sentence_w_punct.replace('stock', ''))
                if len(sentence_w_punct) > 1:
                    if sentence_w_punct[1].isdigit():
                        sentence_w_punct = str(sentence_w_punct.replace(str(sentence_w_punct[:3]), ''))
        if len(sentence_w_punct) < 4:
            sentence_w_punct = str(sentence_w_punct.replace(str(sentence_w_punct[:3]), ''))
        # tokenisation
        import nltk
        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_punct)

        for i in range(len(tokenize_sentence)):
            if tokenize_sentence[i] in words.keys():
                tokenize_sentence[i] = str(tokenize_sentence[i]).replace(str(tokenize_sentence[i]),
                                                                         str(words[tokenize_sentence[i]]))
        # delete stopwords
        with open("D:/doc_stage_2022/infosquare/project/stopwords-fr-master/stopwords-fr-master/stopwords-fr.txt") as f:
            french_stopwords = f.read().splitlines()
        words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]

        # delete nons
        with open("D:/doc_stage_2022/infosquare/project/delete_word.txt") as f:
            words = f.read().splitlines()
        words_clean = [i for i in words_w_stopwords if i not in words]

        # supprimer les mots de un à deux lettres
        # word_2_lettr = [i for i in words_clean if len(i)>=3]

        # limmatization
        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_clean)

        # correction orthographe
        clean_orth = (spell(word) for word in words_lemmatize)

        # stemming
        # stopwords

        # corriger le mot prese tation
        kw = ["presentation", "présentation "]
        clean_orth = find_similar_word(clean_orth, kw)

        # corriger le mot pres crire
        kw = ["prescrir", "préscrir"]
        clean_orth = find_similar_word(clean_orth, kw)

        sentence_clean = ' '.join(w for w in clean_orth)

        preprocess_list.append(sentence_clean)

        return preprocess_list


for frame in (['comment']):
    df[frame].fillna(0, inplace=True)
for index in range(df.shape[0]):
    if df['comment'][index] != 0:
        clean_comment_list.append(preprocess_sentence([str(df['comment'][index])])[0])
        comment_list.append(str(df['comment'][index]))

for index in range(len(clean_comment_list)):
    if clean_comment_list[index] not in resultantList_clean:
        resultantList_clean.append(clean_comment_list[index])
        resultantList.append(comment_list[index])

new_list = [resultantList, resultantList_clean]
df1 = pd.DataFrame(resultantList_clean, columns=(['clean_comment']))
df2 = pd.DataFrame(resultantList, columns=(['comment']))
df2['clean_comment'] = df1

writer = pd.ExcelWriter('../../resource/' + lab_name + '/data/clean_comments30.xlsx')
df2.to_excel(writer, index=False)
writer.save()






