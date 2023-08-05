from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import pandas as pd
import nltk
import string
import re
import sys

# import  the data
lab_name = sys.argv[1]
df = pd.read_excel('../../resource/' + lab_name + '/data/raw_data.xlsx', sheet_name='Data')

comment_list = []
clean_comment_list = []
resultantList_clean = []
resultantList = []

french_stopwords = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'des', 'du', 'elle', 'et', 'eux', 'il', 'ils', 'v', 'ya',
                    'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'me', 'même', 'mes', 'moi', 'mon', 'ella', 'de',
                    'nos', 'notre', 'nous', 'on', 'ou', 'par', 'a', 'l’', "l'", 'qu', 'que', 'qui', 'sa', 'se', 'ses',
                    'son', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd',
                    'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants',
                    'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons',
                    'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions',
                    'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez',
                    'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes',
                    'ayants', "'il", 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura',
                    'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais',
                    'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons',
                    'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', 'mà', 'là', 'nn', 'ra',
                    'ggy']

words = {'puisquelle': 'puisque', 'puisquil': 'puisque', 'tt': 'toute', 'sasatisfait': 'satisfait',
         'satisfai': 'satisfait', 'stisfaite': 'satisfaite', 'rdtour': 'retour', 'pad': 'pas'}
lemmatizer = FrenchLefffLemmatizer()
punctuation = string.punctuation + '’"-°'


def preprocess_sentence(comment_list):
    preprocess_list = []
    for sentence in comment_list:
        import re
        sentence = re.sub(' +', ' ', sentence)

        sentence_w_punct = "".join([i.lower() for i in sentence if i not in punctuation])

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

        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_punct)
        for i in range(len(tokenize_sentence)):
            if tokenize_sentence[i] in words.keys():
                tokenize_sentence[i] = str(tokenize_sentence[i]).replace(str(tokenize_sentence[i]),
                                                                         str(words[tokenize_sentence[i]]))

        words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]

        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

        sentence_clean = ' '.join(w for w in words_lemmatize)

        preprocess_list.append(sentence_clean)

        return preprocess_list


for frame in (['comment']):
    df[frame].fillna(0, inplace=True)

for index in range(df.shape[0]):
    if df['comment'][index] != 0:
        clean_comment_list.append(preprocess_sentence([str(df['comment'][index])])[0])
        comment_list.append(str(df['comment'][index]))

# for element in comment_list_clean:
#     if element not in resultantList:
#         resultantList.append(element)

for index in range(len(clean_comment_list)):
    if clean_comment_list[index] not in resultantList_clean:
        resultantList_clean.append(clean_comment_list[index])
        resultantList.append(comment_list[index])

new_list = [resultantList, resultantList_clean]
df1 = pd.DataFrame(resultantList_clean, columns=(['clean_comment']))
df2 = pd.DataFrame(resultantList, columns=(['comment']))
df2['clean_comment'] = df1
writer = pd.ExcelWriter('../../resource/' + lab_name + '/data/clean_comments.xlsx')
df2.to_excel(writer, index=False)
writer.save()
