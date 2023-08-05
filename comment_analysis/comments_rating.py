import pandas as pd
import pickle
import sys

#lab_name = sys.argv[1]
lab_name = 'pharmavision'
df = pd.read_excel('../../resource/' + lab_name + '/data/raw_data.xlsx', sheet_name='Data')


df['comment_score'] = df['comment']

for frame in (['comment_score']):
    df[frame].fillna(0, inplace=True)

model_name = 'comment_rating2'
model_path = '../../resource/' + lab_name + '/models/comment/' + model_name
model = pickle.load(open(model_path, 'rb'))

for i in range(df.shape[0]):
    if df['comment_score'][i] != 0:
        score = (model.predict([str(df['comment_score'][i])])[0])
        df['comment_score'][i] = score

pd.to_numeric(df['comment_score'], errors='coerce')
print(df.head())
df.to_excel('../../resource/' + lab_name + '/data/raw_data_with_score.xlsx', sheet_name='Data', index=False)
