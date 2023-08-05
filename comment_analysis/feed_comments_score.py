import pandas as pd
import sys

lab_name = 'pharmavision'
df = pd.read_excel('../../resource/' + lab_name + '/data/comments_score.xlsx')
df1 = pd.read_excel('../../resource/' + lab_name + '/data/clean_comments.xlsx')


comments = df['comment'].values.tolist()
new_comments = df1['clean_comment'].values.tolist()
for comment in new_comments:
    if comment not in comments:
        comments.append(comment)

df2 = pd.DataFrame(pd.Series(comments), columns=(['comment']))
df2['score'] = df['score']

writer = pd.ExcelWriter('../../resource/' + lab_name + '/data/comments_score1.xlsx')
df2.to_excel(writer, index=False)
writer.save()
