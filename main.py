import pandas as pd
df=pd.read_csv("earphone_sentiment.csv")
df.dropna()
df['sentiment_value']=df['sentiment_value'].astype(int)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
positive_comments=" ".join(df[df["sentiment_value"]>0]["content"])
font_path="/System/Library/AssetsV2/com_apple_MobileAsset_Font7/54a2ad3dac6cac875ad675d7d273dc425010a877.asset/AssetData/Kaiti.ttc"
wordcloud_positive = WordCloud(width=800, height=400,  background_color='white',font_path=font_path).generate(positive_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('positive_wordcloud')
plt.show()

negative_comments=" ".join(df[df["sentiment_value"]<0]["content"])
wordcloud_negative=WordCloud(width=800, height=400, background_color='black',font_path=font_path).generate(negative_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('negative_wordcloud')
plt.show()

import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
grouped_data=df.groupby("subject")["sentiment_value"].mean().reset_index()
sns.barplot(x='subject', y='sentiment_value', data=grouped_data)
plt.xticks(rotation=45)
plt.title('不同主题下的情感均值')
plt.xlabel('主题')
plt.ylabel('情感均值')
plt.show()

from sklearn.preprocessing import LabelEncoder
df_corr = df.copy()
le = LabelEncoder()
for col in ['subject', 'sentiment_word']:
    if df_corr[col].dtype == 'object':  
        df_corr[col] = le.fit_transform(df_corr[col].astype(str))
corr = df_corr[["subject","sentiment_value","sentiment_word"]].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("不同情感，主题热力图")
plt.show()