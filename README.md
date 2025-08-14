# 数据来自于 "earphone_sentiment.csv"
## 生成词云
# 清洗数据
- 把无效数据剔除(NaN)
- 把"sentiment_value"列转变成**int**类型，
  - 这里面的数据只有“1”和“-1”，代表好或者不好的评价
'''python
df.dropna()
df['sentiment_value']=df['sentiment_value'].astype(int)
# 处理数据为字符串
- 调用 WordCloud 类和 matplotlib.pyplot 类
  -分别实现词云和图片美化的功能
- 正面词云和负面词云分别做成两个不同的字符串
'''python
positive_comments=" ".join(df[df["sentiment_value"]>0]["content"])
negative_comments=" ".join(df[df["sentiment_value"]<0]["content"])
# 生成词云可视化
- 利用terminal找出计算机中中文字体的储存位置，并传入参数
- 使用 Matplotlib 进行图形渲染与展示
'''python
font_path="/System/Library/AssetsV2/com_apple_MobileAsset_Font7/54a2ad3dac6cac875ad675d7d273dc425010a877.asset/AssetData/Kaiti.ttc"
wordcloud_positive = WordCloud(width=800, height=400,  background_color='white',font_path=font_path).generate(positive_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('positive_wordcloud')
plt.show()

## 生成不同主题sentiment_value均值的柱状图
# 字体与符号设置
'''python
mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
# 按主题分组并计算均值
'''python
grouped_data=df.groupby("subject")["sentiment_value"].mean().reset_index()
# 绘制柱状图
sns.barplot(x='subject', y='sentiment_value', data=grouped_data)
plt.xticks(rotation=45)
plt.title('不同主题下的情感均值')
plt.xlabel('主题')
plt.ylabel('情感均值')
plt.show()

## 情感、主题与情感词的相关性热力学图
# 复制数据
- 不对原数据进行修改
df_corr = df.copy()
# 编码非数值字段
- 'subject', 'sentiment_word'列的数据都为string，需要用数字映射便于计算相关性矩阵
le = LabelEncoder()
for col in ['subject', 'sentiment_word']:
    if df_corr[col].dtype == 'object':
        df_corr[col] = le.fit_transform(df_corr[col].astype(str))
# 计算相关系数矩阵
corr = df_corr[["subject", "sentiment_value", "sentiment_word"]].corr()
# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("不同情感、主题与情感词的相关性热力图")
plt.show()