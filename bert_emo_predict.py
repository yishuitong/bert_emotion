from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import torch.nn.functional as F

# 加载保存好的模型和分词器
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
model = BertForSequenceClassification.from_pretrained('./sentiment_model')
model.eval()

df = pd.read_excel(r'D:\Python\bert-weibo-emotional-analysis-main\dataset\weibo_all-0116.xlsx')
df['predict'] = 0.0
df['positive'] = 0.0

for index, row in df.iterrows():
    text_to_predict = row['正文']
    inputs = tokenizer(text_to_predict, return_tensors="pt", padding='max_length', truncation=True, max_length=39)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    prob_class_1 = probs[0, 1].item()  # 第二个元素是类别1的概率

    predicted_class = torch.argmax(logits, dim=1).item()
    df.at[index, 'predict'] = predicted_class
    df.at[index, 'positive'] = prob_class_1


output_excel_path = 'weibo_all-0116_output.xlsx'  # 输出Excel文件的路径
df.to_excel(output_excel_path, index=False)