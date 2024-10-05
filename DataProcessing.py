import pandas as pd
from bs4 import BeautifulSoup

questions_df = pd.read_csv('/content/Questions.csv', encoding='ISO-8859-1')
answers_df = pd.read_csv('/content/Answers.csv', encoding='ISO-8859-1')

def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

questions_df['Cleaned_Questions'] = questions_df['Body'].apply(extract_text)
answers_df['Cleaned_Answers'] = answers_df.loc[answers_df['Score'] > 3, 'Body'].apply(extract_text)

questions_df = questions_df.dropna(subset=['Cleaned_Questions'])
answers_df = answers_df.dropna(subset=['Cleaned_Answers'])

merged_df = pd.merge(questions_df[['Id', 'Cleaned_Questions']], answers_df[['ParentId', 'Cleaned_Answers']], left_on='Id', right_on='ParentId', how='inner')

merged_df = merged_df.apply(lambda x: x.str.encode('utf-8', errors='replace').str.decode('utf-8'), axis=1)

merged_df.to_csv('Cleaned_Questions_Answers_For_Finetuning.csv', index=False, encoding='utf-8-sig')