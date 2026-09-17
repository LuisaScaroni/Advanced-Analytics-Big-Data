import sqlite3
import panda as pd
import chromadb
print('connecting to the database...')
conn=sqlite3.connect('steam_games_reviews_25.sqlite')
games_df=pd.read_sql_query('SELECT*FROM games', conn)
print('we initialize the ChromaDB')
chroma_client=chromadb-PersistentClient(path='./chroma_data')
collection=chroma_client.get_or_create_collection(name='steam_games')
documents=[]
metadatas=[]
ids=[]
print('3 extraction of datas...')
for index, row in games_df.iterrows():
    text_repr=f'Title:{row['title']}. Genres:{row.get('genres', '')}. Description: {row.get('description','')}'
    documents.append(text_repr)
    metadatas.append({'title': row['title'], 'id':row['game_id']})
    ids.append(str(row['game_id']))
print('4 saving vectors....')
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print('we have ended!! vector database created with sucess in folder chroma_data ')
