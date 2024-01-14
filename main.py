import csv
import pandas as pd
# rate=open('./ml-latest-small/ratings.csv')
# csvreader=csv.reader(rate)
# rate_data=[]
# rate_header=next(csvreader)
# for row in csvreader:
#     rate_data.append(row)
# rate.close()
rating=pd.read_csv('./ml-latest-small/ratings.csv')
links=pd.read_csv('./ml-latest-small/links.csv')
tags=pd.read_csv('./ml-latest-small/tags.csv')
movies=pd.read_csv('./ml-latest-small/movies.csv')
print(rating.columns)
print(links.columns)
print(tags.columns)
print(movies.columns)

