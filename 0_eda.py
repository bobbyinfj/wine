from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

import pandas as pd
from pandas_profiling import ProfileReport

#load data to pandas
reader = DataFileReader(open("assignment/takehome.avro", "rb"), DatumReader())

#save to pickled df
records = [record for record in reader]
df = pd.DataFrame.from_records(records)
df.to_pickle("df.pkl")

#quick info
print(df.head())
print(df.info())

#look at large or small values
print(df[(df['rating']<3) | (df['rating']>6)])
test = df

#create test urls in an excel file for later
test['url'] = df.apply(lambda x: "http://127.0.0.1:8080/rating/prediction.json?ph=%s&brightness=%s&chlorides=%s&sugar=%s&sulfates=%s&acidity=%s" % (x['ph'], x['brightness'], x['chlorides'], x['sugar'], x['sulfates'], x['acidity']), axis=1)
test.to_excel('text.xlsx')

#pandas profiling
ProfileReport(df)

#save to html
profile = df.profile_report(title='Wine Profiling Report')
profile.to_file(output_file="wine_profiling_report.html")