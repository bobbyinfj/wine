from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

import pandas as pd
from pandas_profiling import ProfileReport

#load data to pandas
reader = DataFileReader(open("assignment/takehome.avro", "rb"), DatumReader())

records = [record for record in reader]
df = pd.DataFrame.from_records(records)
df.to_pickle("df.pkl")
print(df.head())

print(df.info())

#pandas profiling
ProfileReport(df)

#save to html
profile = df.profile_report(title='Wine Profiling Report')
profile.to_file(output_file="wine_profiling_report.html")