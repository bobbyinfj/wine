from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

import pandas as pd

reader = DataFileReader(open("assignment/takehome.avro", "rb"), DatumReader())

records = [record for record in reader]
df = pd.DataFrame.from_records(records)

print(df.head())

print(df.info())

#pandas profiling?