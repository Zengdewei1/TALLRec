import gzip
import json
meta_data_path = './meta_Books.json.gz'

def parse_meta(path):
  with gzip.open(path, 'r') as g:
    for l in g:
        yield eval(l.replace(b'\n', b''))

i=0
for l in parse_meta(meta_data_path):
    print(l)
    i+=1
    if(i>10):
      break