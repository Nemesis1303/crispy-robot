import glob
import json
import argparse
import os
import pandas as pd


def getDataFiles (path):
    return glob.glob( path + '/**/*.json', recursive=True)

def getPages (pages):
    return [ (page['element_content']) for page in  pages ]

def getSummary (filename):
    try:
        with open( filename ) as user_file:
            return user_file.read()
    except:
        return ""

def getFileData (file):
    try:
        output = {}
        with open( file ) as user_file:
            data = json.load (user_file)
            output['metadata'] = data['metadata']
            output['raw_text'] = " ".join (str(element[0]) for element in [ getPages (page['content']) for page in data['pages']])        
            output['pdf_path'] = data['metadata']['file_path']
            output['json_path'] = os.path.dirname(file)
            output['summary'] = getSummary (os.path.join (output['json_path'], 'summary.txt'))

    except Exception as E:
        import ipdb ; ipdb.set_trace()

    print (os.path.basename(file))
    return output    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', help='Input directory name', required=True)
    parser.add_argument('-p', '--outfile', help='parquet file to save', required=True)
    args = parser.parse_args()

    ignoreFiles = ['default__vector_store.json', 'image__vector_store.json', 
                    'graph_store.json', 'index_store.json', 'docstore.json',
                    'index_store.json'
                ]

    listJsons = getDataFiles (args.indir)
    allData = [getFileData (file) for file in listJsons if os.path.basename(file) not in ignoreFiles]
    allData = [data for data in allData if data['summary'] != ""]

    df = pd.DataFrame.from_dict(allData)
    df = df.reset_index()
    df = df.rename(columns={"index":"pdf_id"})
    df['pdf_id'] = df.index    
    df.to_parquet (args.outfile, engine='pyarrow')


