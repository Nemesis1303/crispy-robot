import argparse
import glob
import time 
import multiprocessing
from datetime import datetime
import json
import time
import fitz
import pathlib

#Propias:
from src.pdf_parser import PDFParser
from src.summarizer import Summarizer
from src.multi_column import column_boxes

class processPDF:

    __inputDir__    = ''
    __outputDir__   = ''
    __workers__     = 0
    __results__     = []
    __listFiles__   = []

    def __init__(self, inputDir, outputDir, workers): 
        self.__inputDir__   = inputDir
        self.__outputDir__  = outputDir
        self.__workers__    = workers


    def __getAllDocs ( self, directory ):

        return glob.glob( directory + '/**/*.pdf', recursive=True)

    def getListFiles (self):
        return self.__listFiles__
    
    #for test:
    @staticmethod
    def processPDFDummy ( path ):
        time.sleep (1)
        return ('el nombre del archivo es %s' % path)

    #simple extract text:
    @staticmethod
    def processPDFSimple ( path ):
        doc = fitz.open( path )
        data = ''
        for page in doc:
            bboxes = column_boxes(page, footer_margin=50, no_image_text=True)
            for rect in bboxes:
                data += (page.get_text(clip=rect, sort=True))

            
        return data 
    
    @staticmethod
    def processPDF (path, summary=False ):

        try:
            # Define the path to the PDF file
            pdf_file = pathlib.Path (path)
            
            # Create a directory to save the extracted content (one directory per PDF file)
            path_save = pathlib.Path('/export/data_ml4ds/thuban/repository/data/output/') / pdf_file.stem
            path_save.mkdir(parents=True, exist_ok=True)
            path_save.joinpath("images").mkdir(parents=True, exist_ok=True)
            path_save.joinpath("tables").mkdir(parents=True, exist_ok=True)
            
            # TODO: This needs to be generic
            
            # instructions = \
            #     """ 
            #     You are a helpful AI assistant working with technical descriptions of air conditioner units. 
                
            #     Please summarize the technical description for the Roof-top air conditioner 680 by sections in such a way that the outputted text can be used as input for a topic modeling algorithm.
            # """
            
            instructions = \
            """You are a helpful AI assistant working with the generation of summaries of PDF documents. Please summarize the given document by sections in such a way that the outputted text can be used as input for a topic modeling algorithm. Dont start with 'The document can be summarized...' or 'The document is about...'. Just start with the first section of the document.
            """
            
            pdf_parser = PDFParser()
            
            if summary:
                summarizer = Summarizer()      
                summarizer.summarize(
                    pdf_file=pdf_file,
                    instructions=instructions,
                    path_save=path_save
                )
            
            content = pdf_parser.parse(
                pdf_path=pdf_file,
                path_save=path_save)
            
            #file ok
            return ({'path':path,'result':True})
        except:
            #file ko
            return ({'path':path,'result':False})
              

    def processFiles ( self ):
        
        listFiles = self.__getAllDocs (self.__inputDir__)
        self.__listFiles__ = listFiles

        num_processes = multiprocessing.cpu_count() if self.__workers__ == 0 else self.__workers__
        print ('using %s workers to process %s files' % (num_processes, len(listFiles)))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(processPDF.processPDF, listFiles)
            if any(results):
                self.__results__ = results
                return results

        return False    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', help='Input directory name', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory name', required=True)
    parser.add_argument('-w', '--workers', help='number of workers', default=0)
    args = parser.parse_args()    

    process = processPDF ( args.indir, args.outdir, args.workers)

    now = datetime.now()
    start_time = time.time()
    data = process.processFiles ()
    end_time = time.time()
    execution_time = end_time - start_time

    logdata ={'datetime':now.strftime('%Y-%m-%d'),
              'executiontime':execution_time //60,
              'inputdir': args.indir,
              'numerfiles': len (data),
              'failedfiles':len ([d for d in data if d['result']==False]),
              'listfiles': data
              }
    with open('/tmp/log.json', 'w') as fp:
        json.dump(logdata, fp)






