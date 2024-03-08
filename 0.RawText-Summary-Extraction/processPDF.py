import argparse
import glob
import time 
import multiprocessing
import pathlib
from src.pdf_parser import PDFParser
from src.summarizer import Summarizer
import fitz
from src.multi_column import column_boxes

class processPDF:

    __inputDir__    = ''
    __outputDir__   = ''
    __workers__     = 0
    __results__     = []

    def __init__(self, inputDir, outputDir, workers): 
        self.__inputDir__   = inputDir
        self.__outputDir__  = outputDir
        self.__workers__    = workers


    def __getAllDocs ( self, directory ):

        return glob.glob( directory + '/**/*.pdf', recursive=True)

    @staticmethod
    def processPDFDummy ( path ):
        time.sleep (1)
        return ('el nombre del archivo es %s' % path)

    @staticmethod
    def processPDF ( path ):
        doc = fitz.open( path )
        data = ''
        for page in doc:
            bboxes = column_boxes(page, footer_margin=50, no_image_text=True)
            for rect in bboxes:
                data += (page.get_text(clip=rect, sort=True))

            
        return data 

    def processFiles ( self ):
        
        listFiles = self.__getAllDocs (self.__inputDir__)


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

    #prueba = processPDF.processPDF ('/export/data_ml4ds/thuban/repository/data/Articulos2/2401.15453.pdf')
    data = process.processFiles ()

    import ipdb ; ipdb.set_trace()




