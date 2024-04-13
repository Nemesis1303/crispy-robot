#!/bin/bash 
python processPDF.py -i $INPUT_DIR -o $OUTPUT_DIR 
python generateParquet.py -i $OUTPUT_DIR -p $PARQUET_FILE 
