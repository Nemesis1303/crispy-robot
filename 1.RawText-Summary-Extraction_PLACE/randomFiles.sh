#!/bin/bash

# Verifica si se pasó un directorio como argumento
if [ -z "$1" ]; then
  echo "Por favor, proporciona un directorio."
  exit 1
fi

# Verifica si el directorio existe
if [ ! -d "$1" ]; then
  echo "El directorio proporcionado no existe."
  exit 1
fi

# Elige 5 ficheros aleatorios del directorio
files=$(find "$1" -type f | shuf -n 10)

# Muestra los archivos seleccionados
echo "Archivos seleccionados al azar:"
echo "$files"

# Para cada archivo seleccionado, ejecuta el script procesa.sh y mide el tiempo
for file in $files; do
  echo "Procesando archivo: $file"
  /usr/bin/time -f "Tiempo de ejecución: %E"  python processOnePDF.py --pdf_path "$file" --path_save /home/sblanco/tmp    
done

