# 0. RawText-Summary-Extraction

This directory contains modules designed to extract the raw content from PDF documents and generate a summary based on that content.

## ``PDFParser``

The PDFParser module is responsible for extracting text, tables, and images from PDFs and generating a JSON file with the extracted content. This functionality is contingent upon the PDFs meeting the following conditions:

- They do not contain watermarks.
- They are not encrypted or subject to any access limitations.
- They consist of plain text along with images and tables.
- They may be single or multi-column documents.
  
Additionally, the module provides the option to generate descriptions for the images and tables in the PDF, if specified. The resulting JSON structure is outlined below:

```json
{
    "metadata": {...},
    "header": "...",
    "footer": "...",
    "pages": [
        {
            "page_number": 0,
            "content": [
                {
                    "element_id": 0,
                    "element_type": "text",
                    "element_content": "...",
                    "element_description": null,
                    "element_path": null
                },
                {
                    "element_id": 1,
                    "element_type": "table",
                    "element_content": "...",
                    "element_description": null,
                    "element_path": "path/to/table"
                },
                {
                    "element_id": 2,
                    "element_type": "image",
                    "element_content": null,
                    "element_description": "...",
                    "element_path": "path/to/image"
                }
            ]
        }
    ]
}
```

In this structure, Element 0 from each page represents the extracted text. Subsequent elements represent any tables or images extracted from the page, preserving their order of appearance.

## ``Summarizer``

The ``Summarizer`` module is responsible for generating a summary of the text extracted from a given PDF document. After invoking the ``summarize(pdf_file, path_save)`` method, a summary of the PDF file will be generated and saved in a ``.txt`` file named ``summary.txt`` in the specified ``path_save`` directory. Additionally, the Llama index will be saved in a directory named index within the ``path_save`` directory.