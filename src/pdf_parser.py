#https://github.com/g-stavrakis/PDF_Text_Extraction/tree/main
#https://medium.com/@hussainshahbazkhawaja/paper-implementation-header-and-footer-extraction-by-page-association-3a499b2552ae
import json
import logging
import os
import pathlib
import re
import time

import fitz
import pandas as pd
import pdfplumber
import PyPDF2
from fuzzywuzzy import fuzz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure
from PIL import Image
from tqdm import tqdm

from src.image_descriptor import ImageDescriptor
from src.utils import compare


class PDFParser(object):
    def __init__(self) -> None:

        self._image_descriptor = ImageDescriptor()
        self._api_key = os.environ['OPENAI_API_KEY']

        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('PDFParser')

    def _extract_text_from_page(self, page):

        text = page.get_text(sort=True)
        text = text.split('\n')
        text = [t.strip() for t in text if t.strip()]

        return text

    def _extract_header(self, header_candidates, WIN):

        header_weights = [1.0, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        all_detected = []
        for i, candidate in enumerate(header_candidates):
            temp = header_candidates[max(
                i-WIN, 1): min(i+WIN, len(header_candidates))]
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] = sublist + [''] * (maxlen - len(sublist))
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn, cm) * header_weights[j]
                    score = score/len(cmp)
                except:
                    score = header_weights[j]
                if score > 0.5:
                    detected.append(cn)
            del temp

            all_detected.extend(detected)

        detected = list(set(all_detected))

        return detected

    def _extract_footer(self, footer_candidates, WIN):

        footer_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0]

        all_detected = []
        for i, candidate in enumerate(footer_candidates):
            temp = footer_candidates[max(
                i-WIN, 1): min(i+WIN, len(footer_candidates))]
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] = [''] * (maxlen - len(sublist)) + sublist
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn, cm)
                    score = score/len(cmp)
                except:
                    score = footer_weights[j]
                if score > 0.5:
                    detected.append(cn)
            del temp

            all_detected.extend(detected)

        detected = list(set(all_detected))

        return detected

    def _parse_text(
        self,
        text
    ) -> str:
        """
        Parse the text extracted from the PDF in order to make it more readable.

        Parameters
        ----------
        text : str
            The text to parse

        Returns
        -------
        str
            The parsed text
        """

        text_formatted = re.sub(' +', ' ', text).strip()

        return text_formatted

    def _extract_text(
        self,
        element
    ) -> str:
        """Extract text from a single element and parse it.

        Parameters
        ----------
        element : LTTextContainer
            The element to extract text from

        Returns
        -------
        str
            The formatted text extracted from the element
        """

        extracted_text = element.get_text()

        if extracted_text is None or not bool(extracted_text.strip()):
            return ""

        # self._logger.info(
        #    f"-- Waiting 3 minutes to avoid OpenAI API rate limit...")
        # time.sleep(60)

        parse_text = self._parse_text(extracted_text)

        for el in self._header + self._footer:
            parse_text = parse_text.replace(
                el, "").replace('-', '–').replace(el, "")

        if parse_text == ".":
            parse_text = ""

        return parse_text

    def _extract_table(
        self,
        pdf_path,
        page_num,
        table_num
    ) -> list:
        """Extract the table given by table_num from the page given by page_num from the pdf given by pdf_path.

        Parameters
        ----------
        pdf_path : str
            The path to the pdf file
        page_num : int
            The page number of the page to extract the table from
        table_num : int
            The table number of the table to extract from the page

        Returns
        -------
        list
            The table extracted from the pdf
        """

        # Open the pdf file
        pdf = pdfplumber.open(pdf_path)
        # Find the examined page
        table_page = pdf.pages[page_num]
        # Extract the appropriate table
        table = table_page.extract_tables()[table_num]

        return table

    def _describe_table(
        self,
        table: pd.DataFrame
    ) -> str:
        """Describe the table given by table.

        Parameters
        ----------
        table : pd.DataFrame
            The table to describe

        Returns
        -------
        str
            The description of the table
        """

        parameters = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of describing in natural language the content of a table given in DataFrame format."
                 },
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "frequency_penalty": 0.0
        }

        parser = OpenAI(api_key=self._api_key)

        gpt_prompt = f"Give a description for the following table: {table}"

        message = [{"role": "user", "content": gpt_prompt}]
        parameters["messages"] = [parameters["messages"][0], *message]

        response = parser.chat.completions.create(
            **parameters
        )
        description = response.choices[0].message.content

        return description

    def _get_label_table(
        self,
        table: pd.DataFrame
    ) -> str:
        """Get the label for the table given by table.

        Parameters
        ----------
        table : pd.DataFrame
            The table to describe

        Returns
        -------
        str
            The description of the table
        """

        parameters = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of giving a descriptive label to a table given in DataFrame format. If characterized by several words, unify it with underscores."
                 },
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "frequency_penalty": 0.0
        }

        parser = OpenAI(api_key=self._api_key)

        gpt_prompt = f"Give me a label to save this table.: {table}"

        message = [{"role": "user", "content": gpt_prompt}]
        parameters["messages"] = [parameters["messages"][0], *message]

        response = parser.chat.completions.create(
            **parameters
        )
        label = response.choices[0].message.content

        return label

    def _table_converter(
        self,
        table,
        pageNr
    ) -> str:
        """Convert the table given by table into a string.

        Parameters
        ----------
        table : list
            The table to convert
        pageNr : int
            The page number of the page the table is on

        Returns
        -------
        str
            The table converted into a string
        """

        table_output_save = None
        description = None

        def eliminate_patterns(text):
            """
            Eliminate fixed patterns found in several utes (e.g., "ley" + month, "u.t.e." prefix/suffix, etc.)

            Parameters
            ----------
            text : str
                String to be processed

            Returns
            -------
            result : str
                String without the fixed patterns
            """

            # Patterns to be removed from the inut string
            # TODO: This should be made general
            PATTERNS = [
                r'^TDS-.*',
                r'Sheet (\d+) of (\d+)',
                r'Date (\d{2}.\d{2})',
                r'Technical Description',
                r'Checked by: [A-Za-z0-9-]+'
            ]
            combined_pattern = re.compile(
                "|".join(PATTERNS), flags=re.IGNORECASE)

            # Remove patterns
            result = re.sub(combined_pattern, "", text)

            return result.strip()

        # Convert to string
        table_string = ""
        # Iterate through each row of the table
        cleaned_table = []
        for row_num in range(len(table)):
            row = table[row_num]
            # Remove the line breaker from the wrapted texts
            cleaned_row = [
                eliminate_patterns(item.replace("\n", " "))
                if item is not None
                else ""
                if item is None
                else item
                for item in row
            ]
            # Convert the table into a string
            table_string += " ".join(cleaned_row)
            cleaned_table.append(cleaned_row)

        # Check if the table is only composed by the header or the footer
        aux = table_string
        for el in self._header + self._footer:
            aux = aux.replace(el, "").strip()
            if not aux:
                break

        if aux == "" or aux is None:
            return None, None, None
        else:
            for el in self._header + self._footer:
                table_string = table_string.replace(el, "")

            table_string = self._parse_text(table_string)
            
            for row in cleaned_table[:]:
                remove_row = False
                for el_hf in self._header + self._footer:
                    for i, el in enumerate(row):
                        if el and fuzz.ratio(el_hf, el) > 50:
                            row[i] = el.replace(el, "")
                            remove_row = True
                all_null_none = all(el is None or el == "" for el in row)
                if remove_row and all_null_none:
                    cleaned_table.remove(row)

            if len(cleaned_table) > 1:
                # Convert to dataframe an save as csv/excel
                df_table = pd.DataFrame(cleaned_table)
                label = self._get_label_table(df_table)
                description = None  # self._describe_table(df_table)
                table_output_save = f"/Users/lbartolome/Documents/GitHub/crispy-robot/data/tables/page_{pageNr}_{label}.xlsx"
                df_table.to_excel(table_output_save)
            else:
                return None, None, None

        return table_string, table_output_save, description

    def _is_text_element_inside_any_table(
        self,
        element,
        page,
        tables
    ) -> bool:
        """
        Check if the element is in any tables present in the page.

        Parameters
        ----------
        element : LTTextContainer
            The element to check
        page : pdfminer.layout.LTPage
            The page the element is on
        tables : list
            The list of tables on the page

        Returns
        -------
        bool
            True if the element is inside any table, False otherwise
        """

        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for table in tables:
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return True
        return False

    def _find_table_for_element(
        self,
        element,
        page,
        tables
    ) -> int:
        """Find the table for a given element. If the element is not inside any table, return None.

        Parameters
        ----------
        element : LTTextContainer
            The element to find the table for
        page : pdfminer.layout.LTPage
            The page the element is on
        tables : list
            The list of tables on the page
        """

        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for i, table in enumerate(tables):
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return i  # Return the index of the table
        return None

    def _extract_image(self, element, pageObj, pageNr, table_info=None):
        """Extract the image given by element from the page given by pageObj from the pdf given by pdf_path. Generate a textual description of the image and save it with a label.

        Parameters
        ----------
        element : LTImage
            The image to extract from the page
        pageObj : pdfminer.layout.LTPage
            The page to extract the image from
        pageNr : int
            The page number of the page to extract the image from

        Returns
        -------
        str
            The path to the image
        str
            The textual description of the image
        """

        # Get the coordinates to crop the image from PDF
        [image_left, image_top, image_right, image_bottom] = [
            element.x0, element.y0, element.x1, element.y1]
        # Crop the page using coordinates (left, bottom, right, top)
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        # Save the cropped page to a new PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)

        # Save the cropped PDF to a new file
        pdf_image_save = '/Users/lbartolome/Documents/GitHub/crispy-robot/data/cropped_image.pdf'
        with open(pdf_image_save, "wb") as cropped_pdf_file:
            cropped_pdf_writer.write(cropped_pdf_file)

        # Temporarily convert the PDF and save as image
        images = convert_from_path(pdf_image_save)
        output_file = "/Users/lbartolome/Documents/GitHub/crispy-robot/data/images/cropped_image.png"
        images[0].save(output_file, 'PNG')

        self._logger.info(
            f"-- Waiting 3 minutes to avoid OpenAI API rate limit...")
        time.sleep(60*3)
        # Get the a textual label to save the image
        label = self._image_descriptor.get_label_image(
            pathlib.Path(output_file))

        if table_info:
            label = f"{table_info}_{label}"

        if not "logo" in label:
            self._logger.info(
                f"-- Waiting 3 minutes to avoid OpenAI API rate limit...")
            time.sleep(60*3)
            # Get the textual description of the image
            description = self._image_descriptor.describe_image(
                pathlib.Path(output_file))
        else:
            description = "logo"

        # Save the image with name "page_{pageNr}_{label}.png"
        if table_info:
            new_output_file = f"/Users/lbartolome/Documents/GitHub/crispy-robot/data/images/{label}.png"
        else:
            new_output_file = f"/Users/lbartolome/Documents/GitHub/crispy-robot/data/images/page_{pageNr}_{label}.png"

        images[0].save(new_output_file, 'PNG')

        # Remove the cropped PDF and temporal image
        os.remove(pdf_image_save)
        os.remove(output_file)

        return new_output_file, description, label

    def _get_metadata(self, pdf_path):

        loader = PyMuPDFLoader(pdf_path.as_posix())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )

        langchain_docs2 = loader.load_and_split(text_splitter)
        return langchain_docs2[0].metadata

    def _generate_json(self, content_per_page):
        document_json = {
            "metadata": self._metadata,
            "header": " ".join(self._header),
            "footer": " ".join(self._footer),
            "pages": content_per_page
        }
        return document_json

    def parser(
        self,
        pdf_path: pathlib.Path
    ) -> dict:

        # Extract the header and footer from the PDF
        pages = fitz.open(pdf_path)
        pages = [self._extract_text_from_page(page) for page in pages]

        header_candidates = []
        footer_candidates = []

        for page in pages:
            header_candidates.append(page[:8])
            footer_candidates.append(page[-8:])

        WIN = 8

        self._header = self._extract_header(header_candidates, WIN)
        self._footer = self._extract_footer(footer_candidates, WIN)
        self._metadata = self._get_metadata(pdf_path)

        # Create a PDF file object
        pdfFileObj = open(pdf_path, 'rb')

        # Create a PDF reader object
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)

        # Create the dictionary where the content extracted from each PDF page will be stored
        content_per_page = []

        # Initialize the number of the examined tables
        table_in_page = -1

        for pagenum, page in tqdm(enumerate(extract_pages(pdf_path))):

            last_element_type = None
            element_id = 0

            self._logger.info(
                f"-- Table extraction from page {pagenum} starts...")

            # Initialize variables
            pageObj = pdfReaded.pages[pagenum]
            text_from_tables = []
            path_tables = []
            description_tables = []
            this_page_content = []

            # Open the PDF file with pdfplumber to extract tables
            pdf = pdfplumber.open(pdf_path)

            # Find the pdfplumber page
            page_tables = pdf.pages[pagenum]

            # Find the number of tables in the page
            tables = page_tables.find_tables()
            if len(tables) != 0:
                table_in_page = 0

            # Extracting the tables of the page
            self._logger.info(
                f"-- Extracting tables from page {pagenum}...")
            for i_table in tqdm(range(len(tables))):

                # Extract the information of the table
                table = self._extract_table(pdf_path, pagenum, i_table)

                # Convert the table information in structured string format, save the table as image and get the description
                table_string, table_output_save, description = \
                    self._table_converter(table, pagenum)

                # TODO: Can be NONE
                # Append the table string, the path to the table image and the description to their corresponding lists
                text_from_tables.append(table_string)
                path_tables.append(table_output_save)
                description_tables.append(description)

            self._logger.info(
                f"-- Table extraction from page {pagenum} finished...")

            # Find all the elements and sort them as they appear in the page
            page_elements = [(element.y1, element)
                             for element in page._objs]
            page_elements.sort(key=lambda a: a[0], reverse=True)

            self._logger.info(
                f"-- Element extraction from page {pagenum} starts...")

            # Find the elements that compose a page
            for i_element, component in tqdm(enumerate(page_elements)):

                # Get the element
                element = component[1]

                # Check the elements for tables
                if table_in_page == -1:
                    pass
                else:
                    if self._is_text_element_inside_any_table(element, page, tables):

                        table_idx = self._find_table_for_element(
                            element, page, tables)

                        if table_idx is not None and table_idx == table_in_page:
                            # If a table is found and it is located in the same page as we are currently extracting the content, we add the table to the content
                            # We do not append them if the table is the header or the footer
                            if text_from_tables[table_idx] is not None:
                                this_page_content.append(
                                    {
                                        "element_id": element_id,
                                        "element_type": "table",
                                        "element_content": text_from_tables[table_idx],
                                        "element_description": description_tables[table_idx],
                                        "element_path": path_tables[table_idx]
                                    }
                                )

                                last_element_type = "table"
                                element_id += 1

                            table_in_page += 1

                        # Pass this iteration because the content of this element was extracted from the tables
                        if isinstance(element, LTFigure):

                            table_label = None
                            if path_tables[table_idx]:
                                table_label = f"in_table_{pathlib.Path(path_tables[table_idx]).stem}"

                            # Extract the image and its description
                            image_path, description, label = self._extract_image(
                                element, pageObj, pagenum, table_label)

                            if "logo" in label:
                                os.remove(image_path)

                            else:
                                # Append the image and its description to the content
                                this_page_content.append(
                                    {
                                        "element_id": element_id,
                                        "element_type": "image",
                                        "element_content": None,
                                        "element_description": description,
                                        "element_path": image_path
                                    }
                                )

                                last_element_type = "image"
                                element_id += 1

                        continue

                if not self._is_text_element_inside_any_table(element, page, tables):

                    if isinstance(element, LTFigure):

                        # Extract the image and its description
                        image_path, description, _ = self._extract_image(
                            element, pageObj, pagenum)

                        # Append the image and its description to the content
                        this_page_content.append(
                            {
                                "element_id": element_id,
                                "element_type": "image",
                                "element_content": None,
                                "element_description": description,
                                "element_path": image_path
                            }
                        )

                        last_element_type = "image"
                        element_id += 1

                    else:
                        # Check if the element is text element
                        # if isinstance(element, LTTextContainer) or isinstance(element, LTTextBoxHorizontal):

                        # Use the function to extract the text and format for each text element
                        try:
                            line_text = self._extract_text(element)
                        except:
                            line_text = ""

                        if line_text != "":

                            if last_element_type == "text":
                                # if the former element was text, append the text to the last element
                                this_page_content[-1]["element_content"] += f" {line_text}"
                            else:

                                # Append the text of each line to the page text
                                this_page_content.append(
                                    {
                                        "element_id": element_id,  # i_element,
                                        "element_type": "text",
                                        "element_content": line_text,
                                        "element_description": None,
                                        "element_path": None
                                    }
                                )

                                element_id += 1

                            last_element_type = "text"

            content_per_page.append(
                {
                    "page_number": pagenum,
                    "content": this_page_content
                }
            )

        document_json = self._generate_json(content_per_page)
        with open(f"/Users/lbartolome/Documents/GitHub/crispy-robot/data/{pdf_path.stem}.json", "w", encoding="utf-8") as json_file:
            json.dump(document_json, json_file, indent=2, ensure_ascii=False)

        return
