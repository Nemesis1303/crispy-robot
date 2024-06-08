# 3. TrainingCorpus-Generation

This directory hosts a system for selecting and filtering documents based on Named Entity Recognition (NER). The output of this process can be used for:

- Retrieving named entities of a certain type from the documents in a given collection, with or without their frequency of appearance.
- Creating a training corpus for topic modeling. Additionally, it includes several ad-hoc topic modeling techniques designed to enhance the quality of the models trained in the subsequent phase.

## Main modules

### DocSelector

The DocSelector module contains the `DocSelector` class, which is responsible for selecting and filtering documents based on Named Entity Recognition (NER) labels. It provides two main methods:

- **filter_docs_by_ners**: This method filters documents by NER labels and can operate in two modes:
  - **with_freq = False** (default): Filters out all words in the document that are not NERs, maintaining the original order and frequency of the NERs.
  - **with_freq = True**: Filters out all words in the document that are not NERs and returns a dictionary with NERs as keys and their frequency of appearance in the document as values.
  
- **filter_docs_if_ner**: This method filters documents based on whether they contain a minimum number of NERs of a specified target label. It returns a DataFrame containing documents that have at least `min_count` NERs of the target label.

These methods allow for flexible and powerful document selection and filtering based on NER criteria, facilitating the creation of high-quality training corpora for subsequent topic modeling phases.

### Preprocessor

These techniques include:

- Elimination of stop words.
- Substitution of specific equivalences within the domain of the analyzed documents.

## Configuration File

The `config.yaml` file contains various parameters used to configure the behavior of the system. Below is an example configuration file and a description of its parameters:

```yaml
mode: 2  # 0, 1, 2, 3
preproc: True

doc_selector:
  lemmas_col: raw_text_LEMMAS
  ner_col: raw_text_SPEC_NERS
  target_label: DRUG
  remove_empty: True

preprocessor:
  exec:
    mode: manual  # manual or auto
    path_stw: wdlists/stw
    path_eq: wdlists/eqs
  object_creation:
    min_lemas: 15
    no_below: 10
    no_above: 0.6
    keep_n: 100000
```

### Parameters

- **mode**: The mode of operation. Possible values are 0, 1, 2, or 3.
- **preproc**: Boolean indicating whether preprocessing is enabled.
- **doc_selector**:
  - **lemmas_col**: The column name with the lemmas of the documents.
  - **ner_col**: The column name with the named entities (NERs) of the documents.
  - **target_label**: The target NER label to filter the documents.
  - **remove_empty**: Boolean indicating whether to remove rows with empty NER data.
- **preprocessor**:
  - **exec**:
    - **mode**: The mode of execution for the preprocessor, either `manual` or `auto`.
    - **path_stw**: Path to the stopwords files.
    - **path_eq**: Path to the equivalence files.
  - **object_creation**:
    - **min_lemas**: Minimum number of lemmas.
    - **no_below**: Minimum frequency threshold.
    - **no_above**: Maximum frequency threshold.
    - **keep_n**: Number of items to keep.

## Usage Modes

The system operates in several modes, determined by the `mode` parameter in the configuration. Here are the details for each mode:

### Mode 0: Filter Docs by NER with Frequency

Documents are filtered based on NERs with frequency. If the parameter 

### Mode 1: Filter Docs by NER without Frequency


### Mode 2: Filter Docs if NER

This mode filters documents if they contain named entities. It uses the `lemmas_col` for preprocessing.

### Mode 3: Just Use Lemmas

In this mode, the system uses lemmas for document selection. It uses the `lemmas_col` for preprocessing.

## Preprocessing

If preprocessing is enabled (`preproc` is True), the system performs additional preprocessing tasks, such as eliminating stop words and substituting specific equivalences. The mode of execution for the preprocessor can be either manual or automatic:

- **Manual Mode**: The user is prompted to select the stop words and equivalent terms files.
- **Automatic Mode**: The system uses all stop words and equivalent terms files in the specified directories.

## Command-Line Arguments

The script can be run with several command-line arguments to override the default configuration values. Here are the arguments that can be used:

- `--config`: Path to the configuration file.
- `--mode`: Mode of operation (0, 1, 2, or 3).
- `--preproc`: Enable preprocessing (True/False).
- `--lemmas_col`: Column name with the lemmas of the documents.
- `--ner_col`: Column name with the NERs of the documents.
- `--target_label`: Target NER label to filter the documents.
- `-s`, `--source`: Input parquet file (required).
- `-o`, `--output`: Output parquet file (required).