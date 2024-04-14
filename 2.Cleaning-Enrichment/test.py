import re
from src.acronym_extractor.acronym_extractor import AcronymExtractor
import pandas as pd

def create_acronym_list(acronyms_data):
    acronyms_list = []
    for acronym, full_form in acronyms_data:
        acronym_pattern = r'\b{}\b'.format(acronym)
        acronyms_list.append((acronym_pattern, full_form))

    # Remove duplicates if any
    acronyms_list_dict = dict(acronyms_list)
    acronyms_list = list(acronyms_list_dict.items())
    
    return acronyms_list

def test_acronym_extractor():
    df = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/data/output4/data.parquet")
    print(len(df))
    
    AE = AcronymExtractor()
    print(df.columns)
    df_aux = df.iloc[0:1]
    df_aux["acronyms"] = df_aux["raw_text"].apply(AE.extract)
    print(df_aux)

    acronyms = df_aux["acronyms"].explode()
    print(acronyms)
    # Create the list of acronyms in the desired format
    en_acronyms_list = create_acronym_list(acronyms)
    print(en_acronyms_list)
    
    test = "AI"
    # Extract acronyms from the test string
    for pattern, full_form in en_acronyms_list:
        matches = re.findall(pattern, test)
        if matches:
            print(f"Acronym '{matches[0]}' found, full form is '{full_form}'")
            # You can break here if you only want to find the first match
            break
    else:
        print("No acronyms found.")

if __name__ == "__main__":
    test_acronym_extractor()
