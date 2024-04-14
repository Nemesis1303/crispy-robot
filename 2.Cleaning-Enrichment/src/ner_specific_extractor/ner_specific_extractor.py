from spacy_llm.util import assemble
from dotenv import load_dotenv

load_dotenv()

nlp = assemble("config.cfg")

doc = nlp("I know of a frat pizza recipe with anchovis")
for ent in doc.ents:
    print(ent.text, ent.label_)