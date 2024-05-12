from pathlib import Path
from spacy_llm.registry import registry
import jinja2
from typing import Iterable
from spacy.tokens import Doc
import os

TEMPLATE_DIR = Path("templates")

@registry.llm_tasks("my_namespace.AcronymExtractTask.v1")
def make_quote_extraction() -> "AcronymExtractTask":
    return AcronymExtractTask()


def read_template(name: str) -> str:
    """Read a template"""

    path = TEMPLATE_DIR / f"{name}.jinja"
    
    if not path.exists():
        try:
            path_ = Path(os.getcwd()) / "src" / "acronym_extractor" / path.as_posix()
            path = path_
        except FileNotFoundError:
            raise FileNotFoundError(f"Neither {path.as_posix()} nor {path_.as_posix()} are a valid template.")
        
    return path.read_text()


class AcronymExtractTask(object):
    def __init__(
        self,
        template: str = "acronym_extract_task",
        field: str = "acronyms"
    ):
        self._template = read_template(template)
        self._field = field

    def _check_doc_extension(self):
        """Add extension if need be."""
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)

    def generate_prompts(
        self,
        docs: Iterable[Doc]
    ) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
            )
            yield prompt

    def parse_responses(
        self,
        docs: Iterable[Doc],
        responses: Iterable[str]
    ) -> Iterable[Doc]:
        self._check_doc_extension()
        for doc, prompt_response in zip(docs, responses):
            try:
                if type(prompt_response) == list and len(prompt_response) == 1:
                    try:
                        prompt_response = eval(prompt_response[0])
                    except:
                        prompt_response = prompt_response[0]
                setattr(
                    doc._,
                    self._field,
                    prompt_response,
                ),
            except ValueError:
                setattr(doc._, self._field, None)

        yield doc
