import os
import pathlib
from wasabi import msg
from dotenv import load_dotenv
import logging

from spacy_llm.util import assemble
from src.acronym_extractor.acronym_extract_task import AcronymExtractTask


class AcronymExtractor(object):
    def __init__(
        self,
        config_path: pathlib.Path = None,
        lang: str = "en",
        logger: logging.Logger = None
    ) -> None:

        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

        if not os.getenv("OPENAI_API_KEY", None):
            msg.fail(
                "OPENAI_API_KEY env variable was not found. "
                "Set it by running 'export OPENAI_API_KEY=...' and try again.",
                exits=1,
            )

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('AcronymExtractor')

        if not config_path:
            config_path = pathlib.Path(os.getcwd()) / "src" / "acronym_extractor" / "config.cfg"
            #examples_path = pathlib.Path(
            #    os.getcwd()) / "src" / "acronym_extractor" / "examples.#json"
        
        self._logger.info(f"-- -- Loading config from {config_path}")
        self._nlp = assemble(
            config_path,
            overrides={
                "nlp.lang": lang,
                #"components.llm.task.examples.path": examples_path.as_posix()
            }
        )

    def extract(self, text): return self._nlp(text)._.acronyms