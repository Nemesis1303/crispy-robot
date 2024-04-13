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
        
        self._logger.info(f"-- -- Loading config from {config_path}")
        self._nlp = assemble(
            config_path,
            overrides={"nlp.lang": lang}
        )

    def extract(self, text): return self._nlp(text)._.acronyms


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Process text to extract acronyms.")
#     parser.add_argument(
#         "text", type=str, help="Text to extract acronyms from.")
#     parser.add_argument("config_path", type=pathlib.Path,
#                         help="Path to the configuration file to use.")
#     parser.add_argument("--lang", default="en", help="Language to use.")
#     parser.add_argument("--verbose", action="store_true",
#                         help="Show extra information.")

#     args = parser.parse_args()

#     extractor = AcronymExtractor(args.config_path, args.lang)
#     acronyms = extractor.extract(args.text)
#     print(acronyms)
