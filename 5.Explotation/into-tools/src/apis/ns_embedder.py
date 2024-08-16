"""
This script defines a Flask RESTful namespace for performing embedding operations.

Author: Lorena Calvo-Bartolomé
Date: 15/08/2024
"""

import logging
import time

from flask_restx import Namespace, Resource, reqparse
from src.core.embedder import Embedder

logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Embedder')

# ======================================================
# Define namespace for embedding operations
# ======================================================
api = Namespace('Embedding operations')

# ======================================================
# Define parsers to take inputs from user
# ======================================================
get_embedding_parser = reqparse.RequestParser()
get_embedding_parser.add_argument("text_to_embed",
                                  help="Text to be embedded",
                                  required=True)
get_embedding_parser.add_argument('sentence_transformer_model',
                                  help='Sentence transformer model to be used for embeddings',
                                  required=False,
                                  default='paraphrase-distilroberta-base-v2')

# ======================================================
# Create Embedder object
# ======================================================
embedder_manager = Embedder(logger=logger)


@api.route('/getEmbedding/')
class getEmbedding(Resource):
    @api.doc(
        parser=get_embedding_parser,
        responses={
            200: 'Success: Embeddings generated successfully',
            504: 'Embeddings generation error: An error occurred while generating the embeddings'
        }
    )
    def get(self):

        start_time = time.time()

        args = get_embedding_parser.parse_args()

        try:
            # Generate embeddings
            embeddings = embedder_manager.infer_embeddings(
                embed_from=[args['text_to_embed']],
                sentence_transformer_model=args["sentence_transformer_model"],
            )

            # Generate string representation of embeddings
            """
            def get_topic_embeddings(vector):
                repr = " ".join(
                    [f"e{idx}|{val}" for idx, val in enumerate(vector)]).rstrip()

                return repr
            """
            
            def get_float_embeddings(vector):
                return [float(val) for _, val in enumerate(vector)]
        
            embeddings_flt = get_float_embeddings(embeddings)

            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time,
            }
            response = {
                "responseHeader": responseHeader,
                "response": embeddings_flt
            }
            logger.info(
                f"-- -- Embeddings generated successfully:{embeddings_flt}")

            return response, sc

        except Exception as e:
            end_time = time.time() - start_time
            sc = 504
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": str(e)
            }
            response = {
                "responseHeader": responseHeader,
                "response": None
            }
            return response, sc
