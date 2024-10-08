"""
This module contains classes for training topic models. More specifically, it contains classes for training Mallet LDA and BERTopic models, grouped under the TMTrainer abstract class. Any new topic model trainer should inherit from TMTrainer and implement the abstract methods train() and infer().
"""
import logging
import os
import pathlib
import sys
import shutil
from subprocess import check_output
import time
from typing import List
from abc import ABC, abstractmethod
import warnings

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from scipy import sparse
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import lil_matrix, csr_matrix
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyLDAvis


from src.utils.utils import file_lines, get_embeddings_from_str, pickler, unpickler
from src.train.topic_labeller import TopicLabeller


class TMTrainer(ABC):
    """
    Abstract base class for topic model trainers.
    """

    def __init__(
        self,
        num_topics: int = 20,
        topn: int = 15,
        model_path: str = None,
        load_trained: bool = False
    ) -> None:
        """
        Initialize the TMTrainer class.

        Parameters
        ----------
        num_topics : int, default=20
            Number of topics to generate.
        topn : int, default=15
            Number of top words per topic.
        model_path : str, optional
            Path to save the trained model.

        """

        # Initialize logger
        self._logger = logging.getLogger(__name__)

        # Create folder for saving model
        self.model_path = pathlib.Path(
            model_path) if model_path else pathlib.Path(__file__).parent / "model"
        
        if not load_trained:
            if self.model_path.exists():
                self._logger.info(
                    f"-- -- Model path {self.model_path} already exists. Saving a copy..."
                )
                old_model_dir = self.model_path.parent / \
                    (self.model_path.name + "_old")
                if not old_model_dir.is_dir():
                    os.makedirs(old_model_dir)
                    shutil.move(self.model_path, old_model_dir)

            self.model_path.mkdir(exist_ok=True)

        # Load OpenAI API key
        try:
            path_env = pathlib.Path(os.getcwd()).parent / '.env'
            load_dotenv(path_env)
            self._api_key = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            self._logger.error(
                f"-- -- Error loading OpenAI API key: {e}. Exiting...")
            self._logger.info(f"-- -- Topic labels will not be generated.")

        # Other attributes
        self.num_topics = num_topics
        self.topn = topn

    def _save_init_params_to_yaml(self) -> None:
        """
        Save the initialization parameters to a YAML file.
        """
        
        not_include = ['train_data', 'df', 'embeddings', 'ids_corpus', '_logger', '_embedding_model', '_umap_model', '_hdbscan_model', '_vectorizer_model', '_ctfidf_model', '_representation_model', '_model']
        
        params = {k: v for k, v in self.__dict__.items(
        ) if not callable(v) and not k.startswith('_') and not k in not_include and not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series, list, dict, pathlib.Path))}
        yaml_path = self.model_path / 'config.yaml'
        with yaml_path.open('w') as yaml_file:
            yaml.dump(params, yaml_file)

    def _save_thr_fig(
        self,
        thetas: np.ndarray,
        plot_file: pathlib.Path
    ) -> None:
        """
        Creates a figure to illustrate the effect of thresholding.

        Parameters
        ----------
        thetas : np.ndarray
            The doc-topics matrix for a topic model.
        plot_file : pathlib.Path
            The name of the file where the plot will be saved.
        """

        all_values = np.sort(thetas.flatten())
        step = int(np.round(len(all_values) / 1000))
        plt.semilogx(all_values[::step], (100 / len(all_values))
                     * np.arange(0, len(all_values))[::step])
        plt.savefig(plot_file)
        plt.close()

    def _save_model_results(
        self,
        thetas: np.ndarray,
        betas: np.ndarray,
        vocab: List[str],
        keys: List[List[str]]
    ) -> None:
        """
        Save the model results.

        Parameters
        ----------
        thetas : np.ndarray
            The doc-topics matrix.
        betas : np.ndarray
            The topic-word distributions.
        vocab : List[str]
            The vocabulary of the model.
        keys : List[List[str]]
            The top words for each topic.
        """

        # self._save_thr_fig(thetas, self.model_path.joinpath('thetasDist.pdf'))
        thetas = sparse.csr_matrix(thetas, copy=True)

        alphas = np.asarray(np.mean(thetas, axis=0)).ravel()

        bow = self.get_bow(vocab)

        sims, sim_rpr = self.calculate_sims(thetas)

        np.save(self.model_path.joinpath('alphas.npy'), alphas)
        np.save(self.model_path.joinpath('betas.npy'), betas)
        sparse.save_npz(self.model_path.joinpath('thetas.npz'), thetas)
        sparse.save_npz(self.model_path.joinpath('bow.npz'), bow)
        sparse.save_npz(self.model_path.joinpath('distances.npz'), sims)
        with open(self.model_path.joinpath('distances.txt'), 'w') as f:
            for item in sim_rpr:
                f.write("%s\n" % item)

        with self.model_path.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        self._logger.info(f"-- -- TOPIC DESCRIPTIONS:")
        self._logger.info('\n'.join(
            [f"TOPIC {id_tpc}: {' '.join(topic)}" for id_tpc, topic in enumerate(keys)]))
        with self.model_path.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([' '.join(topic) for topic in keys]))

        self._logger.info(f"-- -- Generating PyLDAvis visualization...")
        vis_data = self.generate_pyldavis(thetas, betas, vocab, alphas)
        # Save html
        with self.model_path.joinpath("pyLDAvis.html").open("w") as f:
            pyLDAvis.save_html(vis_data, f)

        # Get coordinates of topics in the pyLDAvis visualization
        vis_data_dict = vis_data.to_dict()
        self._coords = list(
            zip(*[vis_data_dict['mdsDat']['x'], vis_data_dict['mdsDat']['y']]))

        with self.model_path.joinpath('tpc_coords.txt').open('w', encoding='utf8') as fout:
            for item in self._coords:
                fout.write(str(item) + "\n")
                
        # Get topic labels
        if self._api_key:
            self._logger.info(f"-- -- Generating topic labels...")
            tpc_labels = self.get_tpc_labels(keys)
        else:
            self._logger.info(f"-- -- OpenAI API key not found. Generating topics labels as 'Topic n'...")
            tpc_labels = [(i, f"Topic {i}") for i in range(self.num_topics)]
            
        with self.model_path.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
            for item in tpc_labels:
                fout.write(str(item) + "\n")
                
    def get_bow(self, vocab: List[str]) -> np.ndarray:
        """
        Get the Bag of Words (BoW) matrix of the documents, maintaining the internal order of the words as in the betas matrix.

        Parameters
        ----------
        vocab : List[str]
            The vocabulary of the model.

        Returns
        -------
        np.ndarray
            The Bag of Words matrix.
        """

        if self.train_data is None:
            self._logger.error(
                f"-- -- Train data not loaded. Cannot create BoW matrix.")
            return np.array([])

        vocab_w2id = {wd: id_wd for id_wd, wd in enumerate(vocab)}

        gensim_dict = Dictionary(self.train_data)
        bow = [gensim_dict.doc2bow(doc) for doc in self.train_data]

        gensim_to_tmt_ids = {
            word_tuple[0]: (vocab_w2id[gensim_dict[word_tuple[0]]]
                            if gensim_dict[word_tuple[0]] in vocab_w2id else None)
            for doc in bow for word_tuple in doc
        }
        gensim_to_tmt_ids = {
            key: value for key, value in gensim_to_tmt_ids.items() if value is not None}

        sorted_bow = [
            sorted([(gensim_to_tmt_ids[gensim_word_id], weight) for gensim_word_id,
                   weight in doc if gensim_word_id in gensim_to_tmt_ids], key=lambda x: x[0])
            for doc in bow
        ]

        num_docs = len(sorted_bow)
        num_words = len(vocab)

        # Create a sparse matrix in LIL format
        bow_mat_sparse = lil_matrix((num_docs, num_words), dtype=np.int32)

        # Populate the sparse matrix
        for doc_id, doc in enumerate(sorted_bow):
            for word_id, weight in doc:
                bow_mat_sparse[doc_id, word_id] = weight
        bow_mat_sparse = csr_matrix(bow_mat_sparse)

        self._logger.info(f"-- -- BoW matrix constructed")

        return bow_mat_sparse

    def calculate_sims(
            self,
            thetas: sparse.csr_matrix,
            topn: int = 300,
            lb: float = 0):
        """Given the path to a TMmodel, it calculates the similarities between documents and saves them in a sparse matrix.

        Parameters
        ----------
        logger : logging.Logger
            Logger.
        thetas : sparse.csr_matrix
            The doc-topics matrix.
        topn : int, optional
            Number of top similar documents to be saved. The default is 300.
        lb : float, optional
            Lower bound for the similarity. The default is 0.6.
        """
        t_start = time.perf_counter()
        thetas_sqrt = np.sqrt(thetas)
        thetas_col = thetas_sqrt.T

        self._logger.info(
            f"-- -- Calculating similarities between documents with topn={topn} and lb={lb}...")
        sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)
        t_end = time.perf_counter()
        t_total = (t_end - t_start)/60
        self._logger.info(f"-- -- Similarities calculated in: {t_total}")

        self._logger.info(
            f"-- -- Getting string representation of similarities...")
        non_zero_indices = sparse.triu(
            sims, k=1).nonzero()  # non-zero elements indices
        sim_str = \
            [' '.join([f"{self.ids_corpus[col]}|{sims[row, col]}" for col in non_zero_indices[1]
                       [non_zero_indices[0] == row]][1:]) for row in range(sims.shape[0])]

        return sims, sim_str

    def get_tpc_labels(self, tpc_descriptions):
        """returns the labels of the topics in the model

        Returns
        -------
        tpc_labels: list of tuples
            Each element is a a term (topic_id, "label for topic topic_id")                    
        """
        
        # Create a topic labeller object
        tl = TopicLabeller(openai_api_key=self._api_key)

        # Get labels
        #labels = tl.get_labels(tpc_descriptions)
        #labels_format = [p for _, p in enumerate(labels)
        try:
            labels = tl.get_labels(tpc_descriptions)
            labels_format = [p for _, p in enumerate(labels)]
        except Exception as e: 
            print("-- -- Something failed with openai. Generating generic labels...")
            labels = [f"Topic {str(el)}" for el in range(len(tpc_descriptions))]


        return labels_format

    def generate_pyldavis(self, thetas: np.ndarray, betas: np.ndarray, vocab: List[str], alphas: np.ndarray) -> None:
        # We will compute the visualization using ndocs random documents
        # In case the model has gone through topic deletion, we may have rows
        # in the thetas matrix that sum up to zero (active topics have been
        # removed for these problematic documents). We need to take this into
        # account
        ndocs = 10000
        validDocs = np.sum(thetas.toarray(), axis=1) > 0
        nValidDocs = np.sum(validDocs)
        if ndocs > nValidDocs:
            ndocs = nValidDocs
        perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocabfreq = np.round(ndocs*(alphas.dot(betas))).astype(int)
        vis_data = pyLDAvis.prepare(
            betas,
            thetas[validDocs, ][perm, ].toarray(),
            doc_len,
            vocab,
            vocabfreq,
            lambda_step=0.05,
            sort_topics=False,
            n_jobs=-1)

        return vis_data

    def _load_train_data(
        self,
        path_to_data: str,
        get_embeddings: bool = False,
        id_col: str = "pdf_id",
        text_data: str = "tr_tokens",
        raw_text_data: str = "raw_text"
    ) -> None:
        """
        Load the training data.

        Parameters
        ----------
        path_to_data : str
            Path to the training data.
        get_embeddings : bool, default=False
            Whether to load embeddings from the data.
        id_col : str, default='pdf_id'
            Column name containing the document IDs.
        text_data : str, default='tr_tokens'
            Column name containing the text data.
        raw_text_data : str, default='raw_text'
            Column name containing the raw text data.
        """

        self.path_to_data = path_to_data
        path_to_data = pathlib.Path(path_to_data)
        self.text_col = text_data
        self.id_col = id_col

        try:
            df = pd.read_parquet(path_to_data)
        except Exception as e:
            self._logger.error(f"-- -- Error loading data: {e}. Exiting...")
            return

        self.df = df
        try:
            self.train_data = [doc.split() for doc in df[text_data]]
        except KeyError:
            self._logger.error(
                f"-- -- Column {text_data} not found in the data. Exiting...")
            return
        try:
            self.ids_corpus = df.pdf_id.values.tolist()
        except KeyError:
            self._logger.error(
                f"-- -- Column pdf_id not found in the data. Exiting...")
            return

        self._logger.info(f"-- -- Loaded processed data from {path_to_data}")

        if get_embeddings:
            if "embeddings" not in df.columns:
                self._logger.warning(
                    f"-- -- Embeddings not present in data. They will be calculated at training time.")
                self.embeddings = None
                self.raw_text = [doc.split() for doc in df[raw_text_data]]
            else:
                self.embeddings = get_embeddings_from_str(df, self._logger)
                self._logger.info(
                    f"-- -- Loaded embeddings from the DataFrame")
        else:
            self.embeddings = None

    @abstractmethod
    def train(self):
        """
        Abstract method to train the topic model.
        """
        pass

    @abstractmethod
    def infer(self):
        """
        Abstract method to perform inference on new documents.
        """
        pass


class MalletLDATrainer(TMTrainer):
    """
    Trainer for Mallet LDA topic model.
    """

    def __init__(
        self,
        num_topics: int = 35,
        alpha: float = 5.0,
        optimize_interval: int = 10,
        num_threads: int = 4,
        num_iters: int = 1000,
        doc_topic_thr: float = 0.0,
        token_regexp: str = "[\p{L}\p{N}][\p{L}\p{N}\p{P}]*\p{L}",
        mallet_path: str = pathlib.Path(
            __file__).parent / "Mallet-202108/bin/mallet",
        topn: int = 15,
        model_path: str = None,
        load_trained: bool = False
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        num_topics : int, default=35
            Number of topics to generate.
        alpha : float, default=5.0
            Alpha parameter for the LDA model.
        optimize_interval : int, default=10
            Interval for optimizing the model parameters.
        num_threads : int, default=4
            Number of threads to use for training.
        num_iters : int, default=1000
            Number of iterations for training.
        doc_topic_thr : float, default=0.0
            Document-topic threshold.
        token_regexp : str, default="[\p{L}\p{N}][\p{L}\p{N}\p{P}]*\p{L}"
            Regular expression for tokenizing the text.
        mallet_path : str
            Path to the Mallet executable.
        topn : int, default=15
            Number of top words per topic.
        model_path : str, optional
            Path to save the trained model.
        load_trained : bool, default=False
            If True, a message is printed indicating that no model will be trained and the inference will be performed.
        """
        super().__init__(num_topics, topn, model_path, load_trained)

        self.mallet_path = pathlib.Path(mallet_path)
        self.alpha = alpha
        self.optimize_interval = optimize_interval
        self.num_threads = num_threads
        self.num_iterations = num_iters
        self.doc_topic_thr = doc_topic_thr
        self.token_regexp = token_regexp

        if not self.mallet_path.is_file():
            self._logger.error(
                f'-- -- Provided mallet path is not valid -- Stop')
            sys.exit()
            
        if load_trained:
            self._logger.info(
                f"-- -- Model will not be trained. Inference will be performed.")
            self.mallet_folder = self.model_path / "modelFiles"

    def train(self, path_to_data: str, text_col: str = "tr_tokens") -> float:
        """
        Train the topic model and save the data to the specified path.

        Parameters
        ----------
        path_to_data : str
            Path to the training data.
        text_col : str, default='tr_tokens'
            Column name containing the text data.

        Returns
        -------
        float
            Time taken to train the model.
        """

        self._load_train_data(
            path_to_data, get_embeddings=False, text_data=text_col)

        t_start = time.perf_counter()

        self.mallet_folder = self.model_path / "modelFiles"
        self.mallet_folder.mkdir(exist_ok=True)

        self._logger.info(f"-- -- Creating Mallet corpus.txt...")
        corpus_txt_path = self.mallet_folder / "corpus.txt"

        with corpus_txt_path.open("w", encoding="utf8") as fout:
            for i, t in zip(self.df[self.id_col], self.df[self.text_col]):
                fout.write(f"{i} 0 {t}\n")
        self._logger.info(f"-- -- Mallet corpus.txt created.")

        self._logger.info(f"-- -- Importing data to Mallet...")
        corpus_mallet = self.mallet_folder / "corpus.mallet"
        cmd = (
            f'{self.mallet_path.as_posix()} import-file --preserve-case --keep-sequence '
            f'--remove-stopwords --token-regex "{self.token_regexp}" '
            f'--input {corpus_txt_path} --output {corpus_mallet}'
        )

        try:
            self._logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- -- Mallet failed to import data. Revise command')
        self._logger.info(f"-- -- Data imported to Mallet.")

        config_mallet = self.mallet_folder / "config.mallet"
        with config_mallet.open('w', encoding='utf8') as fout:
            fout.write(f'input = {corpus_mallet.resolve().as_posix()}\n')
            fout.write(f'num-topics = {self.num_topics}\n')
            fout.write(f'alpha = {self.alpha}\n')
            fout.write(f'optimize-interval = {self.optimize_interval}\n')
            fout.write(f'num-threads = {self.num_threads}\n')
            fout.write(f'num-iterations = {self.num_iterations}\n')
            fout.write(f'doc-topics-threshold = {self.doc_topic_thr}\n')
            fout.write(
                f'output-state = {self.mallet_folder.joinpath("topic-state.gz").resolve().as_posix()}\n')
            fout.write(
                f'output-doc-topics = {self.mallet_folder.joinpath("doc-topics.txt").resolve().as_posix()}\n')
            fout.write(
                f'word-topic-counts-file = {self.mallet_folder.joinpath("word-topic-counts.txt").resolve().as_posix()}\n')
            fout.write(
                f'diagnostics-file = {self.mallet_folder.joinpath("diagnostics.xml").resolve().as_posix()}\n')
            fout.write(
                f'xml-topic-report = {self.mallet_folder.joinpath("topic-report.xml").resolve().as_posix()}\n')
            fout.write(
                f'output-topic-keys = {self.mallet_folder.joinpath("topickeys.txt").resolve().as_posix()}\n')
            fout.write(
                f'inferencer-filename = {self.mallet_folder.joinpath("inferencer.mallet").resolve().as_posix()}\n')

        cmd = f'{self.mallet_path} train-topics --config {config_mallet}'

        try:
            self._logger.info(
                f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Model training failed. Revise command')
            return

        self._logger.info(
            f"-- -- Loading thetas from {self.mallet_folder.joinpath('doc-topics.txt')}")
        thetas_file = self.mallet_folder.joinpath('doc-topics.txt')
        cols = [k for k in np.arange(2, self.num_topics + 2)]
        thetas = np.loadtxt(thetas_file, delimiter='\t',
                            dtype=np.float32, usecols=cols)

        wtcFile = self.mallet_folder.joinpath('word-topic-counts.txt')
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self.num_topics, vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))
        with wtcFile.open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc, i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas, axis=1, norm='l1')

        keys = []
        for k in range(self.num_topics):
            keys.append([vocab[w]
                        for w in np.argsort(betas[k])[::-1][:self.topn]])

        t_end = time.perf_counter() - t_start

        self._save_model_results(thetas, betas, vocab, keys)
        self._save_init_params_to_yaml()

        self._extract_pipe()

        return t_end

    def _extract_pipe(self) -> None:
        """
        Create a pipe based on a small amount of the training data to ensure that the holdout data that may be later inferred is compatible with the training data.
        """

        path_corpus = self.mallet_folder / "corpus.mallet"
        if not path_corpus.is_file():
            self._logger.error(
                f"-- Pipe extraction: Could not locate corpus file")
            return

        path_txt = self.mallet_folder / "corpus.txt"
        with path_txt.open('r', encoding='utf8') as f:
            first_line = f.readline()
        path_aux = self.mallet_folder / "corpus_aux.txt"
        with path_aux.open('w', encoding='utf8') as fout:
            fout.write(first_line + '\n')

        self._logger.info(f"-- Extracting pipeline")
        path_pipe = self.mallet_folder / "import.pipe"

        cmd = (
            f'{self.mallet_path.as_posix()} import-file --use-pipe-from {path_corpus} '
            f'--input {path_aux} --output {path_pipe}'
        )

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Failed to extract pipeline. Revise command')

        path_aux.unlink()

    def infer(
        self,
        docs: List[str],
        num_iterations: int = 1000,
        doc_topic_thr: float = 0.0
    ) -> np.ndarray:
        """
        Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.
        num_iterations : int, optional
            Number of iterations for the inference, by default 1000.
        doc_topic_thr : float, optional
            Document-topic threshold, by default 0.0.

        Returns
        -------
        np.ndarray
            Array of inferred thetas.
        """

        path_pipe = self.mallet_folder / "import.pipe"

        self.inference_folder = self.model_path / "inference"
        self.inference_folder.mkdir(exist_ok=True)

        self._logger.info(f"-- -- Creating Mallet inference corpus.txt...")
        holdout_corpus = self.inference_folder / "corpus.txt"
        with holdout_corpus.open("w", encoding="utf8") as fout:
            for i, t in enumerate(docs):
                fout.write(f"{i} 0 {t}\n")
        self._logger.info(f"-- -- Mallet corpus.txt for inference created.")

        inferencer = self.mallet_folder / "inferencer.mallet"
        corpus_mallet_inf = self.inference_folder / "corpus_inf.mallet"
        doc_topics_file = self.inference_folder / "doc-topics-inf.txt"

        self._logger.info('-- Inference: Mallet Data Import')

        cmd = (
            f'{self.mallet_path.as_posix()} import-file --use-pipe-from {path_pipe} '
            f'--input {holdout_corpus} --output {corpus_mallet_inf}'
        )

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Mallet failed to import data. Revise command')
            return

        self._logger.info('-- Inference: Inferring Topic Proportions')

        cmd = (
            f'{self.mallet_path.as_posix()} infer-topics --inferencer {inferencer} '
            f'--input {corpus_mallet_inf} --output-doc-topics {doc_topics_file} '
            f'--doc-topics-threshold {doc_topic_thr} --num-iterations {num_iterations}'
        )

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Mallet inference failed. Revise command')
            return

        self._logger.info(f"-- -- Inference completed. Loading thetas...")

        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t',
                              dtype=np.float32)[2:]

        self._logger.info(f"-- -- Inferred thetas shape {thetas32.shape}")

        return thetas32


class BERTopicTrainer(TMTrainer):
    """
    Trainer for the BERTopic Topic Model.
    """

    def __init__(
        self,
        model_path: str,
        num_topics: int = 10,
        topn: int = 15,
        no_below: int = 1,
        no_above: float = 1.0,
        stopwords: List[str] = None,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        umap_metric: str = 'cosine',
        hdbscan_min_cluster_size: int = 10,
        hdbscan_metric: str = 'euclidean',
        hdbscan_cluster_selection_method: str = 'eom',
        hbdsan_prediction_data: bool = True,
        load_trained: bool = False,
    ):
        """
        Initialization method.

        Parameters
        ----------
        model_path : str
            Path to save the trained model.
        num_topics : int, default=10
            Number of topics to generate.
        topn : int, default=15
            Number of top words per topic.
        no_below : int, default=1
            Ignore all words that appear in less than `no_below` documents.
        stopwords: List[str], default = None
            Stopwords list to use for the CountVectorizer
        no_above : float, default=1.0
            Ignore all words that appear in more than `no_above` documents.
        sbert_model : str, default='multi-qa-mpnet-base-dot-v1'
            Model to be used for calculating the embeddings.
        umap_n_components : int, default=5
            Number of components to reduce the embeddings to.
        umap_n_neighbors : int, default=15
            Number of neighbors to consider for UMAP.
        umap_min_dist : float, default=0.0
            Minimum distance between points in the UMAP space.
        umap_metric : str, default='cosine'
            Metric to be used for UMAP.
        hdbscan_min_cluster_size : int, default=10
            Minimum number of samples in a cluster.
        hdbscan_metric : str, default='euclidean'
            Metric to be used for HDBSCAN.
        hdbscan_cluster_selection_method : str, default='eom'
            Method to select the number of clusters.
        hbdsan_prediction_data : bool, default=True
            If True, the prediction data is used for HDBSCAN.
        load_trained : bool, default=False
            If True, the model is loaded from model_path / 'model.pickle'.
        """

        super().__init__(num_topics, topn, model_path, load_trained)

        self.sbert_model = sbert_model
        self.no_below = no_below
        self.no_above = no_above
        self.stopwords = stopwords
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method
        self.hbdsan_prediction_data = hbdsan_prediction_data

        word_min_len = 2
        self.word_pattern = (
            f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
            f"[a-zA-Z\u00C0-\u024F]"
            f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{word_min_len - 1},}}"
            f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
            f"(?![a-zA-Z\u00C0-\u024F\d])"
        )
        
        if load_trained:
            if not self.model_path.joinpath('model.pickle').is_file():
                self._logger.error("-- -- No model found in the model path. Exiting...")
            else:
                self._logger.info("-- -- Loading trained model...")
                self._model = unpickler(self.model_path.joinpath('model.pickle'))
            

    def train(self, path_to_data: str, text_col: str = "tr_tokens", raw_text_data: str = "raw_text") -> float:
        """
        Train the topic model and save the data to the specified path.

        Parameters
        ----------
        path_to_data : str
            Path to the training data.
        text_col : str, default='tr_tokens'
            Column name containing the text data.
        raw_text_data: str, defautl='raw_text'

        Returns
        -------
        float
            Time taken to train the model.
        """

        self._load_train_data(path_to_data, get_embeddings=True,
                              text_data=text_col, raw_text_data=raw_text_data)
        
        self.mallet_folder = self.model_path / "modelFiles"
        self.mallet_folder.mkdir(exist_ok=True)

        self._logger.info(f"-- -- Creating Mallet corpus.txt...")
        corpus_txt_path = self.mallet_folder / "corpus.txt"
        
        with corpus_txt_path.open("w", encoding="utf8") as fout:
            for i, t in zip(self.df[self.id_col], self.df[self.text_col]):
                fout.write(f"{i} 0 {t}\n")
        self._logger.info(f"-- -- Mallet corpus.txt created.")

        t_start = time.perf_counter()

        self._logger.info(
            f'-- -- BERTopic Corpus Generation: Using text from col {text_col}')

        if self.embeddings is not None:
            self._logger.info(
                "-- -- Using pre-trained embeddings from the dataset...")
        else:
            self._logger.info(
                f"-- -- Creating SentenceTransformer model with {self.sbert_model}...")

        self._embedding_model = SentenceTransformer(self.sbert_model)

        self._umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric
        )

        self._hdbscan_model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            prediction_data=self.hbdsan_prediction_data
        )

        self._vectorizer_model = CountVectorizer(
            token_pattern=self.word_pattern,
            stop_words=self.stopwords,
        )

        self._ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        self._representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(
                diversity=0.3,
                top_n_words=15
            )
        }

        self._model = BERTopic(
            language="english",
            top_n_words=self.topn,
            nr_topics=self.num_topics,
            umap_model=self._umap_model,
            embedding_model=self._embedding_model,
            verbose=True
        )

        self._logger.info(
            f"-- -- Training BERTopic model with {self.num_topics} topics... ")

        if self.embeddings is not None:
            texts = [" ".join(doc) for doc in self.train_data]
            _, probs = self._model.fit_transform(texts, self.embeddings)
        else:
            texts = [" ".join(doc) for doc in self.raw_text]
            _, probs = self._model.fit_transform(texts)

        thetas_approx, _ = self._model.approximate_distribution(
            texts, use_embedding_model=True)
        self._logger.info(f"-- -- Thetas shape: {thetas_approx.shape}")

        betas = self._model.c_tf_idf_.toarray()
        betas = betas[1:, :]  # drout outlier topic and keep (K-1, V) matrix
        self._logger.info(f"-- -- Betas shape: {betas.shape}")
        vocab = self._model.vectorizer_model.get_feature_names_out()

        keys = []
        for k, v in self._model.get_topics().items():
            keys.append([el[0] for el in v])
        keys = keys[1:]

        model_file = self.model_path.joinpath('model.pickle')
        pickler(model_file, self._model)

        t_end = time.perf_counter() - t_start

        self._save_model_results(thetas_approx, betas, vocab, keys)
        self._save_init_params_to_yaml()

        return t_end

    def infer(self, docs: List[str]) -> np.ndarray:
        """
        Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.

        Returns
        -------
        np.ndarray
            Array of inferred thetas.
        """

        self._logger.info('-- -- Inference: Getting inferred thetas matrix')
        thetas, _ = self._model.approximate_distribution(docs)
        self._logger.info(f"-- -- Inferred thetas shape {thetas.shape}")

        return thetas
