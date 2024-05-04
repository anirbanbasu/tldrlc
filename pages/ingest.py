# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import time
from typing import List
import random
import datetime
import requests
import logging

from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.readers.web import BeautifulSoupWebReader, TrafilaturaWebReader
from llama_index.readers.papers import ArxivReader, PubmedReader
from llama_index.readers.file import PyMuPDFReader

from llama_index_client import Document
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import load_index_from_storage, load_indices_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

import solara
from solara.alias import rv
from solara.lab import task

import wikipedia
from linkpreview import Link, LinkPreview, LinkGrabber

import utils.global_state as global_state
import utils.constants as constants

logger = logging.getLogger(__name__)
# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.basicConfig(
    # See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
    format=constants.LOG_FORMAT,
    level=int(os.getenv(constants.ENV_KEY_LOG_LEVEL)),
    encoding=constants.CHAR_ENCODING_UTF8,
)

# Data sources settings
existing_indices: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
wikipedia_languages: solara.Reactive[list] = solara.reactive([])
wikipedia_language_prefix: solara.Reactive[str] = solara.reactive(
    constants.ISO639SET1_LANGUAGE_ENGLISH
)
wikipedia_article: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
arxiv_query: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
pubmed_query: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
webpage_reader: solara.Reactive[str] = solara.reactive(
    constants.WEBPAGE_READER_BEAUTIFUL_SOUP
)
webpage_url: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
pdf_url: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)

# Generic ingested documents loaded from any supported data source
ingested_documents: solara.Reactive[List[Document]] = solara.reactive([])
last_ingested_data_source: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

status_message: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_colour: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_show: solara.Reactive[bool] = solara.reactive(False)


def show_status_message(message: str, colour: str = "info", timeout: int = 4):
    """Show a status message on the page."""
    status_message.value = message
    status_message_colour.value = colour
    status_message_show.value = True
    if timeout > 0:
        time.sleep(timeout)
        status_message_show.value = False


def load_remote_pdf_data(pdf_url: str) -> List[Document]:
    """Download the PDF from the URL and load its data as text."""
    pdf_response = requests.get(pdf_url, timeout=30)
    pdf_filename = f"__tmp-{global_state.md5_hash(pdf_url)}.pdf"
    with open(pdf_filename, "wb") as pdf:
        pdf.write(pdf_response.content)
        pdf.close()
    reader = PyMuPDFReader()
    pdf_docs = reader.load_data(file_path=pdf_filename)
    # Remove the temporary file
    os.remove(pdf_filename)
    return pdf_docs


def initialise_chat_engine() -> bool:
    """Initialise the chat engine with the knowledge graph index if it has been initialised."""
    post_processors = None
    if (
        global_state.global_settings__language_model_provider.value
        == constants.LLM_PROVIDER_COHERE
    ):
        post_processors = [
            CohereRerank(api_key=global_state.global_settings__cohere_api_key.value)
        ]
    else:
        post_processors = [SentenceTransformerRerank()]
    if global_state.global_knowledge_graph_index.value is not None:
        global_state.global_chat_engine.value = None
        if global_state.global_llamaindex_chat_memory.value is not None:
            global_state.global_llamaindex_chat_memory.value.reset()
        show_status_message(
            message=f"**Initialising chat engine** from index using the _{global_state.global_settings__index_chat_mode.value}_ chat mode."
        )
        global_state.global_chat_engine.value = (
            global_state.global_knowledge_graph_index.value.as_chat_engine(
                chat_mode=global_state.global_settings__index_chat_mode.value,
                llm=Settings.llm,
                verbose=True,
                memory=global_state.global_llamaindex_chat_memory.value,
                system_prompt=global_state.global_settings__llm_system_message.value,
                node_postprocessors=post_processors,
                streaming=True,
            )
        )
        return True
    else:
        raise ValueError("Index from ingested documents is not available.")


def build_index() -> bool:
    """Build the knowledge graph index from the documents."""
    # Try to output some kind of progress bar state?
    global ingested_documents
    chunk_parser = SentenceSplitter.from_defaults(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True,
    )
    chunk_nodes = chunk_parser.get_nodes_from_documents(
        documents=ingested_documents.value, show_progress=True
    )
    show_status_message(
        message=f"**Building index** from {len(chunk_nodes)} extracted chunks.",
        timeout=0,
    )
    global_state.global_knowledge_graph_index.value = KnowledgeGraphIndex(
        nodes=chunk_nodes,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        storage_context=global_state.global_llamaindex_storage_context.value,
        max_triplets_per_chunk=global_state.global_settings__index_max_triplets_per_chunk.value,
        include_embeddings=global_state.global_settings__index_include_embeddings.value,
        show_progress=True,
    )
    global_state.global_knowledge_graph_index.value.storage_context.docstore.add_documents(
        chunk_nodes
    )
    return True


@task
async def ingest_wikipedia_data():
    """Ingest the selected Wikipedia article."""
    global ingested_documents
    try:
        ingested_documents.value = []
        reader = WikipediaReader()
        initialisation_status: bool = False
        articles_list = wikipedia_article.value.split("[ __ ]")
        if len(articles_list) == 0:
            raise ValueError("No Wikipedia article titles found in the input.")
        elif len(articles_list) == 1:
            show_status_message(
                message=f"**Fetching** article '_{wikipedia_article.value}_' from Wikipedia ({wikipedia_language_prefix.value})."
            )
        else:
            show_status_message(
                message=f"""
                **Fetching** {len(articles_list)} articles from Wikipedia ({wikipedia_language_prefix.value}).

                {', '.join([f'\'_{article}_\'' for article in articles_list])}
                """
            )
        if (
            wikipedia_language_prefix.value
            != constants.DEFAULT_DATA_SOURCE_WIKIPEDIA_LANGUAGE_PREFIX
        ):
            # A non-English Wikipedia language prefix is selected
            ingested_documents.value = reader.load_data(
                pages=articles_list,
                lang_prefix=wikipedia_language_prefix.value,
                auto_suggest=False,
            )
        else:
            ingested_documents.value = reader.load_data(
                pages=articles_list,
                auto_suggest=False,
            )
        last_ingested_data_source.value = constants.SOURCE_TYPE_WIKIPEDIA
        initialisation_status = build_index()
        initialisation_status = initialisation_status and initialise_chat_engine()
        if initialisation_status:
            show_status_message(
                message=f"**Done** ingesting article(s) {', '.join([f'\'_{article}_\'' for article in articles_list])} from Wikipedia ({wikipedia_language_prefix.value}). You can now chat with the AI.",
                colour="success",
            )
    except Exception as e:
        show_status_message(
            message=f"**Error** ingesting Wikipedia article(s).<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error ingesting Wikipedia article(s). {str(e)}")


@task
async def ingest_arxiv_data():
    """Ingest the arXiv results."""
    global ingested_documents
    try:
        ingested_documents.value = []
        reader = ArxivReader()
        show_status_message(
            message=f"**Fetching** results of arXiv query `{arxiv_query.value}`.",
        )
        ingested_documents.value = reader.load_data(
            papers_dir=global_state.md5_hash(
                f"{datetime.datetime.now()}-{random.random()}"
            ),
            search_query=arxiv_query.value,
        )
        last_ingested_data_source.value = constants.SOURCE_TYPE_ARXIV
        if len(ingested_documents.value) == 0:
            raise ValueError("No text extracted from the arXiv query.")
        initialisation_status = build_index()
        initialisation_status = initialisation_status and initialise_chat_engine()
        if initialisation_status:
            show_status_message(
                message=f"**Done** ingesting results of arXiv query `{arxiv_query.value}`. You can now chat with the AI.",
                colour="success",
            )
    except Exception as e:
        show_status_message(
            message=f"**Error** ingesting arXiv results.<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error ingesting arXiv results. {str(e)}")


@task
async def ingest_pubmed_data():
    """Ingest the Pubmed results."""
    global ingested_documents
    try:
        ingested_documents.value = []
        reader = PubmedReader()
        show_status_message(
            message=f"**Fetching** results of Pubmed query `{pubmed_query.value}`.",
        )
        ingested_documents.value = reader.load_data(
            search_query=pubmed_query.value,
        )
        last_ingested_data_source.value = constants.SOURCE_TYPE_PUBMED
        if len(ingested_documents.value) == 0:
            raise ValueError("No text extracted from the Pubmed query.")
        initialisation_status = build_index()
        initialisation_status = initialisation_status and initialise_chat_engine()
        if initialisation_status:
            show_status_message(
                message=f"**Done** ingesting results of Pubmed query `{pubmed_query.value}`. You can now chat with the AI.",
                colour="success",
            )
    except Exception as e:
        show_status_message(
            message=f"**Error** ingesting Pubmed results.<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error ingesting Pubmed results. {str(e)}")


@task
async def ingest_webpage_data():
    """Ingest the selected webpage."""
    global ingested_documents
    try:
        ingested_documents.value = []
        initialisation_status: bool = False
        reader: BasePydanticReader = None
        match webpage_reader.value:
            case constants.WEBPAGE_READER_TRAFILATURA:
                reader = TrafilaturaWebReader()
            case constants.WEBPAGE_READER_BEAUTIFUL_SOUP:
                reader = BeautifulSoupWebReader()
        urls_list = webpage_url.value.split(" ")
        if len(urls_list) == 0:
            raise ValueError("No web page URLs found in the input.")
        elif len(urls_list) == 1:
            show_status_message(
                message=f"**Fetching** web page [{urls_list[0]}]({urls_list[0]}) using the `{webpage_reader.value}` reader."
            )
        else:
            show_status_message(
                message=f"""
                **Fetching** {len(urls_list)} web pages using the `{webpage_reader.value}` reader.

                {', '.join([f'[{url}]({url})' for url in urls_list])}
                """
            )
        ingested_documents.value = reader.load_data(
            urls=urls_list,
        )
        last_ingested_data_source.value = constants.SOURCE_TYPE_WEBPAGE
        initialisation_status = build_index()
        initialisation_status = initialisation_status and initialise_chat_engine()
        if initialisation_status:
            show_status_message(
                message=f"**Done** ingesting web page(s) from {', '.join([f'[{url}]({url})' for url in urls_list])}. You can now chat with the AI.",
                colour="success",
            )
    except Exception as e:
        show_status_message(
            message=f"**Error** ingesting web page(s).<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error ingesting web page(s). {str(e)}")


@task
async def ingest_pdfurl_data():
    """Ingest the PDF (URL) results."""
    global ingested_documents
    try:
        ingested_documents.value = []
        show_status_message(
            message=f"**Fetching** PDF [{pdf_url.value}]({pdf_url.value})."
        )
        ingested_documents.value = load_remote_pdf_data(
            pdf_url=pdf_url.value,
        )
        last_ingested_data_source.value = constants.SOURCE_TYPE_PDF
        if len(ingested_documents.value) == 0:
            raise ValueError("No text extracted from the PDF.")
        initialisation_status = build_index()
        initialisation_status = initialisation_status and initialise_chat_engine()
        if initialisation_status:
            show_status_message(
                message=f"**Done** ingesting PDF from [{pdf_url.value}]({pdf_url.value}). You can now chat with the AI.",
                colour="success",
            )
    except Exception as e:
        show_status_message(
            message=f"**Error** ingesting PDF (URL).<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error ingesting PDF (URL). {str(e)}")


@task
async def load_existing_indices():
    """Load the existing indices from the storage context."""
    try:
        initialisation_status: bool = False
        if global_state.global_llamaindex_storage_context.value is not None:
            index_ids = existing_indices.value.split(" ")
            if len(index_ids) == 0:
                raise ValueError("No index IDs found in the input.")
            elif len(index_ids) == 1:
                show_status_message(
                    message=f"**Loading** index with ID _{index_ids[0]}_.",
                    timeout=0,
                )
                global_state.global_knowledge_graph_index.value = load_index_from_storage(
                    storage_context=global_state.global_llamaindex_storage_context.value,
                    index_id=index_ids[0],
                )
            else:
                raise NotImplementedError(
                    "Loading multiple indices is not yet supported."
                )
                # Check: https://github.com/run-llama/llama_index/discussions/13233
                show_status_message(
                    message=f"""
                    **Loading** {len(index_ids)} indices.

                    {', '.join([f'_{index_id}_' for index_id in index_ids])}
                    """,
                    timeout=0,
                )
                load_indices_from_storage(
                    storage_context=global_state.global_llamaindex_storage_context.value,
                    index_ids=index_ids,
                )
            last_ingested_data_source.value = constants.SOURCE_TYPE_INDICES
            initialisation_status = initialise_chat_engine()
            if initialisation_status:
                show_status_message(
                    message=f"**Done** loading indices(s) {', '.join([f'_{index_id}_' for index_id in index_ids])} from storage. You can now chat with the AI.",
                    colour="success",
                )
        else:
            raise ValueError("Storage context is not available.")
    except Exception as e:
        show_status_message(
            message=f"**Error** loading indice(s).<br/>{str(e)}",
            colour="error",
            timeout=8,
        )
        logger.error(f"Error loading indice(s). {str(e)}")


def load_wikipedia_languages():
    """Load the list of supported Wikipedia languages."""
    if len(wikipedia_languages.value) == 0:
        show_status_message(message="**Initialising** Wikipedia languages.")
        wp_lang_dict = wikipedia.languages()
        wikipedia_languages.value = list(
            {"language": f"{v[1]} ({v[0]})", "code": v[0]} for v in wp_lang_dict.items()
        )


@solara.component
def ExistingIndicesSourceComponent():
    """Component for loading existing indices."""

    with solara.Card(
        "Source indices",
        subtitle="""
                Enter the index ID for the index that you wish to load. In order to load multiple indices, provide multiple index IDs separated by space.
                """,
        elevation=0,
    ):
        with solara.Column(align="stretch"):
            solara.InputText(
                "Index ID",
                value=existing_indices,
                disabled=load_existing_indices.pending,
            )
            solara.Button(
                "Load",
                on_click=load_existing_indices,
                disabled=load_existing_indices.pending or not existing_indices.value,
            )
            solara.ProgressLinear(load_existing_indices.pending)


@solara.component
def WikipediaSourceComponent():
    """Component for selecting a Wikipedia article as a data source."""

    task_load_wikipedia_languages = solara.lab.use_task(load_wikipedia_languages)

    with solara.Card(
        "Wikipedia article(s) as data source",
        subtitle="""
                Select the Wikipedia language and input the article title or reference. To ingest multiple articles, separate them with [ __ ], all of which must be in the same selected Wikipedia language.
                """,
        elevation=0,
    ):
        with solara.ColumnsResponsive([2, 10]):
            with solara.Column(align="start"):
                rv.Select(
                    label="Language prefix",
                    multiple=False,
                    loading=task_load_wikipedia_languages.pending,
                    items=wikipedia_languages.value,
                    item_text="language",
                    item_value="code",
                    v_model=wikipedia_language_prefix.value,
                    on_v_model=wikipedia_language_prefix.set,
                    disabled=ingest_wikipedia_data.pending,
                )
            with solara.Column(align="stretch"):
                solara.InputText(
                    "Article",
                    value=wikipedia_article,
                    disabled=(
                        ingest_wikipedia_data.pending
                        or task_load_wikipedia_languages.pending
                    ),
                )
                solara.Button(
                    "Ingest",
                    on_click=ingest_wikipedia_data,
                    disabled=(
                        ingest_wikipedia_data.pending
                        or task_load_wikipedia_languages.pending
                    )
                    or not wikipedia_article.value,
                )
                solara.ProgressLinear(ingest_wikipedia_data.pending)
                if (
                    last_ingested_data_source.value == constants.SOURCE_TYPE_WIKIPEDIA
                    and len(ingested_documents.value) > 0
                ):
                    with rv.ExpansionPanels(
                        inset=True,
                        hover=True,
                        value=0 if len(ingested_documents.value) == 1 else None,
                    ):
                        solara.Text(
                            f"Fetched {len(ingested_documents.value)} Wikipedia {'articles' if len(ingested_documents.value) > 1 else 'article'}"
                        )
                        for doc in ingested_documents.value:
                            wp = wikipedia.page(pageid=doc.doc_id)
                            with rv.ExpansionPanel():
                                with rv.ExpansionPanelHeader():
                                    solara.Markdown(
                                        f"**Article ID**: {doc.doc_id} **Title**: {wp.title}"
                                    )
                                with rv.ExpansionPanelContent():
                                    solara.Markdown("### Article summary")
                                    solara.Text(wp.summary)
                                    solara.Markdown(
                                        f"<a href='{wp.url}' target='_blank'>Read the original article (opens in a new browser tab or window).</a>"
                                    )


@solara.component
def ArxivSourceComponent():
    """Component for selecting an arXiv article or search as data source."""

    with solara.Card(
        "Source article or search query",
        subtitle="""
                Input a specific article or a search query for arXiv.
                """,
        elevation=0,
    ):
        with solara.Column(align="stretch"):
            solara.InputText(
                "Article or search query",
                value=arxiv_query,
                disabled=ingest_arxiv_data.pending,
            )
            solara.Button(
                "Ingest",
                on_click=ingest_arxiv_data,
                disabled=ingest_arxiv_data.pending or not arxiv_query.value,
            )
        solara.ProgressLinear(ingest_arxiv_data.pending)
        if (
            last_ingested_data_source.value == constants.SOURCE_TYPE_ARXIV
            and len(ingested_documents.value) > 0
        ):
            with rv.ExpansionPanels(
                inset=True,
                hover=True,
                value=0 if len(ingested_documents.value) == 1 else None,
            ):
                solara.Text(
                    f"Fetched {len(ingested_documents.value)} {'pages' if len(ingested_documents.value) > 1 else 'page'} of PDF as a result of the arXiv query `{arxiv_query.value}`."
                )
                for doc in ingested_documents.value:
                    with rv.ExpansionPanel():
                        with rv.ExpansionPanelHeader():
                            solara.Markdown(f"**PDF page**: {doc.doc_id}")
                        with rv.ExpansionPanelContent():
                            with solara.Column():
                                solara.Markdown(f"{doc.text}")


@solara.component
def PubmedSourceComponent():
    """Component for selecting a Pubmed search as data source."""

    with solara.Card(
        "Source search query",
        subtitle="""
                Input a search query for Pubmed.
                """,
        elevation=0,
    ):
        with solara.Column(align="stretch"):
            solara.InputText(
                "Search query",
                value=pubmed_query,
                disabled=ingest_pubmed_data.pending,
            )
            solara.Button(
                "Ingest",
                on_click=ingest_pubmed_data,
                disabled=ingest_pubmed_data.pending or not pubmed_query.value,
            )
        solara.ProgressLinear(ingest_pubmed_data.pending)
        if (
            last_ingested_data_source.value == constants.SOURCE_TYPE_PUBMED
            and len(ingested_documents.value) > 0
        ):
            with rv.ExpansionPanels(
                inset=True,
                hover=True,
                value=0 if len(ingested_documents.value) == 1 else None,
            ):
                solara.Text(
                    f"Fetched {len(ingested_documents.value)} {'articles' if len(ingested_documents.value) > 1 else 'article'} as a result of the Pubmed query `{pubmed_query.value}`."
                )
                for doc in ingested_documents.value:
                    with rv.ExpansionPanel():
                        with rv.ExpansionPanelHeader():
                            solara.Markdown(f"**Article**: {doc.doc_id}")
                        with rv.ExpansionPanelContent():
                            with solara.Column():
                                solara.Markdown(f"{doc.text}")


@solara.component
def WebpageSourceComponent():
    """Component for selecting a web page as a data source."""

    with solara.Card(
        "Source web page",
        subtitle="""
                Enter the URL of a web page. You can enter multiple URLs, separated by space. If any URL contains spaces, enter a url-encoded version of that URL.
                """,
        elevation=0,
    ):
        with solara.ColumnsResponsive([2, 10]):
            with solara.Column(align="start"):
                solara.Select(
                    label="Web page reader",
                    values=constants.LIST_OF_WEBPAGE_READERS,
                    value=webpage_reader,
                    disabled=ingest_webpage_data.pending,
                )
            with solara.Column(align="stretch"):
                solara.InputText(
                    "Web page URL",
                    value=webpage_url,
                    disabled=ingest_webpage_data.pending,
                )
                solara.Button(
                    "Ingest",
                    on_click=ingest_webpage_data,
                    disabled=ingest_webpage_data.pending or not webpage_url.value,
                )
                solara.ProgressLinear(ingest_webpage_data.pending)
                if (
                    last_ingested_data_source.value == constants.SOURCE_TYPE_WEBPAGE
                    and len(ingested_documents.value) > 0
                ):
                    with rv.ExpansionPanels(
                        inset=True,
                        hover=True,
                        value=0 if len(ingested_documents.value) == 1 else None,
                    ):
                        solara.Text(
                            f"Fetched {len(ingested_documents.value)} web {'pages' if len(ingested_documents.value) > 1 else 'page'}"
                        )
                        for doc in ingested_documents.value:
                            # Fetch the URL
                            grabber = LinkGrabber(
                                initial_timeout=20,
                                maxsize=10485760,
                                receive_timeout=30,
                                chunk_size=4096,
                            )
                            content, url = grabber.get_content(url=doc.doc_id)
                            link = Link(url=url, content=content)
                            url_preview = LinkPreview(link=link, parser="lxml")
                            with rv.ExpansionPanel():
                                with rv.ExpansionPanelHeader():
                                    solara.Markdown(f"**URL**: {doc.doc_id}")
                                with rv.ExpansionPanelContent():
                                    with solara.Columns([3, 9]):
                                        with solara.Column():
                                            if str(url_preview.image).startswith(
                                                "http"
                                            ):
                                                solara.Image(image=url_preview.image)
                                            elif str(url_preview.image).startswith(
                                                "data"
                                            ):
                                                base64data = url_preview.image.split(
                                                    ","
                                                )[1]
                                                solara.Image(
                                                    image=base64.b64decode(base64data)
                                                )
                                        with solara.Column():
                                            solara.Markdown(f"### {url_preview.title}")
                                            solara.Markdown(
                                                f"_{url_preview.description}_"
                                            )
                                            solara.Markdown(
                                                f"[{doc.doc_id}]({doc.doc_id})"
                                            )


@solara.component
def PDFURLSourceComponent():
    """Component for selecting a PDF (URL) as data source."""

    with solara.Card(
        "Source PDF (URL)",
        subtitle="""
                Input a URL to a PDF.
                """,
        elevation=0,
    ):
        with solara.Column(align="stretch"):
            solara.InputText(
                "PDF URL",
                value=pdf_url,
                disabled=ingest_pdfurl_data.pending,
            )
            solara.Button(
                "Ingest",
                on_click=ingest_pdfurl_data,
                disabled=ingest_pdfurl_data.pending or not pdf_url.value,
            )
            solara.ProgressLinear(ingest_pdfurl_data.pending)
            if (
                last_ingested_data_source.value == constants.SOURCE_TYPE_PDF
                and len(ingested_documents.value) > 0
            ):
                with rv.ExpansionPanels(
                    inset=True,
                    hover=True,
                    value=0 if len(ingested_documents.value) == 1 else None,
                ):
                    solara.Text(
                        f"Fetched {len(ingested_documents.value)} {'pages' if len(ingested_documents.value) > 1 else 'page'} from the PDF at {pdf_url.value}."
                    )
                    for doc in ingested_documents.value:
                        with rv.ExpansionPanel():
                            with rv.ExpansionPanelHeader():
                                solara.Markdown(f"**Page**: {doc.doc_id}")
                            with rv.ExpansionPanelContent():
                                with solara.Column():
                                    solara.Markdown(f"{doc.text}")


@solara.component
def Page():
    """Main settings page."""
    # Remove the "This website runs on Solara" message
    solara.Style(constants.UI_SOLARA_NOTICE_REMOVE)
    global_state.initialise_default_settings()

    with solara.Head():
        solara.Title("Data ingestion")

    with solara.AppBarTitle():
        solara.Markdown("# Ingest data", style={"color": "#FFFFFF"})

    with solara.AppBar():
        solara.lab.ThemeToggle()

    with rv.Snackbar(
        top=True,
        right=True,
        # timeout=status_message_timeout.value,
        timeout=0,
        multi_line=True,
        color=status_message_colour.value,
        v_model=status_message_show.value,
    ):
        solara.Markdown(f"{status_message.value}")
    with solara.lab.Tabs(vertical=True, grow=False, lazy=True):
        with solara.lab.Tab("Existing indices", icon_name="mdi-database"):
            ExistingIndicesSourceComponent()
        with solara.lab.Tab("Web page", icon_name="mdi-web"):
            WebpageSourceComponent()
        with solara.lab.Tab("PDF (URL)", icon_name="mdi-file-pdf-box"):
            PDFURLSourceComponent()
        with solara.lab.Tab("Wikipedia", icon_name="mdi-bookshelf"):
            WikipediaSourceComponent()
        with solara.lab.Tab("arXiv", icon_name="mdi-book-open"):
            ArxivSourceComponent()
        with solara.lab.Tab("Pubmed", icon_name="mdi-book"):
            PubmedSourceComponent()
