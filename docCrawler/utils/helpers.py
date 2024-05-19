import json
from aiohttp import ClientSession
import asyncio
from bs4 import BeautifulSoup
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from duckduckgo_search import DDGS
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from twisted.internet import reactor
from docCrawler.spiders import DocSpider
import os
from typing import Optional
from docCrawler.utils.logger import info_logger as logger
from docCrawler.constants import PROMPT

class Crawler:
    @staticmethod
    def start(start_url, max_depth=1):
        """
        Start the crawler asynchronously using Twisted with given parameters.

        Parameters:
            start_url (str): The url you want as a starting point for your docs.
            allowed_domain (str): The domain you want to crawl.
            max_depth (int): The maximum depth you want to crawl.

        Returns:
            twisted.internet.defer.Deferred: A deferred object representing the asynchronous crawling process.
        """
        logger.info(f"Starting the crawler with start_url: {start_url} and max_depth: {max_depth}")

        # Configure logging
        configure_logging()

        # Create a CrawlerRunner
        runner = CrawlerRunner()

        # Start the spider with the given parameters
        deferred = runner.crawl(DocSpider, start_url=start_url, max_depth=max_depth)

        # Add a callback to stop the Twisted reactor once crawling is completed
        deferred.addCallback(Crawler.stop_reactor)


        return deferred

    @staticmethod
    def stop_reactor(_):
        """
        Callback function to stop the Twisted reactor when crawling is completed.

        Parameters:
            _: This parameter is not used.

        Returns:
            None
        """
        logger.info("Crawling completed. Stopping the reactor.")
        reactor.stop()


class ArticleFetcher:
    MAX_CONCURRENT_REQUESTS = 5  # Adjust as needed

    @staticmethod
    async def fetch_article_summary(semaphore, session, url):
        """
        Fetch the full text of an article from the given URL.

        Parameters:
            semaphore: A semaphore to limit the number of concurrent requests.
            session: A ClientSession object for making HTTP requests.
            url (str): The URL of the article.

        Returns:
            str: The full text of the article.
        """

        logger.info(f"Fetching article summary for {url}")
        async with semaphore:
            try:
                async with session.get(url) as response:
                    html = await response.text()

                soup = BeautifulSoup(html, "html.parser")

                # Find all the text on the page
                text = soup.get_text()
            except Exception as e:
                print(f"Failed to fetch article summary for {url}: {str(e)}")
                return None
            return text

    @staticmethod
    async def process_links(semaphore, links):
        """
        Process a list of links asynchronously to fetch the full text of articles.

        Parameters:
            semaphore: A semaphore to limit the number of concurrent requests.
            links (list): A list of URLs to process.

        Returns:
            list: A list of full text articles.
        """

        logger.info(f"Processing {len(links)} links asynchronously")
        async with ClientSession() as session:
            tasks = []
            for link in links:
                try:
                    task = asyncio.create_task(
                        ArticleFetcher.fetch_article_summary(semaphore, session, link)
                    )
                    tasks.append(task)
                except Exception as e:
                    print(f"Failed to generate summary for {link}: {str(e)}")

            return await asyncio.gather(*tasks)

    def get_full_text(self, path: Optional[str] = None):
        """
        Fetch the full text of articles from the given URLs.

        Parameters:
            path (str): Path to the JSON dir containing the crawled data.

        Returns:
            None
        """

        logger.info("Fetching full text articles")
        path = path or os.getenv("DATA_DIR", "data/")
        path = os.path.join(os.getcwd(), path)
        with open(os.path.join(path, "output_links.json"), "r") as json_file:
            crawled_data = json.load(json_file)

        summary_data_final = {}
        summary_data = []

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        for url in crawled_data:
            links = crawled_data[url]

            # Process links asynchronously with semaphore
            results = asyncio.run(self.process_links(semaphore, links))

            for text in results:
                if text:
                    summary_data.append({"text": text})

        logger.info("Fetching full text articles completed. Writing to OUTPUT file")

        summary_data_final["chatwithdocs"] = tuple(summary_data)


        with open(os.path.join(path, "summary_output.json"), "w") as json_file:
            json.dump(summary_data_final, json_file, indent=4)


class TextSplitter:
    @staticmethod
    def split_docs(documents, chunk_size=500, chunk_overlap=20):
        """
        Split a list of documents into smaller chunks of text.

        Parameters:
            documents (list): A list of text documents.
            chunk_size (int): Maximum size of each chunk (default: 500).
            chunk_overlap (int): Number of characters to overlap between adjacent chunks (default: 20).

        Returns:
            list: A list of split documents.
        """

        logger.info(f"Splitting documents into chunks with size {chunk_size} and overlap {chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        return docs


class DataLoader:
    @staticmethod
    def load_data(path: Optional[str] = None, jq_schema: str = ".chatwithdocs[].text"):
        """
        Load text data from a JSON file using a JSON Loader.

        Parameters:
            path (str): Path to the JSON file containing the data.

        Returns:
            list: A list containing the loaded text data.
        """

        logger.info("Loading data from JSON file")
        path = path or os.getenv("DATA_DIR", "data/")
        current_dir = os.path.join(os.getcwd(), path)
        file_path = os.path.join(current_dir, "summary_output.json")
        loader = JSONLoader(file_path=file_path, jq_schema=jq_schema)
        data = loader.load()
        return data


class DatabaseInitializer:
    @staticmethod
    def init_database(docs, model_name="all-MiniLM-L6-v2"):
        """
        Initialize a vector database using the provided documents and model.

        Parameters:
            docs (list): A list of text documents.
            model_name (str): Name of the Sentence Transformer model (default: "all-MiniLM-L6-v2").

        Returns:
            Chroma: The initialized text database.
        """

        logger.info(f"Initializing database with {len(docs)} documents using model {model_name}")
        embedding = SentenceTransformerEmbeddings(model_name=model_name)
        split_docs_chunked = DatabaseInitializer.split_list(docs, 1000)
        path = os.path.join(os.getcwd(), os.getenv("DATA_DIR", "data/"))

        for split_docs_chunk in split_docs_chunked:
            vectordb = Chroma.from_documents(
                documents=split_docs_chunk,
                embedding=embedding,
                persist_directory=os.path.join(path, "chroma_db")
            )
            vectordb.persist()

    @staticmethod
    def split_list(lst, n):
        """
        Split a list into chunks of size n.

        Args:
            lst (list): The list to split.
            n (int): The size of each chunk.

        Returns:
            list: A list of chunks.
        """
        return [lst[i : i + n] for i in range(0, len(lst), n)]


class SearchEngine:
    @staticmethod
    def search(query):
        """
        Searches for the given query using the DuckDuckGo search engine.

        Args:
            query (str): The search query.

        Returns:
            str: The first result returned by the search engine.
        """

        logger.info(f"Searching for query: {query}")
        with DDGS() as ddgs:
            for r in ddgs.text(query):
                return r


class SimilaritySearch:
    @staticmethod
    def search_similarity(
        query,
        openai_api_key="your openai api key",
        embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        openai_model="gpt-3.5-turbo-16k",
    ):
        """
        Perform a similarity search on the text database using the given query.

        Parameters:
            query (str): The query for the similarity search.
            embeddings (SentenceTransformerEmbeddings): An instance of SentenceTransformerEmbeddings
                with the desired model for generating text embeddings.
            openai_api_key (str): Your OpenAI API key for using the OpenAI language model.
            openai_model (str): The name of the OpenAI language model to use.

        Returns:
            str: The search result as an answer.
        """

        logger.info(f"Performing similarity search for query: {query}")
        path = os.path.join(os.getcwd(), os.getenv("DATA_DIR", "data/"))
        database = Chroma(
            persist_directory=os.path.join(path, "chroma_db"),
            embedding_function=embeddings,
        )
        query = query.lower()

        tools = [
            Tool(
                name="simsearch",
                func=database.similarity_search,
                description="useful for when you need to search documentation for a specific topic using vector database,use the query parameter to specify the input",
            ),
            Tool(
                name="search",
                func=SearchEngine.search,
                description="useful for when you don't get the answer you want from the similarity search,use the query parameter to specify the input",
            ),
        ]

        logger.info("Initializing the agent")
        model_name = openai_model
        llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)
        agent = initialize_agent(
            tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
        )
        answer = agent.run(PROMPT+ query)
        return answer
