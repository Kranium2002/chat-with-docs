import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import json
from urllib.parse import urlparse
import os
from docCrawler.utils import logger

class DocSpider(CrawlSpider):
    """
    A spider class for crawling and extracting links from web documents.

    Attributes:
        name (str): The name of the spider.
        allowed_domains (list): The list of allowed domains for crawling.
        max_depth (int): The maximum depth for crawling.
        links (set): A set to store the extracted links.

    Methods:
        __init__(self, start_url: str, *args, **kwargs): Initializes the spider with a start URL.
        parse_item(self, response): Parses the response and extracts links.
        closed(self, reason): Callback method called when the spider is closed.
        save_to_json(self, data): Saves the extracted links to a JSON file.
    """

    name = "DocSpider"
    allowed_domains = []
    max_depth = 1  # Set the maximum depth to 1
    links = set()

    def __init__(self, start_url: str, *args, **kwargs):
        """
        Initializes the spider with a start URL.

        Args:
            start_url (str): The start URL for crawling.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """

        logger.info(f"Starting the spider with the URL: {start_url}")
        super(DocSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]

    rules = (
        Rule(LinkExtractor(allow=(), unique=True), callback="parse_item", follow=True),
    )

    def parse_item(self, response):
        """
        Parses the response and extracts links.

        Args:
            response (scrapy.http.Response): The response object.

        Returns:
            A generator of scrapy.Request objects for further crawling.
        """

        logger.info(f"Extracting links from: {response.url}")
        new_links = {
            link.url
            for link in LinkExtractor().extract_links(response)
            if urlparse(self.start_urls[0]).netloc in link.url
        }
        self.links.update(new_links)

        depth = response.meta.get("depth", 0)
        if depth < self.max_depth:
            for next_link in new_links:
                yield scrapy.Request(
                    next_link, callback=self.parse_item, meta={"depth": depth + 1}
                )

    def closed(self, reason):
        """
        Callback method called when the spider is closed.

        Args:
            reason (str): The reason for closing the spider.
        """

        logger.info(f"Spider closed: {reason}")
        data = {"links": list(self.links)}
        self.save_to_json(data)
        return

    def save_to_json(self, data):
        """
        Saves the extracted links to a JSON file.

        Args:
            data (dict): The data to be saved.

        Returns:
            None
        """

        logger.info("Saving the extracted links to a JSON file")
        path = os.path.join(os.getcwd(), os.getenv("DATA_DIR", "data/"))
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "output_links.json"), "w") as json_file:
            json.dump(data, json_file, indent=4)
