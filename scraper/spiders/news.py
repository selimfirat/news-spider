import scrapy
from scrapy import Request
from datetime import datetime

from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.spiders import Rule
import datetime

from news_entry import NewsEntry
from newsplease import NewsPlease
from urlparse import urlparse


class NewsSpider(scrapy.Spider):

    name = "news"
    custom_settings = {
        "ITEM_PIPELINES": {
            'data_store_pipeline.DataStorePipeline': 300,
        }
    }

    start_urls = []
    allowed_domains = ['takvim.com.tr']

    def __init__(self, target="https://www.takvim.com.tr/", *args, **kwargs):
        parsed_uri = urlparse(target)
        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

        self.allowed_domains.append(domain)
        self.start_urls.append(target)
        super(NewsSpider, self).__init__(*args, **kwargs)

    def parse(self, response):
        now = datetime.datetime.now()

        article = NewsPlease.from_html(response.text, response.url)
        if article.date_publish is not None and article.text is not None:
            yield NewsEntry(
                full_url=response.url,
                source_domain = article.source_domain,
                date_publish = article.date_publish,
                date_download = str(now),
                title = article.title,
                description = article.description,
                text = article.text
            )

        for link in LxmlLinkExtractor(allow=self.allowed_domains).extract_links(response):
            yield Request(link.url, self.parse)
