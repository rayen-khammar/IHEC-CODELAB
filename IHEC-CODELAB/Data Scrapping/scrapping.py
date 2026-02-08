import requests
from bs4 import BeautifulSoup
import csv
import json
import time
from datetime import datetime
import os

class MultiWebsiteScraper:
    def __init__(self, websites, max_pages_per_site=5):
        self.websites = websites
        self.max_pages_per_site = max_pages_per_site
        self.articles = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def get_article_links_from_page(self, website, page_num=1):
        """Extract all article links from a listing page"""
        try:
            base_url = website['base_url']
            articles_path = website.get('articles_path', '/')
            
            if page_num == 1:
                url = base_url.rstrip('/') + articles_path
            else:
                url = f"{base_url.rstrip('/')}{articles_path}?page={page_num}"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            article_links = []
            
            for article in soup.find_all('article'):
                link = article.find('a', href=True)
                if link:
                    article_links.append(link['href'])
            
            if not article_links:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(pattern in href for pattern in ['/article', '/news', '/actualites', '/post']):
                        article_links.append(href)
            
            if not article_links:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/') and href.count('/') >= 2:
                        article_links.append(href)
            
            article_links = [self.make_absolute_url(link, base_url) for link in article_links]
            article_links = [link for link in article_links if base_url in link]
            article_links = list(set(article_links))
            
            return article_links
            
        except Exception as e:
            return []
    
    def make_absolute_url(self, url, base_url):
        """Convert relative URL to absolute URL"""
        if url.startswith('http'):
            return url
        elif url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return base_url.rstrip('/') + url
        else:
            return base_url.rstrip('/') + '/' + url
    
    def scrape_article(self, article_url, source_name):
        """Scrape a single article"""
        try:
            response = requests.get(article_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            article_data = {}
            
            # Extract title
            title = None
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text(strip=True)
            else:
                title_meta = soup.find('meta', property='og:title')
                if title_meta:
                    title = title_meta.get('content')
                elif soup.title:
                    title = soup.title.string
            article_data['title'] = title if title else ''
            
            # Extract date
            date = None
            time_tags = soup.find_all('time')
            if time_tags:
                date = time_tags[0].get('datetime') or time_tags[0].get_text(strip=True)
            if not date:
                date_meta = soup.find('meta', property='article:published_time')
                if date_meta:
                    date = date_meta.get('content')
            article_data['date'] = date if date else ''
            
            # Extract category
            category = None
            category_span = soup.find('span', class_=['category', 'post-category'])
            if category_span:
                category = category_span.get_text(strip=True)
            article_data['category'] = category if category else ''
            
            # Source
            article_data['source'] = source_name
            
            # Extract main content
            content = ""
            
            article_tag = soup.find('article')
            if article_tag:
                for unwanted in article_tag(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()
                paragraphs = article_tag.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if not content:
                content_div = soup.find('div', class_=['content', 'article-content', 'post-content', 'entry-content'])
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            article_data['content'] = content if content else ''
            
            return article_data
            
        except Exception as e:
            return None
    
    def scrape_all_websites(self):
        """Main scraping function for all websites"""
        for website in self.websites:
            all_article_links = []
            
            for page_num in range(1, self.max_pages_per_site + 1):
                article_links = self.get_article_links_from_page(website, page_num)
                
                if not article_links and page_num == 1:
                    break
                
                all_article_links.extend(article_links)
                time.sleep(2)
            
            all_article_links = list(set(all_article_links))
            
            if all_article_links:
                for article_url in all_article_links:
                    article_data = self.scrape_article(article_url, website['name'])
                    
                    if article_data and article_data.get('content'):
                        self.articles.append(article_data)
                    
                    time.sleep(1.5)
        
        return self.articles
    
    def save_to_csv(self, filename='articles_data.csv'):
        """Save to CSV"""
        if not self.articles:
            return None
        
        output_dir = 'scraped_data'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        fieldnames = ['title', 'date', 'category', 'source', 'content']
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.articles)
        
        return filepath
    
    def save_for_nlp_text(self, filename='nlp_corpus.txt'):
        """Save as plain text - BEST for most NLP tasks"""
        output_dir = 'scraped_data'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in self.articles:
                content = article.get('content', '').strip()
                if content:
                    f.write(content)
                    f.write('\n\n')
        
        return filepath
    
    def save_for_nlp_jsonl(self, filename='nlp_data.jsonl'):
        """Save as JSONL - BEST for NLP with metadata"""
        output_dir = 'scraped_data'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in self.articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        
        return filepath


#main 

WEBSITES = [
        {
            'name': 'irbe7',
            'base_url': 'https://irbe7.com',
            'articles_path': '/actualites/articles'
        },
        {
            'name': 'lapresse',
            'base_url': 'https://www.lapresse.tn/',
            'articles_path': '/category/actualites/'
        },
        {
            'name': 'bourse.tn',
            'base_url': 'http://www.bourse.tn',
            'articles_path': '/actualites'
        },
        {
            'name': 'bvmt',
            'base_url': 'https://www.bvmt.com.tn',
            'articles_path': '/avis-decisions'
        }
    ]
    
MAX_PAGES = 3
    
scraper = MultiWebsiteScraper(WEBSITES, max_pages_per_site=MAX_PAGES)
scraper.scrape_all_websites()
    
    # Save results
scraper.save_for_nlp_text('nlp_corpus.txt')
scraper.save_for_nlp_jsonl('nlp_data.jsonl')