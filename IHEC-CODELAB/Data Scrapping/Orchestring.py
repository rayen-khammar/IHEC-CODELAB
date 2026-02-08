import scrapping

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
    
    # Save in all formats
scraper.save_for_nlp_text('nlp_corpus.txt')