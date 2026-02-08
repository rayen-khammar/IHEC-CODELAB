import re
import json
from collections import Counter
from textblob import TextBlob

class FinancialSentimentAnalyzer:
    def __init__(self, corpus_file='scraped_data/nlp_corpus.txt', jsonl_file='scraped_data/nlp_data.jsonl'):
        """Initialize financial sentiment analyzer for stock market analysis"""
        self.corpus_file = corpus_file
        self.jsonl_file = jsonl_file
        self.articles = []
        self.load_articles()
        
        # Expanded financial keywords with weights for both French and Arabic
        # Weight scale: 1.0 (weak signal) to 3.0 (strong signal)
        
        self.positive_indicators = {
            'growth': {
                # French terms
                'croissance': 2.0, 'augmentation': 1.5, 'hausse': 1.8, 'progression': 1.7,
                'amélioration': 1.6, 'bénéfice': 2.5, 'profit': 2.5, 'gain': 2.0,
                'succès': 1.8, 'performance': 1.5, 'développement': 1.7, 'expansion': 2.0,
                'investissement': 1.8, 'dividende': 2.2, 'rentabilité': 2.3, 'rendement': 2.0,
                'innovation': 1.6, 'acquisition': 2.0, 'partenariat': 1.7, 'contrat': 1.5,
                'accord': 1.5, 'revenus': 2.0, "chiffre d'affaires": 2.5, 'résultats positifs': 2.2,
                'record': 2.5, 'boom': 2.3, 'excédent': 2.0, 'surplus': 1.8, 'valorisation': 2.0,
                'capitalisation': 1.8, 'productivité': 1.7, 'efficacité': 1.6, 'optimisation': 1.5,
                'rebond': 2.0, 'reprise': 2.2, 'relance': 2.0, 'essor': 2.1, 'prospérité': 2.3,
                'dynamisme': 1.8, 'vitalité': 1.7, 'compétitivité': 1.9, 'rentable': 2.2,
                'lucratif': 2.3, 'florissant': 2.4, 'fructueux': 2.2, 'porteur': 1.9,
                # Arabic terms
                'نمو': 2.0, 'زيادة': 1.5, 'ارتفاع': 1.8, 'تحسن': 1.6, 'تطور': 1.7,
                'ربح': 2.5, 'أرباح': 2.5, 'مكاسب': 2.0, 'نجاح': 1.8, 'أداء': 1.5,
                'استثمار': 1.8, 'توسع': 2.0, 'عائد': 2.0, 'مردودية': 2.3, 'إيرادات': 2.0,
                'رقم أعمال': 2.5, 'نتائج إيجابية': 2.2, 'شراكة': 1.7, 'عقد': 1.5,
                'اتفاقية': 1.5, 'ابتكار': 1.6, 'استحواذ': 2.0, 'فائض': 1.8, 'قيمة': 1.8,
                'ازدهار': 2.3, 'انتعاش': 2.2, 'رواج': 2.1, 'تقدم': 1.7, 'تميز': 1.9,
                'كفاءة': 1.6, 'فعالية': 1.6, 'إنتاجية': 1.7, 'تنافسية': 1.9,
            },
            'strong': {
                # French terms
                'solide': 2.0, 'robuste': 2.1, 'forte': 1.8, 'fort': 1.8, 'stable': 1.9,
                'performant': 2.0, 'leader': 2.2, 'dominant': 2.0, 'compétitif': 1.8,
                'stratégique': 1.6, 'efficace': 1.5, 'réussi': 1.9, 'vigoureux': 2.0,
                'puissant': 2.1, 'consolidé': 1.9, 'renforcé': 1.8, 'optimiste': 1.7,
                'prometteur': 2.0, 'encourageant': 1.8, 'favorable': 1.9, 'avantageux': 2.0,
                'bénéfique': 1.9, 'profitable': 2.2, 'sain': 1.8, 'équilibré': 1.7,
                'diversifié': 1.6, 'résistant': 1.8, 'résilient': 1.9, 'pérenne': 1.8,
                # Arabic terms
                'قوي': 1.8, 'متين': 2.1, 'صلب': 2.0, 'مستقر': 1.9, 'فعال': 1.5,
                'رائد': 2.2, 'مهيمن': 2.0, 'تنافسي': 1.8, 'استراتيجي': 1.6, 'ناجح': 1.9,
                'واعد': 2.0, 'مشجع': 1.8, 'إيجابي': 1.7, 'مفيد': 1.9, 'مربح': 2.2,
                'صحي': 1.8, 'متوازن': 1.7, 'متنوع': 1.6, 'صامد': 1.8, 'مستدام': 1.8,
            },
            'market': {
                # French terms
                'marché porteur': 2.3, 'demande forte': 2.2, 'opportunité': 2.0, 'potentiel': 1.9,
                'perspectives favorables': 2.1, 'tendance haussière': 2.5, 'dynamique positive': 2.2,
                'croissance du marché': 2.3, 'part de marché': 1.8, 'leadership': 2.1,
                'position dominante': 2.2, 'avantage concurrentiel': 2.0, 'niche profitable': 2.1,
                'marché en expansion': 2.3, 'secteur en croissance': 2.2, 'demande croissante': 2.1,
                # Arabic terms
                'سوق واعد': 2.3, 'طلب قوي': 2.2, 'فرصة': 2.0, 'إمكانيات': 1.9,
                'توقعات إيجابية': 2.1, 'اتجاه صاعد': 2.5, 'نمو السوق': 2.3, 'حصة سوقية': 1.8,
                'ريادة': 2.1, 'مركز مهيمن': 2.2, 'ميزة تنافسية': 2.0, 'قطاع نامٍ': 2.2,
            }
        }
        
        self.negative_indicators = {
            'decline': {
                # French terms
                'baisse': 1.8, 'chute': 2.5, 'diminution': 1.7, 'recul': 1.9, 'perte': 2.5,
                'pertes': 2.5, 'déficit': 2.3, 'dette': 2.0, 'endettement': 2.2,
                'difficultés': 2.0, 'crise': 2.8, 'problème': 1.6, 'faillite': 3.0,
                'liquidation': 3.0, 'restructuration': 2.2, 'licenciement': 2.3, 'fermeture': 2.7,
                'retrait': 1.9, 'suspension': 2.5, 'sanction': 2.4, 'amende': 2.2,
                'scandale': 2.8, 'enquête': 2.0, 'contentieux': 2.1, 'dépréciation': 2.0,
                'dégradation': 2.2, 'effondrement': 2.9, 'krach': 3.0, 'récession': 2.7,
                'stagnation': 1.9, 'ralentissement': 1.8, 'contraction': 2.1, 'déclin': 2.3,
                'fléchissement': 1.8, 'régression': 2.2, 'affaiblissement': 2.0, 'érosion': 1.9,
                'détérioration': 2.2, 'corruption': 2.7, 'fraude': 2.9, 'illégal': 2.6,
                'violation': 2.4, 'litige': 2.1, 'conflit': 1.9, 'grève': 2.2,
                # Arabic terms
                'انخفاض': 1.8, 'تراجع': 1.9, 'خسارة': 2.5, 'خسائر': 2.5, 'عجز': 2.3,
                'دين': 2.0, 'مديونية': 2.2, 'صعوبات': 2.0, 'أزمة': 2.8, 'مشكلة': 1.6,
                'إفلاس': 3.0, 'تصفية': 3.0, 'إعادة هيكلة': 2.2, 'تسريح': 2.3, 'إغلاق': 2.7,
                'تعليق': 2.5, 'عقوبة': 2.4, 'غرامة': 2.2, 'فضيحة': 2.8, 'تحقيق': 2.0,
                'نزاع': 2.1, 'انهيار': 2.9, 'ركود': 2.7, 'تباطؤ': 1.8, 'تقلص': 2.1,
                'تدهور': 2.2, 'فساد': 2.7, 'احتيال': 2.9, 'انتهاك': 2.4, 'نزاع': 2.1,
                'إضراب': 2.2, 'ضعف': 1.8,
            },
            'weak': {
                # French terms
                'faible': 1.7, 'fragile': 2.0, 'vulnérable': 2.1, 'instable': 2.2,
                'risque': 1.9, 'risqué': 2.0, 'incertitude': 1.8, 'préoccupant': 2.0,
                'inquiétant': 2.1, 'négatif': 1.9, 'défavorable': 2.0, 'médiocre': 2.1,
                'insuffisant': 1.9, 'inadéquat': 1.8, 'limité': 1.6, 'décevant': 2.0,
                'pessimiste': 2.1, 'sombre': 2.0, 'incertain': 1.8, 'volatil': 1.9,
                'précaire': 2.2, 'critique': 2.3, 'alarmant': 2.4, 'dangereux': 2.3,
                # Arabic terms
                'ضعيف': 1.7, 'هش': 2.0, 'معرض للخطر': 2.1, 'غير مستقر': 2.2,
                'خطر': 1.9, 'محفوف بالمخاطر': 2.0, 'عدم يقين': 1.8, 'مقلق': 2.0,
                'سلبي': 1.9, 'غير مواتٍ': 2.0, 'متواضع': 2.1, 'غير كافٍ': 1.9,
                'محدود': 1.6, 'مخيب للآمال': 2.0, 'متشائم': 2.1, 'غامض': 2.0,
                'غير مؤكد': 1.8, 'متقلب': 1.9, 'حرج': 2.3, 'خطير': 2.3,
            },
            'market': {
                # French terms
                'concurrence accrue': 1.9, 'pression': 1.7, 'ralentissement': 1.8,
                'contraction': 2.1, 'perte de marché': 2.3, 'érosion des marges': 2.2,
                'baisse de la demande': 2.2, 'surcapacité': 2.0, 'saturation': 1.9,
                'guerre des prix': 2.1, 'marché en baisse': 2.3, 'secteur en difficulté': 2.4,
                'concurrence féroce': 2.0, 'pression concurrentielle': 1.9,
                # Arabic terms
                'منافسة شديدة': 1.9, 'ضغط': 1.7, 'تباطؤ': 1.8, 'انكماش': 2.1,
                'فقدان حصة سوقية': 2.3, 'انخفاض الطلب': 2.2, 'فائض': 2.0, 'تشبع': 1.9,
                'حرب أسعار': 2.1, 'سوق منخفض': 2.3, 'قطاع في صعوبة': 2.4,
            }
        }
        
        # Financial events with weights
        self.financial_events = {
            'very_positive': {
                # French
                'augmentation du capital': 2.8, 'introduction en bourse': 3.0, 'ipo': 3.0,
                'fusion avantageuse': 2.7, 'dividende exceptionnel': 2.6, "rachat d'actions": 2.5,
                'nouveau contrat majeur': 2.6, 'certification': 2.2, 'notation améliorée': 2.5,
                'partenariat stratégique': 2.4, 'acquisition stratégique': 2.7,
                'levée de fonds': 2.5, 'financement': 2.3, 'investissement majeur': 2.6,
                # Arabic
                'زيادة رأس المال': 2.8, 'طرح في البورصة': 3.0, 'اندماج مفيد': 2.7,
                'أرباح استثنائية': 2.6, 'عقد رئيسي جديد': 2.6, 'شهادة': 2.2,
                'تحسين التصنيف': 2.5, 'شراكة استراتيجية': 2.4, 'استحواذ استراتيجي': 2.7,
                'جمع أموال': 2.5, 'تمويل': 2.3, 'استثمار رئيسي': 2.6,
            },
            'positive': {
                # French
                'nouveau produit': 1.8, 'expansion': 2.0, "croissance du chiffre d'affaires": 2.3,
                'augmentation des bénéfices': 2.5, 'investissement': 1.8, 'innovation': 1.7,
                'lancement': 1.8, 'ouverture': 1.9, 'nomination': 1.5, 'promotion': 1.6,
                'récompense': 1.9, 'prix': 1.8, 'distinction': 1.8, 'reconnaissance': 1.7,
                # Arabic
                'منتج جديد': 1.8, 'توسع': 2.0, 'نمو رقم الأعمال': 2.3,
                'زيادة الأرباح': 2.5, 'استثمار': 1.8, 'ابتكار': 1.7, 'إطلاق': 1.8,
                'افتتاح': 1.9, 'تعيين': 1.5, 'ترقية': 1.6, 'جائزة': 1.8, 'تكريم': 1.7,
            },
            'negative': {
                # French
                'perte nette': 2.5, "baisse du chiffre d'affaires": 2.3, 'dette élevée': 2.2,
                'déficit': 2.3, 'restructuration': 2.2, 'plan social': 2.5,
                'abandon de projet': 2.1, 'retard': 1.8, 'rappel de produit': 2.4,
                'avertissement': 2.0, 'mise en garde': 2.0, 'démission': 2.1,
                'départ': 1.9, 'révision à la baisse': 2.3, 'provisions': 2.0,
                # Arabic
                'خسارة صافية': 2.5, 'انخفاض رقم الأعمال': 2.3, 'دين مرتفع': 2.2,
                'عجز': 2.3, 'إعادة هيكلة': 2.2, 'خطة اجتماعية': 2.5,
                'التخلي عن مشروع': 2.1, 'تأخير': 1.8, 'سحب منتج': 2.4,
                'تحذير': 2.0, 'استقالة': 2.1, 'مراجعة للأسفل': 2.3, 'مخصصات': 2.0,
            },
            'very_negative': {
                # French
                'faillite': 3.0, 'liquidation': 3.0, 'scandale': 2.9, 'fraude': 3.0,
                'défaut de paiement': 2.9, 'radiation': 2.8, 'suspension': 2.7,
                'procédure judiciaire': 2.6, 'condamnation': 2.8, 'tribunal': 2.5,
                'plainte': 2.4, 'accusation': 2.6, 'perquisition': 2.7,
                'saisie': 2.6, 'mise sous tutelle': 2.8, 'redressement judiciaire': 2.9,
                # Arabic
                'إفلاس': 3.0, 'تصفية': 3.0, 'فضيحة': 2.9, 'احتيال': 3.0,
                'تخلف عن السداد': 2.9, 'شطب': 2.8, 'تعليق': 2.7,
                'إجراء قضائي': 2.6, 'إدانة': 2.8, 'محكمة': 2.5,
                'شكوى': 2.4, 'اتهام': 2.6, 'تفتيش': 2.7, 'حجز': 2.6,
                'وضع تحت الوصاية': 2.8, 'تسوية قضائية': 2.9,
            }
        }
    
    def load_articles(self):
        """Load articles from JSONL file"""
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    self.articles.append(article)
        except FileNotFoundError:
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    articles_text = content.split('\n\n')
                    self.articles = [{'content': text, 'title': '', 'source': 'unknown'} 
                                   for text in articles_text if text.strip()]
            except FileNotFoundError:
                self.articles = []
    
    def find_enterprise_mentions(self, enterprise_name):
        """Find all articles mentioning the enterprise"""
        enterprise_name_lower = enterprise_name.lower()
        matching_articles = []
        
        for article in self.articles:
            content = article.get('content', '').lower()
            title = article.get('title', '').lower()
            
            if enterprise_name_lower in content or enterprise_name_lower in title:
                matching_articles.append(article)
        
        return matching_articles
    
    def detect_financial_indicators(self, text):
        """Detect positive and negative financial indicators with weights"""
        text_lower = text.lower()
        
        positive_score = 0.0
        negative_score = 0.0
        
        found_positive = {}
        found_negative = {}
        
        # Check positive indicators
        for category, keywords in self.positive_indicators.items():
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    positive_score += weight * count
                    found_positive[keyword] = {'weight': weight, 'count': count}
        
        # Check negative indicators
        for category, keywords in self.negative_indicators.items():
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    negative_score += weight * count
                    found_negative[keyword] = {'weight': weight, 'count': count}
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'found_positive': found_positive,
            'found_negative': found_negative
        }
    
    def detect_financial_events(self, text):
        """Detect major financial events with weights"""
        text_lower = text.lower()
        
        event_scores = {
            'very_positive': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'very_negative': 0.0
        }
        
        detected_events = {
            'very_positive': {},
            'positive': {},
            'negative': {},
            'very_negative': {}
        }
        
        for impact_level, events in self.financial_events.items():
            for event, weight in events.items():
                if event in text_lower:
                    count = text_lower.count(event)
                    event_scores[impact_level] += weight * count
                    detected_events[impact_level][event] = {'weight': weight, 'count': count}
        
        return {
            'event_scores': event_scores,
            'detected_events': detected_events
        }
    
    def extract_financial_numbers(self, text):
        """Extract financial figures (revenue, profit, etc.)"""
        patterns = {
            'revenue': r"chiffre d'affaires[:\s]+(\d+[,.]?\d*)\s*(millions?|milliards?|md|dt|dinars?)|رقم أعمال[:\s]+(\d+[,.]?\d*)",
            'profit': r'(?:bénéfice|profit|أرباح|ربح)[:\s]+(\d+[,.]?\d*)\s*(millions?|milliards?|md|dt|dinars?)?',
            'loss': r'(?:perte|خسارة)[:\s]+(\d+[,.]?\d*)\s*(millions?|milliards?|md|dt|dinars?)?',
            'percentage': r'(\d+[,.]?\d*)\s*%'
        }
        
        findings = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                findings[key] = matches
        
        return findings
    
    def calculate_sentiment_score(self, indicators, events, sentiment_polarity):
        """Calculate normalized sentiment score between -1 and 1"""
        
        # Base sentiment from TextBlob (already -1 to 1)
        base_score = sentiment_polarity * 0.3
        
        # Financial indicators contribution
        indicator_score = (indicators['positive_score'] - indicators['negative_score']) / 100.0
        
        # Major events contribution (weighted heavily)
        event_score = (
            events['event_scores']['very_positive'] * 1.5 +
            events['event_scores']['positive'] * 0.8 -
            events['event_scores']['negative'] * 0.8 -
            events['event_scores']['very_negative'] * 1.5
        ) / 50.0
        
        # Combine scores
        total_score = base_score + indicator_score + event_score
        
        # Normalize to -1 to +1 using tanh function
        normalized_score = max(-1.0, min(1.0, total_score))
        
        return normalized_score
    
    def interpret_sentiment(self, score):
        """Interpret the sentiment score"""
        if score >= 0.7:
            return 'Very Positive'
        elif score >= 0.3:
            return 'Positive'
        elif score >= 0.1:
            return 'Slightly Positive'
        elif score >= -0.1:
            return 'Neutral'
        elif score >= -0.3:
            return 'Slightly Negative'
        elif score >= -0.7:
            return 'Negative'
        else:
            return 'Very Negative'
    
    def analyze_enterprise_sentiment(self, enterprise_name):
        """Main sentiment analysis for enterprise"""
        
        articles = self.find_enterprise_mentions(enterprise_name)
        
        if not articles:
            return {
                'enterprise': enterprise_name,
                'articles_found': 0,
                'message': f'No articles found mentioning "{enterprise_name}"'
            }
        
        article_analyses = []
        total_score = 0.0
        
        for article in articles:
            content = article.get('content', '')
            title = article.get('title', '')
            full_text = f"{title} {content}"
            
            # Sentiment analysis using TextBlob
            try:
                blob = TextBlob(full_text)
                sentiment_polarity = blob.sentiment.polarity
            except:
                sentiment_polarity = 0.0
            
            # Financial indicators
            indicators = self.detect_financial_indicators(full_text)
            
            # Financial events
            events = self.detect_financial_events(full_text)
            
            # Financial numbers
            numbers = self.extract_financial_numbers(full_text)
            
            # Calculate sentiment score
            sentiment_score = self.calculate_sentiment_score(indicators, events, sentiment_polarity)
            total_score += sentiment_score
            
            # Interpretation
            interpretation = self.interpret_sentiment(sentiment_score)
            
            # Top indicators found
            top_positive = sorted(
                indicators['found_positive'].items(),
                key=lambda x: x[1]['weight'] * x[1]['count'],
                reverse=True
            )[:10]
            
            top_negative = sorted(
                indicators['found_negative'].items(),
                key=lambda x: x[1]['weight'] * x[1]['count'],
                reverse=True
            )[:10]
            
            article_analyses.append({
                'title': title,
                'source': article.get('source', 'unknown'),
                'date': article.get('date', ''),
                'sentiment_score': round(sentiment_score, 4),
                'sentiment_interpretation': interpretation,
                'textblob_polarity': round(sentiment_polarity, 4),
                'positive_indicator_score': round(indicators['positive_score'], 2),
                'negative_indicator_score': round(indicators['negative_score'], 2),
                'top_positive_indicators': [(kw, data['weight'], data['count']) for kw, data in top_positive],
                'top_negative_indicators': [(kw, data['weight'], data['count']) for kw, data in top_negative],
                'event_scores': events['event_scores'],
                'major_events': events['detected_events'],
                'financial_numbers': numbers
            })
        
        # Overall sentiment
        avg_score = total_score / len(articles)
        overall_interpretation = self.interpret_sentiment(avg_score)
        
        # Distribution
        very_positive = sum(1 for a in article_analyses if a['sentiment_score'] >= 0.7)
        positive = sum(1 for a in article_analyses if 0.3 <= a['sentiment_score'] < 0.7)
        slightly_positive = sum(1 for a in article_analyses if 0.1 <= a['sentiment_score'] < 0.3)
        neutral = sum(1 for a in article_analyses if -0.1 <= a['sentiment_score'] < 0.1)
        slightly_negative = sum(1 for a in article_analyses if -0.3 <= a['sentiment_score'] < -0.1)
        negative = sum(1 for a in article_analyses if -0.7 <= a['sentiment_score'] < -0.3)
        very_negative = sum(1 for a in article_analyses if a['sentiment_score'] < -0.7)
        
        results = {
            'enterprise': enterprise_name,
            'articles_analyzed': len(articles),
            'overall_sentiment_score': round(avg_score, 4),
            'overall_sentiment_interpretation': overall_interpretation,
            'sentiment_distribution': {
                'very_positive': very_positive,
                'positive': positive,
                'slightly_positive': slightly_positive,
                'neutral': neutral,
                'slightly_negative': slightly_negative,
                'negative': negative,
                'very_negative': very_negative
            },
            'articles': article_analyses
        }
        
        return results
    
    def print_sentiment_analysis(self, results):
        """Print sentiment analysis results"""
        if results['articles_analyzed'] == 0:
            print(f"\n{results['message']}")
            return
        
        print(f"\n{'='*80}")
        print(f"FINANCIAL SENTIMENT ANALYSIS")
        print(f"{'='*80}")
        print(f"Enterprise: {results['enterprise']}")
        print(f"Articles Analyzed: {results['articles_analyzed']}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL SENTIMENT")
        print(f"{'='*80}")
        print(f"Sentiment Score: {results['overall_sentiment_score']:.4f} (Range: -1.0 to +1.0)")
        print(f"Interpretation: {results['overall_sentiment_interpretation']}")
        
        print(f"\n{'='*80}")
        print(f"SENTIMENT DISTRIBUTION")
        print(f"{'='*80}")
        dist = results['sentiment_distribution']
        total = results['articles_analyzed']
        
        if dist['very_positive'] > 0:
            print(f"Very Positive: {dist['very_positive']} articles ({dist['very_positive']/total*100:.1f}%)")
        if dist['positive'] > 0:
            print(f"Positive: {dist['positive']} articles ({dist['positive']/total*100:.1f}%)")
        if dist['slightly_positive'] > 0:
            print(f"Slightly Positive: {dist['slightly_positive']} articles ({dist['slightly_positive']/total*100:.1f}%)")
        if dist['neutral'] > 0:
            print(f"Neutral: {dist['neutral']} articles ({dist['neutral']/total*100:.1f}%)")
        if dist['slightly_negative'] > 0:
            print(f"Slightly Negative: {dist['slightly_negative']} articles ({dist['slightly_negative']/total*100:.1f}%)")
        if dist['negative'] > 0:
            print(f"Negative: {dist['negative']} articles ({dist['negative']/total*100:.1f}%)")
        if dist['very_negative'] > 0:
            print(f"Very Negative: {dist['very_negative']} articles ({dist['very_negative']/total*100:.1f}%)")
        
        print(f"\n{'='*80}")
        print(f"DETAILED ARTICLE ANALYSIS")
        print(f"{'='*80}")
        
        for idx, article in enumerate(results['articles'], 1):
            print(f"\nArticle {idx}:")
            if article['title']:
                print(f"  Title: {article['title'][:70]}...")
            print(f"  Source: {article['source']} | Date: {article['date']}")
            print(f"  Sentiment Score: {article['sentiment_score']:.4f}")
            print(f"  Interpretation: {article['sentiment_interpretation']}")
            print(f"  TextBlob Polarity: {article['textblob_polarity']:.4f}")
            
            print(f"\n  Indicator Scores:")
            print(f"    Positive: {article['positive_indicator_score']:.2f}")
            print(f"    Negative: {article['negative_indicator_score']:.2f}")
            
            if article['top_positive_indicators']:
                print(f"\n  Top Positive Indicators (keyword, weight, count):")
                for kw, weight, count in article['top_positive_indicators'][:5]:
                    print(f"    - {kw}: weight={weight}, count={count}")
            
            if article['top_negative_indicators']:
                print(f"\n  Top Negative Indicators (keyword, weight, count):")
                for kw, weight, count in article['top_negative_indicators'][:5]:
                    print(f"    - {kw}: weight={weight}, count={count}")
            
            # Event scores
            event_scores = article['event_scores']
            if any(event_scores.values()):
                print(f"\n  Event Scores:")
                if event_scores['very_positive'] > 0:
                    print(f"    Very Positive Events: {event_scores['very_positive']:.2f}")
                if event_scores['positive'] > 0:
                    print(f"    Positive Events: {event_scores['positive']:.2f}")
                if event_scores['negative'] > 0:
                    print(f"    Negative Events: {event_scores['negative']:.2f}")
                if event_scores['very_negative'] > 0:
                    print(f"    Very Negative Events: {event_scores['very_negative']:.2f}")
            
            print(f"  {'-'*78}")
        
        print(f"\n{'='*80}")
        print(f"NOTE: Sentiment scores range from -1.0 (very negative) to +1.0 (very positive)")
        print(f"This is a quantitative sentiment analysis for informational purposes.")
        print(f"{'='*80}\n")
    
    def save_results(self, results, filename):
        """Save analysis results to JSON"""
        import os
        output_dir = 'sentiment_analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    
    analyzer = FinancialSentimentAnalyzer(
        corpus_file='scraped_data/nlp_corpus.txt',
        jsonl_file='scraped_data/nlp_data.jsonl'
    )
    
    print("="*80)
    print("FINANCIAL SENTIMENT ANALYSIS TOOL")
    print("="*80)
    enterprise_name = input("\nEnter company name (e.g., Attijari Bank, UBCI, Alkimia): ").strip()
    
    # Analyze sentiment
    results = analyzer.analyze_enterprise_sentiment(enterprise_name)
    
    # Display results
    analyzer.print_sentiment_analysis(results)
    
    # Save results
    if results['articles_analyzed'] > 0:
        analyzer.save_results(results, f'{enterprise_name.replace(" ", "_")}_sentiment_analysis.json')