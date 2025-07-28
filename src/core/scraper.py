import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from urllib.parse import urljoin, urlparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_news_list(base_url, max_pages=5):
    """
    Scrape news listings from AFNC website
    
    Args:
        base_url (str): Base URL of the AFNC website
        max_pages (int): Maximum number of pages to scrape
        
    Returns:
        list: List of dictionaries containing news data
    """
    news_data = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    for page in range(1, max_pages + 1):
        try:
            # Construct URL for each page (this is a placeholder - actual pagination structure needs to be determined)
            page_url = f"{base_url}?page={page}" if page > 1 else base_url
            
            logger.info(f"Scraping page {page}: {page_url}")
            response = session.get(page_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles - this needs to be adjusted based on actual site structure
            # Looking for common selectors for news articles
            articles = soup.find_all(['article', '.news-item', '.post', '.article'])
            
            # If no articles found with common selectors, try to find links that might lead to articles
            if not articles:
                links = soup.find_all('a', href=True)
                article_links = [link for link in links if '/news/' in link['href'] or '/article/' in link['href']]
                articles = [{'link': link['href']} for link in article_links]
            
            if not articles:
                logger.warning(f"No articles found on page {page}")
                continue
                
            for article in articles:
                try:
                    # Extract headline/title
                    headline_elem = article.find(['h1', 'h2', 'h3', 'h4', '.title', '.headline'])
                    headline = headline_elem.get_text(strip=True) if headline_elem else "N/A"
                    
                    # Extract link
                    link_elem = article.find('a', href=True)
                    link = urljoin(base_url, link_elem['href']) if link_elem else "N/A"
                    
                    # Extract date if available
                    date_elem = article.find(['time', '.date', '.published'])
                    date = date_elem.get_text(strip=True) if date_elem else "N/A"
                    
                    # For now, we'll assume all scraped articles are real news
                    # In a real implementation, we'd need to determine the label based on AFNC's classification
                    label = "Real"  # Placeholder
                    
                    news_data.append({
                        'headline': headline,
                        'link': link,
                        'date': date,
                        'label': label
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
                    
            # Be respectful to the server
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error scraping page {page}: {e}")
            continue
    
    return news_data

def scrape_article_content(url):
    """
    Scrape the full content of a single article
    
    Args:
        url (str): URL of the article
        
    Returns:
        dict: Dictionary containing article content
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to extract article content with common selectors
        content_selectors = [
            'article', 
            '.content', 
            '.article-content', 
            '.post-content',
            '.entry-content',
            'main'
        ]
        
        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break
                
        if not content_elem:
            # Fallback: get all paragraph elements
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        else:
            content = content_elem.get_text(strip=True)
            
        return {
            'url': url,
            'content': content[:5000]  # Limit content length
        }
        
    except Exception as e:
        logger.error(f"Error scraping article {url}: {e}")
        return {
            'url': url,
            'content': "Error scraping content"
        }

def create_fake_news_dataset():
    """
    Create a synthetic fake news dataset to supplement real news data
    This is a placeholder implementation - in a real scenario, 
    fake news would come from the AFNC website or other sources
    """
    fake_headlines = [
        "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",
        "หมอเผยผลข้างเคียงของการฉีดวัคซีน mRNA ที่ไม่เคยมีคนพูดถึงมาก่อน",
        "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%",
        "การใช้สมาร์ทโฟนมากกว่า 2 ชั่วโมงต่อวันทำให้สมองย่อขนาดลง",
        "รัฐบาลเตรียมเก็บภาษีใหม่จากการใช้โซเชียลมีเดีย",
        "อาหารเสริมใหม่ช่วยให้ลืมความทรงจำที่ไม่ดีใน 3 วัน",
        "พบดาวเคราะห์ใหม่ที่มีสภาพเหมือนโลกอยู่ห่างจากเราเพียง 10 แสงปี",
        "การนอนหลับมากกว่า 8 ชั่วโมงต่อคืนทำให้เสี่ยงเป็นโรคอัลไซเมอร์"
    ]
    
    return pd.DataFrame({
        'headline': fake_headlines,
        'label': ['Fake'] * len(fake_headlines),
        'date': ['N/A'] * len(fake_headlines),
        'link': ['N/A'] * len(fake_headlines)
    })

def main():
    """Main function to scrape and prepare news data"""
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    base_url = 'https://www.antifakenewscenter.com/'
    
    # Scrape real news data
    logger.info("Starting to scrape real news data...")
    real_news_data = scrape_news_list(base_url, max_pages=3)
    
    if real_news_data:
        real_df = pd.DataFrame(real_news_data)
        logger.info(f"Scraped {len(real_df)} real news items")
    else:
        logger.warning("No real news data scraped, creating empty dataframe")
        real_df = pd.DataFrame(columns=['headline', 'link', 'date', 'label'])
    
    # Create fake news data
    logger.info("Creating synthetic fake news data...")
    fake_df = create_fake_news_dataset()
    logger.info(f"Created {len(fake_df)} fake news items")
    
    # Combine datasets
    combined_df = pd.concat([real_df, fake_df], ignore_index=True)
    
    # Save raw data
    raw_file_path = 'data/raw/news.csv'
    combined_df.to_csv(raw_file_path, index=False)
    logger.info(f"Raw data saved to {raw_file_path}")
    
    # Basic data preparation
    logger.info("Preparing processed data...")
    processed_df = combined_df.copy()
    
    # Clean headline text
    processed_df['headline'] = processed_df['headline'].str.strip()
    processed_df = processed_df[processed_df['headline'].notna() & (processed_df['headline'] != '')]
    
    # Save processed data
    processed_file_path = 'data/processed/news.csv'
    processed_df.to_csv(processed_file_path, index=False)
    logger.info(f"Processed data saved to {processed_file_path}")
    
    print(f"Data acquisition and preparation completed!")
    print(f"Total articles: {len(processed_df)}")
    print(f"Real news: {len(processed_df[processed_df['label'] == 'Real'])}")
    print(f"Fake news: {len(processed_df[processed_df['label'] == 'Fake'])}")

if __name__ == '__main__':
    main()
