"""
Feature extraction module for phishing website detection.
Extracts various features from URLs for model training and prediction.
"""

import re
import ssl
import socket
import urllib.parse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import whois
from Levenshtein import distance as levenshtein_distance


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection."""
    
    # Common brands for typosquatting detection
    COMMON_BRANDS = [
        'google', 'facebook', 'amazon', 'paypal', 'apple', 'microsoft',
        'netflix', 'gmail', 'yahoo', 'linkedin', 'twitter', 'instagram',
        'bankofamerica', 'chase', 'wellsfargo', 'citibank', 'amex',
        'dropbox', 'github', 'spotify', 'uber', 'airbnb'
    ]
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def extract_all_features(self, url: str) -> Dict[str, float]:
        """Extract all features from a URL."""
        features = {}
        
        # Basic URL features
        features.update(self._extract_length_features(url))
        features.update(self._extract_special_char_features(url))
        features.update(self._extract_https_features(url))
        features.update(self._extract_ip_features(url))
        features.update(self._extract_subdomain_features(url))
        
        # Domain-based features
        features.update(self._extract_domain_age_features(url))
        features.update(self._extract_typosquatting_features(url))
        
        # Content-based features (optional, requires fetching)
        # features.update(self._extract_content_features(url))
        
        return features
    
    def _extract_length_features(self, url: str) -> Dict[str, float]:
        """Extract URL length related features."""
        return {
            'url_length': len(url),
            'domain_length': len(urllib.parse.urlparse(url).netloc),
            'path_length': len(urllib.parse.urlparse(url).path),
        }
    
    def _extract_special_char_features(self, url: str) -> Dict[str, float]:
        """Extract special character count features."""
        return {
            'at_count': url.count('@'),
            'double_slash_count': url.count('//'),
            'dash_count': url.count('-'),
            'dot_count': url.count('.'),
            'underscore_count': url.count('_'),
            'question_count': url.count('?'),
            'equal_count': url.count('='),
            'ampersand_count': url.count('&'),
            'exclamation_count': url.count('!'),
            'space_count': url.count(' ') + url.count('%20'),
            'tilde_count': url.count('~'),
            'comma_count': url.count(','),
            'plus_count': url.count('+'),
            'asterisk_count': url.count('*'),
            'hash_count': url.count('#'),
            'dollar_count': url.count('$'),
            'percent_count': url.count('%'),
        }
    
    def _extract_https_features(self, url: str) -> Dict[str, float]:
        """Extract HTTPS related features."""
        parsed = urllib.parse.urlparse(url)
        has_https = parsed.scheme == 'https'
        
        features = {
            'has_https': 1.0 if has_https else 0.0,
            'https_in_domain': 1.0 if 'https' in parsed.netloc.lower() else 0.0,
        }
        
        # Check SSL certificate validity if HTTPS
        if has_https:
            features['has_valid_ssl'] = 1.0 if self._check_ssl_certificate(parsed.netloc) else 0.0
        else:
            features['has_valid_ssl'] = 0.0
            
        return features
    
    def _check_ssl_certificate(self, domain: str) -> bool:
        """Check if domain has a valid SSL certificate."""
        try:
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    if cert and 'notAfter' in cert:
                        return True
            return False
        except Exception:
            return False
    
    def _extract_ip_features(self, url: str) -> Dict[str, float]:
        """Extract IP address related features."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check for IP address in domain
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        has_ip = bool(re.match(ip_pattern, domain))
        
        # Check for IP address in URL (phishing indicator)
        ip_in_url = bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url))
        
        return {
            'has_ip_address': 1.0 if has_ip else 0.0,
            'ip_in_url': 1.0 if ip_in_url else 0.0,
        }
    
    def _extract_subdomain_features(self, url: str) -> Dict[str, float]:
        """Extract subdomain related features."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Remove www. prefix for counting
        if domain.startswith('www.'):
            domain = domain[4:]
        
        parts = domain.split('.')
        num_subdomains = max(0, len(parts) - 2)  # Exclude domain and TLD
        
        return {
            'num_subdomains': num_subdomains,
            'has_www': 1.0 if parsed.netloc.startswith('www.') else 0.0,
        }
    
    def _extract_domain_age_features(self, url: str) -> Dict[str, float]:
        """Extract domain age related features using WHOIS."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        # Remove port and www if present
        if ':' in domain:
            domain = domain.split(':')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        
        features = {
            'domain_age_days': -1.0,  # -1 indicates unknown
            'domain_registration_length': -1.0,
        }
        
        try:
            w = whois.whois(domain)
            
            # Get creation date
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            # Get expiration date
            expiration_date = w.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            
            if creation_date:
                if isinstance(creation_date, str):
                    try:
                        creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
                    except ValueError:
                        pass
                
                if isinstance(creation_date, datetime):
                    age = (datetime.now() - creation_date).days
                    features['domain_age_days'] = float(age)
            
            if expiration_date and creation_date:
                if isinstance(expiration_date, str):
                    try:
                        expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
                    except ValueError:
                        pass
                
                if isinstance(expiration_date, datetime) and isinstance(creation_date, datetime):
                    reg_length = (expiration_date - creation_date).days
                    features['domain_registration_length'] = float(reg_length)
                    
        except Exception:
            # WHOIS lookup failed, keep default values
            pass
        
        return features
    
    def _extract_typosquatting_features(self, url: str) -> Dict[str, float]:
        """Extract typosquatting detection features using Levenshtein distance."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port, www, and TLD
        if ':' in domain:
            domain = domain.split(':')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove TLD
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            # Handle subdomains - take the main domain part
            domain = domain_parts[-2]
        elif len(domain_parts) == 2:
            domain = domain_parts[0]
        
        min_distance = float('inf')
        closest_brand = None
        
        for brand in self.COMMON_BRANDS:
            dist = levenshtein_distance(domain, brand)
            if dist < min_distance:
                min_distance = dist
                closest_brand = brand
        
        # Normalize by length of domain
        normalized_distance = min_distance / max(len(domain), 1)
        
        return {
            'levenshtein_distance': float(min_distance),
            'normalized_levenshtein': normalized_distance,
            'typosquatting_score': 1.0 if min_distance <= 2 and len(domain) > 3 else 0.0,
        }
    
    def _extract_content_features(self, url: str) -> Dict[str, float]:
        """Extract features from webpage content (requires HTTP request)."""
        features = {
            'has_form': 0.0,
            'has_password_field': 0.0,
            'num_external_links': 0.0,
            'num_images': 0.0,
            'favicon_from_other_domain': 0.0,
        }
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for forms
            forms = soup.find_all('form')
            features['has_form'] = 1.0 if forms else 0.0
            
            # Check for password fields
            password_inputs = soup.find_all('input', {'type': 'password'})
            features['has_password_field'] = 1.0 if password_inputs else 0.0
            
            # Count external links
            parsed_url = urllib.parse.urlparse(url)
            base_domain = parsed_url.netloc
            
            external_links = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') and base_domain not in href:
                    external_links += 1
            features['num_external_links'] = float(external_links)
            
            # Count images
            images = soup.find_all('img')
            features['num_images'] = float(len(images))
            
            # Check favicon
            favicon = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
            if favicon and 'href' in favicon.attrs:
                favicon_href = favicon['href']
                if favicon_href.startswith('http') and base_domain not in favicon_href:
                    features['favicon_from_other_domain'] = 1.0
            
        except Exception:
            # Content fetch failed, return default values
            pass
        
        return features


def extract_features_from_urls(urls: List[str]) -> List[Dict[str, float]]:
    """Extract features from a list of URLs."""
    extractor = URLFeatureExtractor()
    return [extractor.extract_all_features(url) for url in urls]
