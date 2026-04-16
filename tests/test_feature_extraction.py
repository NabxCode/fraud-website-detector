"""
Unit tests for feature extraction module.
"""

import pytest
from src.feature_extraction import URLFeatureExtractor


class TestFeatureExtraction:
    """Test cases for URLFeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        return URLFeatureExtractor()
    
    def test_url_length_features(self, extractor):
        """Test URL length feature extraction."""
        url = "https://www.google.com/search?q=test"
        features = extractor._extract_length_features(url)
        
        assert features['url_length'] == len(url)
        assert features['domain_length'] == len("www.google.com")
        assert features['path_length'] == len("/search")
    
    def test_special_char_features(self, extractor):
        """Test special character counting."""
        url = "http://test-site.com//path?a=1&b=2@test"
        features = extractor._extract_special_char_features(url)
        
        assert features['dash_count'] == 1
        assert features['at_count'] == 1
        assert features['double_slash_count'] == 2
        assert features['dot_count'] >= 2
        assert features['question_count'] == 1
        assert features['ampersand_count'] == 1
    
    def test_https_features(self, extractor):
        """Test HTTPS feature detection."""
        https_url = "https://secure.example.com"
        http_url = "http://insecure.example.com"
        
        https_features = extractor._extract_https_features(https_url)
        http_features = extractor._extract_https_features(http_url)
        
        assert https_features['has_https'] == 1.0
        assert http_features['has_https'] == 0.0
    
    def test_ip_features(self, extractor):
        """Test IP address detection."""
        ip_url = "http://192.168.1.1/login"
        domain_url = "http://example.com/login"
        
        ip_features = extractor._extract_ip_features(ip_url)
        domain_features = extractor._extract_ip_features(domain_url)
        
        assert ip_features['has_ip_address'] == 1.0
        assert domain_features['has_ip_address'] == 0.0
    
    def test_subdomain_features(self, extractor):
        """Test subdomain counting."""
        simple_url = "https://google.com"
        www_url = "https://www.google.com"
        multi_sub_url = "https://a.b.c.google.com"
        
        simple_features = extractor._extract_subdomain_features(simple_url)
        www_features = extractor._extract_subdomain_features(www_url)
        multi_features = extractor._extract_subdomain_features(multi_sub_url)
        
        assert simple_features['num_subdomains'] == 0
        assert www_features['has_www'] == 1.0
        assert multi_features['num_subdomains'] == 3  # a, b, c
    
    def test_typosquatting_features(self, extractor):
        """Test typosquatting detection."""
        # Suspicious URL mimicking google
        typo_url = "https://g00gle-security.com"
        legitimate_url = "https://google.com"
        
        typo_features = extractor._extract_typosquatting_features(typo_url)
        legit_features = extractor._extract_typosquatting_features(legitimate_url)
        
        assert typo_features['typosquatting_score'] == 1.0
        assert typo_features['closest_brand'] == 'google'
        assert typo_features['levenshtein_distance'] <= 2
        
        assert legit_features['typosquatting_score'] == 0.0
    
    def test_extract_all_features(self, extractor):
        """Test complete feature extraction."""
        url = "https://www.google.com/search?q=test"
        features = extractor.extract_all_features(url)
        
        # Check that all expected features are present
        expected_numeric_features = [
            'url_length', 'domain_length', 'path_length',
            'has_https', 'has_ip_address', 'num_subdomains',
            'levenshtein_distance', 'typosquatting_score'
        ]
        
        for feature in expected_numeric_features:
            assert feature in features, f"Missing feature: {feature}"
            assert isinstance(features[feature], (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
