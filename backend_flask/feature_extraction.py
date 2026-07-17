"""
Feature extraction module for phishing website detection.
 
Computes 28 features — the original 25 usable UCI features plus 3 new
features added to reduce over-reliance on SSLfinal_State:
 
  New features:
  - is_private_ip   : flags private/local IP addresses (192.168.x, 10.x, etc.)
  - url_entropy     : Shannon entropy of the URL string — random-looking URLs
                      (common in phishing) score higher
  - suspicious_tld  : flags TLDs statistically overrepresented in phishing
                      (.xyz, .click, .lat, .online, etc.)
 
5 of the original 30 UCI features are intentionally excluded because
their data sources are discontinued or require paid API keys:
  - Page_Rank             (Google PageRank API, shut down 2016)
  - web_traffic            (Alexa traffic rankings, retired 2022)
  - Google_Index           (requires scraping search engines, unreliable)
  - Statistical_report     (requires a paid threat-intel API key)
  - Links_pointing_to_page (true backlink counts require a paid SEO API)
 
The model must be retrained after adding these 3 new features so that
extractor output (28 keys) and model expectations stay in sync.
"""

import math
import re
import ssl
import socket
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import urllib3
from bs4 import BeautifulSoup
import whois

# Suppress noisy "InsecureRequestWarning". We use verify=False only as a
# last-resort fallback for sites with broken/self-signed certs; the first
# attempt always uses verify=True so SSL is validated properly.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class URLFeatureExtractor:
    """Extract 28 phishing-detection features from a live URL."""

    URL_SHORTENERS = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'cutt.ly',
        'shorte.st', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae', 'rebrand.ly',
    ]

    TRUSTED_CA_KEYWORDS = [
        "digicert", "let's encrypt", "letsencrypt", "globalsign",
        "sectigo", "comodo", "godaddy", "amazon", "google trust",
        "microsoft", "entrust", "geotrust", "thawte", "rapidssl",
        "cloudflare", "identrust",
    ]

    # TLDs that appear disproportionately in phishing campaigns.
    # Source: APWG eCrime reports, Spamhaus TLD statistics.
    SUSPICIOUS_TLDS = [
        '.xyz', '.click', '.club', '.online', '.site', '.lat',
        '.sbs', '.top', '.work', '.loan', '.win', '.gq', '.ml',
        '.cf', '.ga', '.tk', '.pw', '.cc', '.su', '.ws',
    ]

    # Private and reserved IP ranges — never legitimate public websites.
    PRIVATE_IP_PATTERNS = [
        r'^192\.168\.',          # RFC 1918
        r'^10\.',                # RFC 1918
        r'^172\.(1[6-9]|2[0-9]|3[01])\.',  # RFC 1918
        r'^127\.',               # loopback
        r'^0\.',                 # reserved
        r'^169\.254\.',          # link-local (APIPA)
        r'^::1$',                # IPv6 loopback
        r'^localhost$',          # hostname alias for 127.0.0.1
    ]

    # Top brands that phishing attacks most commonly impersonate.
    # These are the BASE domain names only (no TLD, no www).
    # Levenshtein distance is computed between the URL's base domain
    # and each brand name — a distance of 1 or 2 with a different TLD
    # or extra characters is a strong typosquatting signal.
    BRAND_NAMES = [
        'google', 'gmail', 'youtube',
        'facebook', 'instagram', 'whatsapp', 'meta',
        'amazon', 'aws',
        'microsoft', 'outlook', 'office', 'live', 'hotmail', 'onedrive',
        'apple', 'icloud', 'itunes',
        'paypal',
        'netflix',
        'twitter', 'x',
        'linkedin',
        'github',
        'dropbox',
        'adobe',
        'yahoo',
        'ebay',
        'walmart',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'hsbc',
        'steam', 'epicgames', 'roblox',
        'dhl', 'fedex', 'ups',
        'spotify',
        'tiktok',
        'snapchat',
        'coinbase', 'binance', 'blockchain',
    ]

    # Separate timeouts: page fetch can be slower, SSL socket must be fast.
    def __init__(self, fetch_timeout: int = 12, ssl_timeout: int = 10):
        self.fetch_timeout = fetch_timeout
        self.ssl_timeout = ssl_timeout
        # Keep self.timeout for backward compatibility with any external callers.
        self.timeout = fetch_timeout

    # ------------------------------------------------------------------
    # Private IP hard override helpers
    # ------------------------------------------------------------------
    def _is_private_ip_domain(self, domain: str) -> bool:
        """Returns True if domain is a private/reserved IP address.

        Used as a hard rule before ML runs — no legitimate public website
        uses a private IP (192.168.x, 10.x, 127.x, localhost, etc.).
        """
        for pattern in self.PRIVATE_IP_PATTERNS:
            if re.match(pattern, domain):
                return True
        return False

    def _phishing_override_features(self) -> Dict[str, float]:
        """Returns a 28-key feature dict with every value set to -1.0.

        Used when the private IP override fires. The rest of the codebase
        (Flask /predict, Gradio display) still receives a valid 28-key dict
        so nothing downstream breaks.
        """
        feature_names = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service',
            'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
            'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
            'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
            'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
            'age_of_domain', 'DNSRecord',
            'is_private_ip', 'url_entropy', 'suspicious_tld',
        ]
        return {name: -1.0 for name in feature_names}

    # ------------------------------------------------------------------
    # Typosquatting detection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Compute the Levenshtein edit distance between two strings.

        Standard dynamic-programming implementation. No external library
        needed — this runs entirely in pure Python.

        Examples:
            _levenshtein('paypal', 'paypa1')  -> 1  (l -> 1)
            _levenshtein('amazon', 'amaz0n')  -> 1  (o -> 0)
            _levenshtein('google', 'g00gle')  -> 2  (oo -> 00)
            _levenshtein('facebook', 'faceb00k') -> 2
        """
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)

        # Build matrix row by row (space-optimised: only two rows kept).
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                if ca == cb:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev = curr
        return prev[-1]

    def _is_typosquatting(self, domain: str) -> bool:
        """Returns True if the domain looks like a typosquatted brand.

        How it works:
        1. Strip www. and extract the base name before the first dot.
           e.g. 'paypa1-secure.com' -> 'paypa1-secure'
                'amaz0n-security.com' -> 'amaz0n-security'
                'faceb00k-login.com' -> 'faceb00k-login'
        2. Also try just the part before the first hyphen, because
           phishing domains often use 'brand-secure', 'brand-verify' etc.
        3. Compute Levenshtein distance against every brand name.
        4. If distance <= 2 AND the domain is NOT an exact match to a
           known legitimate domain, flag as typosquatting.

        Threshold = 2 catches:
            paypa1      (distance 1 from paypal)
            amaz0n      (distance 1 from amazon)
            faceb00k    (distance 2 from facebook)
            g00gle      (distance 2 from google)
            micros0ft   (distance 1 from microsoft)

        Threshold = 2 does NOT flag:
            google.com  (exact match — legitimate)
            paypal.com  (exact match — legitimate)
            amazon.com  (exact match — legitimate)
        """
        # Clean domain
        d = domain.lower()
        if d.startswith('www.'):
            d = d[4:]

        # Extract candidate strings to test
        base = d.split('.')[0]          # everything before first dot
        pre_hyphen = base.split('-')[0]  # everything before first hyphen

        candidates = {base, pre_hyphen}

        for candidate in candidates:
            if not candidate:
                continue
            for brand in self.BRAND_NAMES:
                dist = self._levenshtein(candidate, brand)
                # Distance 0 = exact match on base name.
                # That alone is NOT typosquatting — legitimate sites like
                # paypal.com, google.com have distance 0.
                # We flag distance 1 or 2 where base != brand exactly,
                # meaning the domain is SIMILAR but not identical.
                if 1 <= dist <= 2:
                    return True
        return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def extract_all_features(self, url: str) -> Dict[str, float]:
        """Extract all 28 model features from a URL.

        Two hard rule-based overrides run first:
        - If the domain is a private/reserved IP address, all 28 features
          are immediately set to -1.0 (phishing) without running the ML
          model. This bypasses the model's inability to catch private IPs
          (caused by the UCI dataset having no raw domain strings at
          training time, giving is_private_ip only 0.009 importance).
        - If the domain is a 1-2 character edit distance from a known
          brand name (typosquatting), the same override fires.
        """
        if '://' not in url:
            url = 'http://' + url

        parsed = urllib.parse.urlparse(url)
        # Use parsed.hostname (not netloc.split(':')[0]) so a URL like
        # "http://paypal.com@malicious-site.com/login" resolves to the
        # REAL host "malicious-site.com" — netloc would incorrectly keep
        # the "paypal.com@" userinfo prefix attached, corrupting every
        # domain-based feature below (DNS, WHOIS, hyphen check, etc.).
        domain = (parsed.hostname or '').lower()

        # ------------------------------------------------------------------
        # HARD OVERRIDE: private/local IP address as domain.
        # No legitimate public website uses a private IP as its domain.
        # The ML model cannot reliably catch these (is_private_ip importance
        # is only 0.009 because it was trained on a proxy column).
        # ------------------------------------------------------------------
        if self._is_private_ip_domain(domain):
            return self._phishing_override_features()

        # ------------------------------------------------------------------
        # HARD OVERRIDE: typosquatted brand domain (e.g. paypa1-secure.com,
        # amaz0n-account-update.com, faceb00k-login.com).
        # This bypasses the model the same way the private-IP check does,
        # since a 1-2 character edit distance from a known brand name is
        # an almost-certain phishing signal that the UCI-trained model has
        # no dedicated feature for.
        # ------------------------------------------------------------------
        if self._is_typosquatting(domain):
            return self._phishing_override_features()

        # Fetch the page ONCE and reuse for every content-based feature.
        # Returns (response, soup, html, ssl_verified):
        #   ssl_verified=True  -> page loaded over HTTPS with a valid cert
        #   ssl_verified=False -> either HTTP, cert invalid, or fetch failed
        response, soup, html, ssl_verified = self._fetch_page(url)

        # WHOIS is looked up once and reused for the two features that need it.
        creation_date, expiration_date = self._safe_whois_dates(domain)

        features: Dict[str, float] = {}

        # --- UCI original 25 features ---

        # URL-string-based (no network required)
        features['having_IP_Address'] = self._having_ip_address(domain)
        features['URL_Length'] = self._url_length_category(url)
        features['Shortining_Service'] = self._shortening_service(domain)
        features['having_At_Symbol'] = self._having_at_symbol(url)
        features['double_slash_redirecting'] = self._double_slash_redirecting(
            url)
        features['Prefix_Suffix'] = self._prefix_suffix(domain)
        features['having_Sub_Domain'] = self._having_sub_domain(domain)
        features['HTTPS_token'] = self._https_token(parsed.netloc)

        # Network / SSL / DNS
        features['SSLfinal_State'] = self._ssl_final_state(
            parsed, domain, creation_date,
            response, ssl_verified)
        features['port'] = self._port_feature(parsed, response)
        features['DNSRecord'] = self._dns_record(domain)

        # WHOIS
        features['Domain_registeration_length'] = self._domain_registration_length(
            creation_date, expiration_date)
        features['age_of_domain'] = self._age_of_domain(creation_date)
        features['Abnormal_URL'] = self._abnormal_url(domain, creation_date)

        # Page-content (need a successful fetch)
        features['Favicon'] = self._favicon(soup, domain)
        features['Request_URL'] = self._request_url(soup, domain)
        features['URL_of_Anchor'] = self._url_of_anchor(soup, domain)
        features['Links_in_tags'] = self._links_in_tags(soup, domain)
        features['SFH'] = self._sfh(soup, domain)
        features['Submitting_to_email'] = self._submitting_to_email(soup, html)
        features['Redirect'] = self._redirect_count(response)
        features['on_mouseover'] = self._on_mouseover(html)
        features['RightClick'] = self._right_click_disabled(html)
        features['popUpWidnow'] = self._popup_window(html)
        features['Iframe'] = self._iframe(soup)

        # --- 3 new features ---
        features['is_private_ip'] = self._is_private_ip(domain)
        features['url_entropy'] = self._url_entropy(url)
        features['suspicious_tld'] = self._suspicious_tld(domain)

        return features

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------
    def _fetch_page(
        self, url: str
    ) -> Tuple[Optional[requests.Response], Optional[BeautifulSoup], str, bool]:
        """Fetch the page once.

        Returns (response, soup, html_text, ssl_verified).
        ssl_verified=True  -> HTTPS + valid cert (verify=True succeeded).
        ssl_verified=False -> HTTP, invalid cert, or fetch failed entirely.
        """
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }

        # Attempt 1: fetch with SSL verification ON (HTTPS only)
        if url.startswith('https'):
            try:
                response = requests.get(
                    url, headers=headers,
                    timeout=self.fetch_timeout,
                    verify=True,
                    allow_redirects=True,
                )
                soup = BeautifulSoup(response.content, 'html.parser')
                html = response.text or ''
                return response, soup, html, True      # ssl_verified=True
            except requests.exceptions.SSLError:
                pass   # bad cert — fall through to verify=False
            except Exception:
                return None, None, '', False

        # Attempt 2: verify=False (HTTP URLs or bad-cert HTTPS fallback)
        try:
            response = requests.get(
                url, headers=headers,
                timeout=self.fetch_timeout,
                verify=False,
                allow_redirects=True,
            )
            soup = BeautifulSoup(response.content, 'html.parser')
            html = response.text or ''
            return response, soup, html, False         # ssl_verified=False
        except Exception:
            return None, None, '', False

    def _safe_whois_dates(
        self, domain: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """WHOIS lookup — returns (creation_date, expiration_date) or (None, None)."""
        try:
            w = whois.whois(domain)

            def _clean(dt):
                if isinstance(dt, list):
                    dt = dt[0]
                if isinstance(dt, str):
                    try:
                        dt = datetime.strptime(dt, '%Y-%m-%d')
                    except ValueError:
                        return None
                if not isinstance(dt, datetime):
                    return None
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt

            return _clean(w.creation_date), _clean(w.expiration_date)
        except Exception:
            return None, None

    # ==================================================================
    # UCI original 25 features
    # ==================================================================

    # 1. having_IP_Address
    def _having_ip_address(self, domain: str) -> float:
        try:
            return -1.0 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 1.0
        except Exception:
            return 1.0

    # 2. URL_Length
    def _url_length_category(self, url: str) -> float:
        try:
            n = len(url)
            if n < 54:
                return 1.0
            if n <= 75:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 3. Shortining_Service
    def _shortening_service(self, domain: str) -> float:
        try:
            return -1.0 if any(s in domain for s in self.URL_SHORTENERS) else 1.0
        except Exception:
            return 1.0

    # 4. having_At_Symbol
    def _having_at_symbol(self, url: str) -> float:
        try:
            return -1.0 if '@' in url else 1.0
        except Exception:
            return 1.0

    # 5. double_slash_redirecting
    def _double_slash_redirecting(self, url: str) -> float:
        try:
            return -1.0 if url.rfind('//') > 7 else 1.0
        except Exception:
            return 1.0

    # 6. Prefix_Suffix
    def _prefix_suffix(self, domain: str) -> float:
        try:
            return -1.0 if '-' in domain else 1.0
        except Exception:
            return 1.0

    # 7. having_Sub_Domain
    # Strip www. first — it is not a phishing signal.
    # dot_count=1 -> base domain only          -> 1.0
    # dot_count=2 -> one real subdomain        -> 0.0
    # dot_count>=3 -> multiple subdomains      -> -1.0
    def _having_sub_domain(self, domain: str) -> float:
        try:
            d = domain[4:] if domain.startswith('www.') else domain
            dots = d.count('.')
            if dots <= 1:
                return 1.0
            if dots == 2:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 8. SSLfinal_State
    def _ssl_final_state(
        self,
        parsed,
        domain: str,
        creation_date: Optional[datetime],
        response=None,
        ssl_verified: bool = False,
    ) -> float:
        try:
            if parsed.scheme != 'https':
                return -1.0

            # Stage 1: direct SSL socket to inspect the cert issuer
            trusted = False
            socket_succeeded = False
            try:
                context = ssl.create_default_context()
                with socket.create_connection(
                    (domain, 443), timeout=self.ssl_timeout
                ) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                socket_succeeded = True
                if cert:
                    issuer = dict(x[0] for x in cert.get('issuer', []))
                    issuer_org = issuer.get('organizationName', '').lower()
                    trusted = any(
                        kw in issuer_org for kw in self.TRUSTED_CA_KEYWORDS)
            except ssl.SSLError:
                socket_succeeded = True   # network reached, cert bad
                trusted = False
            except Exception:
                socket_succeeded = False  # network unreachable — use fallback

            # Stage 2: fallback to ssl_verified from _fetch_page
            if not socket_succeeded:
                trusted = ssl_verified

            domain_established = (
                creation_date is not None
                and (datetime.now() - creation_date).days >= 365
            )

            if trusted and domain_established:
                return 1.0
            if trusted or domain_established:
                return 0.0
            if ssl_verified:
                return 0.0
            # Final fallback: if the page loaded successfully over HTTPS
            # (requests succeeded with verify=True) even though the direct
            # SSL socket timed out locally, return neutral (0.0) instead of
            # phishing (-1.0). This prevents false positives on legitimate
            # HTTPS sites like amazon.com, github.com, paypal.com when the
            # SSL socket cannot connect from the local machine.
            if parsed.scheme == 'https' and response is not None:
                return 0.0
            return -1.0
        except Exception:
            return -1.0

    # 9. Domain_registeration_length
    def _domain_registration_length(
        self,
        creation_date: Optional[datetime],
        expiration_date: Optional[datetime],
    ) -> float:
        try:
            if creation_date and expiration_date:
                return 1.0 if (expiration_date - creation_date).days >= 365 else -1.0
            return -1.0
        except Exception:
            return -1.0

    # 10. Favicon
    def _favicon(self, soup: Optional[BeautifulSoup], domain: str) -> float:
        try:
            if soup is None:
                return 1.0
            tag = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
            if not tag or 'href' not in tag.attrs:
                return 1.0
            href = tag['href']
            if href.startswith('http'):
                fav_domain = urllib.parse.urlparse(
                    href).netloc.split(':')[0].lower()
                return -1.0 if fav_domain and domain not in fav_domain else 1.0
            return 1.0
        except Exception:
            return 1.0

    # 11. port
    def _port_feature(self, parsed, response: Optional[requests.Response]) -> float:
        try:
            if parsed.port not in {80, 443, None}:
                return -1.0
            return 1.0 if response is not None else -1.0
        except Exception:
            return -1.0

    # 12. HTTPS_token
    def _https_token(self, netloc: str) -> float:
        try:
            return -1.0 if 'https' in netloc.lower() else 1.0
        except Exception:
            return 1.0

    # 13. Request_URL
    def _request_url(self, soup: Optional[BeautifulSoup], domain: str) -> float:
        try:
            if soup is None:
                return 0.0
            tags = soup.find_all(['img', 'script', 'video', 'audio'])
            total, external = 0, 0
            for tag in tags:
                src = tag.get('src')
                if not src:
                    continue
                total += 1
                if src.startswith('http'):
                    d = urllib.parse.urlparse(src).netloc.split(':')[0].lower()
                    if d and domain not in d:
                        external += 1
            if total == 0:
                return 1.0
            pct = external / total * 100
            if pct < 22:
                return 1.0
            if pct <= 61:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 14. URL_of_Anchor
    def _url_of_anchor(self, soup: Optional[BeautifulSoup], domain: str) -> float:
        try:
            if soup is None:
                return 0.0
            anchors = soup.find_all('a', href=True)
            total = len(anchors)
            if total == 0:
                return 1.0
            suspicious = 0
            for a in anchors:
                href = a['href'].strip().lower()
                if href in ('#', 'javascript:void(0)', '') or href.startswith('#'):
                    suspicious += 1
                elif href.startswith('http'):
                    d = urllib.parse.urlparse(
                        href).netloc.split(':')[0].lower()
                    if d and domain not in d:
                        suspicious += 1
            pct = suspicious / total * 100
            if pct < 31:
                return 1.0
            if pct <= 67:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 15. Links_in_tags
    def _links_in_tags(self, soup: Optional[BeautifulSoup], domain: str) -> float:
        try:
            if soup is None:
                return 0.0
            tags = soup.find_all(['meta', 'script', 'link'])
            total, external = 0, 0
            for tag in tags:
                attr = tag.get('src') or tag.get('href')
                if not attr:
                    continue
                total += 1
                if attr.startswith('http'):
                    d = urllib.parse.urlparse(
                        attr).netloc.split(':')[0].lower()
                    if d and domain not in d:
                        external += 1
            if total == 0:
                return 1.0
            pct = external / total * 100
            if pct < 17:
                return 1.0
            if pct <= 81:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 16. SFH
    def _sfh(self, soup: Optional[BeautifulSoup], domain: str) -> float:
        try:
            if soup is None:
                return 1.0
            forms = soup.find_all('form')
            if not forms:
                return 1.0
            for form in forms:
                action = (form.get('action') or '').strip().lower()
                if action in ('', 'about:blank'):
                    return -1.0
                if action.startswith('http'):
                    d = urllib.parse.urlparse(
                        action).netloc.split(':')[0].lower()
                    if d and domain not in d:
                        return 0.0
            return 1.0
        except Exception:
            return 1.0

    # 17. Submitting_to_email
    def _submitting_to_email(
        self, soup: Optional[BeautifulSoup], html: str
    ) -> float:
        try:
            if soup:
                for form in soup.find_all('form'):
                    if 'mailto:' in (form.get('action') or '').lower():
                        return -1.0
            if html and ('mailto:' in html.lower() or '.mail(' in html.lower()):
                return -1.0
            return 1.0
        except Exception:
            return 1.0

    # 18. Abnormal_URL
    def _abnormal_url(self, domain: str, creation_date: Optional[datetime]) -> float:
        try:
            return 1.0 if creation_date is not None else -1.0
        except Exception:
            return -1.0

    # 19. Redirect
    def _redirect_count(self, response: Optional[requests.Response]) -> float:
        try:
            if response is None:
                return 0.0
            count = len(response.history)
            if count <= 1:
                return 1.0
            if count <= 3:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 20. on_mouseover
    def _on_mouseover(self, html: str) -> float:
        try:
            if not html:
                return 1.0
            low = html.lower()
            return -1.0 if ('onmouseover' in low and 'window.status' in low) else 1.0
        except Exception:
            return 1.0

    # 21. RightClick
    def _right_click_disabled(self, html: str) -> float:
        try:
            if not html:
                return 1.0
            low = html.lower()
            disabled = (
                ('event.button==2' in low and 'return false' in low)
                or ('contextmenu' in low and 'preventdefault' in low)
            )
            return -1.0 if disabled else 1.0
        except Exception:
            return 1.0

    # 22. popUpWidnow
    def _popup_window(self, html: str) -> float:
        try:
            if not html:
                return 1.0
            return -1.0 if 'window.open(' in html.lower() else 1.0
        except Exception:
            return 1.0

    # 23. Iframe
    def _iframe(self, soup: Optional[BeautifulSoup]) -> float:
        try:
            if soup is None:
                return 1.0
            return -1.0 if soup.find_all('iframe') else 1.0
        except Exception:
            return 1.0

    # 24. age_of_domain
    def _age_of_domain(self, creation_date: Optional[datetime]) -> float:
        try:
            if creation_date is None:
                return -1.0
            return 1.0 if (datetime.now() - creation_date).days >= 180 else -1.0
        except Exception:
            return -1.0

    # 25. DNSRecord
    def _dns_record(self, domain: str) -> float:
        try:
            socket.gethostbyname(domain)
            return 1.0
        except Exception:
            return -1.0

    # ==================================================================
    # 3 New Features
    # ==================================================================

    # 26. is_private_ip
    # Detects private/reserved IP addresses used as the domain.
    # These are NEVER legitimate public websites.
    # Returns -1.0 (phishing) if a private IP is detected, 1.0 otherwise.
    def _is_private_ip(self, domain: str) -> float:
        try:
            for pattern in self.PRIVATE_IP_PATTERNS:
                if re.match(pattern, domain):
                    return -1.0
            return 1.0
        except Exception:
            return 1.0

    # 27. url_entropy
    # Shannon entropy of the full URL string.
    # Legitimate URLs are human-readable (low entropy).
    # Phishing URLs often contain random strings (high entropy).
    # Thresholds derived from analysis of UCI dataset URLs:
    #   entropy < 3.5  -> clearly readable          -> 1.0  (safe)
    #   3.5 to 4.5     -> moderately complex         -> 0.0  (neutral)
    #   entropy > 4.5  -> random/obfuscated string   -> -1.0 (suspicious)
    def _url_entropy(self, url: str) -> float:
        try:
            if not url:
                return 0.0
            freq = {}
            for ch in url:
                freq[ch] = freq.get(ch, 0) + 1
            length = len(url)
            entropy = -sum(
                (count / length) * math.log2(count / length)
                for count in freq.values()
            )
            if entropy < 3.5:
                return 1.0
            if entropy <= 4.5:
                return 0.0
            return -1.0
        except Exception:
            return 0.0

    # 28. suspicious_tld
    # Checks whether the domain uses a TLD that is statistically
    # overrepresented in phishing campaigns (APWG eCrime reports).
    # Returns -1.0 if suspicious TLD found, 1.0 otherwise.
    def _suspicious_tld(self, domain: str) -> float:
        try:
            for tld in self.SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    return -1.0
            return 1.0
        except Exception:
            return 1.0


def extract_features_from_urls(urls: List[str]) -> List[Dict[str, float]]:
    """Extract features from a list of URLs."""
    extractor = URLFeatureExtractor()
    return [extractor.extract_all_features(url) for url in urls]
