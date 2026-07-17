"""
Static data for the Gradio UI: example URLs, the 28-feature plain-English
guide, feature key ordering/labels, categorisation, fallback importance
values, and short importance-chart labels.
Split out of gradio_app.py to keep that file focused on layout/wiring.
"""

EXAMPLES = [
    ["https://www.paypal.com"],
    ["https://www.amazon.com"],
    ["https://www.google.com"],
    ["http://192.168.1.1/admin"],
    ["http://192.168.1.50/"],
    ["http://paypa1-secure.com/verify"],
    ["http://xn--pypal-4ve.com/verify"],
    ["http://tinyurl.com/xyz123"],
    ["http://bit.ly/3xR9kLm"],
    ["http://malicious-site.xyz/free-money"],
    ["http://evil.tk/steal-data"],
    ["https://paypal-login.xyz/secure"],
    ["http://login.free-money-now.online/claim"],
    ["http://a23f8k2p9qr.club/win"],
    ["http://verify-account.sbs/update"],
    ["http://update-your-info.lat/confirm"],
    ["http://example.com@malicious.com/login"],
]

# ------------------------------------------------------------------
# Feature explanations — 28 total (25 UCI + 3 new)
# ------------------------------------------------------------------
FEATURE_GUIDE = [
    ("IP address as domain",
     "Legitimate websites always have a human-readable name like 'google.com' or 'paypal.com'. "
     "The model checks whether the domain part of the URL is a raw number like '192.168.1.1' instead of a proper name. "
     "Attackers use IP addresses to avoid registering a domain (which leaves a traceable paper trail) and to make it harder "
     "for security tools to block them by name. If you see a number where a site name should be, treat it as a serious red flag."),

    ("URL length",
     "The model measures the total number of characters in the URL. Phishing URLs tend to be much longer than normal ones. "
     "Attackers pad the address with extra words — for example 'secure-login-paypal-account-verify.com/confirm/identity/step2' — "
     "so that the fake part gets buried and the URL looks like it belongs to a trusted brand. "
     "Legitimate websites keep their URLs short and clean. Anything unusually long deserves a closer look."),

    ("URL shortener used",
     "Services like bit.ly, tinyurl.com, and t.co replace long URLs with a short anonymous link. "
     "The model detects when a known shortening service is used in the URL. "
     "Phishers use these to hide the real destination — you can't see where you're actually going until it's too late. "
     "Shortened links in emails or messages asking you to log in or verify your account are especially dangerous."),

    ("@ symbol in URL",
     "The @ symbol has a very specific meaning in URLs: your browser treats everything before it as login credentials, "
     "and everything after it as the actual destination. So a URL like 'paypal.com@evil.com/login' takes you to evil.com — not PayPal. "
     "The model flags any URL containing @ because legitimate websites never use this character in their links. "
     "This is a deliberate trick to confuse people who scan a URL quickly."),

    ("Hidden redirect (//) in URL",
     "A double-slash '//' is supposed to appear only at the start of a URL (after 'https:'). "
     "If '//' appears again somewhere else in the URL path, browsers can interpret it as a redirect instruction, "
     "silently sending you to a completely different site than what the URL appears to show. "
     "The model flags this pattern because legitimate websites never construct their links this way."),

    ("Hyphen in domain name",
     "Real brands and companies own clean, hyphen-free domain names: 'paypal.com', 'amazon.com', 'bankofamerica.com'. "
     "Attackers can't register those domains, so instead they create fake ones with hyphens: 'paypal-secure-login.com', "
     "'bank-of-america-verify.com'. The hyphen makes it look official at a glance but no real brand owns it. "
     "The model checks whether the domain name contains a hyphen, which is a well-known phishing pattern."),

    ("Too many sub-parts in domain",
     "A domain normally has one or two dot-separated parts: 'mail.google.com' has three parts and is genuine. "
     "Phishing sites often stack many parts to bury the real (fake) domain at the very end: "
     "'login.secure.paypal.verify.support.xyz' — the only part that actually matters is 'support.xyz', which is the fake domain. "
     "The model counts how many dot-separated sub-parts exist and flags URLs with an unusually high number."),

    ("HTTPS / security certificate",
     "The padlock icon in your browser means the site has been issued a valid security certificate from a trusted authority — "
     "this proves the site's identity and encrypts your connection. "
     "The model checks three things: whether HTTPS is present, whether the certificate is from a trusted issuer, and whether it is valid. "
     "Many phishing sites either skip HTTPS entirely or obtain cheap, untrusted certificates just to show a padlock. "
     "A padlock alone is not a guarantee of safety — the model looks deeper at certificate quality."),

    ("How long domain is registered",
     "When a company registers a domain, they typically pay for it 2, 5, or even 10 years in advance — because their business depends on it. "
     "Phishing sites are registered for the minimum period (often just 1 month or 1 year) because the attacker knows it will be shut down soon. "
     "The model checks the WHOIS registration record to see how long the domain was paid for. "
     "A domain registered for less than a year is a warning sign, especially when combined with other phishing signals."),

    ("Website icon source",
     "Every website shows a small icon in your browser tab — called a favicon. "
     "The model checks where this icon is loaded from. On a legitimate site, the favicon comes from the site itself. "
     "On a cloned phishing page, the favicon is often loaded from the real brand's actual website because the attacker copied the page "
     "but didn't bother hosting all the files themselves. "
     "If the favicon domain doesn't match the page domain, that's a strong sign the page was copied from somewhere else."),

    ("Unusual port number",
     "Web servers listen on specific port numbers. Legitimate public websites use port 80 (HTTP) or port 443 (HTTPS). "
     "If a URL specifies a different port number — like ':8080', ':4443', or ':8888' — it is almost certainly not a real public website. "
     "Attackers use unusual ports to host phishing pages on servers that haven't been configured for proper public hosting, "
     "or to bypass corporate firewalls and security tools that only monitor standard ports."),

    ("'https' written in domain name",
     "A clever trick where the attacker puts the word 'https' directly inside the domain name itself — "
     "for example, 'https-paypal-secure.com' or 'https.bankverify.net'. "
     "This makes someone scanning the URL quickly think the connection is secure, when in fact 'https' here is just part of a fake domain name. "
     "The model checks whether the domain name itself contains the string 'https' or 'http', which no real website would ever need to do."),

    ("External images and scripts",
     "When you open a webpage, it loads many resources to build what you see — images, buttons, fonts, and icons. "
     "On a genuine site, most of these come from the site's own server. "
     "On a cloned phishing page, most images and elements are still loaded from the original real website's server, "
     "because the attacker copy-pasted the page's HTML without actually hosting the files. "
     "The model measures what percentage of loaded resources come from external domains and flags pages where this is unusually high."),

    ("Where links on the page go",
     "The model looks at every clickable link (anchor tag) on the page and checks where each one actually goes. "
     "On a real website, most links stay within the same domain. "
     "On a phishing page, most links either go nowhere (they're empty or broken) or they redirect to a completely different website. "
     "Attackers clone a real page's look but can't replicate the full site behind it, so the links are often just hollow copies."),

    ("Where page resources come from",
     "Besides images, webpages also load CSS stylesheets and JavaScript files that control the page's layout and behaviour. "
     "The model checks what percentage of these resources come from external domains that don't belong to the site. "
     "A high percentage of externally sourced scripts and stylesheets is a strong sign the page was cloned — "
     "the attacker copied the visual shell but left all the behind-the-scenes files pointing back to the real site."),

    ("Where forms send your data",
     "A web form is what you interact with when you type your email, password, or credit card number and click Submit. "
     "The model checks the form's 'action' attribute — the destination server that receives whatever you type. "
     "On a phishing page, this destination is a completely different server controlled by the attacker. "
     "You may be looking at a page that appears to be your bank's login, but your credentials are silently sent to a criminal's server."),

    ("Form submits to email address",
     "Instead of sending your data to a web server, some basic phishing forms use a 'mailto:' action, "
     "which delivers whatever you type directly to an email inbox. "
     "This is one of the oldest and most primitive phishing techniques — it requires almost no technical infrastructure. "
     "The model flags any form whose action is an email address rather than a proper web server URL."),

    ("Domain has no public record",
     "Every domain registered through a legitimate registrar is supposed to have a public WHOIS record "
     "that shows who registered it, when, and for how long. "
     "If this record doesn't exist or is completely hidden, it means the domain was registered very recently or through unusual channels. "
     "The model treats a missing or unresolvable WHOIS record as a strong phishing signal, "
     "because established businesses always have a traceable and verifiable domain history."),

    ("Number of page redirects",
     "A redirect is when one URL automatically sends you to another URL without you clicking anything. "
     "Some redirects are normal — one or two hops is common for login flows. "
     "But phishing infrastructure often chains together many redirects to confuse security scanners, "
     "hide the origin server, and make the URL chain hard to trace back to the attacker. "
     "The model flags URLs that cause more than 3 redirects before landing on the final page."),

    ("Status bar manipulation",
     "When you hover your mouse over a link in a browser, the actual destination URL appears in the browser's status bar at the bottom. "
     "Some phishing pages inject JavaScript that intercepts this and shows you a fake, trusted-looking URL in the status bar "
     "while the real link points somewhere else entirely. "
     "The model detects whether the page uses this JavaScript trick to manipulate what you see on hover."),

    ("Right-click disabled",
     "Phishing pages sometimes use JavaScript to block your right-click menu entirely. "
     "This is done to prevent you from opening the browser's 'Inspect Element' tool, "
     "which would reveal the fake structure of the page, or from copying the URL to check it elsewhere. "
     "Disabling right-click is never needed on a legitimate website — it only benefits someone trying to hide something from you."),

    ("Pop-up windows",
     "A page that launches pop-up windows immediately on load is a classic warning sign. "
     "Pop-ups are used by phishers to display fake login prompts, fake security warnings, or fake prize notifications "
     "to pressure you into entering your credentials or downloading malware. "
     "Modern legitimate websites avoid pop-ups because browsers block them by default — "
     "a site that forces them through is almost certainly malicious."),

    ("Hidden frames on page",
     "An iframe is a webpage embedded invisibly inside another webpage. "
     "Phishing pages use hidden iframes to load content from a malicious server in the background "
     "while displaying something that looks trustworthy in the foreground. "
     "The malicious content running in the hidden iframe can steal cookies, capture keystrokes, or run exploit code "
     "without you ever seeing it or knowing it's happening."),

    ("How old the domain is",
     "The model checks the WHOIS record to find out when the domain was first created. "
     "A domain that is less than 6 months old is a red flag. "
     "Phishing domains are disposable — attackers register them, run the scam for a few weeks, "
     "and abandon them before they get blocked. Legitimate businesses build trust over years, not weeks. "
     "Domain age alone is not proof of safety, but combined with other signals it is a very reliable indicator."),

    ("Domain exists in DNS",
     "DNS (Domain Name System) is the internet's address book — it translates domain names into IP addresses. "
     "If a domain has no DNS record at all, it means no one can currently find or resolve it on the internet. "
     "This can happen when a domain was registered so recently it hasn't propagated through the DNS system yet, "
     "or when an attacker is using a domain they haven't properly set up. "
     "The model treats a missing DNS record as a strong phishing signal."),

    ("Private/local IP as domain",
     "IP addresses in the ranges 192.168.x.x, 10.x.x.x, and 127.x.x.x are reserved for private internal networks — "
     "your home Wi-Fi router, your company's internal systems, or your own computer. "
     "These addresses can never be reached from the public internet. "
     "If a URL uses one of these as its 'website', it is either targeting your local network devices "
     "or trying to trick you into making a request to something on your own machine. This is always a phishing or attack attempt."),

    ("How random the URL looks",
     "The model calculates the 'entropy' of the URL — a mathematical measure of how random or unpredictable the characters are. "
     "Human-written URLs are low-entropy because they use real words: 'paypal.com/login'. "
     "Auto-generated phishing URLs are high-entropy because they use random strings: 'a7f2k9xQr3.club/xZ9pLm'. "
     "Phishing tools often generate URLs programmatically, producing random-looking patterns that have a measurably different "
     "character distribution from legitimate URLs."),

    ("High-risk domain ending",
     "The final part of a domain name — .com, .org, .net, .xyz — is called the Top-Level Domain (TLD). "
     "Some TLDs are provided for free or for fractions of a cent, making them extremely popular with phishers: "
     ".tk (Tokelau), .xyz, .click, .sbs, .lat, .online, .club, and .top are among the most abused. "
     "The model checks whether the domain uses one of these high-risk TLDs. "
     "Legitimate businesses almost always use .com, .org, .net, or their country's official TLD — not disposable free ones."),
]

# ------------------------------------------------------------------
# Raw feature keys, in the same order as FEATURE_GUIDE above —
# lets us zip a label/explanation onto every raw model feature key.
# ------------------------------------------------------------------
FEATURE_KEY_ORDER = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "is_private_ip", "url_entropy", "suspicious_tld",
]
FEATURE_LABELS = {key: label for key, (label, _) in zip(
    FEATURE_KEY_ORDER, FEATURE_GUIDE)}

# Grouping used on the Feature Extraction tab
FEATURE_CATEGORIES = {
    "📁 URL Structure": [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "HTTPS_token",
    ],
    "📁 SSL / Network": ["SSLfinal_State", "port", "DNSRecord"],
    "📁 Domain / WHOIS": ["Domain_registeration_length", "age_of_domain", "Abnormal_URL"],
    "📁 Page Content": [
        "Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "Redirect", "on_mouseover", "RightClick",
        "popUpWidnow", "Iframe",
    ],
    "📁 New Features": ["is_private_ip", "url_entropy", "suspicious_tld"],
}

# Fallback XGBoost feature importances — used only if /feature-importance
# can't be reached (API down, model not loaded yet). The live values from
# the Flask endpoint are preferred whenever available.
FALLBACK_FEATURE_IMPORTANCE = [
    ("SSLfinal_State", 0.4198), ("URL_of_Anchor", 0.1970), ("Prefix_Suffix", 0.0722),
    ("SFH", 0.0329), ("Links_in_tags", 0.0252), ("having_Sub_Domain", 0.0220),
    ("Shortining_Service", 0.0177), ("Domain_registeration_length", 0.0143),
    ("suspicious_tld", 0.0126), ("Request_URL", 0.0126), ("age_of_domain", 0.0126),
    ("double_slash_redirecting", 0.0118), ("URL_Length", 0.0113), ("Favicon", 0.0111),
    ("HTTPS_token", 0.0110), ("having_IP_Address", 0.0106), ("Abnormal_URL", 0.0103),
    ("Redirect", 0.0099), ("popUpWidnow", 0.0099), ("Iframe", 0.0096),
    ("Submitting_to_email", 0.0096), ("is_private_ip",
                                      0.0090), ("on_mouseover", 0.0088),
    ("DNSRecord", 0.0087), ("url_entropy", 0.0083), ("RightClick", 0.0078),
    ("port", 0.0071), ("having_At_Symbol", 0.0066),
]

# Short 3-5 word descriptions used as bracket labels on the
# Model Insights -> Feature Importance chart.
IMPORTANCE_SHORT_LABELS = {
        "having_IP_Address":           "site uses a raw IP number instead of a name — a common phishing tactic",
        "URL_Length":                  "URL is unusually long, often to bury the fake part deep inside",
        "Shortining_Service":          "URL shortener hides the real destination link from the user",
        "having_At_Symbol":            "@ symbol tricks the browser into ignoring the trusted-looking part",
        "double_slash_redirecting":    "double slash after domain silently sends you to a different site",
        "Prefix_Suffix":               "hyphen in domain fakes a brand name — e.g. secure-paypal.com",
        "having_Sub_Domain":           "too many dot-separated parts bury the real fake domain at the end",
        "SSLfinal_State":              "site lacks a valid HTTPS certificate from a trusted authority",
        "Domain_registeration_length": "domain registered for only a short time — phishing sites are disposable",
        "Favicon":                     "browser icon loaded from a different site — copied from a real brand",
        "port":                        "unusual port number used to bypass standard security filters",
        "HTTPS_token":                 "the word 'https' is placed inside the domain name itself to mislead",
        "Request_URL":                 "most images and buttons load from external domains — page was cloned",
        "URL_of_Anchor":               "clickable links on the page go to different or empty destinations",
        "Links_in_tags":               "stylesheets and scripts are hosted on external unrelated domains",
        "SFH":                         "form sends your typed data to a completely different website",
        "Submitting_to_email":         "form delivers your data directly to an email — a primitive phishing method",
        "Abnormal_URL":                "domain has no public WHOIS record — registered secretly or very recently",
        "Redirect":                    "page passes you through many redirects to confuse security scanners",
        "on_mouseover":                "page hides the real link by changing what shows in the browser status bar",
        "RightClick":                  "right-click is blocked to stop you from inspecting or reporting the page",
        "popUpWidnow":                 "page opens pop-up windows — used to steal credentials or install malware",
        "Iframe":                      "hidden invisible frames load malicious content from another site",
        "age_of_domain":               "domain is less than 6 months old — phishing domains are short-lived",
        "DNSRecord":                   "domain has no DNS record — it was just registered and not yet publicly listed",
        "is_private_ip":               "IP address is a private network address — can never be a real public website",
        "url_entropy":                 "URL contains random-looking characters — a sign of auto-generated phishing links",
        "suspicious_tld":              "domain ending like .xyz or .tk is cheap or free and heavily used by phishers",
}
