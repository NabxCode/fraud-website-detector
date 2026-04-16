# Fraud Website Detection System - Design Document

## 1. System Overview

### 1.1 Purpose
A machine learning-based web service that detects phishing websites by analyzing URL characteristics and features. The system provides both a REST API and a web interface for real-time URL analysis.

### 1.2 Key Capabilities
- Real-time URL analysis and classification
- Batch URL processing
- Feature extraction and visualization
- Model comparison (Random Forest vs XGBoost)
- Web UI and CLI interfaces

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENTS                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Web UI    │  │    CLI      │  │  External Systems   │ │
│  │  (Browser)  │  │  (Terminal) │  │    (API Calls)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼────────────┘
          │                │                    │
          └────────────────┴────────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │      FLASK REST API SERVER        │
          │         (Port: 5000)              │
          │  ┌──────────┐  ┌──────────────┐  │
          │  │  /predict │  │  /features   │  │
          │  │ /predict/ │  │/feature-     │  │
          │  │   batch   │  │ importance   │  │
          │  └──────────┘  └──────────────┘  │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│  FEATURE        │ │   MODEL      │ │   DATA       │
│  EXTRACTION     │ │   LAYER      │ │   LAYER      │
│  MODULE         │ │              │ │              │
│ ┌─────────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ URL Parsing │ │ │ │ Random   │ │ │ │ UCI      │ │
│ │ Special Chars│ │ │ │ Forest   │ │ │ │ Dataset  │ │
│ │ HTTPS Check │ │ │ │ (Default)│ │ │ │ Loader   │ │
│ │ WHOIS Lookup│ │ │ ├──────────┤ │ │ ├──────────┤ │
│ │ Levenshtein │ │ │ │ XGBoost  │ │ │ │ Custom   │ │
│ │ Distance    │ │ │ │ (Adv)    │ │ │ │ Dataset  │ │
│ └─────────────┘ │ │ └──────────┘ │ │ │ Loader   │ │
└─────────────────┘ └──────────────┘ └──────────────┘
```

### 2.2 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Web UI      │  │  CLI Tool    │  │  Flask API   │  │
│  │  (HTML/JS)   │  │  (Python)    │  │  (REST)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC LAYER                 │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Feature Extraction Engine               ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ││
│  │  │  Basic   │ │  Domain  │ │ Network  │ │ Content│ ││
│  │  │  URL     │ │  Analysis│ │  Checks  │ │  Scrap │ ││
│  │  │  Features│ │  (WHOIS) │ │          │ │ (Opt)  │ ││
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ ││
│  └─────────────────────────────────────────────────────┘│
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │              ML Prediction Engine                    ││
│  │         (Random Forest / XGBoost)                   ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Model File  │  │  Training    │  │  Feature     │  │
│  │  (.pkl)      │  │  Dataset     │  │  Cache       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Feature Engineering Design

### 3.1 Feature Categories

| Category | Features | Description | Phishing Indicator |
|----------|----------|-------------|-------------------|
| **Structural** | url_length | Total URL length | > 75 chars |
| | domain_length | Domain part length | > 30 chars |
| | path_length | Path component length | Unusually long |
| **Special Chars** | at_count | '@' symbol count | > 0 (suspicious) |
| | double_slash_count | '//' occurrences | > 1 (redirect) |
| | dash_count | '-' count | > 1 (subdomain) |
| | dot_count | '.' count | > 3 (subdomain) |
| **Security** | has_https | HTTPS presence | 0 (insecure) |
| | has_valid_ssl | SSL certificate valid | 0 (suspicious) |
| | has_ip_address | IP in URL | 1 (suspicious) |
| **Domain** | num_subdomains | Subdomain count | > 2 (suspicious) |
| | domain_age_days | Domain registration age | < 30 days |
| | domain_registration_length | Registration period | Very short |
| **Typosquatting** | levenshtein_distance | Edit distance to brands | <= 2 |
| | typosquatting_score | Brand impersonation flag | 1 (phishing) |
| | closest_brand | Most similar brand | Known brand |

### 3.2 Feature Extraction Pipeline

```
Input URL
    │
    ▼
┌─────────────────┐
│  URL Parsing    │──→ Extract: protocol, domain, path, query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Basic Features  │──→ Lengths, character counts
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Network│ │ Domain │
│ Checks │ │  Info  │
│        │ │        │
│• HTTPS │ │• WHOIS │
│• SSL   │ │• Age   │
│• IP    │ │• Subdomains
└────────┘ └────────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Typosquatting   │──→ Levenshtein distance to brand list
│ Detection       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Vector  │──→ Normalized numeric features
│ (20+ features)  │
└─────────────────┘
         │
         ▼
    ML Model
```

---

## 4. Machine Learning Design

### 4.1 Model Selection

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| **Random Forest** | Baseline / Default | Interpretable, robust, fast inference | May overfit on small data |
| **XGBoost** | Advanced accuracy | Best performance, handles imbalanced data | Slightly less interpretable |

### 4.2 Model Architecture

```
┌─────────────────────────────────────────┐
│           RANDOM FOREST                 │
│                                         │
│  Input: Feature Vector (20+ features)   │
│              │                          │
│              ▼                          │
│  ┌───────────────────────────────┐      │
│  │   StandardScaler (normalize)  │      │
│  └───────────────┬───────────────┘      │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────┐      │
│  │  100 Decision Trees (ensemble)│      │
│  │  • Max depth: 10               │      │
│  │  • Min samples split: 5        │      │
│  │  • Bootstrap sampling          │      │
│  └───────────────┬───────────────┘      │
│                  │                      │
│  ┌───────────────┴───────────────┐      │
│  │     Majority Voting           │      │
│  │  (Classification)              │      │
│  └───────────────┬───────────────┘      │
│                  │                      │
│  Output: [Phishing Prob, Legit Prob]    │
│                    │                    │
│                    ▼                    │
│              Threshold: 0.5             │
│                    │                    │
│                    ▼                    │
│           Final Prediction              │
│        (0 = Legit, 1 = Phishing)        │
└─────────────────────────────────────────┘
```

### 4.3 Training Pipeline

```
Raw Dataset (URLs + Labels)
            │
            ▼
┌─────────────────────┐
│   Data Preprocessing │
│   • Handle missing   │
│   • Normalize        │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌────────┐  ┌────────┐
│ Train  │  │  Test  │
│  80%   │  │  20%   │
└───┬────┘  └────────┘
    │
    ▼
┌─────────────────────┐
│  Feature Extraction │
│  (20+ features)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Training    │
│   • Random Forest   │
│   • Cross-validation│
│   • Grid search     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Evaluation  │
│   • Accuracy        │
│   • Precision/Recall│
│   • F1 Score        │
│   • ROC-AUC         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Persistence │
│   (Save as .pkl)    │
└─────────────────────┘
```

---

## 5. API Design

### 5.1 REST Endpoints

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/health` | GET | Health check | - | `{status, model_loaded}` |
| `/predict` | POST | Single URL check | `{"url": "..."}` | Prediction + features |
| `/predict/batch` | POST | Multiple URLs | `{"urls": [...]}` | List of predictions |
| `/features` | POST | Extract features | `{"url": "..."}` | Feature dictionary |
| `/feature-importance` | GET | Model insights | - | Feature rankings |
| `/` | GET | Web UI | - | HTML page |

### 5.2 Prediction Response Format

```json
{
  "url": "https://suspicious-site.com/login",
  "is_phishing": true,
  "confidence": 0.92,
  "phishing_probability": 0.92,
  "features": {
    "url_length": 35,
    "has_https": 0.0,
    "has_ip_address": 0.0,
    "num_subdomains": 1,
    "typosquatting_score": 1.0,
    "closest_brand": "legitimate-site",
    "levenshtein_distance": 2,
    "domain_age_days": 5
  }
}
```

---

## 6. Data Flow Diagram

### 6.1 Single URL Analysis

```
User enters URL
      │
      ▼
┌─────────────┐
│  Validate   │──→ Check URL format
│    URL      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Feature         │──→ Extract 20+ features
│ Extraction      │    (local processing)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Model      │──→ Load .pkl from disk
│ (if not cached) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocess      │──→ Scale/normalize features
│ Features        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Inference │──→ Random Forest prediction
│                 │    ( < 100ms typical)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Format Response │──→ Build JSON response
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return Result   │──→ Display to user
└─────────────────┘
```

---

## 7. Security Considerations

| Concern | Mitigation |
|---------|------------|
| **URL Validation** | Parse and validate URL format before processing |
| **WHOIS Rate Limits** | Cache results, implement timeouts (10s) |
| **SSL Verification** | Allow verification bypass for expired certs |
| **Content Fetching** | Optional, with timeout and user-agent headers |
| **Model Persistence** | Secure pickle loading, validate file integrity |
| **Input Sanitization** | URL encoding, length limits |
| **Rate Limiting** | Implement on API endpoints (future) |

---

## 8. Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Feature Extraction (basic) | 1-5ms | Local string operations |
| WHOIS Lookup | 1-3s | Network dependent, cached |
| SSL Check | 0.5-2s | Network dependent |
| Model Inference | 10-50ms | Random Forest 100 trees |
| **End-to-end** | 100ms-3s | Depending on network checks |

---

## 9. Scalability Considerations

| Aspect | Current | Future Enhancement |
|--------|---------|-------------------|
| **Model** | Single instance | Model ensemble, A/B testing |
| **Caching** | None | Redis cache for WHOIS/results |
| **Deployment** | Flask dev server | Gunicorn + Nginx |
| **Database** | File-based | PostgreSQL for logs/analysis |
| **Processing** | Synchronous | Async queue (Celery) for batch |

---

## 10. Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.8+ |
| **Web Framework** | Flask |
| **ML Library** | scikit-learn, XGBoost |
| **Feature Extraction** | python-whois, Levenshtein |
| **Data Processing** | pandas, numpy |
| **Testing** | pytest |
| **Frontend** | HTML5, vanilla JS |
| **Model Serialization** | joblib |

---

## 11. Error Handling Strategy

```
Input Validation Errors
         │
         ▼
┌─────────────────┐
│ 4xx Client Error│──→ 400 Bad Request
│                 │    (Invalid URL, missing params)
└─────────────────┘

Network/External Errors
         │
         ▼
┌─────────────────┐
│ Service Degraded│──→ Continue without optional features
│                 │    (WHOIS timeout, SSL check fail)
└─────────────────┘

Model/System Errors
         │
         ▼
┌─────────────────┐
│ 5xx Server Error│──→ 503 Service Unavailable
│                 │    (Model not loaded, exception)
└─────────────────┘
```

---

## 12. Deployment Architecture

```
┌─────────────────────────────────────────┐
│              CLIENTS                    │
│         (Browser / CLI / API)         │
└──────────────────┬────────────────────┘
                   │
                   │ HTTP/HTTPS
                   ▼
          ┌─────────────────┐
          │   Nginx/Apache  │  (Reverse Proxy, SSL)
          │   (Port 80/443) │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Gunicorn      │  (WSGI Server)
          │   (Workers: 4)  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Flask App     │  (Application Server)
          │   (Port 5000)   │
          └────────┬────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │  Model │ │  Logs  │ │  Cache │
   │  File  │ │  Files │ │ (Redis)│
   └────────┘ └────────┘ └────────┘
```

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Current | Initial release with RF and XGBoost |

---

**Document Version:** 1.0.0  
**Last Updated:** April 2026  
**Author:** Project Team
