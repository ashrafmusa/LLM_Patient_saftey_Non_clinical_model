# Deployment Guide: From Prototype to Production

## Overview

This guide provides step-by-step instructions for deploying the Patient Safety LLM pipeline in production environments.

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Configuration](#environment-configuration)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Governance & Compliance](#governance--compliance)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

### Clinical Review
- [ ] Clinician review of model performance
- [ ] Assessment of false positive/negative rates acceptable for use case
- [ ] Approval from clinical leadership
- [ ] Sign-off from medical director or chief safety officer

### Technical Review
- [ ] Code review completed
- [ ] All tests passing (pytest)
- [ ] Security review (bandit, safety)
- [ ] Performance testing (response time, throughput)
- [ ] Load testing for expected usage volume

### Regulatory & Compliance
- [ ] IRB approval or exemption documentation
- [ ] HIPAA Business Associate Agreement (BAA) if applicable
- [ ] Data governance and retention policies documented
- [ ] Model monitoring and update procedures established
- [ ] Incident reporting procedures in place

### Data & Privacy
- [ ] De-identification validated by privacy officer
- [ ] PHI handling procedures documented
- [ ] Data retention policy implemented
- [ ] Access controls configured
- [ ] Audit logging enabled

---

## Environment Configuration

### Required Secrets & Credentials

Create `.env` file (do NOT commit to version control):

```bash
# Database (if applicable)
DATABASE_URL=postgresql://user:password@localhost/patient_safety_llm
REDIS_URL=redis://localhost:6379

# LLM Backend
LLM_SERVER_URL=http://localhost:8080
LLM_API_KEY=sk-your-api-key

# Monitoring & Logging
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret

# Email Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ALERT_EMAIL=alerts@your-org.com
```

### Configuration File

Create `config/production.yaml`:

```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 30

model:
  path: /models/risk_model.pkl
  vectorizer_path: /models/vectorizer.pkl
  batch_size: 32
  cache_predictions: true

logging:
  level: INFO
  file: /var/log/patient_safety_llm.log
  max_size: 10485760  # 10 MB
  backup_count: 5

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 60

security:
  require_auth: true
  rate_limit: 100  # requests per minute
  cors_origins: ["https://your-domain.com"]
```

---

## Local Deployment

### Quick Start

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Start API server
uvicorn src.app:app --host 0.0.0.0 --port 8000

# 4. In another terminal, start Streamlit UI
streamlit run src/ui.py --server.port 8501
```

### Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Test assessment endpoint
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient denies chest pain. BP normal."}'

# Access UI
open http://localhost:8501
```

---

## Docker Deployment

### Build Image

```bash
# Build production image
docker build -t patient-safety-llm:latest .

# Tag for registry
docker tag patient-safety-llm:latest your-registry/patient-safety-llm:latest
```

### Run Container

```bash
# Single container with API
docker run -p 8000:8000 \
  -e DATABASE_URL=$DATABASE_URL \
  -e LOG_LEVEL=INFO \
  -v /var/log/app:/app/logs \
  patient-safety-llm:latest \
  uvicorn src.app:app --host 0.0.0.0 --port 8000

# Using docker-compose (recommended)
docker-compose up -d
```

### Docker Compose Services

```bash
# Start all services (UI, API, Notebook)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Cloud Deployment

### AWS ECS (Recommended)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/patient-safety-llm:latest

# 2. Create ECS task definition (task-definition.json)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. Create/update ECS service
aws ecs create-service \
  --cluster patient-safety \
  --service-name patient-safety-llm \
  --task-definition patient-safety-llm:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/patient-safety-llm

# Deploy to Cloud Run
gcloud run deploy patient-safety-llm \
  --image gcr.io/PROJECT_ID/patient-safety-llm \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Push to ACR
az acr build --registry your-registry --image patient-safety-llm:latest .

# Deploy to ACI
az container create \
  --resource-group your-group \
  --name patient-safety-llm \
  --image your-registry.azurecr.io/patient-safety-llm:latest \
  --cpu 2 --memory 4
```

---

## Monitoring & Logging

### Application Monitoring

```python
# Monitor key metrics
from prometheus_client import Counter, Histogram, Gauge

# Request counter
request_count = Counter('assessment_requests_total', 'Total assessment requests')

# Response time histogram
response_time = Histogram('assessment_duration_seconds', 'Assessment response time')

# Active predictions gauge
active_predictions = Gauge('active_predictions', 'Currently processing predictions')

# Error rate
errors = Counter('assessment_errors_total', 'Total assessment errors')
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger('patient_safety_llm')

# Log all predictions (for audit trail)
logger.info(f"Assessment: text={text[:50]}... risk={risk_level} score={score}")

# Log errors with context
logger.error(f"Assessment failed: {error}", extra={
    'user_id': user_id,
    'text_hash': hash(text),
    'timestamp': datetime.now()
})

# Log model performance metrics
logger.info(f"Daily metrics: accuracy={accuracy:.2%}, auc={auc:.2%}")
```

### Alerting

Set up alerts for:
- API latency > 5 seconds
- Error rate > 1%
- Model accuracy drop > 5%
- Memory usage > 80%
- Disk usage > 85%

---

## Governance & Compliance

### Model Governance

**Track all model versions**:
```python
model_registry = {
    'model_id': 'risk_model_v2.3',
    'version': '2.3',
    'training_date': '2025-01-15',
    'accuracy': 0.92,
    'deployed_date': '2025-01-20',
    'approved_by': 'Dr. Smith, Chief Medical Officer',
    'validation_status': 'approved',
    'next_review': '2026-01-20'
}
```

### Regular Performance Reviews

**Schedule**:
- Daily: Monitor error rates and latency
- Weekly: Accuracy and prediction distribution review
- Monthly: Detailed performance metrics and fairness analysis
- Quarterly: Full model validation and update assessment

### Incident Reporting

**Document**:
1. What happened (error, prediction error, etc.)
2. When it occurred
3. User/patient impact
4. Root cause
5. Corrective action taken
6. Preventive measures

### Continuous Improvement

**Process**:
1. Collect edge cases and errors
2. Quarterly: Review error patterns
3. Retrain model with additional data if needed
4. Validate improvements before deployment
5. Document all changes

---

## Troubleshooting

### Common Issues

**Issue**: API slow response times
```bash
# Check CPU/memory usage
docker stats

# Increase workers
uvicorn src.app:app --workers 8

# Enable caching
redis-cli FLUSHALL  # Start fresh cache
```

**Issue**: De-identification failure
```bash
# Test de-identification
python -c "
from src.deid import deidentify_text
text = 'Your problematic text here'
result = deidentify_text(text)
print(result)
"
```

**Issue**: Model prediction errors
```bash
# Check model file
ls -la models/risk_model.pkl

# Test model loading
python -c "
import joblib
model = joblib.load('models/risk_model.pkl')
print('Model loaded successfully')
"
```

**Issue**: Database connection errors
```bash
# Test database
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.environ.get('DATABASE_URL'))
connection = engine.connect()
print('Database connected')
connection.close()
"
```

---

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
import redis

# Cache de-identified text
redis_client = redis.Redis(host='localhost', port=6379)

def assess_risk_cached(text: str) -> dict:
    """Assess risk with result caching."""
    cache_key = f"assessment:{hash(text)}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute if not cached
    result = assess_risk(text)
    redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
    
    return result
```

### Batch Processing

```python
def assess_risk_batch(texts: List[str]) -> List[dict]:
    """Assess multiple texts efficiently."""
    # De-identify all at once
    deidentified = [deidentify_text(t) for t in texts]
    
    # Vectorize all at once (better memory usage)
    vectors = vectorizer.transform(deidentified)
    
    # Predict batch
    predictions = model.predict(vectors)
    scores = model.predict_proba(vectors)
    
    return [{'risk_level': p, 'score': s} for p, s in zip(predictions, scores)]
```

---

## Rollback Procedures

If deployment encounters critical issues:

```bash
# Immediate rollback to previous version
docker pull your-registry/patient-safety-llm:previous
docker run ... patient-safety-llm:previous

# Or with docker-compose
git checkout HEAD~1  # Previous commit
docker-compose down
docker-compose up -d
```

---

## Support & Documentation

- **Issues**: Report via GitHub Issues
- **Questions**: GitHub Discussions
- **Manuscript**: See docs/study_draft.md for methods
- **API Docs**: See FastAPI auto-generated docs at `/docs`

---

**Version**: 1.0  
**Last Updated**: December 25, 2025
