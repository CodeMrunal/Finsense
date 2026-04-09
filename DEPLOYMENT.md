# Deployment Guide

This guide covers deploying the FinSense Financial Intelligence System locally and in the cloud.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key (for RAG chatbot features)

## Local Deployment

### Option 1: Direct Python Execution

1. **Clone and setup:**
```bash
git clone <repository-url>
cd FinSense
```

2. **Create virtual environment:**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp config/env.example .env
# Edit .env with your API keys
```

5. **Run backend:**
```bash
python run_backend.py
```
Backend available at: `http://localhost:8000`

6. **Run dashboard (in another terminal):**
```bash
streamlit run dashboard/app.py
```
Dashboard available at: `http://localhost:8501`

### Option 2: Docker Compose (Recommended)

1. **Create .env file:**
```bash
cp config/env.example .env
# Edit .env with your API keys
```

2. **Start services:**
```bash
docker-compose up -d
```

3. **View logs:**
```bash
docker-compose logs -f
```

4. **Stop services:**
```bash
docker-compose down
```

### Option 3: Docker (Individual Containers)

1. **Build image:**
```bash
docker build -t finsense:latest .
```

2. **Run backend:**
```bash
docker run -d \
  --name finsense-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  finsense:latest python run_backend.py
```

3. **Run dashboard:**
```bash
docker run -d \
  --name finsense-dashboard \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  finsense:latest streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```

## Cloud Deployment

### AWS EC2 / Google Cloud Compute / Azure VM

1. **Launch instance:**
   - Ubuntu 22.04 LTS
   - Minimum: 2 CPU, 4GB RAM
   - Recommended: 4 CPU, 8GB RAM

2. **SSH into instance:**
```bash
ssh user@your-instance-ip
```

3. **Install Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

4. **Clone repository:**
```bash
git clone <repository-url>
cd FinSense
```

5. **Configure environment:**
```bash
cp config/env.example .env
nano .env  # Add your API keys
```

6. **Start with Docker Compose:**
```bash
docker-compose up -d
```

7. **Configure firewall:**
```bash
# Allow ports 8000 and 8501
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
sudo ufw enable
```

8. **Access services:**
   - Backend: `http://your-instance-ip:8000`
   - Dashboard: `http://your-instance-ip:8501`

### AWS Elastic Beanstalk

1. **Install EB CLI:**
```bash
pip install awsebcli
```

2. **Initialize EB:**
```bash
eb init -p python-3.10 finsense-app
```

3. **Create .ebextensions/01_python.config:**
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: backend.main:app
```

4. **Deploy:**
```bash
eb create finsense-env
eb deploy
```

### Google Cloud Run

1. **Build and push image:**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/finsense
```

2. **Deploy backend:**
```bash
gcloud run deploy finsense-backend \
  --image gcr.io/PROJECT-ID/finsense \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key \
  --port 8000
```

3. **Deploy dashboard:**
```bash
gcloud run deploy finsense-dashboard \
  --image gcr.io/PROJECT-ID/finsense \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key \
  --port 8501 \
  --command streamlit \
  --args "run,dashboard/app.py,--server.port=8501,--server.address=0.0.0.0"
```

### Azure Container Instances

1. **Build and push to Azure Container Registry:**
```bash
az acr build --registry your-registry --image finsense:latest .
```

2. **Deploy backend:**
```bash
az container create \
  --resource-group your-rg \
  --name finsense-backend \
  --image your-registry.azurecr.io/finsense:latest \
  --dns-name-label finsense-backend \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your-key \
  --command-line "python run_backend.py"
```

### Heroku

1. **Create Procfile:**
```
web: python run_backend.py
dashboard: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create runtime.txt:**
```
python-3.10.12
```

3. **Deploy:**
```bash
heroku create finsense-app
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

## Environment Variables

Required environment variables:

- `OPENAI_API_KEY`: OpenAI API key for RAG chatbot
- `ALPHA_VANTAGE_API_KEY`: (Optional) Alpha Vantage API key
- `PORT`: Backend port (default: 8000)
- `HOST`: Backend host (default: 0.0.0.0)

## Health Checks

- Backend: `GET http://localhost:8000/health`
- Dashboard: Check if Streamlit is running on port 8501

## Monitoring

### Logs

**Docker:**
```bash
docker-compose logs -f backend
docker-compose logs -f dashboard
```

**Local:**
```bash
tail -f logs/api.log
```

### Metrics

Monitor:
- API response times
- Model prediction accuracy
- Resource usage (CPU, memory)
- Error rates

## Scaling

### Horizontal Scaling

Use a load balancer (nginx, AWS ALB) to distribute traffic:

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

### Vertical Scaling

Increase instance resources:
- CPU: For model training/prediction
- RAM: For large datasets
- Storage: For model files and data

## Security Best Practices

1. **Use secrets management:**
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager

2. **Enable HTTPS:**
   - Use reverse proxy (nginx) with SSL certificates
   - Or use cloud load balancer with SSL termination

3. **Restrict access:**
   - Use firewall rules
   - Implement authentication for dashboard
   - Rate limiting for API endpoints

4. **Keep dependencies updated:**
```bash
pip list --outdated
pip install --upgrade package-name
```

## Troubleshooting

### Port already in use
```bash
# Find process using port
lsof -i :8000
# Kill process
kill -9 <PID>
```

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a
# Rebuild without cache
docker build --no-cache -t finsense:latest .
```

### Out of memory
- Increase Docker memory limit
- Use smaller batch sizes for models
- Enable model quantization

## Production Checklist

- [ ] Set up environment variables securely
- [ ] Configure logging and monitoring
- [ ] Set up backup for models and data
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up health checks
- [ ] Configure auto-scaling (if needed)
- [ ] Set up CI/CD pipeline
- [ ] Document API endpoints
- [ ] Set up error alerting

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- Review documentation: `README.md`
- Open an issue on GitHub













