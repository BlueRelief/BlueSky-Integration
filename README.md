# BlueSky Disaster Monitoring

Automated data collection from BlueSky with AI-powered disaster analysis using PostgreSQL for efficient storage and deduplication.

## 🎯 Features

- **Automated Data Collection**: Daily scheduled pulls from BlueSky
- **AI Analysis**: Google Gemini powered disaster severity assessment
- **PostgreSQL Database**: Efficient storage with automatic deduplication
- **REST API**: Access collected data, statistics, and analysis
- **Task Queue**: Celery + Redis for robust background processing
- **Docker**: Fully containerized with docker-compose

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   BlueSky   │────>│  FastAPI App │────>│ PostgreSQL  │
│     API     │     │   + Celery   │     │  Database   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Gemini    │
                    │      AI      │
                    └──────────────┘
```

## 📊 Database Schema

### Collections
- `collection_runs`: Tracks each data collection run
- `posts`: Stores BlueSky posts (deduplicated by post ID)
- `disasters`: AI-extracted disaster information

## 🚀 Quick Start

### 1. Setup Environment

```bash
cp env.example .env
# Edit .env with your credentials
```

### 2. Start Services

```bash
docker-compose up -d
```

This will start:
- **PostgreSQL** (port 5432): Database
- **Redis** (port 6379): Message broker
- **API** (port 8000): FastAPI server
- **Celery Worker**: Background task processor
- **Celery Beat**: Task scheduler

### 3. Verify Services

```bash
# Check all services are running
docker-compose ps

# Test API
curl http://localhost:8000/api/

# Check statistics
curl http://localhost:8000/api/stats
```

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Service info and endpoint list |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Collection statistics |

### Data Collection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trigger` | POST | Manually trigger data collection |
| `/api/task/{task_id}` | GET | Get task status |

### Data Access

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/disasters` | GET | Get recent disasters |
| `/api/disasters/{id}` | GET | Get specific disaster |
| `/api/posts/recent` | GET | Get recent posts |
| `/api/runs` | GET | Get collection runs |

### Examples

```bash
# Get statistics
curl http://localhost:8000/api/stats

# Trigger manual collection
curl -X POST http://localhost:8000/api/trigger

# Get recent disasters
curl http://localhost:8000/api/disasters?limit=10

# Get recent posts
curl http://localhost:8000/api/posts/recent?limit=20
```

## 🗄️ Database Benefits

### Before (Files)
- ❌ 213KB JSON files per run
- ❌ Duplicate posts across runs
- ❌ No query capability
- ❌ Manual deduplication needed
- ❌ Slow file parsing

### After (PostgreSQL)
- ✅ Efficient normalized storage
- ✅ Automatic deduplication
- ✅ Fast queries and filtering
- ✅ Historical tracking
- ✅ Structured data analysis

## 🔧 Configuration

Edit `.env` file:

```env
# BlueSky Credentials
BlueSky_Username=your-username.bsky.social
BlueSky_Password=your-app-password

# Google Gemini API
GOOGLE_API_KEY=your-google-api-key

# Collection Settings
SEARCH_HASHTAG=#earthquake
POST_LIMIT=50
SCHEDULE_HOURS=24

# Database
DATABASE_URL=postgresql://bluesky:bluesky123@postgres:5432/bluesky

# Redis
REDIS_URL=redis://redis:6379/0
```

## 🛠️ Development

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f celery-beat
```

### Restart Services

```bash
# All services
docker-compose restart

# Specific service
docker-compose restart api
```

### Access Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U bluesky -d bluesky

# Example queries
SELECT COUNT(*) FROM posts;
SELECT * FROM disasters ORDER BY extracted_at DESC LIMIT 10;
SELECT * FROM collection_runs ORDER BY started_at DESC;
```

## 📈 Data Flow

1. **Celery Beat** triggers task every 24 hours
2. **Celery Worker** executes collection task:
   - Creates new collection run in DB
   - Fetches posts from BlueSky API
   - Saves posts (deduplicated by post ID)
   - Sends posts to Gemini AI
   - Parses and saves disaster information
   - Marks run as completed
3. **API** serves data from PostgreSQL

## 🧹 Maintenance

### Clean Up Old Data

```sql
-- Delete old collection runs (keeps last 30 days)
DELETE FROM collection_runs 
WHERE started_at < NOW() - INTERVAL '30 days';

-- Vacuum database
VACUUM ANALYZE;
```

### Backup Database

```bash
docker-compose exec postgres pg_dump -U bluesky bluesky > backup.sql
```

### Restore Database

```bash
docker-compose exec -T postgres psql -U bluesky bluesky < backup.sql
```

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres psql -U bluesky -d bluesky -c "SELECT 1;"
```

### Task Not Running

```bash
# Check Celery worker
docker-compose logs celery-worker

# Check Celery beat
docker-compose logs celery-beat

# Verify Redis
docker-compose exec redis redis-cli ping
```

### API Errors

```bash
# Check API logs
docker-compose logs api

# Restart API
docker-compose restart api
```

## 📦 Tech Stack

- **Python 3.13**: Core language
- **FastAPI**: Web framework
- **PostgreSQL 16**: Database
- **SQLAlchemy**: ORM
- **Celery**: Task queue
- **Redis**: Message broker
- **Google Gemini**: AI analysis
- **BlueSky API**: Data source
- **Docker**: Containerization

## 📝 Project Structure

```
BlueSky-Integration/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── models.py            # SQLAlchemy models
│   ├── celery_app.py        # Celery config
│   ├── tasks.py             # Celery tasks
│   ├── routes/
│   │   └── api.py           # API endpoints
│   └── services/
│       ├── bluesky.py       # BlueSky integration
│       ├── analysis.py      # AI analysis
│       └── database_service.py  # Database operations
├── demo.ipynb               # Jupyter notebook
├── docker-compose.yml       # Multi-service setup
├── Dockerfile               # Container image
├── requirements.txt         # Python dependencies
├── env.example              # Environment template
└── README.md                # This file
```

## 🎓 Next Steps

1. **Implement Filtering**: Add location/severity filters
2. **Add Notifications**: Email/Slack alerts for high-severity events
3. **Visualization**: Create dashboard with charts
4. **Export**: Add CSV/JSON export endpoints
5. **Search**: Full-text search on posts
6. **Aggregations**: Daily/weekly summaries

## 📄 License

MIT License - feel free to use and modify!