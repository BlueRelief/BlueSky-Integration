from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery_app = Celery(
    "bluesky_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
)

celery_app.conf.beat_schedule = {
    "collect-bluesky-data": {
        "task": "app.tasks.collect_and_analyze",
        "schedule": crontab(hour=f"*/{settings.SCHEDULE_HOURS}"),
    },
}

celery_app.conf.timezone = "UTC"
