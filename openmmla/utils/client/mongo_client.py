import logging
import yaml
from datetime import datetime, timezone
from typing import Any

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
except ModuleNotFoundError:
    MongoClient = None
    ASCENDING = 1

    class ConnectionFailure(Exception):
        """fallback when pymongo is unavailable during lightweight tests."""

    class DuplicateKeyError(Exception):
        """fallback when pymongo is unavailable during lightweight tests."""

from openmmla.utils.constants import MONGODB_DEFAULT_DB

logger = logging.getLogger(__name__)


class MongoDBClientWrapper:
    """MongoDB client that loads configuration from a YAML file and provides session management."""

    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, 'r'))
        mongo_config = config['MongoDB']

        self.url = mongo_config['url']
        self.db_name = mongo_config.get('db', MONGODB_DEFAULT_DB)
        if MongoClient is None:
            raise ModuleNotFoundError("pymongo is required to use MongoDBClientWrapper")

        try:
            self.client = MongoClient(self.url)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self._initialize_collections()
            logger.info("MongoDB connected: %s, db: %s", self.url, self.db_name)
        except ConnectionFailure as e:
            logger.exception("MongoDB connection failed: %s", e)
            raise

    def _initialize_collections(self):
        self.sessions = self.db['sessions']
        self.sessions.create_index([("session_id", ASCENDING)], unique=True)

    # ---- session CRUD ----

    def create_session(self, session_id: str, experiment_id: str, group_id: str,
                       participants: list[dict[str, str]] | None = None,
                       metadata: dict[str, Any] | None = None) -> bool:
        try:
            session_doc = {
                "session_id": session_id,
                "experiment_id": experiment_id,
                "group_id": group_id,
                "participants": participants or [],
                "start_time": datetime.now(timezone.utc),
                "end_time": None,
                "status": "active",
                "metadata": metadata or {},
            }
            self.sessions.insert_one(session_doc)
            logger.info("session created: %s", session_id)
            return True
        except DuplicateKeyError:
            logger.warning("session already exists: %s", session_id)
            return False
        except Exception as e:
            logger.warning("create_session failed: %s", e)
            return False

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            return self.sessions.find_one({"session_id": session_id}, {"_id": 0})
        except Exception as e:
            logger.warning("get_session failed: %s", e)
            return None

    def get_all_sessions(self) -> list[dict[str, Any]]:
        try:
            return list(self.sessions.find({}, {"_id": 0}).sort("start_time", -1))
        except Exception as e:
            logger.warning("get_all_sessions failed: %s", e)
            return []

    def get_sessions_by_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        try:
            return list(self.sessions.find(
                {"experiment_id": experiment_id}, {"_id": 0}
            ).sort("start_time", -1))
        except Exception as e:
            logger.warning("get_sessions_by_experiment failed: %s", e)
            return []

    def end_session(self, session_id: str) -> bool:
        try:
            result = self.sessions.update_one(
                {"session_id": session_id},
                {"$set": {
                    "end_time": datetime.now(timezone.utc),
                    "status": "ended",
                }},
            )
            return result.acknowledged
        except Exception as e:
            logger.warning("end_session failed: %s", e)
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            result = self.sessions.delete_one({"session_id": session_id})
            if result.deleted_count > 0:
                logger.info("session deleted: %s", session_id)
                return True
            return False
        except Exception as e:
            logger.warning("delete_session failed: %s", e)
            return False

    def close(self):
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")
