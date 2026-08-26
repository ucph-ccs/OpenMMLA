import redis

from openmmla.utils.config import load_config_with_system_services


class RedisClientWrapper(redis.Redis):
    """Extended Redis client that loads configuration from a file and adds custom functionalities."""

    def __init__(self, config_path):
        """Initialize a Redis client with configurations"""
        config = load_config_with_system_services(config_path)
        super().__init__(
            host=config['Redis']['host'],
            port=int(config['Redis']['port']),
            db=int(config['Redis']['db']),
            health_check_interval=10,
            socket_timeout=10,
            socket_keepalive=True,
            socket_connect_timeout=10,
            retry_on_timeout=True
        )

    def subscribe(self, topic):
        """Subscribe to a topic and return a pubsub object"""
        p = self.pubsub()
        p.subscribe(topic)
        return p
