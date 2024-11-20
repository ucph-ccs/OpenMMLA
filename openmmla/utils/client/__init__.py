def __getattr__(name):
    if name == "InfluxDBClientWrapper":
        from .influx_client import InfluxDBClientWrapper
        return InfluxDBClientWrapper
    elif name == "MQTTClientWrapper":
        from .mqtt_client import MQTTClientWrapper
        return MQTTClientWrapper
    elif name == "RedisClientWrapper":
        from .redis_client import RedisClientWrapper
        return RedisClientWrapper
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
