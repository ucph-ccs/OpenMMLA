import configparser

import paho.mqtt.client as mqtt


class MQTTClientWrapper(mqtt.Client):
    """Extended MQTT client that loads configuration from a file and adds custom functionalities."""

    def __init__(self, config_path, user_data=None, on_message=None, topics=None):
        """Initialize MQTT client with configurations and optional user data, message callback, and topic
        subscription."""
        super().__init__(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

        self.config_path = config_path
        self.current_topics = []

        config = configparser.ConfigParser()
        config.read(config_path)
        self.connect(config['MQTT']['mqtt_host'], int(config['MQTT']['mqtt_port']), 60)

        if user_data:
            self.user_data_set(user_data)
        if on_message:
            self.on_message = on_message
        if topics:
            self.subscribe(topics)

    def subscribe(self, topics, qos=0, options=None, properties=None):
        """Override: subscribe to new topics, ensuring previous subscriptions are removed."""
        if isinstance(topics, str):
            topics = [(topics, qos)]
        elif isinstance(topics, tuple):
            topics = [topics]

        if self.current_topics:
            self.unsubscribe(self.current_topics)
        for topic, qos in topics:
            super().subscribe(topic, qos)
        self.current_topics = topics

    def unsubscribe_current(self):
        """Unsubscribe from all currently subscribed topics."""
        if self.current_topics:
            for topic, _ in self.current_topics:
                self.unsubscribe(topic)
            self.current_topics = []

    def reinitialise(self, user_data=None, on_message=None, topics=None):
        """Override: reinitialize the client with new user data, on_message callback, and optionally change the topic
        subscription."""
        self._reset_sockets()
        self.__init__(config_path=self.config_path, user_data=user_data, on_message=on_message, topics=topics)
