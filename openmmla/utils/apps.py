from flask import Flask, jsonify


def create_app(class_type, endpoint, method_name, class_args=None, route_args=None):
    """Create Flask app for deploying restful services.

    Args:
        class_type (class): The class to instantiate (e.g., AudioEnhancer or AudioInferer).
        endpoint (str): The API endpoint name for the Flask route.
        method_name (str): The method name to call on the instantiated class.
        class_args (dict): Additional arguments required to instantiate the class.
        route_args (dict): Arguments needed to define the route function.

    Returns:
        app (Flask): The Flask application.
    """
    app = Flask(__name__)

    if class_args is None:
        class_args = {}
    if route_args is None:
        route_args = {}

    processor = class_type(**class_args)

    @app.route(f'/{endpoint}', methods=['POST'])
    def process_route():
        method = getattr(processor, method_name)
        return method(**route_args)

    # what the service runs (its backend, model, language ...), for the bases to note in
    # their session: a component's entry in the session document, `services`
    @app.route(f'/{endpoint}/info', methods=['GET'])
    def info_route():
        describe = getattr(processor, 'describe', None)
        info = describe() if callable(describe) else {'service': type(processor).__name__}
        return jsonify(info)

    return app
