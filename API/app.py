"""
Main application and routing logic
"""
# Standard imports
import os

#  Database + Heroku + Postgres

from flask import Flask, jsonify, request
from models.nearest_neighbors_model import predict


def create_app():
    """Create and configure an instance of the Flask application"""
    app = Flask(__name__)

    @app.route('/')
    def root():
        return "Welcome to Med Cab"

    @app.route("/test", methods=['POST', 'GET'])
    def predict_strain():
        text = request.get_json(force=True)
        predictions = predict(text)
        return jsonify(predictions)
    
    return app
