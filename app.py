"""
Golden Cross Scanner — web server.

This file does NOT run scans. It serves the dashboard and hands back
whatever results.json currently holds, so every request returns
instantly regardless of how big the ticker universe gets.

The actual scanning happens in run_scan.py, executed on a schedule by
GitHub Actions (see .github/workflows/scan.yml), which commits an
updated results.json back to the repo.
"""

import json
import os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


@app.route("/")
def dashboard():
    return send_from_directory("static", "index.html")


@app.route("/scan")
def scan():
    """Return the most recent saved scan. Instant — no live scanning."""
    try:
        with open(RESULTS_PATH) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return jsonify({
            "last_run": None,
            "results": [],
            "error": "No scan results yet — the scheduled job hasn't run.",
        }), 200
    except json.JSONDecodeError:
        return jsonify({
            "last_run": None,
            "results": [],
            "error": "Results file is corrupted.",
        }), 200

    return jsonify(payload)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
