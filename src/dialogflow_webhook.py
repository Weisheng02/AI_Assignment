"""Minimal, credential-free Dialogflow ES webhook for the assignment demo.

The handler returns parameter-aware links to maintained official TAR UMT pages.
It does not scrape pages, store personal data, or claim that the endpoint is
already deployed. Deploy behind HTTPS before connecting it in Dialogflow ES.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PROGRAMME_URL = "https://www.tarc.edu.my/admissions/programmes/programme-offered-a-z/undergraduate-programme/"
INTAKE_URL = "https://dace.tarc.edu.my/programmes/intakes"
CONTACT_URL = "https://www.tarc.edu.my/contact-us/"
ALLOWED_ACTIONS = {"programme.lookup", "campus.service.lookup"}


def _parameter_text(parameters, name):
    value = parameters.get(name, "") if isinstance(parameters, dict) else ""
    if isinstance(value, dict):
        value = value.get("resolvedValue") or value.get("name") or ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()[:100]


def build_fulfillment_response(payload):
    """Validate a Dialogflow ES V2 request and create a safe response."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    query_result = payload.get("queryResult")
    if not isinstance(query_result, dict):
        raise ValueError("queryResult is required")
    action = str(query_result.get("action") or "").strip()
    if action not in ALLOWED_ACTIONS:
        return {
            "fulfillmentText": "I cannot perform that action. Please continue with the chatbot's verified FAQ responses.",
            "source": "assignment-dialogflow-webhook",
        }

    parameters = query_result.get("parameters") or {}
    if action == "programme.lookup":
        programme = _parameter_text(parameters, "programme")
        intake = _parameter_text(parameters, "intake")
        subject = programme or "your selected programme"
        intake_note = f" for the {intake} intake" if intake else ""
        text = (
            f"Please verify {subject}{intake_note} in TAR UMT's official programme directory: "
            f"{PROGRAMME_URL} Current intake notices are available at {INTAKE_URL}"
        )
    else:
        service = _parameter_text(parameters, "service")
        channel = _parameter_text(parameters, "contact_channel")
        subject = service or "the required campus service"
        channel_note = f" by {channel}" if channel else ""
        text = (
            f"Use TAR UMT's official contact directory to find {subject}{channel_note}: "
            f"{CONTACT_URL} Confirm the current details before contacting the department."
        )
    return {"fulfillmentText": text, "source": "assignment-dialogflow-webhook"}


class DialogflowWebhookHandler(BaseHTTPRequestHandler):
    def _write_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/dialogflow-webhook":
            self._write_json(404, {"error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 65536)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._write_json(200, build_fulfillment_response(payload))
        except (ValueError, json.JSONDecodeError) as error:
            self._write_json(400, {"error": str(error)})

    def log_message(self, format, *args):
        return


def main(host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, port), DialogflowWebhookHandler)
    print(f"Dialogflow webhook listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
