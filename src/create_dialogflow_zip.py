"""Build and validate an importable Dialogflow ES agent archive.

The local classifier consumes ``data/intents.json`` directly.  Dialogflow-only
metadata in that file is optional and is translated here into ES intent,
parameter, entity, welcome, fallback, and fulfillment settings.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "intents.json"
ENTITIES_PATH = PROJECT_ROOT / "data" / "entities.json"
OUTPUT_ZIP_PATH = PROJECT_ROOT / "dialogflow_agent.zip"

LANGUAGE = "en"
UUID_NAMESPACE = uuid.UUID("a7a95d2d-384f-4c71-85c5-a2f52bc956db")
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)

FALLBACK_RESPONSES = [
    "I could not match that request. Please rephrase it or ask about TAR UMT programmes, admissions, fees, intakes, accommodation or campus services.",
    "I am not certain what you mean. Try a specific question such as 'Where is the official fee guide?' or 'How do I contact Admissions?'",
    "Sorry, I did not understand. For an important or time-sensitive matter, use TAR UMT's official contact directory: https://www.tarc.edu.my/contact-us/",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _stable_uuid(label: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, label))


def _normalise_pattern(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _safe_filename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe Dialogflow display name: {value!r}")
    return value


def _entity_name(data_type: str) -> str:
    if not isinstance(data_type, str) or not data_type.startswith("@"):
        raise ValueError(f"Entity data type must start with '@': {data_type!r}")
    return data_type[1:]


def _validate_source_data(
    intent_data: Mapping[str, Any], entity_data: Mapping[str, Any]
) -> Dict[str, Any]:
    intents = intent_data.get("intents")
    entities = entity_data.get("entities")
    if not isinstance(intents, list) or not intents:
        raise ValueError("data/intents.json must contain a non-empty 'intents' list")
    if not isinstance(entities, list) or not entities:
        raise ValueError("data/entities.json must contain a non-empty 'entities' list")

    entity_names: set[str] = set()
    entity_values = 0
    for entity in entities:
        name = entity.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every entity requires a non-empty name")
        if name in entity_names:
            raise ValueError(f"Duplicate entity name: {name}")
        entity_names.add(name)
        entries = entity.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Entity {name!r} has no entries")
        seen_values: set[str] = set()
        for entry in entries:
            value = entry.get("value")
            synonyms = entry.get("synonyms")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Entity {name!r} has an empty value")
            key = value.casefold()
            if key in seen_values:
                raise ValueError(f"Entity {name!r} repeats value {value!r}")
            seen_values.add(key)
            if not isinstance(synonyms, list) or not all(
                isinstance(item, str) and item.strip() for item in synonyms
            ):
                raise ValueError(f"Entity {name!r} value {value!r} needs synonyms")
            entity_values += 1

    tags: set[str] = set()
    patterns: Dict[str, str] = {}
    parameter_count = 0
    webhook_intents = 0
    welcome_intents = 0
    response_count = 0
    for intent in intents:
        tag = intent.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError("Every intent requires a non-empty tag")
        if tag in tags:
            raise ValueError(f"Duplicate intent tag: {tag}")
        tags.add(tag)

        intent_patterns = intent.get("patterns")
        responses = intent.get("responses")
        if not isinstance(intent_patterns, list) or len(intent_patterns) < 2:
            raise ValueError(f"Intent {tag!r} needs at least two patterns")
        if not isinstance(responses, list) or not responses or not all(
            isinstance(item, str) and item.strip() for item in responses
        ):
            raise ValueError(f"Intent {tag!r} needs at least one response")
        response_count += len(responses)

        for pattern in intent_patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(f"Intent {tag!r} has an empty pattern")
            key = _normalise_pattern(pattern)
            if key in patterns:
                raise ValueError(
                    f"Duplicate/conflicting pattern {pattern!r} in {tag!r}; "
                    f"already used by {patterns[key]!r}"
                )
            patterns[key] = tag

        dialogflow = intent.get("dialogflow", {})
        if not isinstance(dialogflow, dict):
            raise ValueError(f"Intent {tag!r} dialogflow metadata must be an object")
        if "WELCOME" in dialogflow.get("events", []):
            welcome_intents += 1
        if dialogflow.get("webhook_used"):
            webhook_intents += 1

        parameters = intent.get("parameters", [])
        if not isinstance(parameters, list):
            raise ValueError(f"Intent {tag!r} parameters must be a list")
        parameter_names: set[str] = set()
        for parameter in parameters:
            parameter_name = parameter.get("name")
            if not isinstance(parameter_name, str) or not parameter_name:
                raise ValueError(f"Intent {tag!r} has an unnamed parameter")
            if parameter_name in parameter_names:
                raise ValueError(
                    f"Intent {tag!r} repeats parameter {parameter_name!r}"
                )
            parameter_names.add(parameter_name)
            referenced_entity = _entity_name(parameter.get("entity", ""))
            if referenced_entity not in entity_names:
                raise ValueError(
                    f"Intent {tag!r} references missing entity {referenced_entity!r}"
                )
            prompts = parameter.get("prompts", [])
            if parameter.get("required") and not prompts:
                raise ValueError(
                    f"Required parameter {parameter_name!r} in {tag!r} needs a prompt"
                )
            parameter_count += 1

    if welcome_intents != 1:
        raise ValueError(
            f"Exactly one intent must handle the WELCOME event; found {welcome_intents}"
        )
    if webhook_intents < 1:
        raise ValueError("At least one intent must enable webhook fulfillment")
    if not any(
        intent.get("dialogflow", {}).get("webhook_used")
        and intent.get("parameters")
        for intent in intents
    ):
        raise ValueError("A webhook intent with structured parameters is required")

    return {
        "intents": len(intents),
        "patterns": len(patterns),
        "responses": response_count,
        "entities": len(entities),
        "entity_values": entity_values,
        "parameters": parameter_count,
        "webhook_intents": webhook_intents,
    }


def _build_entity_synonym_index(
    entities: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for entity in entities:
        synonyms: Dict[str, str] = {}
        for entry in entity["entries"]:
            for phrase in [entry["value"], *entry["synonyms"]]:
                synonyms.setdefault(phrase.casefold(), phrase)
        index[entity["name"]] = sorted(
            synonyms.values(), key=lambda value: (-len(value), value.casefold())
        )
    return index


def _find_parameter_spans(
    pattern: str,
    parameters: Sequence[Mapping[str, Any]],
    synonym_index: Mapping[str, Sequence[str]],
) -> List[Tuple[int, int, Mapping[str, Any]]]:
    spans: List[Tuple[int, int, Mapping[str, Any]]] = []
    occupied: List[Tuple[int, int]] = []
    for parameter in parameters:
        entity_name = _entity_name(parameter["entity"])
        for synonym in synonym_index[entity_name]:
            expression = re.compile(
                rf"(?<!\w){re.escape(synonym)}(?!\w)", re.IGNORECASE
            )
            match = expression.search(pattern)
            if not match:
                continue
            start, end = match.span()
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            spans.append((start, end, parameter))
            occupied.append((start, end))
            break
    return sorted(spans, key=lambda item: item[0])


def _build_usersays_data(
    pattern: str,
    parameters: Sequence[Mapping[str, Any]],
    synonym_index: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    spans = _find_parameter_spans(pattern, parameters, synonym_index)
    if not spans:
        return [{"text": pattern, "userDefined": False}]

    data: List[Dict[str, Any]] = []
    cursor = 0
    for start, end, parameter in spans:
        if start > cursor:
            data.append({"text": pattern[cursor:start], "userDefined": False})
        data.append(
            {
                "text": pattern[start:end],
                "alias": parameter["name"],
                "meta": parameter["entity"],
                "userDefined": True,
            }
        )
        cursor = end
    if cursor < len(pattern):
        data.append({"text": pattern[cursor:], "userDefined": False})
    return data


def _build_parameter(tag: str, parameter: Mapping[str, Any]) -> Dict[str, Any]:
    name = parameter["name"]
    return {
        "id": _stable_uuid(f"intent:{tag}:parameter:{name}"),
        "required": bool(parameter.get("required", False)),
        "dataType": parameter["entity"],
        "name": name,
        "value": f"${name}",
        "defaultValue": "",
        "isList": bool(parameter.get("is_list", False)),
        "prompts": [
            {"lang": LANGUAGE, "value": prompt}
            for prompt in parameter.get("prompts", [])
        ],
        "promptMessages": [],
        "noMatchPromptMessages": [],
        "noInputPromptMessages": [],
        "outputDialogContexts": [],
    }


def _build_intent(
    item: Mapping[str, Any], synonym_index: Mapping[str, Sequence[str]]
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], int]:
    tag = item["tag"]
    dialogflow = item.get("dialogflow", {})
    display_name = _safe_filename(dialogflow.get("display_name", tag))
    parameters = item.get("parameters", [])
    action = dialogflow.get("action", "")

    intent_json = {
        "id": _stable_uuid(f"intent:{display_name}"),
        "name": display_name,
        "auto": True,
        "contexts": [],
        "responses": [
            {
                "resetContexts": False,
                "action": action,
                "affectedContexts": [],
                "parameters": [
                    _build_parameter(tag, parameter) for parameter in parameters
                ],
                "messages": [
                    {
                        "type": 0,
                        "lang": LANGUAGE,
                        "condition": "",
                        "speech": item["responses"],
                    }
                ],
                "defaultResponsePlatforms": {},
                "speech": [],
            }
        ],
        "priority": 500000,
        "webhookUsed": bool(dialogflow.get("webhook_used", False)),
        "webhookForSlotFilling": bool(
            dialogflow.get("webhook_for_slot_filling", False)
        ),
        "fallbackIntent": False,
        "events": [
            {"name": event_name} for event_name in dialogflow.get("events", [])
        ],
        "conditionalResponses": [],
        "condition": "",
        "conditionalFollowupEvents": [],
    }

    usersays: List[Dict[str, Any]] = []
    annotated_phrases = 0
    for position, pattern in enumerate(item["patterns"]):
        phrase_data = _build_usersays_data(pattern, parameters, synonym_index)
        if any(part.get("userDefined") for part in phrase_data):
            annotated_phrases += 1
        usersays.append(
            {
                "id": _stable_uuid(
                    f"intent:{display_name}:usersays:{position}:{pattern}"
                ),
                "data": phrase_data,
                "isTemplate": False,
                "count": 0,
                "updated": 0,
            }
        )
    return display_name, intent_json, usersays, annotated_phrases


def _build_fallback_intent() -> Dict[str, Any]:
    display_name = "Default Fallback Intent"
    return {
        "id": _stable_uuid(f"intent:{display_name}"),
        "name": display_name,
        "auto": True,
        "contexts": [],
        "responses": [
            {
                "resetContexts": False,
                "action": "input.unknown",
                "affectedContexts": [],
                "parameters": [],
                "messages": [
                    {
                        "type": 0,
                        "lang": LANGUAGE,
                        "condition": "",
                        "speech": FALLBACK_RESPONSES,
                    }
                ],
                "defaultResponsePlatforms": {},
                "speech": [],
            }
        ],
        "priority": 500000,
        "webhookUsed": False,
        "webhookForSlotFilling": False,
        "fallbackIntent": True,
        "events": [],
        "conditionalResponses": [],
        "condition": "",
        "conditionalFollowupEvents": [],
    }


def _build_agent_json() -> Dict[str, Any]:
    return {
        "description": (
            "TAR UMT University Inquiry Chatbot student prototype. "
            "Time-sensitive facts route to official primary sources."
        ),
        "language": LANGUAGE,
        "shortDescription": "TAR UMT inquiry prototype",
        "examples": "",
        "linkToDocs": "",
        "disableInteractionLogs": False,
        "disableStackdriverLogs": True,
        "googleAssistant": {
            "googleAssistantCompatible": False,
            "welcomeIntentSignInRequired": False,
            "startIntents": [],
            "systemIntents": [],
            "endIntentIds": [],
            "oAuthLinking": {
                "required": False,
                "providerId": "",
                "authorizationUrl": "",
                "tokenUrl": "",
                "scopes": "",
                "privacyPolicyUrl": "",
                "grantType": "AUTH_CODE_GRANT",
            },
            "voiceType": "MALE_1",
            "capabilities": [],
            "env": "",
            "protocolVersion": "V2",
            "autoPreviewEnabled": False,
            "isDeviceAgent": False,
        },
        "defaultTimezone": "Asia/Kuala_Lumpur",
        "webhook": {
            "url": "",
            "username": "",
            "headers": {},
            "available": False,
            "useForDomains": False,
            "cloudFunctionsEnabled": False,
            "cloudFunctionsInitialized": False,
        },
        "isPrivate": True,
        "customClassifierMode": "use.after",
        "mlMinConfidence": 0.3,
        "supportedLanguages": [],
        "onePlatformApiVersion": "v2",
        "analyzeQueryTextSentiment": False,
        "enabledKnowledgeBaseNames": [],
        "knowledgeServiceConfidenceAdjustment": -0.4,
        "dialogBuilderMode": False,
    }


def _write_json(
    archive: zipfile.ZipFile, archive_path: str, value: Any
) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    info = zipfile.ZipInfo(archive_path, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload.encode("utf-8"))


def _validate_archive(
    archive_path: Path, expected: Mapping[str, int]
) -> Dict[str, int]:
    with zipfile.ZipFile(archive_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise ValueError(f"Corrupt ZIP member: {bad_file}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        if "agent.json" not in names:
            raise ValueError("ZIP is missing agent.json")

        intent_files = [
            name
            for name in names
            if name.startswith("intents/")
            and name.endswith(".json")
            and "_usersays_" not in name
        ]
        usersays_files = [
            name
            for name in names
            if name.startswith("intents/") and name.endswith("_usersays_en.json")
        ]
        entity_files = [
            name
            for name in names
            if name.startswith("entities/")
            and name.endswith(".json")
            and "_entries_" not in name
        ]
        entity_entry_files = [
            name
            for name in names
            if name.startswith("entities/") and name.endswith("_entries_en.json")
        ]

        # Source intents plus the generated default fallback intent.
        if len(intent_files) != expected["intents"] + 1:
            raise ValueError(
                f"Expected {expected['intents'] + 1} intent definitions, "
                f"found {len(intent_files)}"
            )
        if len(usersays_files) != expected["intents"]:
            raise ValueError(
                f"Expected {expected['intents']} usersays files, "
                f"found {len(usersays_files)}"
            )
        if len(entity_files) != expected["entities"]:
            raise ValueError(
                f"Expected {expected['entities']} entity definitions, "
                f"found {len(entity_files)}"
            )
        if len(entity_entry_files) != expected["entities"]:
            raise ValueError(
                f"Expected {expected['entities']} entity entry files, "
                f"found {len(entity_entry_files)}"
            )

        parsed = {
            name: json.loads(archive.read(name).decode("utf-8")) for name in names
        }
        intents = [parsed[name] for name in intent_files]
        if sum(intent.get("name") == "Default Welcome Intent" for intent in intents) != 1:
            raise ValueError("Archive must contain exactly one Default Welcome Intent")
        if sum(intent.get("fallbackIntent") is True for intent in intents) != 1:
            raise ValueError("Archive must contain exactly one fallback intent")
        if sum(intent.get("webhookUsed") is True for intent in intents) < 1:
            raise ValueError("Archive contains no webhook-enabled intent")

        annotated_segments = 0
        for name in usersays_files:
            for phrase in parsed[name]:
                annotated_segments += sum(
                    part.get("userDefined") is True for part in phrase.get("data", [])
                )
        if annotated_segments < 1:
            raise ValueError("No custom entity annotations were exported")

    return {
        "zip_members": len(names),
        "intent_definitions": len(intent_files),
        "usersays_files": len(usersays_files),
        "entity_definitions": len(entity_files),
        "entity_entry_files": len(entity_entry_files),
        "annotated_segments": annotated_segments,
    }


def build_dialogflow_zip() -> Dict[str, int]:
    intent_data = _load_json(DATASET_PATH)
    entity_data = _load_json(ENTITIES_PATH)
    source_counts = _validate_source_data(intent_data, entity_data)
    entities = entity_data["entities"]
    synonym_index = _build_entity_synonym_index(entities)

    fd, temporary_name = tempfile.mkstemp(
        prefix="dialogflow_agent_", suffix=".zip", dir=PROJECT_ROOT
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    annotated_phrases = 0
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            _write_json(archive, "agent.json", _build_agent_json())

            for entity in entities:
                name = _safe_filename(entity["name"])
                entity_definition = {
                    "id": _stable_uuid(f"entity:{name}"),
                    "name": name,
                    "isOverridable": bool(entity.get("is_overridable", True)),
                    "isEnum": bool(entity.get("is_enum", False)),
                    "automatedExpansion": bool(
                        entity.get("automated_expansion", False)
                    ),
                    "allowFuzzyExtraction": bool(
                        entity.get("allow_fuzzy_extraction", False)
                    ),
                }
                _write_json(archive, f"entities/{name}.json", entity_definition)
                _write_json(
                    archive,
                    f"entities/{name}_entries_{LANGUAGE}.json",
                    entity["entries"],
                )

            for item in intent_data["intents"]:
                display_name, intent_json, usersays, annotated = _build_intent(
                    item, synonym_index
                )
                annotated_phrases += annotated
                _write_json(archive, f"intents/{display_name}.json", intent_json)
                _write_json(
                    archive,
                    f"intents/{display_name}_usersays_{LANGUAGE}.json",
                    usersays,
                )

            fallback_name = "Default Fallback Intent"
            _write_json(
                archive,
                f"intents/{fallback_name}.json",
                _build_fallback_intent(),
            )

        archive_counts = _validate_archive(temporary_path, source_counts)
        os.replace(temporary_path, OUTPUT_ZIP_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    result = {**source_counts, **archive_counts}
    result["annotated_phrases"] = annotated_phrases
    print(
        "Dialogflow ES archive created: "
        f"{OUTPUT_ZIP_PATH}\n"
        f"  source intents={result['intents']}, patterns={result['patterns']}, "
        f"responses={result['responses']}\n"
        f"  custom entities={result['entities']}, "
        f"entity values={result['entity_values']}, parameters={result['parameters']}\n"
        f"  webhook intents={result['webhook_intents']}, "
        f"annotated phrases={result['annotated_phrases']}, "
        f"ZIP members={result['zip_members']}"
    )
    return result


if __name__ == "__main__":
    build_dialogflow_zip()
