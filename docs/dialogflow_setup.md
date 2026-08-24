# Dialogflow ES import, setup and evidence guide

## What this package contains

`dialogflow_agent.zip` is generated locally from:

- `data/intents.json`: 36 source intents, 432 unique training patterns and 36
  static responses.
- `data/entities.json`: 5 custom entity types with 41 canonical values.
- `src/create_dialogflow_zip.py`: deterministic ES JSON exporter and validator.

The ZIP contains 37 intent definitions because the exporter adds one complete
`Default Fallback Intent`. The source `greeting` intent is exported as the
complete `Default Welcome Intent` with the `WELCOME` event. It also contains 36
English usersays files, 5 entity definitions and 5 English entity-entry files.

## Import into Dialogflow ES

1. Create or select a **Dialogflow ES** agent, not Dialogflow CX.
2. Set the agent language to English and timezone to Asia/Kuala_Lumpur.
3. Open **Agent settings > Export and Import**.
4. Use **Import from ZIP** for a merge, or **Restore from ZIP** only when it is
   acceptable to replace the draft agent.
5. Select `dialogflow_agent.zip` and allow training to finish.
6. In **Intents**, confirm there is exactly one `Default Welcome Intent` and one
   `Default Fallback Intent`.
7. In **Entities**, confirm `course_programme`, `campus_service`,
   `campus_service_directory`, `intake`, and `contact_channel` are present with
   their synonyms.

Google's ES import/export documentation:
https://cloud.google.com/dialogflow/es/docs/agents-settings

## Custom entities and parameters

The entities are connected to actual intent parameters and annotated training
phrases; they are not decorative exports.

| Entity | Parameter use | Example phrase |
|---|---|---|
| `@course_programme` | `course.programme` | "do you offer computer science" |
| `@intake` | `course.intake`, `intake.intake` | "programmes available for the June intake" |
| `@campus_service` | `transport.service`, `assessment_rules.service` | "campus shuttle" |
| `@campus_service_directory` | `campus_service_lookup.service`; canonical values are official URLs | "library" resolves to the official Library opening-hours URL |
| `@contact_channel` | `contact.contact_channel`, `campus_service_lookup.contact_channel` | "contact student affairs by phone" |

The `service` slot is required in `campus_service_lookup`. A query such as "I
need help from a campus service" should therefore prompt the user to select a
service.

## Static-response and slot-filling design

This submission deliberately does **not** use webhook fulfillment. All intents
return controlled static responses, so the imported agent has no external
service, credential, deployment, or network dependency.

- `course` extracts optional `programme` and `intake` parameters and returns a
  controlled response containing the official programme finder.
- `campus_service_lookup` extracts required `service` and optional
  `contact_channel`. When `service` is absent, Dialogflow prompts the user to
  choose a campus service. The dedicated directory entity normalises that answer
  to the relevant official URL; the static response uses `$service.original`
  for the user's service name and `$service` for the official URL.

After import, confirm that **Enable webhook call for this intent** is off. The
parameters, entity annotations, required-slot prompt, welcome/fallback handling,
and official-source static responses are the platform extensions demonstrated
for this assignment.

## Local reproducibility and validation

From the project root:

```bash
python3 src/create_dialogflow_zip.py
unzip -t dialogflow_agent.zip
python3 -m json.tool data/intents.json >/dev/null
python3 -m json.tool data/entities.json >/dev/null
```

The exporter fails before replacement if it finds a duplicate normalized
training phrase, duplicate tag/entity, missing response, unknown entity
reference, required slot without a prompt, enabled webhook call, missing WELCOME
event, corrupt JSON, or an incomplete ZIP structure.

## Cloud evidence checklist

After a real ES import, capture dated screenshots or screen recordings of:

1. Agent settings showing **Dialogflow ES**, language and timezone.
2. The five custom entity pages with representative values/synonyms.
3. `Default Welcome Intent` showing the WELCOME event and a successful welcome
   invocation.
4. `Default Fallback Intent` responding to at least two out-of-scope phrases.
5. `course` showing its action, parameters, annotations, static response, and
   disabled webhook toggle.
6. `campus_service_lookup` showing required slot filling, extracted `service`
   and optional `contact_channel`.
7. `transport` correctly routing bus, shuttle, route and timetable queries to
   the official live bus-service page, without a cached timetable.
8. `assessment_rules` correctly routing passing-mark, grading and result-review
   questions to the Student Intranet/official Examination FAQ without claiming
   one universal passing mark.
9. A test table containing query, expected intent, matched intent, confidence,
   extracted parameters, response, date and pass/fail result.

## Evidence boundary

Local checks prove JSON validity, archive integrity, deterministic package
structure, intent/entity counts, parameter wiring and entity annotations. They
do **not** prove that Google accepted the import, trained the cloud model,
or produced a particular cloud confidence score. Do not label
local simulator results as a live Dialogflow test. Cloud claims require the
dated evidence listed above.
