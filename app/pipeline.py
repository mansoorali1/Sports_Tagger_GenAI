"""
pipeline.py — Core Sports Auto-Tagger Pipeline

Converts football commentary (structured or unstructured) into
multi-format match highlights using TF-IDF+SVM classification
and LLaMA-3 via Groq API.

Used by both api.py (FastAPI) and streamlit_app.py (Streamlit UI).
"""

# import os
# import re
# import json
# import joblib
# import pickle
# import unicodedata
# import numpy as np
# from groq import Groq
# import httpx
    

# # ── Label and impact mappings ──────────────────────────────────────

# IMPACT_SCORES = {
#     'Goal': 5, 'Red Card': 5, 'Penalty': 4,
#     'Yellow Card': 3, 'Substitution': 2,
#     'Attempt': 1, 'Corner': 1, 'Foul': 1,
#     'Free Kick Won': 1, 'Offside': 0, 'Hand Ball': 1
# }

# NON_PLAYER_PHRASES = {
#     'italy serie', 'france ligue', 'spain la liga', 'la liga',
#     'england premier league', 'premier league', 'germany bundesliga',
#     'bundesliga', 'serie a', 'ligue 1', 'the match', 'the hosts',
#     'the visitors', 'the saints', 'the home', 'the away'
# }

# SKIP_KEYWORDS = [
#     'serie', 'ligue', 'liga', 'league', 'bundesliga', 'premier',
#     'cup', 'champions', 'europa', 'match', 'half', 'minute', 'season'
# ]


# # ── Model loading ──────────────────────────────────────────────────

# def load_artifacts(artifacts_dir: str = 'artifacts'):
#     """
#     Loads classifier, vectorizer, and label mappings.
#     Called once at app startup and cached.
#     """
#     svm_clf   = joblib.load(f'{artifacts_dir}/svm_classifier.joblib')
#     tfidf_vec = joblib.load(f'{artifacts_dir}/tfidf_vectorizer.joblib')

#     with open(f'{artifacts_dir}/id2label.pkl', 'rb') as f:
#         id2label = pickle.load(f)
#     with open(f'{artifacts_dir}/label2id.pkl', 'rb') as f:
#         label2id = pickle.load(f)

#     return svm_clf, tfidf_vec, id2label, label2id


# # ── Text segmentation ──────────────────────────────────────────────

# # def segment_commentary(raw_text: str) -> list:
# #     """
# #     Splits raw commentary block into individual event segments.
# #     Extracts minute timestamps where present.
# #     """
# #     segments = []
# #     lines    = raw_text.strip().split('\n')

# #     for line in lines:
# #         line = line.strip()
# #         if not line or len(line.split()) < 4:
# #             continue
# #         if re.match(r'^\d+[\'\:]?\s*$', line):
# #             continue

# #         minute = None
# #         text   = line

# #         for pattern in [r'^\[?(\d+)\+?\d*\]?[\'\:\-\s]+',
# #                         r'^\((\d+)\)[\s]+']:
# #             match = re.match(pattern, line)
# #             if match:
# #                 minute = int(match.group(1))
# #                 text   = line[match.end():].strip()
# #                 break

# #         if len(text.split()) >= 4:
# #             segments.append({'minute': minute, 'text': text})

# #     return segments
# def segment_commentary(raw_text: str) -> list:
#     segments = []
#     lines = raw_text.strip().split('\n')
    
#     current_minute = None  # State variable to track isolated timestamps

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
            
#         # Catch isolated minute markers like 90'+6' or 82'
#         minute_match = re.match(r'^(\d+)(?:\+\d*)?\'?\s*$', line)
#         if minute_match:
#             current_minute = int(minute_match.group(1))
#             continue # Move to the next line containing the text

#         if len(line.split()) < 4:
#             continue

#         minute = current_minute
#         text = line

#         # Standard inline regex check if timestamp is on the same line
#         for pattern in [r'^\[?(\d+)\+?\d*\]?[\'\:\-\s]+', r'^\((\d+)\)[\s]+']:
#             match = re.match(pattern, line)
#             if match:
#                 minute = int(match.group(1))
#                 text = line[match.end():].strip()
#                 break

#         if len(text.split()) >= 4:
#             segments.append({'minute': minute, 'text': text})
#             current_minute = None # Reset after assignment

#     return segments

# # ── Classification ─────────────────────────────────────────────────

# def classify_segments(segments: list,
#                        svm_clf,
#                        tfidf_vec,
#                        id2label: dict) -> list:
#     """
#     Classifies each segment using TF-IDF + SVM.
#     Returns segments with label, confidence, and impact score.
#     """
#     if not segments:
#         return []

#     texts     = [seg['text'] for seg in segments]
#     X         = tfidf_vec.transform(texts)
#     preds     = svm_clf.predict(X)
#     probas    = svm_clf.predict_proba(X)
#     max_proba = probas.max(axis=1)

#     return [
#         {
#             'minute'    : seg['minute'],
#             'text'      : seg['text'],
#             'label'     : id2label[preds[i]],
#             'confidence': round(float(max_proba[i]), 3),
#             'impact'    : IMPACT_SCORES.get(id2label[preds[i]], 0)
#         }
#         for i, seg in enumerate(segments)
#     ]


# # ── Timeline builder ───────────────────────────────────────────────

# def build_timeline(classified_segments: list,
#                    home_team: str = 'Home Team',
#                    away_team: str = 'Away Team',
#                    score: str = 'Unknown',
#                    competition: str = '',
#                    match_date: str = '',
#                    min_impact: int = 2,
#                    min_confidence: float = 0.6) -> tuple:
#     """
#     Filters classified segments and builds structured timeline
#     string for the LLM prompt.
#     """
#     filtered = [
#         s for s in classified_segments
#         if s['impact'] >= min_impact and s['confidence'] >= min_confidence
#     ]
#     if not filtered:
#         filtered = [
#             s for s in classified_segments
#             if s['impact'] >= 1 and s['confidence'] >= 0.5
#         ]

#     filtered.sort(key=lambda x: x['minute'] if x['minute'] else 999)

#     header = (
#         f"Match: {home_team} vs {away_team}\n"
#         f"Score: {score}\n"
#         f"Competition: {competition}\n"
#         f"Date: {match_date}\n\n"
#         f"Key Events:\n"
#     )
#     lines = [
#         f"  {'Minute ' + str(s['minute']) if s['minute'] else 'Unknown'}: "
#         f"{s['label'].upper()} — {s['text']}"
#         for s in filtered
#     ]

#     return header + '\n'.join(lines), filtered


# # ── LLM generation ─────────────────────────────────────────────────

# SYSTEM_PROMPT = (
#     "You are a senior sports journalist writing for DAZN, "
#     "a global sports streaming platform. Convert structured match event data "
#     "into compelling, accurate, publish-ready content.\n\n"
#     "STRICT RULES:\n"
#     "- Report ONLY events present in the data. Never invent details.\n"
#     "- Include ALL goals in key_moments, including late goals.\n"
#     "- Do not claim a player scored more goals than shown in the data.\n"
#     "- Red cards must be mentioned in the full summary if present.\n"
#     "- Push notification must be under 120 characters.\n\n"
#     "Return ONLY valid JSON with exactly these keys:\n"
#     '{\n'
#     '  "full_summary": "150-200 word match narrative",\n'
#     '  "executive_summary": "2-3 sentence version",\n'
#     '  "push_notification": "under 120 characters",\n'
#     '  "player_of_match": "Name and one sentence reason",\n'
#     '  "key_moments": [\n'
#     '    {"minute": "45", "event": "Goal", "description": "one line"}\n'
#     '  ]\n'
#     '}'
# )


# def generate_highlights(timeline_text: str, groq_api_key: str) -> dict:
#     """
#     Sends timeline to LLaMA-3 via Groq and returns parsed highlights.
#     """
#     http_client = httpx.Client()
#     client = Groq(api_key=groq_api_key, http_client=http_client)

#     response = client.chat.completions.create(
#         model='llama-3.1-8b-instant',
#         temperature=0.3,
#         max_tokens=1500,
#         messages=[
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user',
#              'content': f'Generate match content:\n\n{timeline_text}'}
#         ]
#     )
#     raw = response.choices[0].message.content
#     return parse_llm_output(raw)


# def parse_llm_output(raw: str) -> dict:
#     """Parses JSON from LLM response handling markdown fences."""
#     text = raw.strip()
#     if text.startswith('```'):
#         lines = text.split('\n')
#         text  = '\n'.join(lines[1:-1]) if lines[-1] == '```' \
#                 else '\n'.join(lines[1:])
#     start = text.find('{')
#     end   = text.rfind('}') + 1
#     if start >= 0 and end > start:
#         text = text[start:end]
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         return {
#             'full_summary'     : raw,
#             'executive_summary': raw[:300],
#             'push_notification': raw[:120],
#             'player_of_match'  : 'See full summary',
#             'key_moments'      : []
#         }


# # ── Guardrail ──────────────────────────────────────────────────────

# def run_guardrail(highlights: dict,
#                   classified_segments: list) -> dict:
#     """
#     Post-generation fact verification guardrail.
#     Checks LLM output against classified source events.

#     Adapted for unstructured input — uses classified segments
#     as ground truth instead of events.csv columns.

#     Returns verdict: PASS / WARN / FAIL
#     """
#     full_text  = highlights.get('full_summary', '').lower()
#     key_moments = highlights.get('key_moments', [])
#     hard_issues = []
#     warnings    = []

#     # Ground truth from classified segments
#     goal_events = [s for s in classified_segments
#                    if s['label'] == 'Goal' and s['impact'] >= 4]
#     red_events  = [s for s in classified_segments
#                    if s['label'] == 'Red Card']

#     # Check goal count in key moments
#     goals_in_moments = sum(
#         1 for km in key_moments
#         if 'goal' in km.get('event', '').lower()
#     )
#     if abs(goals_in_moments - len(goal_events)) > 1:
#         warnings.append(
#             f'Goal count mismatch: LLM={goals_in_moments}, '
#             f'classified={len(goal_events)}'
#         )

#     # Check red cards mentioned
#     for rc in red_events:
#         # Try to find any distinctive word from the red card text
#         rc_words = [w for w in rc['text'].lower().split()
#                     if len(w) > 4 and w not in ['footed', 'shot', 'from']]
#         found = any(word in full_text for word in rc_words[:3])
#         if not found:
#             warnings.append('Red card event may not be mentioned in summary')

#     # Check push notification length
#     notif = highlights.get('push_notification', '')
#     if len(notif) > 120:
#         hard_issues.append(
#             f'Push notification too long: {len(notif)} chars (limit 120)'
#         )

#     verdict = 'FAIL' if hard_issues else 'WARN' if warnings else 'PASS'

#     return {
#         'verdict'         : verdict,
#         'hard_issues'     : hard_issues,
#         'warnings'        : warnings,
#         'guardrail_passed': verdict != 'FAIL'
#     }


# # ── Main pipeline entry point ──────────────────────────────────────

# def run_pipeline(raw_text: str,
#                  home_team: str,
#                  away_team: str,
#                  score: str,
#                  competition: str,
#                  match_date: str,
#                  groq_api_key: str,
#                  artifacts_dir: str = 'artifacts') -> dict:
#     """
#     Full end-to-end pipeline for unstructured commentary input.
#     Called by both FastAPI and Streamlit.

#     Returns dict with all outputs and guardrail result.
#     """
#     svm_clf, tfidf_vec, id2label, label2id = load_artifacts(artifacts_dir)

#     segments   = segment_commentary(raw_text)
#     classified = classify_segments(segments, svm_clf, tfidf_vec, id2label)
#     timeline, key_events = build_timeline(
#         classified, home_team, away_team, score, competition, match_date
#     )
#     highlights = generate_highlights(timeline, groq_api_key)
#     guardrail  = run_guardrail(highlights, classified)

#     return {
#         'match_info'      : {
#             'home_team'  : home_team,
#             'away_team'  : away_team,
#             'score'      : score,
#             'competition': competition,
#             'date'       : match_date
#         },
#         'segments_found'  : len(segments),
#         'key_events_used' : len(key_events),
#         'classified'      : classified,
#         'timeline'        : timeline,
#         'highlights'      : highlights,
#         'guardrail'       : guardrail
#     }
"""
pipeline.py — Core Sports Auto-Tagger Pipeline

Converts football commentary (structured or unstructured) into
multi-format match highlights using TF-IDF+SVM classification
and LLaMA-3 via Groq API.

Used by both api.py (FastAPI) and streamlit_app.py (Streamlit UI).
"""

import os
import re
import json
import joblib
import pickle
import unicodedata
import numpy as np
from groq import Groq


# ── Label and impact mappings ──────────────────────────────────────

IMPACT_SCORES = {
    'Goal': 5, 'Red Card': 5, 'Penalty': 4,
    'Yellow Card': 3, 'Substitution': 2,
    'Attempt': 1, 'Corner': 1, 'Foul': 1,
    'Free Kick Won': 1, 'Offside': 0, 'Hand Ball': 1
}

NON_PLAYER_PHRASES = {
    'italy serie', 'france ligue', 'spain la liga', 'la liga',
    'england premier league', 'premier league', 'germany bundesliga',
    'bundesliga', 'serie a', 'ligue 1', 'the match', 'the hosts',
    'the visitors', 'the saints', 'the home', 'the away'
}

SKIP_KEYWORDS = [
    'serie', 'ligue', 'liga', 'league', 'bundesliga', 'premier',
    'cup', 'champions', 'europa', 'match', 'half', 'minute', 'season'
]


# ── Model loading ──────────────────────────────────────────────────

def load_artifacts(artifacts_dir: str = 'artifacts'):
    """
    Loads classifier, vectorizer, and label mappings.
    Called once at app startup and cached.
    """
    svm_clf   = joblib.load(f'{artifacts_dir}/svm_classifier.joblib')
    tfidf_vec = joblib.load(f'{artifacts_dir}/tfidf_vectorizer.joblib')

    with open(f'{artifacts_dir}/id2label.pkl', 'rb') as f:
        id2label = pickle.load(f)
    with open(f'{artifacts_dir}/label2id.pkl', 'rb') as f:
        label2id = pickle.load(f)

    return svm_clf, tfidf_vec, id2label, label2id


# ── Text segmentation ──────────────────────────────────────────────

def segment_commentary(raw_text: str) -> list:
    """
    Splits raw commentary block into individual event segments.
    Extracts minute timestamps where present. Handles multi-line breaks.
    """
    segments = []
    lines    = raw_text.strip().split('\n')
    current_minute = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for isolated minute timestamps (e.g., 90'+6' or 82')
        minute_match = re.match(r'^(\d+)(?:\+\d*)?\'?\s*$', line)
        if minute_match:
            current_minute = int(minute_match.group(1))
            continue

        if len(line.split()) < 4:
            continue
        # In segment_commentary, add this filter before the word count check
        # Skip common noise lines that carry no event information
        noise_patterns = [
            r'second half ends', r'first half ends', r'full time',
            r'half time', r'second half begins', r'first half begins',
            r'lineups are announced', r'fourth official',
            r'delay in match', r'delay over', r'they are ready'
        ]
        if any(re.search(p, line.lower()) for p in noise_patterns):
            continue
        if re.match(r'^\d+[\'\:]?\s*$', line):
            continue

        minute = current_minute if current_minute is not None else None
        text   = line

        for pattern in [r'^\[?(\d+)\+?\d*\]?[\'\:\-\s]+',
                        r'^\((\d+)\)[\s]+']:
            match = re.match(pattern, line)
            if match:
                minute = int(match.group(1))
                text   = line[match.end():].strip()
                break

        if len(text.split()) >= 4:
            segments.append({'minute': minute, 'text': text})
            current_minute = None # Reset after application

    return segments


# ── Classification ─────────────────────────────────────────────────

def classify_segments(segments: list,
                       svm_clf,
                       tfidf_vec,
                       id2label: dict) -> list:
    """
    Classifies each segment using TF-IDF + SVM.
    Returns segments with label, confidence, and impact score.
    """
    if not segments:
        return []

    texts     = [seg['text'] for seg in segments]
    X         = tfidf_vec.transform(texts)
    preds     = svm_clf.predict(X)
    probas    = svm_clf.predict_proba(X)
    max_proba = probas.max(axis=1)

    return [
        {
            'minute'    : seg['minute'],
            'text'      : seg['text'],
            'label'     : id2label[preds[i]],
            'confidence': round(float(max_proba[i]), 3),
            'impact'    : IMPACT_SCORES.get(id2label[preds[i]], 0)
        }
        for i, seg in enumerate(segments)
    ]


# ── Timeline builder ───────────────────────────────────────────────

def build_timeline(classified_segments: list,
                   home_team: str = 'Home Team',
                   away_team: str = 'Away Team',
                   score: str = 'Unknown',
                   competition: str = '',
                   min_impact: int = 2,
                   min_confidence: float = 0.6) -> tuple:
    """
    Filters classified segments and builds structured timeline
    string for the LLM prompt.
    """
    filtered = [
        s for s in classified_segments
        if s['impact'] >= min_impact and s['confidence'] >= min_confidence
    ]
    if not filtered:
        filtered = [
            s for s in classified_segments
            if s['impact'] >= 1 and s['confidence'] >= 0.5
        ]

    filtered.sort(key=lambda x: x['minute'] if x['minute'] else 999)

    header = (
        f"Match: {home_team} vs {away_team}\n"
        f"Score: {score}\n"
        f"Competition: {competition}\n\n"
        f"Key Events:\n"
    )
    lines = [
        f"  {'Minute ' + str(s['minute']) if s['minute'] else 'Unknown'}: "
        f"{s['label'].upper()} — {s['text']}"
        for s in filtered
    ]

    return header + '\n'.join(lines), filtered


# ── Match Metadata Extraction From Commentary ──────────────────────

def extract_match_details_from_commentary(raw_text: str, groq_api_key: str) -> dict:
    """
    Analyzes the raw commentary feed text using Groq to extract the match participants 
    and final score if they aren't explicitly provided by user.
    """
    import httpx
    http_client = httpx.Client()
    client = Groq(api_key=groq_api_key, http_client=http_client)
    
    # Grab a snippet of the beginning and end where metadata is usually concentrated
    text_lines = raw_text.strip().split('\n')
    snippet = "\n".join(text_lines[:25] + text_lines[-25:])

    extract_prompt = (
        "Analyze this snippet of football match live commentary text and accurately extract the following details:\n"
        "1. Home Team name\n"
        "2. Away Team name\n"
        "3. Final Score or latest score found (format like '2-1' or '0-3')\n"
        "4. Competition name if mentioned (e.g. Premier League, Champions League, Friendly). Default to 'Football Match' if unsure.\n\n"
        "Return ONLY a clean valid JSON block with these exact keys:\n"
        '{\n'
        '  "home_team": "Team Name",\n'
        '  "away_team": "Team Name",\n'
        '  "score": "X-Y",\n'
        '  "competition": "League Name"\n'
        '}'
    )

    try:
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            temperature=0.1,
            max_tokens=200,
            messages=[
                {'role': 'system', 'content': 'You are a precise data extraction assistant.'},
                {'role': 'user', 'content': f'{extract_prompt}\n\nCommentary Snippet:\n{snippet}'}
            ]
        )
        res_text = response.choices[0].message.content.strip()
        start = res_text.find('{')
        end = res_text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(res_text[start:end])
    except Exception:
        pass
        
    return {
        "home_team": "Home Team",
        "away_team": "Away Team",
        "score": "Unknown",
        "competition": "Football Match"
    }


# ── LLM generation ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior sports journalist writing for DAZN, "
    "a global sports streaming platform. Convert structured match event data "
    "into compelling, accurate, publish-ready content.\n\n"
    "STRICT RULES:\n"
    "- Report ONLY events present in the data. Never invent details.\n"
    "- Include ALL goals in key_moments, including late goals.\n"
    "- Do not claim a player scored more goals than shown in the data.\n"
    "- Red cards must be mentioned in the full summary if present.\n"
    "- Push notification must be under 120 characters.\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    '{\n'
    '  "full_summary": "150-200 word match narrative",\n'
    '  "executive_summary": "2-3 sentence version",\n'
    '  "push_notification": "under 120 characters",\n'
    '  "player_of_match": "Name and one sentence reason",\n'
    '  "key_moments": [\n'
    '    {"minute": "45", "event": "Goal", "description": "one line"}\n'
    '  ]\n'
    '}'
)


def generate_highlights(timeline_text: str, groq_api_key: str) -> dict:
    """
    Sends timeline to LLaMA-3 via Groq and returns parsed highlights.
    """
    import httpx
    http_client = httpx.Client()
    
    client = Groq(
        api_key=groq_api_key,
        http_client=http_client
    )

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        temperature=0.3,
        max_tokens=1500,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',
             'content': f'Generate match content:\n\n{timeline_text}'}
        ]
    )
    raw = response.choices[0].message.content
    return parse_llm_output(raw)


def parse_llm_output(raw: str) -> dict:
    """Parses JSON from LLM response handling markdown fences."""
    text = raw.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text  = '\n'.join(lines[1:-1]) if lines[-1] == '```' \
                else '\n'.join(lines[1:])
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            'full_summary'     : raw,
            'executive_summary': raw[:300],
            'push_notification': raw[:120],
            'player_of_match'  : 'See full summary',
            'key_moments'      : []
        }


# ── Guardrail ──────────────────────────────────────────────────────

def run_guardrail(highlights: dict,
                  classified_segments: list) -> dict:
    """
    Post-generation fact verification guardrail.
    """
    full_text  = highlights.get('full_summary', '').lower()
    key_moments = highlights.get('key_moments', [])
    hard_issues = []
    warnings    = []

    goal_events = [s for s in classified_segments
                   if s['label'] == 'Goal' and s['impact'] >= 4]
    red_events  = [s for s in classified_segments
                   if s['label'] == 'Red Card']

    goals_in_moments = sum(
        1 for km in key_moments
        if 'goal' in km.get('event', '').lower()
    )
    if abs(goals_in_moments - len(goal_events)) > 1:
        warnings.append(
            f'Goal count mismatch: LLM={goals_in_moments}, '
            f'classified={len(goal_events)}'
        )

    for rc in red_events:
        rc_words = [w for w in rc['text'].lower().split()
                    if len(w) > 4 and w not in ['footed', 'shot', 'from']]
        found = any(word in full_text for word in rc_words[:3])
        if not found:
            warnings.append('Red card event may not be mentioned in summary')

    notif = highlights.get('push_notification', '')
    if len(notif) > 120:
        hard_issues.append(
            f'Push notification too long: {len(notif)} chars (limit 120)'
        )

    verdict = 'FAIL' if hard_issues else 'WARN' if warnings else 'PASS'

    return {
        'verdict'         : verdict,
        'hard_issues'     : hard_issues,
        'warnings'        : warnings,
        'guardrail_passed': verdict != 'FAIL'
    }


# ── Main pipeline entry point ──────────────────────────────────────

def run_pipeline(raw_text: str,
                 home_team: str,
                 away_team: str,
                 score: str,
                 competition: str,
                 groq_api_key: str,
                 artifacts_dir: str = 'artifacts') -> dict:
    """
    Full end-to-end pipeline for unstructured commentary input.
    Auto-fills missing metadata fields from text when left blank.
    """
    # If parameters are empty strings, perform dynamic LLM discovery
    if not home_team.strip() or not away_team.strip() or not score.strip() or score == "Unknown":
        extracted = extract_match_details_from_commentary(raw_text, groq_api_key)
        if not home_team.strip(): home_team = extracted.get('home_team', 'Home Team')
        if not away_team.strip(): away_team = extracted.get('away_team', 'Away Team')
        if not score.strip() or score == "Unknown": score = extracted.get('score', 'Unknown')
        if not competition.strip() or competition == "Football Match": competition = extracted.get('competition', 'Football Match')

    svm_clf, tfidf_vec, id2label, label2id = load_artifacts(artifacts_dir)

    segments   = segment_commentary(raw_text)
    classified = classify_segments(segments, svm_clf, tfidf_vec, id2label)
    timeline, key_events = build_timeline(
        classified, home_team, away_team, score, competition
    )
    highlights = generate_highlights(timeline, groq_api_key)
    guardrail  = run_guardrail(highlights, classified)

    return {
        'match_info'      : {
            'home_team'  : home_team,
            'away_team'  : away_team,
            'score'      : score,
            'competition': competition
        },
        'segments_found'  : len(segments),
        'key_events_used' : len(key_events),
        'classified'      : classified,
        'timeline'        : timeline,
        'highlights'      : highlights,
        'guardrail'       : guardrail
    }
