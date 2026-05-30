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

# def segment_commentary(raw_text: str) -> list:
#     """
#     Splits raw commentary block into individual event segments.
#     Extracts minute timestamps where present. Handles multi-line breaks.
#     """
#     segments = []
#     lines    = raw_text.strip().split('\n')
#     current_minute = None

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check for isolated minute timestamps (e.g., 90'+6' or 82')
#         minute_match = re.match(r'^(\d+)(?:\+\d*)?\'?\s*$', line)
#         if minute_match:
#             current_minute = int(minute_match.group(1))
#             continue

#         if len(line.split()) < 4:
#             continue
#         # In segment_commentary, add this filter before the word count check
#         # Skip common noise lines that carry no event information
#         noise_patterns = [
#             r'second half ends', r'first half ends', r'full time',
#             r'half time', r'second half begins', r'first half begins',
#             r'lineups are announced', r'fourth official',
#             r'delay in match', r'delay over', r'they are ready'
#         ]
#         if any(re.search(p, line.lower()) for p in noise_patterns):
#             continue
#         if re.match(r'^\d+[\'\:]?\s*$', line):
#             continue

#         minute = current_minute if current_minute is not None else None
#         text   = line

#         for pattern in [r'^\[?(\d+)\+?\d*\]?[\'\:\-\s]+',
#                         r'^\((\d+)\)[\s]+']:
#             match = re.match(pattern, line)
#             if match:
#                 minute = int(match.group(1))
#                 text   = line[match.end():].strip()
#                 break

#         if len(text.split()) >= 4:
#             segments.append({'minute': minute, 'text': text})
#             current_minute = None # Reset after application

#     return segments
# def segment_commentary(raw_text: str) -> list:
#     """
#     Parses raw football commentary copied from websites into
#     clean individual event segments.

#     Handles two real-world copy-paste formats:

#     Format A — inline minute and text on same line:
#         33' Goal! Brighton 0-1 Manchester United...
#         45+3' Yellow Card. Kobbie Mainoo shown yellow...

#     Format B — multi-line blocks from modern sites like BBC Sport:
#         82'
#         Substitution
#         Luke Shaw
#         Luke Shaw
#         Tyrell Malacia
#         Tyrell Malacia
#         Substitution, Manchester United. Tyrell Malacia replaces Luke Shaw.

#     For Format B the function:
#         - Detects the minute on its own line
#         - Identifies the event label line immediately after
#         - Skips duplicate player name lines
#         - Keeps only the full descriptive sentence at the end of the block
#     """

#     # Known event label words that appear as standalone lines
#     # after the minute in multi-line format commentary
#     EVENT_LABELS = {
#         'goal', 'goal!', 'yellow card', 'red card', 'substitution',
#         'sub', 'penalty', 'offside', 'corner', 'foul', 'var',
#         'attempt', 'free kick', 'second yellow', 'sending off',
#         'assist', 'own goal', 'var review', 'var decision',
#         'manchester united club badge', 'club badge',
#         'second half begins', 'first half begins',
#         'second half ends', 'first half ends',
#         'half time', 'full time', 'kick off',
#         'fourth official', 'delay in match', 'delay over',
#         'they are ready to continue', 'lineups are announced',
#         'players are warming up'
#     }

#     # Lines to skip entirely — they carry no event information
#     NOISE_PATTERNS = [
#         r'^second half ends',
#         r'^first half ends',
#         r'^full time',
#         r'^half time',
#         r'^second half begins',
#         r'^first half begins',
#         r'^lineups are announced',
#         r'^players are warming',
#         r'^fourth official',
#         r'^delay in match',
#         r'^delay over',
#         r'^they are ready',
#         r'^kick off',
#         r'^var review$',
#         r'^var decision$',
#         r'^goal awarded',
#         r'^club badge',
#         r'club badge$',
#     ]

#     lines = raw_text.strip().split('\n')
#     segments = []

#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()

#         if not line:
#             i += 1
#             continue

#         # Check if this line is a standalone minute timestamp
#         # Handles: 82'   90'+2'   90+5'   45'+3'   45'
#         minute_only = re.match(
#             r'^(\d+)(?:\+\d+)?[\'′\u2019]?\s*$',
#             line
#         )

#         if minute_only:
#             minute = int(minute_only.group(1))

#             # Look ahead to collect the block belonging to this minute
#             i += 1
#             block_lines = []

#             while i < len(lines):
#                 next_line = lines[i].strip()

#                 # Stop when we hit the next minute marker
#                 next_minute = re.match(
#                     r'^(\d+)(?:\+\d+)?[\'′\u2019]',
#                     next_line
#                 )
#                 if next_minute:
#                     break

#                 if next_line:
#                     block_lines.append(next_line)
#                 i += 1

#             if not block_lines:
#                 continue

#             # From the block find the best descriptive line.
#             # Priority: longest line that is not a noise/label line
#             # and contains at least 6 words.
#             best_line = None
#             for bl in reversed(block_lines):
#                 bl_lower = bl.lower().strip()

#                 # Skip noise
#                 is_noise = any(
#                     re.search(p, bl_lower) for p in NOISE_PATTERNS
#                 )
#                 if is_noise:
#                     continue

#                 # Skip pure event label lines
#                 if bl_lower in EVENT_LABELS:
#                     continue

#                 # Skip short lines (likely player name duplicates)
#                 if len(bl.split()) < 5:
#                     continue

#                 # Skip lines that are just a score like "0 - 3"
#                 if re.match(r'^\d+\s*[-–]\s*\d+$', bl.strip()):
#                     continue

#                 # Skip lines that are team + score like
#                 # "Brighton and Hove Albion 0 - 3 Manchester United"
#                 if re.match(
#                     r'^[A-Za-z\s]+ \d+ [-–] \d+ [A-Za-z\s]+$',
#                     bl.strip()
#                 ):
#                     continue

#                 best_line = bl
#                 break

#             if best_line and len(best_line.split()) >= 5:
#                 segments.append({
#                     'minute': minute,
#                     'text'  : best_line
#                 })

#             continue

#         # Format A — minute and text on the same line
#         # Handles: 33' Goal! Brighton...   or   45'+3' Kobbie Mainoo...
#         inline_match = re.match(
#             r'^(\d+)(?:\+\d+)?[\'′\u2019]\s+(.+)$',
#             line
#         )

#         if inline_match:
#             minute    = int(inline_match.group(1))
#             text_part = inline_match.group(2).strip()

#             # Skip noise lines
#             is_noise = any(
#                 re.search(p, text_part.lower()) for p in NOISE_PATTERNS
#             )
#             if not is_noise and len(text_part.split()) >= 5:
#                 segments.append({
#                     'minute': minute,
#                     'text'  : text_part
#                 })

#             i += 1
#             continue

#         # Line has no minute — skip it
#         # (orphaned player names, badges, etc.)
#         i += 1

#     return segments
import re


# def segment_commentary(raw_text: str) -> list:
#     """
#     Parses raw football commentary copied from two website formats into
#     clean individual event segments, returning only meaningful events
#     (goals, yellow cards, red cards, substitutions, attempts, fouls, etc.)

#     Handles:
#       Format A  — minute and text on same line:
#           33' Goal! Brighton 0-1 Manchester United...

#       Format B  — BBC Sport multi-line blocks:
#           82'
#           Substitution
#           Luke Shaw
#           Luke Shaw
#           Tyrell Malacia
#           Tyrell Malacia
#           Substitution, Manchester United. Tyrell Malacia replaces Luke Shaw.

#       Format C  — sports journalism site (e.g. Real Madrid commentary):
#           88‎'‎
#           Goal!
#           21
#           Brahim Díaz
#           Attacker
#           THERE'S THE FOURTH FOR MADRID!! Brahim Diaz rounds off...
#           Shot type
#           Left foot
#           xG
#           0.74
#           xGOT
#           0.91
#     """

#     # ------------------------------------------------------------------ #
#     #  Regex: matches any minute stamp on its own line or leading a line   #
#     #  Covers: 3'  45'  45'+2'  90+3'  90+3‎'‎  (plain ' or unicode variants)
#     # ------------------------------------------------------------------ #
#     # MINUTE_PATTERN = re.compile(
#     #     r'^(\d{1,3})(?:\+\d{1,3})?[\'\u2019\u2018\u201b\u02bc\xb4]'
#     #     r'(?:\u200e|\u200f|\u202a|\u202c)*'   # swallow any bidi marks
#     #     r'\s*$'
#     # )
#     MINUTE_PATTERN = re.compile(
#     r'^(\d{1,3})(?:[\'\u2019\u2018\u201b\u02bc\xb4]\+\d{1,3})?'
#     r'[\'\u2019\u2018\u201b\u02bc\xb4]'
#     r'(?:\u200e|\u200f|\u202a|\u202c)*'
#     r'\s*$'
#     )

#     MINUTE_INLINE = re.compile(
#         r'^(\d{1,3})(?:\+\d{1,3})?[\'\u2019\u2018\u201b\u02bc\xb4]'
#         r'(?:\u200e|\u200f|\u202a|\u202c)*'
#         r'\s+(.+)$'
#     )

#     # ------------------------------------------------------------------ #
#     #  Event labels that appear as standalone lines right after a minute.  #
#     #  When found alone they are stripped; the real sentence follows.      #
#     # ------------------------------------------------------------------ #
#     STANDALONE_EVENT_LABELS = {
#         'goal', 'goal!', 'yellow card', 'red card', 'substitution', 'sub',
#         'penalty', 'penalty missed', 'penalty saved', 'offside', 'corner',
#         'foul', 'var', 'attempt', 'free kick', 'second yellow', 'sending off',
#         'own goal', 'var review', 'var decision', 'assist', 'highlight',
#         'second half begins', 'first half begins', 'second half ends',
#         'first half ends', 'half time', 'full time', 'kick off',
#         'fourth official', 'delay in match', 'delay over',
#         'they are ready to continue', 'lineups are announced',
#         'players are warming up',
#     }

#     # ------------------------------------------------------------------ #
#     #  Patterns for lines that carry zero useful event information.        #
#     # ------------------------------------------------------------------ #
#     NOISE_LINE_PATTERNS = [
#         # half/full time banners
#         r'^second half ends',
#         r'^first half ends',
#         r'^full time',
#         r'^half[-\s]?time',
#         r'^second half begins',
#         r'^first half begins',
#         r'^lineups are announced',
#         r'^players are warming',
#         r'^fourth official',
#         r'^delay in match',
#         r'^delay over',
#         r'^they are ready',
#         r'^kick off',
#         r'^var review$',
#         r'^var decision$',
#         r'^goal awarded',
#         # club badge lines
#         r'club badge',
#         # score-only lines:  "0 - 3"  or  "West Ham United 3 - 0 Leeds United"
#         r'^\d+\s*[-–]\s*\d+$',
#         r'^[a-z\s]+ \d+ [-–] \d+ [a-z\s]+$',
#         # xG / shot stat lines (from journalism sites)
#         r'^shot type$',
#         r'^xg$',
#         r'^xgot$',
#         r'^right foot$',
#         r'^left foot$',
#         r'^header$',
#         r'^\d+\.\d+$',            # bare decimal numbers like 0.74
#         r'^\d+$',                 # bare integers like squad numbers
#         # position labels
#         r'^striker$', r'^attacker$', r'^midfielder$', r'^defender$',
#         r'^goalkeeper$', r'^right winger$', r'^left winger$',
#         r'^right[-\s]back$', r'^left[-\s]back$',
#         r'^central midfielder$', r'^left midfielder$', r'^right midfielder$',
#         # half-time summary block (long analyst paragraph without a minute)
#         r'^referee \w+ brings an end',
#         r'^the players are back out',
#     ]

#     # ------------------------------------------------------------------ #
#     #  Helper: does a (lowercased) line match any noise pattern?           #
#     # ------------------------------------------------------------------ #
#     def is_noise(line_lower: str) -> bool:
#         return any(re.search(p, line_lower) for p in NOISE_LINE_PATTERNS)

#     # ------------------------------------------------------------------ #
#     #  Helper: is this line a standalone event label?                      #
#     # ------------------------------------------------------------------ #
#     def is_event_label(line_lower: str) -> bool:
#         return line_lower.strip() in STANDALONE_EVENT_LABELS

#     # ------------------------------------------------------------------ #
#     #  Helper: is this line a duplicate player-name line?                  #
#     #  Heuristic: ≤4 words, Title Case or ALL CAPS, no punctuation.       #
#     # ------------------------------------------------------------------ #
#     def is_player_name_dupe(line: str) -> bool:
#         stripped = line.strip()
#         if len(stripped.split()) > 5:
#             return False
#         # No sentence-ending punctuation and no parentheses
#         if re.search(r'[().,:;!?]', stripped):
#             return False
#         # Looks like a name (each word capitalised or all-caps)
#         words = stripped.split()
#         if all(w[0].isupper() for w in words if w):
#             return True
#         return False

#     # ------------------------------------------------------------------ #
#     #  Helper: pick the best descriptive sentence from a block of lines.  #
#     #                                                                      #
#     #  Priority rules (in order):                                          #
#     #   1. A line that starts with a known event keyword followed by a     #
#     #      capital letter and team/player name  →  the "clean" sentence.  #
#     #      e.g. "Goal! West Ham United 3, Leeds United 0. Callum Wilson…" #
#     #      e.g. "Substitution, Leeds United. Joël Piroe replaces Ao…"     #
#     #      e.g. "Kobbie Mainoo (Manchester United) is shown the yellow…"  #
#     #   2. Longest line with ≥ 7 words that is not noise / label / name.  #
#     # ------------------------------------------------------------------ #
#     CLEAN_SENTENCE_PREFIXES = re.compile(
#         r'^(goal!|substitution,|substitution\b|yellow card|red card|'
#         r'attempt\s+(saved|missed|blocked)|foul\s+by|'
#         r'corner,|offside,|penalty\b|free\s+kick\b|'
#         r'var\s*[-–]?\s*(review|decision|goal awarded)|'
#         r'[a-záéíóúñüàèìòùäöüç]+\s+\w+\s+\()',  # "Player Name (Club)"
#         re.IGNORECASE
#     )

#     def best_line_from_block(block_lines: list) -> str | None:
#         candidates = []
#         for bl in block_lines:
#             bl_lower = bl.lower().strip()
#             if is_noise(bl_lower):
#                 continue
#             if is_event_label(bl_lower):
#                 continue
#             if is_player_name_dupe(bl):
#                 continue
#             if len(bl.split()) < 5:
#                 continue
#             candidates.append(bl)

#         if not candidates:
#             return None

#         # Priority 1: find a line that starts with a clean-sentence prefix
#         for c in candidates:
#             if CLEAN_SENTENCE_PREFIXES.match(c.strip()):
#                 return c

#         # Priority 2: longest candidate
#         return max(candidates, key=lambda x: len(x.split()))

#     # ------------------------------------------------------------------ #
#     #  Main parsing loop                                                   #
#     # ------------------------------------------------------------------ #
#     lines = raw_text.strip().split('\n')
#     segments = []
#     seen_minutes: dict[int, list] = {}   # minute → list of texts (dedupe)

#     i = 0
#     while i < len(lines):
#         raw_line = lines[i]
#         line = raw_line.strip()

#         if not line:
#             i += 1
#             continue

#         # ---- Check for standalone minute line (Format B / C) ----
#         m_only = MINUTE_PATTERN.match(line)
#         if m_only:
#             minute = int(m_only.group(1))
#             i += 1
#             block_lines = []

#             while i < len(lines):
#                 next_raw = lines[i]
#                 next_line = next_raw.strip()

#                 # Stop at next minute marker (standalone or inline)
#                 if MINUTE_PATTERN.match(next_line) or MINUTE_INLINE.match(next_line):
#                     break

#                 if next_line:
#                     block_lines.append(next_line)
#                 i += 1

#             best = best_line_from_block(block_lines)
#             if best:
#                 _add_segment(segments, seen_minutes, minute, best)
#             continue

#         # ---- Check for inline minute + text (Format A) ----
#         m_inline = MINUTE_INLINE.match(line)
#         if m_inline:
#             minute = int(m_inline.group(1))
#             text_part = m_inline.group(2).strip()
#             text_lower = text_part.lower()

#             if not is_noise(text_lower) and len(text_part.split()) >= 5:
#                 _add_segment(segments, seen_minutes, minute, text_part)

#             i += 1
#             continue

#         # ---- No minute on this line — skip ----
#         i += 1

#     # Return in chronological order
#     return sorted(segments, key=lambda s: s['minute'])


# # ------------------------------------------------------------------ #
# #  Helper outside the main function to avoid closure issues            #
# # ------------------------------------------------------------------ #
# def _add_segment(
#     segments: list,
#     seen: dict,
#     minute: int,
#     text: str,
# ) -> None:
#     """
#     Add a segment only if we haven't already added the same text
#     for the same minute (prevents duplicates from Assist+Goal blocks).
#     """
#     existing = seen.get(minute, [])
#     # Normalise for comparison
#     norm = re.sub(r'\s+', ' ', text.strip().lower())
#     for e in existing:
#         if re.sub(r'\s+', ' ', e.strip().lower()) == norm:
#             return
#         # Also skip if one is a substring of the other (near-dupe)
#         if norm in re.sub(r'\s+', ' ', e.strip().lower()):
#             return
#     existing.append(text)
#     seen[minute] = existing
#     segments.append({'minute': minute, 'text': text})


def segment_commentary(raw_text: str) -> list:
    """
    Parses raw football commentary into clean individual event segments.

    Each returned segment dict contains:
        minute_label : str   e.g. "1'", "45'+2'", "90'+4'"
        sort_key     : float e.g. 1.0, 45.2, 90.4  (for chronological sort)
        minute       : int   base minute (kept for backward compatibility)
        added        : int   added time (0 if none)
        text         : str   the descriptive event sentence
    """

    # ------------------------------------------------------------------ #
    #  Regex: standalone minute line                                       #
    #  Captures group(1)=base, group(2)=added (or None)                   #
    #  Handles both apostrophe styles:  45'   45'+2'   90+3'   90'+4'    #
    # ------------------------------------------------------------------ #
    # MINUTE_PATTERN = re.compile(
    #     r'^(\d{1,3})'                                    # base minute
    #     r'[\'\u2019\u2018\u201b\u02bc\xb4]?'            # optional apostrophe after base
    #     r'(?:\+(\d{1,3}))?'                              # optional +added group
    #     r'[\'\u2019\u2018\u201b\u02bc\xb4]'             # closing apostrophe (required)
    #     r'(?:\u200e|\u200f|\u202a|\u202c)*'              # swallow bidi marks
    #     r'\s*$'
    # )
    # FIXED — handles bidi marks (‎) on either side of the apostrophe
    # Website 1:  45'   90+3'   45'+2'
    # Website 2:  88‎'‎   90+3‎'‎   45+1‎'‎  (Left-to-Right Marks wrap the apostrophe)
    _BIDI = r'(?:\u200e|\u200f|\u202a|\u202c)*'
    _APO  = r'[\'\u2019\u2018\u201b\u02bc\xb4]'
    
    MINUTE_PATTERN = re.compile(
        r'^(\d{1,3})' + _BIDI + _APO + r'?' + _BIDI +
        r'(?:\+(\d{1,3}))?' + _BIDI + _APO + _BIDI + r'\s*$'
    )

    # MINUTE_INLINE = re.compile(
    #     r'^(\d{1,3})'                                    # base minute
    #     r'[\'\u2019\u2018\u201b\u02bc\xb4]?'            # optional apostrophe after base
    #     r'(?:\+(\d{1,3}))?'                              # optional +added group
    #     r'[\'\u2019\u2018\u201b\u02bc\xb4]'             # closing apostrophe (required)
    #     r'(?:\u200e|\u200f|\u202a|\u202c)*'              # swallow bidi marks
    #     r'\s+(.+)$'                                      # space then event text
    # )
    MINUTE_INLINE = re.compile(
    r'^(\d{1,3})' + _BIDI + _APO + r'?' + _BIDI +
    r'(?:\+(\d{1,3}))?' + _BIDI + _APO + _BIDI + r'\s+(.+)$'
    )
    

    # ------------------------------------------------------------------ #
    #  Helper: build a human-readable label and a float sort key          #
    #  from the two regex groups.                                          #
    #                                                                      #
    #  sort_key formula:  base + added/100                                 #
    #    →  90'+4'  becomes 90.04  (sorts correctly after 90.03 etc.)     #
    #    →  45'+2'  becomes 45.02                                          #
    # ------------------------------------------------------------------ #
    def parse_minute(base_str: str, added_str: str | None):
        base = int(base_str)
        added = int(added_str) if added_str else 0
        sort_key = base + added / 100
        if added:
            label = f"{base}'+{added}'"
        else:
            label = f"{base}'"
        return base, added, sort_key, label

    # ------------------------------------------------------------------ #
    #  Event labels / noise patterns (unchanged from original)            #
    # ------------------------------------------------------------------ #
    STANDALONE_EVENT_LABELS = {
        'goal', 'goal!', 'yellow card', 'red card', 'substitution', 'sub',
        'penalty', 'penalty missed', 'penalty saved', 'offside', 'corner',
        'foul', 'var', 'attempt', 'free kick', 'second yellow', 'sending off',
        'own goal', 'var review', 'var decision', 'assist', 'highlight',
        'second half begins', 'first half begins', 'second half ends',
        'first half ends', 'half time', 'full time', 'kick off',
        'fourth official', 'delay in match', 'delay over',
        'they are ready to continue', 'lineups are announced',
        'players are warming up',
    }

    NOISE_LINE_PATTERNS = [
        r'^second half ends', r'^first half ends', r'^full time',
        r'^full[-\s]?time',
        r'^half[-\s]?time', r'^second half begins', r'^first half begins',
        r'^lineups are announced', r'^players are warming',
        r'^fourth official', r'^delay in match', r'^delay over',
        r'^they are ready', r'^kick off',
        r'^var review$', r'^var decision$', r'^goal awarded',
        r'club badge',
        r'^\d+\s*[-–]\s*\d+$',
        r'^[a-z\s]+ \d+ [-–] \d+ [a-z\s]+$',
        r'^shot type$', r'^xg$', r'^xgot$',
        r'^right foot$', r'^left foot$', r'^header$',
        r'^\d+\.\d+$', r'^\d+$',
        r'^striker$', r'^attacker$', r'^midfielder$', r'^defender$',
        r'^goalkeeper$', r'^right winger$', r'^left winger$',
        r'^right[-\s]back$', r'^left[-\s]back$',
        r'^central midfielder$', r'^left midfielder$', r'^right midfielder$',
        r'^referee \w+ brings an end', r'^the players are back out',
    ]

    def is_noise(line_lower: str) -> bool:
        return any(re.search(p, line_lower) for p in NOISE_LINE_PATTERNS)

    def is_event_label(line_lower: str) -> bool:
        return line_lower.strip() in STANDALONE_EVENT_LABELS

    def is_player_name_dupe(line: str) -> bool:
        stripped = line.strip()
        if len(stripped.split()) > 5:
            return False
        if re.search(r'[().,:;!?]', stripped):
            return False
        words = stripped.split()
        if all(w[0].isupper() for w in words if w):
            return True
        return False

    CLEAN_SENTENCE_PREFIXES = re.compile(
        r'^(goal!|substitution,|substitution\b|yellow card|red card|'
        r'attempt\s+(saved|missed|blocked)|foul\s+by|'
        r'corner,|offside,|penalty\b|free\s+kick\b|'
        r'var\s*[-–]?\s*(review|decision|goal awarded)|'
        r'[a-záéíóúñüàèìòùäöüç]+\s+\w+\s+\()',
        re.IGNORECASE
    )

    def best_line_from_block(block_lines: list) -> str | None:
        candidates = []
        for bl in block_lines:
            bl_lower = bl.lower().strip()
            if is_noise(bl_lower):
                continue
            if is_event_label(bl_lower):
                continue
            if is_player_name_dupe(bl):
                continue
            if len(bl.split()) < 5:
                continue
            candidates.append(bl)

        if not candidates:
            return None

        for c in candidates:
            if CLEAN_SENTENCE_PREFIXES.match(c.strip()):
                return c

        return max(candidates, key=lambda x: len(x.split()))

    # ------------------------------------------------------------------ #
    #  Main parsing loop                                                   #
    # ------------------------------------------------------------------ #
    lines = raw_text.strip().split('\n')
    segments = []
    # Key is now sort_key (float) to distinguish 90' from 90'+1', 90'+4' etc.
    seen: dict[float, list] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # ---- Standalone minute line (Format B / C) ----
        m_only = MINUTE_PATTERN.match(line)
        if m_only:
            base, added, sort_key, label = parse_minute(
                m_only.group(1), m_only.group(2)
            )
            i += 1
            block_lines = []

            while i < len(lines):
                next_line = lines[i].strip()
                # Stop at next minute marker
                if MINUTE_PATTERN.match(next_line) or MINUTE_INLINE.match(next_line):
                    break
                if next_line:
                    block_lines.append(next_line)
                i += 1

            best = best_line_from_block(block_lines)
            if best:
                _add_segment(segments, seen, base, added, sort_key, label, best)
            continue

        # ---- Inline minute + text (Format A) ----
        m_inline = MINUTE_INLINE.match(line)
        if m_inline:
            base, added, sort_key, label = parse_minute(
                m_inline.group(1), m_inline.group(2)
            )
            text_part = m_inline.group(3).strip()

            if not is_noise(text_part.lower()) and len(text_part.split()) >= 5:
                _add_segment(segments, seen, base, added, sort_key, label, text_part)

            i += 1
            continue

        i += 1

    # Chronological order using sort_key
    return sorted(segments, key=lambda s: s['sort_key'])


# ------------------------------------------------------------------ #
#  Updated _add_segment                                                #
# ------------------------------------------------------------------ #
def _add_segment(
    segments: list,
    seen: dict,
    base: int,
    added: int,
    sort_key: float,
    label: str,
    text: str,
) -> None:
    existing = seen.get(sort_key, [])
    norm = re.sub(r'\s+', ' ', text.strip().lower())
    for e in existing:
        if re.sub(r'\s+', ' ', e.strip().lower()) == norm:
            return
        if norm in re.sub(r'\s+', ' ', e.strip().lower()):
            return
    existing.append(text)
    seen[sort_key] = existing
    segments.append({
        'minute':       base,       # int, backward compatible
        'added':        added,      # int, 0 if none
        'sort_key':     sort_key,   # float, for ordering
        'minute_label': label,      # str, e.g. "90'+4'" — use this in all outputs
        'text':         text,
    })

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

    # return [
    #     {
    #         'minute'    : seg['minute'],
    #         'text'      : seg['text'],
    #         'label'     : id2label[preds[i]],
    #         'confidence': round(float(max_proba[i]), 3),
    #         'impact'    : IMPACT_SCORES.get(id2label[preds[i]], 0)
    #     }
    #     for i, seg in enumerate(segments)
    # ]
    # FIX — pass through all minute fields
    return [
        {
            'minute'      : seg['minute'],
            'added'       : seg['added'],
            'sort_key'    : seg['sort_key'],
            'minute_label': seg['minute_label'],   # ← carry this forward
            'text'        : seg['text'],
            'label'       : id2label[preds[i]],
            'confidence'  : round(float(max_proba[i]), 3),
            'impact'      : IMPACT_SCORES.get(id2label[preds[i]], 0)
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

    # filtered.sort(key=lambda x: x['minute'] if x['minute'] else 999)
    filtered.sort(key=lambda x: x['sort_key'] if x.get('sort_key') is not None else 999)


    header = (
        f"Match: {home_team} vs {away_team}\n"
        f"Score: {score}\n"
        f"Competition: {competition}\n\n"
        f"Key Events:\n"
    )
    # lines = [
    #     f"  {'Minute ' + str(s['minute']) if s['minute'] else 'Unknown'}: "
    #     f"{s['label'].upper()} — {s['text']}"
    #     for s in filtered
    # ]
    lines = [
    f"  {s.get('minute_label', str(s['minute']) + chr(39))}: "  # ← "90'+4'"
    f"{s['label'].upper()} — {s['text']}"
    for s in filtered
    ]

    return header + '\n'.join(lines), filtered
                       
# ── Pre-LLM Fact Computation ───────────────────────────────────────

def compute_match_facts(classified_segments: list,
                        home_team: str,
                        away_team: str) -> dict:
    """
    Deterministically computes ground-truth facts from classified segments
    BEFORE the LLM sees anything. These facts are passed into the prompt
    as constraints so the LLM cannot contradict them.
    """
    goal_events = [
        s for s in classified_segments
        if s['label'] == 'Goal' and s['confidence'] >= 0.5
    ]
    red_events = [
        s for s in classified_segments
        if s['label'] == 'Red Card'
    ]
    yellow_events = [
        s for s in classified_segments
        if s['label'] == 'Yellow Card'
    ]

    # ── Scorer extraction ─────────────────────────────────────────
    # Pattern: "Firstname Lastname (Club)" or "Firstname Lastname right footed..."
    NAME_PATTERN = re.compile(
        r'^([A-Z][a-záéíóúñüàèìòùä]+(?:\s+[A-Z][a-záéíóúñüàèìòùä]+){1,3})'
        r'(?:\s+\(|\s+right|\s+left|\s+header|\s+scores|\s+converts|\s+taps)'
    )

    scorer_counts: dict[str, int] = {}

    for g in goal_events:
        text = g['text'].strip()
        # Try regex first
        match = NAME_PATTERN.match(text)
        if match:
            name = match.group(1).strip()
            scorer_counts[name] = scorer_counts.get(name, 0) + 1
        else:
            # Fallback: first 2-3 Title Case words at start of text
            words = text.split()
            name_words = []
            for w in words[:4]:
                if w[0].isupper() and w.isalpha():
                    name_words.append(w)
                else:
                    break
            if len(name_words) >= 2:
                name = ' '.join(name_words)
                scorer_counts[name] = scorer_counts.get(name, 0) + 1

    # ── Classify scorers ──────────────────────────────────────────
    braces     = [p for p, c in scorer_counts.items() if c == 2]
    hat_tricks = [p for p, c in scorer_counts.items() if c >= 3]

    # ── Team goal split (heuristic from score string) ─────────────
    # We don't have team attribution per goal from raw text reliably,
    # so we just report total goals as ground truth.

    return {
        'total_goals'   : len(goal_events),
        'scorer_counts' : scorer_counts,
        'braces'        : braces,
        'hat_tricks'    : hat_tricks,
        'red_card_count': len(red_events),
        'yellow_card_count': len(yellow_events),
        'goal_scorers_list': list(scorer_counts.keys()),
    }

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


# def generate_highlights(timeline_text: str, groq_api_key: str) -> dict:
#     """
#     Sends timeline to LLaMA-3 via Groq and returns parsed highlights.
#     """
#     import httpx
#     http_client = httpx.Client()
    
#     client = Groq(
#         api_key=groq_api_key,
#         http_client=http_client
#     )

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

def generate_highlights(timeline_text: str,
                        groq_api_key: str,
                        match_facts: dict | None = None) -> dict:
    """
    Sends timeline to LLaMA-3 via Groq and returns parsed highlights.
    match_facts are injected as hard constraints into the prompt.
    """
    import httpx
    http_client = httpx.Client()

    client = Groq(
        api_key=groq_api_key,
        http_client=http_client
    )

    # ── Build grounded facts block ────────────────────────────────
    facts_block = ""
    if match_facts:
        lines = [
            "\n\nGROUND TRUTH FACTS — these are computed from source data.",
            "You MUST NOT contradict any of these in your output:\n",
            f"- Total goals in this match: {match_facts['total_goals']}",
            f"- Goal scorers and counts: {match_facts['scorer_counts']}",
        ]
        if match_facts['braces']:
            lines.append(f"- Players who scored a brace (exactly 2 goals): {match_facts['braces']}")
        if match_facts['hat_tricks']:
            lines.append(f"- Players who scored a hat-trick (3+ goals): {match_facts['hat_tricks']}")
        if not match_facts['braces'] and not match_facts['hat_tricks']:
            lines.append("- No player scored more than once. Do NOT use the words brace or hat-trick.")
        lines.append(f"- Red cards issued: {match_facts['red_card_count']}")
        if match_facts['red_card_count'] == 0:
            lines.append("- No red cards. Do NOT mention any red card in the summary.")
        facts_block = "\n".join(lines)

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        temperature=0.3,
        max_tokens=1500,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',
             'content': f'Generate match content:\n\n{timeline_text}{facts_block}'}
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

# def run_guardrail(highlights: dict,
#                   classified_segments: list) -> dict:
#     """
#     Post-generation fact verification guardrail.
#     """
#     full_text  = highlights.get('full_summary', '').lower()
#     key_moments = highlights.get('key_moments', [])
#     hard_issues = []
#     warnings    = []

#     goal_events = [s for s in classified_segments
#                    if s['label'] == 'Goal' and s['impact'] >= 4]
#     red_events  = [s for s in classified_segments
#                    if s['label'] == 'Red Card']

#     goals_in_moments = sum(
#         1 for km in key_moments
#         if 'goal' in km.get('event', '').lower()
#     )
#     if abs(goals_in_moments - len(goal_events)) > 1:
#         warnings.append(
#             f'Goal count mismatch: LLM={goals_in_moments}, '
#             f'classified={len(goal_events)}'
#         )

#     for rc in red_events:
#         rc_words = [w for w in rc['text'].lower().split()
#                     if len(w) > 4 and w not in ['footed', 'shot', 'from']]
#         found = any(word in full_text for word in rc_words[:3])
#         if not found:
#             warnings.append('Red card event may not be mentioned in summary')

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

def run_guardrail(highlights: dict,
                  classified_segments: list,
                  match_facts: dict | None = None) -> dict:
    """
    Post-generation fact verification guardrail.
    Now cross-checks against pre-computed match_facts for scorer-level accuracy.
    """
    full_text   = highlights.get('full_summary', '').lower()
    key_moments = highlights.get('key_moments', [])
    hard_issues = []
    warnings    = []

    # ── Use pre-computed facts if available, else recompute ───────
    if match_facts is None:
        match_facts = compute_match_facts(classified_segments, '', '')

    total_goals      = match_facts['total_goals']
    scorer_counts    = match_facts['scorer_counts']
    braces           = match_facts['braces']
    hat_tricks       = match_facts['hat_tricks']
    red_card_count   = match_facts['red_card_count']

    # ── 1. Goal count check ───────────────────────────────────────
    goals_in_moments = sum(
        1 for km in key_moments
        if 'goal' in km.get('event', '').lower()
    )
    if abs(goals_in_moments - total_goals) > 1:
        warnings.append(
            f'Goal count mismatch: LLM key_moments has {goals_in_moments} '
            f'goals, classified data has {total_goals}'
        )

    # ── 2. Brace / hat-trick hallucination check ──────────────────
    if 'hat-trick' in full_text or 'hat trick' in full_text:
        if not hat_tricks:
            hard_issues.append(
                'Hallucination: summary mentions hat-trick but no player '
                'scored 3+ goals in the classified events'
            )

    if 'brace' in full_text:
        if not braces:
            hard_issues.append(
                'Hallucination: summary mentions brace but no player '
                'scored exactly 2 goals in the classified events'
            )

    # ── 3. Scorer name check ──────────────────────────────────────
    # If a player name appears in scorer_counts with count=1,
    # make sure the summary doesn't pair them with brace/hat-trick language
    for player, count in scorer_counts.items():
        first_name = player.split()[0].lower()
        if count == 1:
            # Check if the summary incorrectly inflates their tally
            if first_name in full_text:
                surrounding = full_text[max(0, full_text.find(first_name)-30):
                                        full_text.find(first_name)+60]
                if 'brace' in surrounding or 'hat-trick' in surrounding or 'hat trick' in surrounding:
                    warnings.append(
                        f'Possible hallucination: {player} scored 1 goal '
                        f'but summary may attribute multiple goals to them'
                    )

    # ── 4. Red card check ─────────────────────────────────────────
    if red_card_count == 0 and ('red card' in full_text or 'sent off' in full_text or 'sending off' in full_text):
        hard_issues.append(
            'Hallucination: summary mentions red card but none found '
            'in classified events'
        )

    if red_card_count > 0:
        red_events = [s for s in classified_segments if s['label'] == 'Red Card']
        for rc in red_events:
            rc_words = [w for w in rc['text'].lower().split()
                        if len(w) > 4 and w not in ['footed', 'shot', 'from']]
            found = any(word in full_text for word in rc_words[:3])
            if not found:
                warnings.append('Red card event may not be mentioned in summary')

    # ── 5. Push notification length ───────────────────────────────
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
        'guardrail_passed': verdict != 'FAIL',
        'match_facts'     : match_facts,   # expose for debugging in UI
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
    # highlights = generate_highlights(timeline, groq_api_key)
    # guardrail  = run_guardrail(highlights, classified)

    # ── Compute ground-truth facts BEFORE calling LLM ─────────────
    match_facts = compute_match_facts(classified, home_team, away_team)

    # ── Pass facts as constraints into LLM prompt ──────────────────
    highlights  = generate_highlights(timeline, groq_api_key, match_facts)

    # ── Validate LLM output against those same facts ───────────────
    guardrail   = run_guardrail(highlights, classified, match_facts)

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
