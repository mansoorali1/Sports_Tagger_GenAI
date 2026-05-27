"""
api.py — FastAPI Backend

Exposes the sports commentary pipeline as a REST API.
In production at DAZN this endpoint would receive live
JSON event feeds. In the demo it receives pasted commentary text.

Endpoints:
  POST /generate-highlights  — main pipeline endpoint
  GET  /health               — health check
  GET  /                     — API info
"""

# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional

# from app.pipeline import run_pipeline



# app = FastAPI(
#     title='Sports Auto-Tagger API',
#     description=(
#         'Converts raw football commentary into multi-format '
#         'match highlights using NLP classification and LLM summarization.'
#     ),
#     version='1.0.0'
# )

# # Allow Streamlit frontend to call this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_methods=['*'],
#     allow_headers=['*']
# )

# GROQ_API_KEY   = os.environ.get('GROQ_API_KEY', '')
# ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', '/app/artifacts')




# class CommentaryRequest(BaseModel):
#     """Request body for the highlights generation endpoint."""
#     commentary  : str
#     home_team   : str                = 'Home Team'
#     away_team   : str                = 'Away Team'
#     score       : str                = 'Unknown'
#     competition : Optional[str]      = 'Football Match'
#     match_date  : Optional[str]      = ''


# class HighlightsResponse(BaseModel):
#     """Response from the highlights generation endpoint."""
#     match_info      : dict
#     segments_found  : int
#     key_events_used : int
#     highlights      : dict
#     guardrail       : dict
#     timeline        : str


# @app.get('/')
# def root():
#     return {
#         'service'    : 'Sports Auto-Tagger API',
#         'version'    : '1.0.0',
#         'status'     : 'running',
#         'endpoints'  : {
#             'POST /generate-highlights': 'Main pipeline endpoint',
#             'GET  /health'             : 'Health check'
#         }
#     }


# @app.get('/health')
# def health():
#     return {
#         'status'         : 'healthy',
#         'groq_configured': bool(GROQ_API_KEY),
#         'artifacts_dir'  : ARTIFACTS_DIR
#     }


# @app.post('/generate-highlights', response_model=HighlightsResponse)
# def generate_highlights(request: CommentaryRequest):
#     """
#     Main pipeline endpoint.

#     Accepts raw football commentary text and match metadata.
#     Returns multi-format highlights and guardrail verdict.

#     In production this would be called by:
#     - DAZN editorial team dashboard
#     - Automated post-match content pipeline
#     - Social media scheduling tools
#     """
#     if not GROQ_API_KEY:
#         raise HTTPException(
#             status_code=500,
#             detail='GROQ_API_KEY not configured'
#         )

#     if len(request.commentary.strip()) < 50:
#         raise HTTPException(
#             status_code=400,
#             detail='Commentary text too short. Provide at least one match event.'
#         )

#     try:
#         result = run_pipeline(
#             raw_text    = request.commentary,
#             home_team   = request.home_team,
#             away_team   = request.away_team,
#             score       = request.score,
#             competition = request.competition,
#             match_date  = request.match_date,
#             groq_api_key= GROQ_API_KEY,
#             artifacts_dir= ARTIFACTS_DIR
#         )
#         return result

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

"""
api.py — FastAPI Backend

Exposes the sports commentary pipeline as a REST API.
"""

import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.pipeline import run_pipeline

app = FastAPI(
    title='Sports Auto-Tagger API',
    description=(
        'Converts raw football commentary into multi-format '
        'match highlights using NLP classification and LLM summarization.'
    ),
    version='1.0.0'
)

# Allow Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', '/app/artifacts')


class CommentaryRequest(BaseModel):
    commentary: str
    home_team: str = 'Home Team'
    away_team: str = 'Away Team'
    score: str = 'Unknown'
    competition: Optional[str] = 'Football Match'
    match_date: Optional[str] = ''


class HighlightsResponse(BaseModel):
    match_info: dict
    segments_found: int
    key_events_used: int
    highlights: dict
    guardrail: dict
    timeline: str


@app.get('/')
def root():
    return {
        'service': 'Sports Auto-Tagger API',
        'version': '1.0.0',
        'status': 'running'
    }


@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'groq_configured': bool(GROQ_API_KEY),
        'artifacts_dir': ARTIFACTS_DIR
    }


@app.post('/generate-highlights')
def generate_highlights(request: CommentaryRequest):

    print("\n========== NEW REQUEST ==========")
    print("Commentary length:", len(request.commentary))
    print("Home team:", request.home_team)
    print("Away team:", request.away_team)
    print("Artifacts dir:", ARTIFACTS_DIR)
    print("Groq configured:", bool(GROQ_API_KEY))

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail='GROQ_API_KEY not configured'
        )

    if len(request.commentary.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail='Commentary text too short.'
        )

    try:
        print("Running pipeline...")

        result = run_pipeline(
            raw_text=request.commentary,
            home_team=request.home_team,
            away_team=request.away_team,
            score=request.score,
            competition=request.competition,
            match_date=request.match_date,
            groq_api_key=GROQ_API_KEY,
            artifacts_dir=ARTIFACTS_DIR
        )

        print("Pipeline completed successfully.")

        return result

    except Exception as e:

        print("\n========== PIPELINE ERROR ==========")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
