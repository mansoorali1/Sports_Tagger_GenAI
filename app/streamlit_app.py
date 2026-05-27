"""
streamlit_app.py — Demo Interface

Interactive UI for the Sports Auto-Tagger pipeline.
Demonstrates the CLEBS (Centralized Live Event Broadcast Service)
architecture where one pipeline serves multiple output formats
to different business units simultaneously.
"""

import os
import requests
import streamlit as st
from app.pipeline import run_pipeline


API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# ── Page config ────────────────────────────────────────────────────

st.set_page_config(
    page_title='Sports Auto-Tagger',
    page_icon='⚽',
    layout='wide'
)

st.title('⚽ Sports Auto-Tagger & Highlight Generator')
st.markdown(
    'Paste raw football commentary → get instant multi-format '
    'match highlights powered by NLP classification and LLM summarization.'
)
st.divider()

# ── Sidebar — match metadata ───────────────────────────────────────

with st.sidebar:
    st.header('Match Details')
    st.caption('Provide match context to improve summary quality')

    home_team   = st.text_input('Home Team',   value='Manchester City')
    away_team   = st.text_input('Away Team',   value='Manchester United')
    score       = st.text_input('Final Score', value='1-0',
                                help='Format: 2-1')
    competition = st.text_input('Competition', value='Premier League')
    match_date  = st.date_input('Match Date')

    st.divider()
    st.caption(
        'This tool demonstrates an automated sports content pipeline. '
        'Generated content is verified by a post-generation guardrail '
        'before display.'
    )

# ── Main area — commentary input ───────────────────────────────────

st.subheader('Paste Match Commentary')

sample = """45' Goal! Erling Haaland (Manchester City) right footed shot from the centre of the box to the bottom right corner. Assisted by Kevin De Bruyne.
67' Yellow card shown to Bruno Fernandes (Manchester United) for a bad foul.
78' Substitution, Manchester City. Phil Foden replaces Bernardo Silva.
85' Red card for Harry Maguire (Manchester United) for violent conduct."""

commentary = st.text_area(
    'Commentary text',
    height=200,
    placeholder=sample,
    help='One event per line. Minutes optional but improve output quality.'
)

generate_btn = st.button('Generate Highlights', type='primary',
                          use_container_width=True)

# ── Generation ─────────────────────────────────────────────────────

if generate_btn:
    if not commentary.strip():
        st.error('Please paste some commentary text first.')
        st.stop()

    with st.spinner('Classifying events and generating highlights...'):
        try:
            response = requests.post(
                f'{API_URL}/generate-highlights',
                json={
                    'commentary' : commentary,
                    'home_team'  : home_team,
                    'away_team'  : away_team,
                    'score'      : score,
                    'competition': competition,
                    'match_date' : str(match_date)
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                'Cannot connect to API. '
                'Make sure the FastAPI server is running.'
            )
            st.stop()
        except Exception as e:
            st.error(f'Error: {e}')
            st.stop()

    # Guardrail status
    guardrail = data.get('guardrail', {})
    verdict   = guardrail.get('verdict', 'UNKNOWN')

    if verdict == 'PASS':
        st.success('Guardrail: PASS — All facts verified against source events')
    elif verdict == 'WARN':
        st.warning(
            'Guardrail: WARN — Content may need review before publishing. '
            + ' | '.join(guardrail.get('warnings', []))
        )
    else:
        st.error(
            'Guardrail: FAIL — Content blocked. '
            + ' | '.join(guardrail.get('hard_issues', []))
        )

    highlights = data.get('highlights', {})

    # Pipeline stats
    col1, col2, col3 = st.columns(3)
    col1.metric('Segments Found',    data.get('segments_found', 0))
    col2.metric('Key Events Used',   data.get('key_events_used', 0))
    col3.metric('Guardrail Verdict', verdict)

    st.divider()

    # Output tabs — one per business unit / use case
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '📰 Full Summary',
        '⚡ Executive',
        '📱 Push Notification',
        '🏆 Player of Match',
        '⏱ Key Moments'
    ])

    with tab1:
        st.subheader('Full Match Summary')
        st.caption('For editorial team — website match report section')
        st.write(highlights.get('full_summary', 'Not generated'))

    with tab2:
        st.subheader('Executive Summary')
        st.caption('For match result cards and in-app match center')
        st.info(highlights.get('executive_summary', 'Not generated'))

    with tab3:
        st.subheader('Push Notification')
        st.caption('For Growth/Marketing team → Braze/Firebase mobile push')
        notif = highlights.get('push_notification', 'Not generated')
        st.success(notif)
        char_count = len(notif)
        color = 'green' if char_count <= 120 else 'red'
        st.markdown(
            f'Character count: :{color}[**{char_count}**] / 120'
        )

    with tab4:
        st.subheader('Player of the Match')
        st.caption('For post-match stats cards and social media')
        st.write(highlights.get('player_of_match', 'Not generated'))

    with tab5:
        st.subheader('Key Moments')
        st.caption(
            'For Video Player Engineering team — '
            'VOD chapter markers on replay timeline'
        )
        moments = highlights.get('key_moments', [])
        if moments:
            for km in moments:
                icon = '⚽' if 'goal' in km.get('event','').lower() \
                       else '🟥' if 'red' in km.get('event','').lower() \
                       else '🟨' if 'yellow' in km.get('event','').lower() \
                       else '🔄'
                st.markdown(
                    f"{icon} **{km.get('minute','')}\'** "
                    f"*{km.get('event','')}* — "
                    f"{km.get('description','')}"
                )
        else:
            st.write('No key moments generated')

    # Raw classified events expander
    with st.expander('View classified events (pipeline internals)'):
        st.caption(
            'Each commentary line classified by TF-IDF + SVM. '
            'High-impact events (score ≥ 2) sent to LLM.'
        )
        classified = data.get('classified', [])
        if classified:
            import pandas as pd
            df_cls = pd.DataFrame(classified)[
                ['minute', 'label', 'confidence', 'impact', 'text']
            ]
            st.dataframe(df_cls, use_container_width=True)
