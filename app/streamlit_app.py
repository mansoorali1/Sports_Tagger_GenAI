"""
streamlit_app.py — Demo Interface

Interactive UI for the Sports Auto-Tagger pipeline.
Demonstrates the CLEBS (Centralized Live Event Broadcast Service)
architecture where one pipeline serves multiple output formats
to different business units simultaneously.
"""

# import os
# import requests
# import streamlit as st
# from pipeline import run_pipeline


# API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# # ── Page config ────────────────────────────────────────────────────

# st.set_page_config(
#     page_title='Sports Auto-Tagger',
#     page_icon='⚽',
#     layout='wide'
# )

# st.title('⚽ Sports Auto-Tagger & Highlight Generator')
# st.markdown(
#     'Paste raw football commentary → get instant multi-format '
#     'match highlights powered by NLP classification and LLM summarization.'
# )
# st.divider()

# # ── Sidebar — match metadata ───────────────────────────────────────

# with st.sidebar:
#     st.header('Match Details')
#     st.caption('Provide match context to improve summary quality')

#     home_team   = st.text_input('Home Team',   value='Manchester City')
#     away_team   = st.text_input('Away Team',   value='Manchester United')
#     score       = st.text_input('Final Score', value='1-0',
#                                 help='Format: 2-1')
#     competition = st.text_input('Competition', value='Premier League')
#     match_date  = st.date_input('Match Date')

#     st.divider()
#     st.caption(
#         'This tool demonstrates an automated sports content pipeline. '
#         'Generated content is verified by a post-generation guardrail '
#         'before display.'
#     )

# # ── Main area — commentary input ───────────────────────────────────

# st.subheader('Paste Match Commentary')

# sample = """45' Goal! Erling Haaland (Manchester City) right footed shot from the centre of the box to the bottom right corner. Assisted by Kevin De Bruyne.
# 67' Yellow card shown to Bruno Fernandes (Manchester United) for a bad foul.
# 78' Substitution, Manchester City. Phil Foden replaces Bernardo Silva.
# 85' Red card for Harry Maguire (Manchester United) for violent conduct."""

# commentary = st.text_area(
#     'Commentary text',
#     height=200,
#     placeholder=sample,
#     help='One event per line. Minutes optional but improve output quality.'
# )

# generate_btn = st.button('Generate Highlights', type='primary',
#                           use_container_width=True)

# # ── Generation ─────────────────────────────────────────────────────

# if generate_btn:
#     if not commentary.strip():
#         st.error('Please paste some commentary text first.')
#         st.stop()

#     with st.spinner('Classifying events and generating highlights...'):
#         try:
#             response = requests.post(
#                 f'{API_URL}/generate-highlights',
#                 json={
#                     'commentary' : commentary,
#                     'home_team'  : home_team,
#                     'away_team'  : away_team,
#                     'score'      : score,
#                     'competition': competition,
#                     'match_date' : str(match_date)
#                 },
#                 timeout=60
#             )
#             response.raise_for_status()
#             data = response.json()

#         except requests.exceptions.ConnectionError:
#             st.error(
#                 'Cannot connect to API. '
#                 'Make sure the FastAPI server is running.'
#             )
#             st.stop()
#         except Exception as e:
#             st.error(f'Error: {e}')
#             st.stop()

#     # Guardrail status
#     guardrail = data.get('guardrail', {})
#     verdict   = guardrail.get('verdict', 'UNKNOWN')

#     if verdict == 'PASS':
#         st.success('Guardrail: PASS — All facts verified against source events')
#     elif verdict == 'WARN':
#         st.warning(
#             'Guardrail: WARN — Content may need review before publishing. '
#             + ' | '.join(guardrail.get('warnings', []))
#         )
#     else:
#         st.error(
#             'Guardrail: FAIL — Content blocked. '
#             + ' | '.join(guardrail.get('hard_issues', []))
#         )

#     highlights = data.get('highlights', {})

#     # Pipeline stats
#     col1, col2, col3 = st.columns(3)
#     col1.metric('Segments Found',    data.get('segments_found', 0))
#     col2.metric('Key Events Used',   data.get('key_events_used', 0))
#     col3.metric('Guardrail Verdict', verdict)

#     st.divider()

#     # Output tabs — one per business unit / use case
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         '📰 Full Summary',
#         '⚡ Executive',
#         '📱 Push Notification',
#         '🏆 Player of Match',
#         '⏱ Key Moments'
#     ])

#     with tab1:
#         st.subheader('Full Match Summary')
#         st.caption('For editorial team — website match report section')
#         st.write(highlights.get('full_summary', 'Not generated'))

#     with tab2:
#         st.subheader('Executive Summary')
#         st.caption('For match result cards and in-app match center')
#         st.info(highlights.get('executive_summary', 'Not generated'))

#     with tab3:
#         st.subheader('Push Notification')
#         st.caption('For Growth/Marketing team → Braze/Firebase mobile push')
#         notif = highlights.get('push_notification', 'Not generated')
#         st.success(notif)
#         char_count = len(notif)
#         color = 'green' if char_count <= 120 else 'red'
#         st.markdown(
#             f'Character count: :{color}[**{char_count}**] / 120'
#         )

#     with tab4:
#         st.subheader('Player of the Match')
#         st.caption('For post-match stats cards and social media')
#         st.write(highlights.get('player_of_match', 'Not generated'))

#     with tab5:
#         st.subheader('Key Moments')
#         st.caption(
#             'For Video Player Engineering team — '
#             'VOD chapter markers on replay timeline'
#         )
#         moments = highlights.get('key_moments', [])
#         if moments:
#             for km in moments:
#                 icon = '⚽' if 'goal' in km.get('event','').lower() \
#                        else '🟥' if 'red' in km.get('event','').lower() \
#                        else '🟨' if 'yellow' in km.get('event','').lower() \
#                        else '🔄'
#                 st.markdown(
#                     f"{icon} **{km.get('minute','')}\'** "
#                     f"*{km.get('event','')}* — "
#                     f"{km.get('description','')}"
#                 )
#         else:
#             st.write('No key moments generated')

#     # Raw classified events expander
#     with st.expander('View classified events (pipeline internals)'):
#         st.caption(
#             'Each commentary line classified by TF-IDF + SVM. '
#             'High-impact events (score ≥ 2) sent to LLM.'
#         )
#         classified = data.get('classified', [])
#         if classified:
#             import pandas as pd
#             df_cls = pd.DataFrame(classified)[
#                 ['minute', 'label', 'confidence', 'impact', 'text']
#             ]
#             st.dataframe(df_cls, use_container_width=True)


# import os
# import requests
# import streamlit as st

# API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# # ── Page config ────────────────────────────────────────────────────

# st.set_page_config(
#     page_title='Sports Auto-Tagger',
#     page_icon='⚽',
#     layout='wide'
# )

# st.title('⚽ Sports Auto-Tagger & Highlight Generator')
# st.markdown(
#     'Paste raw football commentary → get instant multi-format '
#     'match highlights. Sidebar details are fully optional and will be auto-extracted from text if left blank.'
# )
# st.divider()

# # ── Sidebar — match metadata ───────────────────────────────────────

# with st.sidebar:
#     st.header('Match Details (Optional)')
#     st.caption('Leave blank to auto-detect teams and scores from text')

#     home_team   = st.text_input('Home Team',   value='', placeholder='e.g. Brighton')
#     away_team   = st.text_input('Away Team',   value='', placeholder='e.g. Man United')
#     score       = st.text_input('Final Score', value='', placeholder='e.g. 0-3')
#     competition = st.text_input('Competition', value='', placeholder='e.g. Premier League')

#     st.divider()
#     st.caption(
#         'This tool demonstrates an automated sports content pipeline. '
#         'Generated content is verified by a post-generation guardrail algorithm.'
#     )

# # ── Main UI Layout ──────────────────────────────────────────────────

# col_in, col_out = st.columns([1, 1])

# with col_in:
#     st.subheader('Input Commentary Feed')
#     raw_commentary = st.text_area(
#         'Paste text copy-pasted directly from feed provider:',
#         height=450,
#         placeholder="90'+6' Full Time...\n48' Goal! Bruno Fernandes..."
#     )
    
#     generate_btn = st.button('🚀 Generate Highlights', use_container_width=True)

# with col_out:
#     st.subheader('Generated Editorial Artifacts')
    
#     if generate_btn:
#         if not raw_commentary.strip():
#             st.error('Please paste some text commentary first.')
#         else:
#             with st.spinner('Running AI Pipeline (Extracting context, classifying events & summarizing)...'):
#                 payload = {
#                     'commentary': raw_commentary,
#                     'home_team': home_team,
#                     'away_team': away_team,
#                     'score': score if score.strip() else 'Unknown',
#                     'competition': competition if competition.strip() else 'Football Match'
#                 }
                
#                 try:
#                     res = requests.post(f"{API_URL}/generate-highlights", json=payload)
#                     if res.status_code == 200:
#                         data = res.json()
#                         highlights = data.get('highlights', {})
#                         guardrail = data.get('guardrail', {})
#                         match_info = data.get('match_info', {})
                        
#                         # Show what was used/extracted
#                         st.success(
#                             f"Processed Match: **{match_info.get('home_team')}** vs **{match_info.get('away_team')}** "
#                             f"({match_info.get('score')}) — *{match_info.get('competition')}*"
#                         )
                        
#                         # Guardrail Status Card
#                         verdict = guardrail.get('verdict', 'PASS')
#                         if verdict == 'PASS':
#                             st.success('✅ Guardrail Status: PASS (Context Factuality Verified)')
#                         elif verdict == 'WARN':
#                             st.warning('⚠️ Guardrail Status: WARNING (Minor deviations caught)')
#                             for w in guardrail.get('warnings', []):
#                                 st.write(f"- {w}")
#                         else:
#                             st.error('❌ Guardrail Status: FAIL (Hallucinations blocked)')
#                             for h in guardrail.get('hard_issues', []):
#                                 st.write(f"- {h}")
                        
#                         # Tabs for editorial content channels
#                         tab1, tab2, tab3, tab4, tab5 = st.tabs([
#                             'Full Summary', 'Executive Summary', 
#                             'Push Notification', 'Player of Match', 'Timeline Chapter Markers'
#                         ])
                        
#                         with tab1:
#                             st.markdown(highlights.get('full_summary', 'Not generated'))
#                         with tab2:
#                             st.write(highlights.get('executive_summary', 'Not generated'))
#                         with tab3:
#                             st.code(highlights.get('push_notification', 'Not generated'))
#                         with tab4:
#                             st.write(highlights.get('player_of_match', 'Not generated'))
#                         with tab5:
#                             moments = highlights.get('key_moments', [])
#                             if moments:
#                                 for km in moments:
#                                     icon = '⚽' if 'goal' in km.get('event','').lower() \
#                                            else '🟥' if 'red' in km.get('event','').lower() \
#                                            else '🟨' if 'yellow' in km.get('event','').lower() \
#                                            else '🔄'
#                                     st.markdown(f"{icon} **{km.get('minute','')}'** *{km.get('event','')}* — {km.get('description','')}")
#                             else:
#                                 st.write('No key moments generated')
                                
#                         # Internal breakdown expander
#                         with st.expander('View classified events (pipeline internals)'):
#                             st.json(data.get('classified', []))
                            
#                     else:
#                         st.error(f"Backend Error ({res.status_code}): {res.text}")
#                 except Exception as e:
#                     st.error(f"Could not connect to FastAPI server: {str(e)}")
#     else:
#         st.info('Paste match text on the left and click Generate to run.')




import os
import requests
import streamlit as st

# API runs as separate process in same container
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# ── Page config ──────────────────────────────────────────────────

st.set_page_config(
    page_title='MatchLens AI',
    page_icon='⚽',
    layout='wide'
)

st.title('⚽ MatchLens AI — Sports Commentary Auto-Tagger')
st.markdown(
    'Paste raw football commentary and get instant multi-format '
    'match highlights powered by NLP classification and LLM summarization.'
)
st.divider()

# ── Sidebar — optional match metadata ───────────────────────────

with st.sidebar:
    st.header('Match Details')
    st.caption(
        'Optional — fill in for better accuracy. '
        'Leave blank to auto-detect from commentary.'
    )

    # Empty defaults — no pre-filled values that could mislead
    home_team   = st.text_input('Home Team',   value='',
                                placeholder='e.g. Brighton')
    away_team   = st.text_input('Away Team',   value='',
                                placeholder='e.g. Manchester United')
    score       = st.text_input('Final Score', value='',
                                placeholder='e.g. 0-3',
                                help='Format: HomeGoals-AwayGoals')
    competition = st.text_input('Competition', value='',
                                placeholder='e.g. Premier League')

    st.divider()
    st.caption(
        'Generated content is verified by a post-generation '
        'guardrail before display.'
    )

# ── Main area ────────────────────────────────────────────────────

st.subheader('Paste Match Commentary')
st.caption(
    'Copy from any football website, live feed, or text document. '
    'One event per line works best.'
)

commentary = st.text_area(
    label='Commentary text',
    height=250,
    placeholder=(
        "33' Goal! Brighton 0-1 Manchester United. Patrick Dorgu header...\n"
        "44' Goal! Brighton 0-2 Manchester United. Bryan Mbeumo...\n"
        "48' Goal! Brighton 0-3 Manchester United. Bruno Fernandes...\n"
        "45+3' Yellow Card. Kobbie Mainoo shown yellow for a bad foul."
    ),
    help='Paste full or partial match commentary. Minutes optional.'
)

generate_btn = st.button(
    'Generate Highlights',
    type='primary',
    use_container_width=True
)

# ── Generation ───────────────────────────────────────────────────

if generate_btn:
    if not commentary.strip():
        st.error('Please paste some commentary text first.')
        st.stop()

    if len(commentary.strip()) < 50:
        st.error('Commentary too short. Paste at least a few match events.')
        st.stop()

    # Build request — use empty string if user left fields blank
    # Pipeline handles empty strings by extracting from commentary
    payload = {
        'commentary' : commentary,
        'home_team'  : home_team.strip(),
        'away_team'  : away_team.strip(),
        'score'      : score.strip() if score.strip() else 'Unknown',
        'competition': competition.strip() if competition.strip() else 'Football Match'
    }

    with st.spinner('Classifying events and generating highlights...'):
        try:
            response = requests.post(
                f'{API_URL}/generate-highlights',
                json=payload,
                timeout=90
            )

            if response.status_code != 200:
                st.error(
                    f'API Error {response.status_code}: '
                    f'{response.json().get("error", response.text)}'
                )
                st.stop()

            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error('Cannot connect to API backend.')
            st.stop()
        except requests.exceptions.Timeout:
            st.error('Request timed out. Try with shorter commentary.')
            st.stop()
        except Exception as e:
            st.error(f'Unexpected error: {e}')
            st.stop()

    # ── Match info banner ────────────────────────────────────────
    match_info = data.get('match_info', {})
    st.success(
        f"**{match_info.get('home_team', '?')}** vs "
        f"**{match_info.get('away_team', '?')}** — "
        f"Score: {match_info.get('score', '?')} — "
        f"{match_info.get('competition', '')}"
    )

    # ── Guardrail status ─────────────────────────────────────────
    guardrail = data.get('guardrail', {})
    verdict   = guardrail.get('verdict', 'UNKNOWN')

    if verdict == 'PASS':
        st.success('Guardrail: PASS — All facts verified')
    elif verdict == 'WARN':
        issues = ' | '.join(guardrail.get('warnings', []))
        st.warning(f'Guardrail: WARN — Review before publishing. {issues}')
    else:
        issues = ' | '.join(guardrail.get('hard_issues', []))
        st.error(f'Guardrail: FAIL — Content blocked. {issues}')

    # ── Pipeline metrics ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric('Segments Found',  data.get('segments_found', 0))
    col2.metric('Key Events Used', data.get('key_events_used', 0))
    col3.metric('Guardrail',       verdict)

    st.divider()

    highlights = data.get('highlights', {})

    # ── Output tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '📰 Full Summary',
        '⚡ Executive',
        '📱 Push Notification',
        '🏆 Player of Match',
        '⏱ Key Moments'
    ])

    with tab1:
        st.subheader('Full Match Summary')
        st.caption('For editorial team — website match report')
        st.write(highlights.get('full_summary', 'Not generated'))

    with tab2:
        st.subheader('Executive Summary')
        st.caption('For match result cards and in-app match center')
        st.info(highlights.get('executive_summary', 'Not generated'))

    with tab3:
        st.subheader('Push Notification')
        st.caption('For Growth/Marketing team → mobile push alerts')
        notif      = highlights.get('push_notification', 'Not generated')
        char_count = len(notif)
        st.success(notif)
        color = 'green' if char_count <= 120 else 'red'
        st.markdown(f'Character count: :{color}[**{char_count}**] / 120')

    with tab4:
        st.subheader('Player of the Match')
        st.caption('For post-match stats cards and social media')
        st.write(highlights.get('player_of_match', 'Not generated'))

    with tab5:
        st.subheader('Key Moments')
        st.caption('For Video Player Engineering — VOD chapter markers')
        moments = highlights.get('key_moments', [])
        if moments:
            for km in moments:
                event_lower = km.get('event', '').lower()
                icon = ('⚽' if 'goal' in event_lower
                        else '🟥' if 'red' in event_lower
                        else '🟨' if 'yellow' in event_lower
                        else '🔄')
                st.markdown(
                    f"{icon} **{km.get('minute', '?')}'** "
                    f"*{km.get('event', '')}* — "
                    f"{km.get('description', '')}"
                )
        else:
            st.write('No key moments generated')

    # ── Pipeline internals expander ──────────────────────────────
    with st.expander('View classified events (pipeline internals)'):
        st.caption(
            'Each commentary line classified by TF-IDF + SVM. '
            'High-impact events (impact ≥ 2) sent to LLM.'
        )
        classified = data.get('classified', [])
        if classified:
            import pandas as pd
            df_cls = pd.DataFrame(classified)
            cols   = [c for c in ['minute', 'label', 'confidence',
                                   'impact', 'text'] if c in df_cls.columns]
            st.dataframe(df_cls[cols], use_container_width=True)
