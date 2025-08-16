# app.py
# Fantasy Combine – Lobby + TRUE Modal Entry (uses st.dialog)

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page & Style ----------
st.set_page_config(page_title="Fantasy Combine Scorer", page_icon="🏈", layout="wide")
ACCENT = "#7c3aed"  # kept for other accents if needed

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 700px at 5% 0%, #0b1220 0%, #0f172a 45%, #0a0f1a 100%) !important;
  color: #e5e7eb !important;
  font-variant-numeric: tabular-nums;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

/* === Inputs: black text on white background === */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
input[type="text"],
input[type="number"] {{
  color: #111 !important;
  background-color: #fff !important;
  caret-color: #111 !important;
  border: 1px solid rgba(0,0,0,.3) !important;
}}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
input[type="text"]::placeholder,
input[type="number"]::placeholder {{
  color: #111 !important;
  opacity: .6 !important;
}}

/* Force text color in all input contexts */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
  color: #111 !important;
  background-color: #fff !important;
}}

/* === Input-like buttons (Name/Weight displays): white bg, black text === */
.input-like .stButton > button {{
  color: #000 !important;
  background: #ffffff !important;
  border: 2px solid #333 !important;
  border-radius: 12px !important;
  font-weight: 900 !important;
  font-size: 14px !important;
  text-shadow: none !important;
  box-shadow: none !important;
  transition: none !important;
  filter: none !important;
}}
.input-like .stButton > button:hover {{
  color: #000 !important;
  background: #f8f8f8 !important;
  border: 2px solid #333 !important;
  box-shadow: none !important;
  filter: none !important;
  transform: none !important;
}}
.input-like .stButton > button:active {{ 
  color: #000 !important;
  background: #f0f0f0 !important;
  border: 2px solid #333 !important;
  box-shadow: none !important;
  filter: none !important;
  transform: none !important;
}}
.input-like .stButton > button:focus {{
  color: #000 !important;
  background: #ffffff !important;
  border: 2px solid #333 !important;
  box-shadow: none !important;
  filter: none !important;
}}

/* Nuclear option - force black text everywhere in input-like buttons */
.input-like *,
.input-like button,
.input-like button *,
.input-like .stButton,
.input-like .stButton *,
.input-like .stButton > button,
.input-like .stButton > button *,
.input-like .stButton > button > div,
.input-like .stButton > button > div *,
.input-like .stButton > button span,
.input-like .stButton > button p,
.input-like [data-testid="baseButton-secondary"],
.input-like [data-testid="baseButton-secondary"] *,
.input-like [class*="Button"],
.input-like [class*="Button"] * {{
  color: #000 !important;
  text-shadow: none !important;
  filter: none !important;
  opacity: 1 !important;
}}

/* === Red animated action buttons (Go to Lobby, Start, Enter/Clear, etc.) === */
.accent .stButton > button {{
  color: #fff !important;
  background: linear-gradient(180deg, #ef4444, #dc2626) !important; /* red */
  border: none !important;
  border-radius: 12px !important;
  box-shadow: 0 10px 25px rgba(239,68,68,.35);
  transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
}}
.accent .stButton > button:hover {{
  filter: brightness(1.05);
  box-shadow: 0 14px 35px rgba(239,68,68,.45);
}}
.accent .stButton > button:active {{ transform: translateY(1px) scale(.997); }}

/* Compact red buttons for row actions */
.accent-small .stButton > button {{
  color: #fff !important;
  background: linear-gradient(180deg, #ef4444, #dc2626) !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 8px 10px !important;
  box-shadow: 0 6px 18px rgba(239,68,68,.35);
  transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
}}
.accent-small .stButton > button:hover {{
  filter: brightness(1.05);
  box-shadow: 0 10px 24px rgba(239,68,68,.45);
}}
.accent-small .stButton > button:active {{ transform: translateY(1px) scale(.997); }}

/* Cards/table/leader styles kept */
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.04));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  margin-bottom: 16px;
  animation: fadeInUp .35s ease-out;
}}
.h-title {{ font-weight: 800; letter-spacing: .2px; }}
.subtle {{ opacity: .85; }}
a, .linkish {{ color: #c4b5fd; font-weight: 800; text-decoration: none; }}
.linkish:hover {{ color: #e9d5ff; text-decoration: underline; }}

.badge {{
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  font-weight: 700; font-size: 11px; letter-spacing: .3px;
  border: 1px solid rgba(255,255,255,.15);
}}
.badge.done {{ background: rgba(34,197,94,.15); color: #a7f3d0; border-color: rgba(34,197,94,.35); }}
.badge.pending {{ background: rgba(59,130,246,.15); color: #bfdbfe; border-color: rgba(59,130,246,.35); }}

.table-compact table {{ width: 100%; }}
.table-compact thead th {{ font-size: 12px !important; }}
.table-compact tbody td {{ font-size: 13px !important; padding-top: 6px !important; padding-bottom: 6px !important; }}

.leader-row {{
  display: grid; grid-template-columns: 56px 1.2fr .8fr .8fr; gap: 10px;
  align-items: center; padding: 12px 14px; border-radius: 12px;
  border: 1px solid rgba(255,255,255,.1);
  background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.03));
  box-shadow: 0 8px 18px rgba(0,0,0,.25);
  margin-bottom: 8px; animation: popIn .2s ease-out;
}}
.leader-rank-badge {{
  width: 44px; height: 44px; display:flex; align-items:center; justify-content:center;
  border-radius: 999px; font-weight: 800; background: rgba(124,58,237,.15);
  border: 1px solid rgba(124,58,237,.35);
}}
.small-btn {{
  border-radius: 10px; padding: 8px 10px; font-weight: 700; border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.06); color: #e5e7eb;
  box-shadow: 0 6px 16px rgba(0,0,0,.25);
}}
.small-btn:hover {{ filter: brightness(1.05); }}

@keyframes fadeInUp {{ from {{ transform: translateY(6px); opacity: 0; }} to {{ transform: translateY(0); opacity: 1; }} }}
@keyframes popIn {{ from {{ transform: scale(.98); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='h-title' style='font-size:32px; margin-bottom:0;'>🏈 Fantasy Combine Scorer</h1>"
    "<div class='subtle' style='margin-top:4px;'>Lobby + true modal entry everywhere (names, weights, events).</div>",
    unsafe_allow_html=True
)

# ---------- Constants & State ----------
DEFAULT_N = 12
EVENTS = ["40", "3C", "Broad", "Accuracy", "PPK"]
EVENT_TITLES = {"40":"40 yd dash","3C":"3-Cone","Broad":"Broad Jump","Accuracy":"Accuracy","PPK":"PPK (total yd)"}
EVENT_COLS   = {"40":"40 yd (s)","3C":"3-cone (s)","Broad":"Broad Jump (in)","Accuracy":"Accuracy pts","PPK":"PPK (yd)"}
EVENT_MIN    = {"40":0.0,"3C":0.0,"Broad":0.0,"Accuracy":0,"PPK":0}

def _init_state():
    ss = st.session_state
    if "page" not in ss: ss.page = "setup"  # setup | lobby | event | event_results
    if "event" not in ss: ss.event = None
    if "roster_size" not in ss: ss.roster_size = DEFAULT_N
    if "players" not in ss:
        ss.players = pd.DataFrame({
            "ID":   list(range(1, DEFAULT_N+1)),
            "Name": [f"Player {i+1}" for i in range(DEFAULT_N)],
            "Weight (lb)": [np.nan]*DEFAULT_N,
            "40 yd (s)": [np.nan]*DEFAULT_N,
            "3-cone (s)": [np.nan]*DEFAULT_N,
            "Broad Jump (in)": [np.nan]*DEFAULT_N,
            "Accuracy pts": [np.nan]*DEFAULT_N,
            "PPK (yd)": [np.nan]*DEFAULT_N,
        })
    if "next_id" not in ss: ss.next_id = len(ss.players) + 1
_init_state()

# ---------- Metrics & Scoring ----------
def to_num(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""): return np.nan
        return float(x)
    except Exception:
        return np.nan

def speed_score(W, t40):
    W = to_num(W); t = to_num(t40)
    if np.isnan(W) or np.isnan(t) or t <= 0: return np.nan
    return (W * 200.0) / (t ** 4)

def agility_score(W, t3c):
    W = to_num(W); t = to_num(t3c)
    if np.isnan(W) or np.isnan(t) or t <= 0: return np.nan
    return (W * 200.0) / (t ** 4)

def broad_jump_index(W, inches):
    W = to_num(W); L = to_num(inches)
    if np.isnan(W) or np.isnan(L) or L < 0: return np.nan
    return L * ((W / 150.0) ** 0.67)

def rank_desc(values):
    s = pd.Series(values, dtype=float)
    if s.notna().sum() == 0: return pd.Series([np.nan]*len(s))
    return s.rank(method="average", ascending=False)

def points_from_rank(rs):
    s = rs.copy()
    N = s.notna().sum()
    if N == 0: return pd.Series([0]*len(s), index=s.index)
    out = np.round(100.0 * (N + 1 - s) / N)
    out[s.isna()] = 0
    return out.astype(int)

def compute_scoreboard(df: pd.DataFrame):
    speed = df.apply(lambda r: speed_score(r["Weight (lb)"], r["40 yd (s)"]), axis=1)
    agil  = df.apply(lambda r: agility_score(r["Weight (lb)"], r["3-cone (s)"]), axis=1)
    broad = df.apply(lambda r: broad_jump_index(r["Weight (lb)"], r["Broad Jump (in)"]), axis=1)
    acc   = df["Accuracy pts"]
    ppk   = df["PPK (yd)"]

    r40, r3c, rbr = rank_desc(speed), rank_desc(agil), rank_desc(broad)
    racc, rppk = rank_desc(acc), rank_desc(ppk)

    pts40, pts3c, ptsbr = points_from_rank(r40), points_from_rank(r3c), points_from_rank(rbr)
    ptsac, ptspp = points_from_rank(racc), points_from_rank(rppk)

    total = (pts40 + pts3c + ptsbr + ptsac + ptspp).astype(int)
    overall = rank_desc(total)

    board = df.copy()
    board["SpeedScore"]   = speed
    board["AgilityScore"] = agil
    board["BroadIndex"]   = broad
    board["Pts 40"]       = pts40
    board["Pts 3C"]       = pts3c
    board["Pts Broad"]    = ptsbr
    board["Pts Acc"]      = ptsac
    board["Pts PPK"]      = ptspp
    board["Total"]        = total
    board["Overall Rank"] = overall

    leader = board[["ID","Name","Weight (lb)","Total"]].copy().sort_values("Total", ascending=False).reset_index(drop=True)
    leader.index = leader.index + 1
    return board, leader

def compute_event_results(df: pd.DataFrame, event: str):
    """Compute results for a specific event only"""
    col = EVENT_COLS[event]
    
    if event == "40":
        score = df.apply(lambda r: speed_score(r["Weight (lb)"], r["40 yd (s)"]), axis=1)
        pts_col = "Pts 40"
    elif event == "3C":
        score = df.apply(lambda r: agility_score(r["Weight (lb)"], r["3-cone (s)"]), axis=1)
        pts_col = "Pts 3C"
    elif event == "Broad":
        score = df.apply(lambda r: broad_jump_index(r["Weight (lb)"], r["Broad Jump (in)"]), axis=1)
        pts_col = "Pts Broad"
    elif event == "Accuracy":
        score = df["Accuracy pts"]
        pts_col = "Pts Acc"
    elif event == "PPK":
        score = df["PPK (yd)"]
        pts_col = "Pts PPK"
    
    rank = rank_desc(score)
    pts = points_from_rank(rank)
    
    results = df[["ID", "Name", "Weight (lb)", col]].copy()
    if event in ["40", "3C", "Broad"]:
        results["Score"] = score
    results["Rank"] = rank
    results["Points"] = pts
    
    # Sort by rank (best first)
    results = results.sort_values("Rank").reset_index(drop=True)
    results.index = results.index + 1
    
    return results

def event_status(df: pd.DataFrame, e: str):
    col = EVENT_COLS[e]
    if df[col].isna().all(): return "pending"
    return "done" if df[col].notna().all() else "pending"

def names_ok(df):   return df["Name"].astype(str).str.strip().ne("").all()
def weights_ok(df): return df["Weight (lb)"].notna().all()

# ---------- MODAL (uses st.dialog) ----------
def open_modal_edit(row_id: int, col: str, label: str, is_text: bool, minv: float = 0.0):
    df = st.session_state.players
    matches = df.index[df["ID"] == row_id].tolist()
    if not matches:
        st.warning("Could not find that row."); return
    ridx = matches[0]

    @st.dialog("Enter Value")
    def _modal():
        st.caption(label)
        if is_text:
            val = st.text_input("Value", value=str(df.at[ridx, col] if pd.notna(df.at[ridx, col]) else ""))
        else:
            preset = "" if pd.isna(df.at[ridx, col]) else str(df.at[ridx, col])
            st.markdown(
                "<div style='font-size:14px;opacity:.8;margin-bottom:6px;'>Numbers only"
                f"{' (min '+str(minv)+')' if minv is not None else ''}.</div>",
                unsafe_allow_html=True
            )
            val = st.text_input("Value", value=preset, label_visibility="collapsed", key="modal_number")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", use_container_width=True):
                if is_text:
                    v = val.strip()
                    if v == "":
                        st.warning("Name cannot be empty.")
                        st.stop()
                    st.session_state.players.at[ridx, col] = v
                    st.rerun()
                else:
                    try:
                        f = float(val)
                        if f < minv:
                            st.warning(f"Value must be ≥ {minv}.")
                            st.stop()
                        st.session_state.players.at[ridx, col] = f
                        st.rerun()
                    except Exception:
                        st.warning("Please enter a valid number.")
                        st.stop()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()

    _modal()

# ---------- PAGES ----------
def page_setup():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='h-title' style='font-size:20px;'>Setup: Names & Weights</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,3], vertical_alignment="top")
    with c1:
        n = st.number_input("Roster size", min_value=2, max_value=24,
                            value=st.session_state.roster_size, step=1)
        if n != st.session_state.roster_size:
            st.session_state.roster_size = int(n)
            cur = st.session_state.players.copy()
            if len(cur) < n:
                add = n - len(cur)
                extra = pd.DataFrame({
                    "ID":   list(range(st.session_state.next_id, st.session_state.next_id + add)),
                    "Name": [f"Player {i+1}" for i in range(len(cur), n)],
                    "Weight (lb)": [np.nan]*add,
                    "40 yd (s)": [np.nan]*add,
                    "3-cone (s)": [np.nan]*add,
                    "Broad Jump (in)": [np.nan]*add,
                    "Accuracy pts": [np.nan]*add,
                    "PPK (yd)": [np.nan]*add,
                })
                st.session_state.next_id += add
                st.session_state.players = pd.concat([cur, extra], ignore_index=True)
            else:
                st.session_state.players = cur.iloc[:n].reset_index(drop=True)
    with c2:
        st.caption("Click any **Name** or **Weight** to edit via modal.")
        header = st.columns([6, 3, 3])
        header[0].markdown("**Name**")
        header[1].markdown("**Weight (lb)**")
        header[2].markdown("")

        df = st.session_state.players

        # Input-like white/black buttons scope with inline styles
        for _, row in df.iterrows():
            cols = st.columns([6, 3, 3])
            with cols[0]:
                st.markdown(
                    f'<div style="background: white; color: black; padding: 8px 12px; border-radius: 12px; border: 1px solid #ccc; text-align: center; font-weight: bold; margin: 2px 0;">{row["Name"]}</div>',
                    unsafe_allow_html=True
                )
                if st.button("Edit Name", key=f"setup_name_{row['ID']}", use_container_width=True):
                    open_modal_edit(row_id=row["ID"], col="Name",
                                    label=f"Edit Name — Player ID {row['ID']}", is_text=True)
            with cols[1]:
                wlabel = "—" if pd.isna(row["Weight (lb)"]) else f"{int(row['Weight (lb)'])}"
                st.markdown(
                    f'<div style="background: white; color: black; padding: 8px 12px; border-radius: 12px; border: 1px solid #ccc; text-align: center; font-weight: bold; margin: 2px 0;">{wlabel}</div>',
                    unsafe_allow_html=True
                )
                if st.button("Edit Weight", key=f"setup_w_{row['ID']}", use_container_width=True):
                    open_modal_edit(row_id=row["ID"], col="Weight (lb)",
                                    label=f"Edit Weight (lb) — {row['Name']}", is_text=False, minv=0.0)
            cols[2].markdown("&nbsp;")

    st.markdown("</div>", unsafe_allow_html=True)

    # Red animated action button
    st.markdown("<div class='accent'>", unsafe_allow_html=True)
    if st.button("Enter Lobby ▶", type="primary", use_container_width=False):
        if not names_ok(st.session_state.players):
            st.warning("Please enter all player names.")
        elif not weights_ok(st.session_state.players):
            st.warning("Please enter all weights.")
        else:
            st.session_state.page = "lobby"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def leader_list_clickable(leader_df: pd.DataFrame):
    # Input-like styling for name/weight in leaderboard with inline styles
    for idx, row in leader_df.reset_index().iterrows():
        rank = idx + 1
        pid  = int(row["ID"])
        name = row["Name"]
        wt   = row["Weight (lb)"]
        total= int(row["Total"])
        st.markdown("<div class='leader-row'>", unsafe_allow_html=True)
        c0, c1, c2, c3 = st.columns([0.6, 1.2, 1.0, 1.0])
        c0.markdown(f"<div class='leader-rank-badge'>{rank}</div>", unsafe_allow_html=True)
        with c1:
            st.markdown(
                f'<div style="background: white; color: black; padding: 8px 12px; border-radius: 12px; border: 1px solid #ccc; text-align: center; font-weight: bold; margin: 2px 0;">{name}</div>',
                unsafe_allow_html=True
            )
            if st.button("Edit Name", key=f"lb_name_{pid}", use_container_width=True):
                open_modal_edit(row_id=pid, col="Name", label=f"Edit Name — Rank #{rank}", is_text=True)
        with c2:
            wlabel = "—" if pd.isna(wt) else f"{int(wt)} lb"
            st.markdown(
                f'<div style="background: white; color: black; padding: 8px 12px; border-radius: 12px; border: 1px solid #ccc; text-align: center; font-weight: bold; margin: 2px 0;">{wlabel}</div>',
                unsafe_allow_html=True
            )
            if st.button("Edit Weight", key=f"lb_w_{pid}", use_container_width=True):
                open_modal_edit(row_id=pid, col="Weight (lb)", label=f"Edit Weight (lb) — {name}", is_text=False, minv=0.0)
        c3.markdown(f"**{total} pts**")
        st.markdown("</div>", unsafe_allow_html=True)

def event_leader_list(event_results: pd.DataFrame, event: str):
    # Display event-specific results in a similar format
    for idx, row in event_results.reset_index().iterrows():
        rank = idx + 1
        pid = int(row["ID"])
        name = row["Name"]
        col = EVENT_COLS[event]
        raw_val = row[col]
        points = int(row["Points"])
        
        st.markdown("<div class='leader-row'>", unsafe_allow_html=True)
        c0, c1, c2, c3 = st.columns([0.6, 1.2, 1.0, 1.0])
        c0.markdown(f"<div class='leader-rank-badge'>{rank}</div>", unsafe_allow_html=True)
        c1.markdown(f"**{name}**")
        
        # Format the raw value nicely
        if pd.isna(raw_val):
            val_display = "—"
        elif event in ["Accuracy", "PPK"]:
            val_display = f"{int(raw_val)}"
        else:
            val_display = f"{raw_val:.2f}"
        c2.markdown(val_display)
        c3.markdown(f"**{points} pts**")
        st.markdown("</div>", unsafe_allow_html=True)

def page_lobby():
    df = st.session_state.players.copy()
    board, leader = compute_scoreboard(df)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='h-title' style='font-size:20px;'>Lobby</h3>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, e in enumerate(EVENTS):
        status = event_status(df, e)
        cls = "done" if status == "done" else "pending"
        label = {"40":"40 yd","3C":"3-Cone","Broad":"Broad Jump","Accuracy":"Accuracy","PPK":"PPK"}[e]
        with cols[i]:
            st.markdown(f"<span class='badge {cls}'>{label}: {status.upper()}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 class='h-title' style='font-size:18px;'>Leaderboard</h4>", unsafe_allow_html=True)
    leader_list_clickable(leader)
    st.caption("Tip: click a Name or Weight here to edit via modal.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card table-compact'>", unsafe_allow_html=True)
    st.markdown("<h4 class='h-title' style='font-size:18px;'>Roster & Current Stats</h4>", unsafe_allow_html=True)
    show_cols = [
        "Name","Weight (lb)","40 yd (s)","3-cone (s)","Broad Jump (in)",
        "Accuracy pts","PPK (yd)","Pts 40","Pts 3C","Pts Broad","Pts Acc","Pts PPK","Total"
    ]
    st.dataframe(board[show_cols], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Red animated action buttons row
    st.markdown("<div class='accent'>", unsafe_allow_html=True)
    colA, colB, colC, colD, colE, colF = st.columns(6)
    with colA:
        if st.button("Edit Names & Weights", use_container_width=True):
            st.session_state.page = "setup"; st.rerun()
    with colB:
        if st.button("Start 40 yd dash", use_container_width=True):
            st.session_state.page = "event"; st.session_state.event = "40"; st.rerun()
    with colC:
        if st.button("Start 3-Cone", use_container_width=True):
            st.session_state.page = "event"; st.session_state.event = "3C"; st.rerun()
    with colD:
        if st.button("Start Broad Jump", use_container_width=True):
            st.session_state.page = "event"; st.session_state.event = "Broad"; st.rerun()
    with colE:
        if st.button("Start Accuracy", use_container_width=True):
            st.session_state.page = "event"; st.session_state.event = "Accuracy"; st.rerun()
    with colF:
        if st.button("Start PPK", use_container_width=True):
            st.session_state.page = "event"; st.session_state.event = "PPK"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def page_event():
    df = st.session_state.players.copy()
    e = st.session_state.event
    col = EVENT_COLS[e]
    title = EVENT_TITLES[e]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='h-title' style='font-size:20px;'>Event: {title}</h3>", unsafe_allow_html=True)
    st.caption("Click **Enter** to input a result for a player. Save/Cancel in the modal.")
    header = st.columns([5, 2, 2, 2])
    header[0].markdown("**Name**")
    header[1].markdown("**Weight (lb)**")
    header[2].markdown(f"**{col}**")
    header[3].markdown("")

    for _, row in df.iterrows():
        cols = st.columns([5, 2, 2, 2])
        cols[0].markdown(row["Name"])
        cols[1].markdown(f"{int(row['Weight (lb)']) if not pd.isna(row['Weight (lb)']) else '—'}")
        val = row[col]
        cols[2].markdown("—" if pd.isna(val) else (f"{val:g}" if isinstance(val, (int,float,np.floating)) else str(val)))

        with cols[3]:
            st.markdown("<div class='accent-small'>", unsafe_allow_html=True)
            if st.button("Enter", key=f"enter_{e}_{row['ID']}", use_container_width=True):
                open_modal_edit(row_id=row["ID"], col=col, label=f"{row['Name']} — {col}",
                                is_text=False, minv=EVENT_MIN[e])
            if st.button("Clear", key=f"clear_{e}_{row['ID']}", use_container_width=True):
                st.session_state.players.loc[st.session_state.players["ID"]==row["ID"], col] = np.nan
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='accent'>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Lobby", use_container_width=True):
            st.session_state.page = "lobby"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='accent'>", unsafe_allow_html=True)
        if st.button("Finish Event ✓", use_container_width=True):
            st.session_state.page = "event_results"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def page_event_results():
    df = st.session_state.players.copy()
    e = st.session_state.event
    title = EVENT_TITLES[e]
    col = EVENT_COLS[e]
    
    # Get event-specific results instead of overall leaderboard
    event_results = compute_event_results(df, e)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='h-title' style='font-size:20px;'>{title} Results</h3>", unsafe_allow_html=True)
    
    # Show event-specific leaderboard
    st.markdown("<h4 class='h-title' style='font-size:16px; margin-bottom:12px;'>Event Leaderboard</h4>", unsafe_allow_html=True)
    event_leader_list(event_results, e)
    st.markdown("</div>", unsafe_allow_html=True)

    # Show detailed event results table
    st.markdown("<div class='card table-compact'>", unsafe_allow_html=True)
    st.markdown("<h4 class='h-title' style='font-size:18px;'>Event Details</h4>", unsafe_allow_html=True)
    
    # Select columns to show based on event type
    display_cols = ["Name", "Weight (lb)", col, "Rank", "Points"]
    if e in ["40", "3C", "Broad"]:
        display_cols.insert(-2, "Score")  # Insert score before rank for calculated events
    
    st.dataframe(event_results[display_cols], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='accent'>", unsafe_allow_html=True)
    if st.button("Return to Lobby ▶", use_container_width=True):
        st.session_state.page = "lobby"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Router ----------
page = st.session_state.page
if page == "setup":
    page_setup()
elif page == "lobby":
    page_lobby()
elif page == "event":
    page_event()
elif page == "event_results":
    page_event_results()
else:
    st.session_state.page = "setup"; st.rerun()