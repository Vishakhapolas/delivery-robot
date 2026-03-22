import streamlit as st
import random
import numpy as np
import pickle
import os
import time

# ─────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Delivery Robot · Q-Learning",
    page_icon="🤖",
    layout="centered",
)

# ─────────────────────────────────────────
# Inject CSS — same dark navy theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A !important;
    font-family: 'Share Tech Mono', monospace;
}
[data-testid="stHeader"] { background: transparent; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Stat cards row */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
}
.stat-card {
    flex: 1;
    background: #131C2E;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 10px 6px 8px 6px;
    text-align: center;
    position: relative;
}
.stat-bar {
    height: 4px;
    border-radius: 2px;
    margin: 0 auto 8px auto;
    width: 80%;
}
.stat-value {
    font-size: 1.45rem;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.6rem;
    color: #64748B;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-family: 'Share Tech Mono', monospace;
}

/* Grid table */
.grid-wrap {
    background: #0A0E1A;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
}
table.maze {
    border-collapse: collapse;
    margin: 0 auto;
    table-layout: fixed;
}
table.maze td {
    width: 80px;
    height: 80px;
    text-align: center;
    vertical-align: middle;
    font-size: 1.9rem;
    border: 1px solid #1F2D45;
    position: relative;
}
.cell-even { background: #111827; }
.cell-odd  { background: #0F1620; }
.cell-obstacle { background: #2A0A0A; border: 2px solid #EF4444 !important; }
.cell-package  { background: #1A1200; border: 2px solid #F59E0B !important; }
.cell-house    { background: #001A0A; border: 2px solid #22C55E !important; }
.cell-robot    { background: #0A1830; border: 2px solid #3B82F6 !important; }
.cell-robot-pkg{ background: #1A1200; border: 2px solid #F59E0B !important; }

/* Reward bar */
.reward-bar {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    padding: 6px;
    border-radius: 6px;
    margin-bottom: 10px;
}

/* Title */
.title-block {
    text-align: center;
    padding: 18px 0 8px 0;
}
.title-main {
    font-size: 1.7rem;
    font-weight: bold;
    color: #3B82F6;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.12em;
}
.title-sub {
    font-size: 0.8rem;
    color: #64748B;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 2px;
}

/* Status banner */
.banner-delivered {
    background: #001A0A;
    border: 2px solid #22C55E;
    border-radius: 10px;
    text-align: center;
    padding: 12px;
    color: #22C55E;
    font-size: 1.2rem;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 10px;
}
.banner-obstacle {
    background: #2A0A0A;
    border: 2px solid #EF4444;
    border-radius: 10px;
    text-align: center;
    padding: 12px;
    color: #EF4444;
    font-size: 1.2rem;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 10px;
}

/* Start screen */
.start-card {
    background: #131C2E;
    border: 2px solid #3B82F6;
    border-radius: 16px;
    padding: 36px 30px 30px 30px;
    text-align: center;
    max-width: 420px;
    margin: 40px auto 0 auto;
}
.start-title {
    font-size: 2rem;
    font-weight: bold;
    color: #3B82F6;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.start-sub {
    color: #64748B;
    font-size: 0.85rem;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 20px;
}
.start-divider {
    border: none;
    border-top: 1px solid #1E3A5F;
    margin: 16px 0;
}
.info-row {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    padding: 4px 0;
}

/* Streamlit button override */
div.stButton > button {
    background-color: #3B82F6 !important;
    color: #0A0E1A !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: bold !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 32px !important;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background-color: #2563EB !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Q-Learning constants  (UNCHANGED)
# ─────────────────────────────────────────
GRID_SIZE  = 6
PACKAGE    = (4, 1)
HOUSE      = (5, 5)
OBSTACLES  = [(1,3),(2,2),(3,4),(1,1),(4,3)]

alpha   = 0.1
gamma   = 0.9
epsilon = 0.2
actions = [0, 1, 2, 3]

QTABLE_FILE = "delivery_qtable.pkl"

# ─────────────────────────────────────────
# Q-Learning functions  (UNCHANGED)
# ─────────────────────────────────────────
def load_qtable():
    if os.path.exists(QTABLE_FILE):
        with open(QTABLE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_qtable(q_table):
    with open(QTABLE_FILE, "wb") as f:
        pickle.dump(q_table, f)

def choose_action(state, q_table):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    return int(np.argmax(q_table.get(state, [0,0,0,0])))

def update_q(q_table, state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = [0,0,0,0]
    if next_state not in q_table:
        q_table[next_state] = [0,0,0,0]
    predict = q_table[state][action]
    target  = reward + gamma * max(q_table[next_state])
    q_table[state][action] += alpha * (target - predict)
    return q_table

def move_robot(pos, action):
    r, c = pos
    new_r, new_c = r, c
    if action == 0 and r > 0:               new_r -= 1
    elif action == 1 and r < GRID_SIZE - 1: new_r += 1
    elif action == 2 and c > 0:             new_c -= 1
    elif action == 3 and c < GRID_SIZE - 1: new_c += 1
    return (new_r, new_c), (new_r, new_c) != (r, c)

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
def init_state():
    if "started" not in st.session_state:
        st.session_state.started      = False
    if "robot_pos" not in st.session_state:
        st.session_state.robot_pos    = (0, 0)
    if "has_package" not in st.session_state:
        st.session_state.has_package  = False
    if "runs" not in st.session_state:
        st.session_state.runs         = 1
    if "deliveries" not in st.session_state:
        st.session_state.deliveries   = 0
    if "best_steps" not in st.session_state:
        st.session_state.best_steps   = 999
    if "current_steps" not in st.session_state:
        st.session_state.current_steps= 0
    if "last_reward" not in st.session_state:
        st.session_state.last_reward  = 0
    if "q_table" not in st.session_state:
        st.session_state.q_table      = load_qtable()
    if "status" not in st.session_state:
        st.session_state.status       = "running"   # running | delivered | obstacle
    if "auto_run" not in st.session_state:
        st.session_state.auto_run     = True

init_state()

# ─────────────────────────────────────────
# Grid renderer
# ─────────────────────────────────────────
CELL_EMOJI = {
    "robot":        "🤖",
    "robot_pkg":    "🤖",
    "package":      "📦",
    "house":        "🏠",
    "obstacle":     "🚧",
    "empty_even":   "",
    "empty_odd":    "",
}

def render_grid():
    rp   = st.session_state.robot_pos
    has  = st.session_state.has_package

    rows = []
    for r in range(GRID_SIZE):
        cells = []
        for c in range(GRID_SIZE):
            pos = (r, c)
            is_robot = (pos == rp)

            if is_robot and has:
                css, emoji = "cell-robot-pkg", "🤖📦"
            elif is_robot:
                css, emoji = "cell-robot", "🤖"
            elif pos in OBSTACLES:
                css, emoji = "cell-obstacle", "🚧"
            elif pos == PACKAGE and not has:
                css, emoji = "cell-package", "📦"
            elif pos == HOUSE:
                css, emoji = "cell-house", "🏠"
            else:
                css   = "cell-even" if (r+c)%2==0 else "cell-odd"
                emoji = str(c) if r == GRID_SIZE-1 else ""
                if emoji:
                    emoji = f'<span style="font-size:0.55rem;color:#64748B">{emoji}</span>'

            cells.append(f'<td class="{css}">{emoji}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    html = (
        '<div class="grid-wrap">'
        '<table class="maze">'
        + "".join(rows) +
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Stat cards
# ─────────────────────────────────────────
def render_stats():
    best = "-" if st.session_state.best_steps == 999 else str(st.session_state.best_steps)
    cards = [
        ("#3B82F6", "RUN",        str(st.session_state.runs)),
        ("#22C55E", "DELIVERIES", str(st.session_state.deliveries)),
        ("#F59E0B", "BEST ROUTE", best),
        ("#F1F5F9", "STEPS",      str(st.session_state.current_steps)),
    ]
    html = '<div class="stat-row">'
    for color, label, value in cards:
        html += f"""
        <div class="stat-card">
            <div class="stat-bar" style="background:{color}"></div>
            <div class="stat-value" style="color:{color}">{value}</div>
            <div class="stat-label">{label}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Reward indicator
# ─────────────────────────────────────────
def render_reward():
    rw = st.session_state.last_reward
    if rw > 0:
        color, icon = "#22C55E", "✦"
    elif rw < -10:
        color, icon = "#EF4444", "✖"
    else:
        color, icon = "#64748B", "·"
    st.markdown(
        f'<div class="reward-bar" style="color:{color};background:#0D1220;">'
        f'{icon} &nbsp; Last Reward: &nbsp; <b>{rw:+}</b> &nbsp; {icon}</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────
# One Q-learning step  (UNCHANGED logic)
# ─────────────────────────────────────────
def step():
    s   = st.session_state
    qt  = s.q_table

    state  = (s.robot_pos, s.has_package)
    action = choose_action(state, qt)

    new_pos, actually_moved = move_robot(s.robot_pos, action)
    s.robot_pos  = new_pos
    next_state   = (s.robot_pos, s.has_package)
    reward       = -1

    # Obstacle hit
    if s.robot_pos in OBSTACLES:
        reward           = -50
        s.last_reward    = reward
        s.q_table        = update_q(qt, state, action, reward, next_state)
        save_qtable(s.q_table)
        s.runs          += 1
        s.status         = "obstacle"
        s.robot_pos      = (0, 0)
        s.has_package    = False
        s.current_steps  = 0
        return

    if actually_moved:
        s.current_steps += 1

    # Package pickup
    if s.robot_pos == PACKAGE and not s.has_package:
        reward        = 30
        s.has_package = True

    # Delivery
    elif s.robot_pos == HOUSE and s.has_package:
        reward = 100

    s.q_table     = update_q(qt, state, action, reward, next_state)
    s.last_reward = reward

    if reward == 100:
        s.deliveries += 1
        if s.current_steps < s.best_steps:
            s.best_steps = s.current_steps
        save_qtable(s.q_table)
        s.runs      += 1
        s.status     = "delivered"
        s.robot_pos  = (0, 0)
        s.has_package= False
        s.current_steps = 0
    else:
        s.status = "running"

# ─────────────────────────────────────────
# START SCREEN
# ─────────────────────────────────────────
if not st.session_state.started:
    st.markdown("""
    <div class="title-block">
        <div class="title-main">DELIVERY ROBOT</div>
        <div class="title-sub">Q-Learning Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="start-card">
        <div class="start-title">🤖</div>
        <div class="start-title">DELIVERY ROBOT</div>
        <div class="start-sub">Q-Learning Agent</div>
        <hr class="start-divider">
        <div class="info-row">
            <span style="color:#64748B">Grid Size</span>
            <span style="color:#F1F5F9">{GRID_SIZE} × {GRID_SIZE}</span>
        </div>
        <div class="info-row">
            <span style="color:#64748B">Obstacles</span>
            <span style="color:#EF4444">{len(OBSTACLES)}</span>
        </div>
        <div class="info-row">
            <span style="color:#64748B">Objective</span>
            <span style="color:#22C55E">Pick up &amp; Deliver</span>
        </div>
        <hr class="start-divider">
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶   START TRAINING"):
            st.session_state.started = True
            st.rerun()

    st.stop()

# ─────────────────────────────────────────
# MAIN GAME SCREEN
# ─────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <div class="title-main">DELIVERY ROBOT</div>
    <div class="title-sub">Q-Learning Agent</div>
</div>
""", unsafe_allow_html=True)

render_stats()

# Status banners
if st.session_state.status == "delivered":
    st.markdown(
        f'<div class="banner-delivered">✓ &nbsp; DELIVERED! &nbsp; — &nbsp; '
        f'Total: {st.session_state.deliveries}</div>',
        unsafe_allow_html=True
    )
elif st.session_state.status == "obstacle":
    st.markdown(
        '<div class="banner-obstacle">✖ &nbsp; HIT AN OBSTACLE! &nbsp; Resetting...</div>',
        unsafe_allow_html=True
    )

render_grid()
render_reward()

# ─────────────────────────────────────────
# Speed slider
# ─────────────────────────────────────────
speed = st.slider(
    "⚡ Speed",
    min_value=1, max_value=10, value=4,
    help="1 = slowest  |  10 = fastest"
)
# Map slider 1-10 → sleep seconds 1.2 → 0.05
delay = round(1.2 - (speed - 1) * (1.15 / 9), 3)

# ─────────────────────────────────────────
# Controls row
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    step_btn = st.button("⏭ Step Once")
with col2:
    auto_label = "⏸ Pause" if st.session_state.auto_run else "▶ Resume"
    if st.button(auto_label):
        st.session_state.auto_run = not st.session_state.auto_run
        st.rerun()
with col3:
    if st.button("🔄 Reset All"):
        for key in ["robot_pos","has_package","runs","deliveries",
                    "best_steps","current_steps","last_reward","status","auto_run"]:
            del st.session_state[key]
        st.rerun()

# Manual step
if step_btn:
    step()
    st.rerun()

# Auto-run — 1 step per rerun, speed controlled by slider
if st.session_state.auto_run:
    step()
    time.sleep(delay)
    st.rerun()
