import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NBA 2K26 Rating Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Cool/Compact" Dark Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Support Box - High Visibility Gradient */
    .support-box {
        background: linear-gradient(135deg, #2E8651, #1F6A3D); /* Celtics Green Gradient */
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(46, 134, 81, 0.4);
        border: 1px solid #4CAF50;
    }
    .support-box h3 {
        color: #FFD700 !important; /* Gold text */
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 24px;
    }
    .support-box p {
        font-size: 16px;
        margin-bottom: 15px;
    }
    .support-box a {
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        background-color: #000000;
        padding: 12px 25px;
        border-radius: 25px;
        display: inline-block;
        margin-top: 10px;
        border: 2px solid #FFD700;
        transition: all 0.3s ease;
    }
    .support-box a:hover {
        background-color: #FFD700;
        color: black !important;
        transform: scale(1.05);
    }

    /* Metric Cards */
    .metric-card {
        background-color: #262730;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .metric-card:hover {
        border-color: #4CAF50;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 13px;
        color: #aaa;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Badge Tags */
    .badge-item {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    .badge-hof { background: linear-gradient(135deg, #E040FB, #AA00FF); color: white; }
    .badge-gold { background: linear-gradient(135deg, #FFD700, #FDB931); }
    .badge-silver { background: linear-gradient(135deg, #E0E0E0, #BDBDBD); }
    .badge-bronze { background: linear-gradient(135deg, #CD7F32, #A0522D); color: white; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; 
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1E1E1E;
        border-radius: 5px;
        color: #FFF;
        flex-grow: 1; /* Make tabs standard width */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC & MODELS (From Original App)
# -----------------------------------------------------------------------------

# Constants
BADGE_MAP = {0: None, 1: 'Bronze', 2: 'Silver', 3: 'Gold', 4: 'HOF'}
ALL_BADGES = [
    'Ankle Assassin', 'Bail Out', 'Dimer', 'Float Game', 'Glove',
    'Handles For Days', 'Immovable Enforcer', 'Interceptor', 'Layup Mixmaster',
    'Mini Marksman', 'On-Ball Menace', 'Pick Dodger', 'Versatile Visionary',
    'Challenger', 'Deadeye', 'Lightning Launch', 'Set Shot Specialist',
    'Shifty Shooter', 'Strong Handle', 'Unpluckable', 'Break Starter',
    'Limitless Range', 'Off-Ball Pest', 'Brick Wall', 'Boxout Beast',
    'Slippery Off-Ball', 'Rise Up', 'Aerial Wizard', 'High-Flying Denier',
    'Paint Patroller', 'Pogo Stick', 'Post Lockdown', 'Rebound Chaser',
    'Physical Finisher', 'Posterizer', 'Post Fade Phenom', 'Post-Up Poet',
    'Post Powerhouse', 'Hook Specialist', 'Paint Prodigy'
]

# Attribute Organization for UI
ATTRIBUTES = {
    "Scoring": ['Close Shot', 'Mid-Range Shot', 'Three-Point Shot', 'Free Throw', 'Shot IQ', 'Offensive Consistency', 'Layup', 'Standing Dunk', 'Driving Dunk', 'Post Hook', 'Post Fade', 'Post Control', 'Draw Foul'],
    "Playmaking": ['Pass Accuracy', 'Ball Handle', 'Speed with Ball', 'Pass IQ', 'Pass Vision', 'Hands'],
    "Defense": ['Interior Defense', 'Perimeter Defense', 'Steal', 'Block', 'Help Defense IQ', 'Pass Perception', 'Defensive Consistency'],
    "Athleticism": ['Speed', 'Agility', 'Strength', 'Vertical', 'Stamina', 'Hustle', 'Overall Durability'],
    "Rebounding": ['Offensive Rebound', 'Defensive Rebound']
}
ALL_ATTRIBUTES = [attr for group in ATTRIBUTES.values() for attr in group]

@st.cache_resource
def load_models():
    """Load all models, scaler, and feature list."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try finding models in local 'models/v7' (Deployment) or parent '../models/v7' (Dev)
    local_path = os.path.join(base_dir, 'models', 'v7')
    parent_path = os.path.join(base_dir, '..', 'models', 'v7')
    
    if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, 'scaler.pkl')):
        MODELS_DIR = local_path
    elif os.path.exists(parent_path):
        MODELS_DIR = parent_path
    else:
        MODELS_DIR = local_path # Default to local for error message clarity if both fail

    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))
    except FileNotFoundError:
        st.error(f"⚠️ Critical files missing! Checked locations:\\n1. {local_path}\\n2. {parent_path}")
        return None, [], {}, {}
    
    attr_models = {}
    for attr in ALL_ATTRIBUTES:
        fname = f"attr_{attr.replace(' ', '_')}.pkl"
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            attr_models[attr] = joblib.load(path)
    
    badge_models = {}
    for badge in ALL_BADGES:
        fname = f"badge_{badge.replace(' ', '_').replace('-', '_')}.pkl"
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            badge_models[badge] = joblib.load(path)
    
    return scaler, features, attr_models, badge_models

def build_features(user_inputs, feature_list):
    """Build feature vector matching training features (CRITICAL for accuracy)."""
    f = {}
    eps = 0.001
    
    # Unpack
    height = user_inputs['height']
    weight = user_inputs['weight']
    wingspan = user_inputs['wingspan']
    gp = user_inputs['gp']
    mins = user_inputs['min']
    pos = user_inputs['position']
    pts = user_inputs['pts']
    fgm = user_inputs['fgm']
    fga = user_inputs['fga']
    fg3m = user_inputs['fg3m']
    fg3a = user_inputs['fg3a']
    ftm = user_inputs['ftm']
    fta = user_inputs['fta']
    oreb = user_inputs['oreb']
    dreb = user_inputs['dreb']
    reb = oreb + dreb
    ast = user_inputs['ast']
    tov = user_inputs['tov']
    stl = user_inputs['stl']
    blk = user_inputs['blk']
    pf = user_inputs['pf']
    
    # Basic Feature Mapping
    f['Height_CM'] = height
    f['Weight_KG'] = weight
    f['Wingspan_CM'] = wingspan
    f['GP'] = gp
    f['MIN'] = mins
    f['PTS'] = pts
    f['FGM'] = fgm
    f['FGA'] = fga
    f['FG_PCT'] = fgm / (fga + eps) if fga > 0 else 0
    f['FG3M'] = fg3m
    f['FG3A'] = fg3a
    f['FG3_PCT'] = fg3m / (fg3a + eps) if fg3a > 0 else 0
    f['FTM'] = ftm
    f['FTA'] = fta
    f['FT_PCT'] = ftm / (fta + eps) if fta > 0 else 0
    f['OREB'] = oreb
    f['DREB'] = dreb
    f['REB'] = reb
    f['AST'] = ast
    f['TOV'] = tov
    f['STL'] = stl
    f['BLK'] = blk
    f['PF'] = pf
    f['USG_PCT'] = 0.2  # Default
    f['PACE'] = 100  # Default
    
    # Position encoding
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'G-F', 'F-C', 'F-G', 'C-F']
    for p in positions:
        f[f'POS_{p}'] = 1 if p in pos else 0
    
    f['IS_GUARD'] = 1 if 'G' in pos else 0
    f['IS_FORWARD'] = 1 if 'F' in pos else 0
    f['IS_CENTER'] = 1 if pos == 'C' else 0
    
    # Era
    f['IS_MODERN'] = 1
    f['IS_CLASSIC'] = 0
    f['ERA_FACTOR'] = 1.0
    
    # Per-36 stats
    min_ratio = 36.0 / max(mins, 1)
    for stat, val in [('PTS', pts), ('REB', reb), ('AST', ast), ('STL', stl), 
                       ('BLK', blk), ('TOV', tov), ('FGM', fgm), ('FGA', fga),
                       ('FG3M', fg3m), ('FG3A', fg3a), ('FTM', ftm), ('FTA', fta),
                       ('OREB', oreb), ('DREB', dreb), ('PF', pf)]:
        f[f'{stat}_p36'] = min(val * min_ratio, 50)
    
    # Efficiency metrics
    f['TS_PCT'] = pts / (2 * (fga + 0.44 * fta + eps))
    f['EFG_PCT'] = (fgm + 0.5 * fg3m) / (fga + eps)
    f['AST_TO'] = ast / (tov + eps)
    f['PTS_per_FGA'] = pts / (fga + eps)
    
    # Shot profile
    total_fga = fga + eps
    f['FG3_RATE'] = fg3a / total_fga
    f['FT_RATE'] = fta / total_fga
    f['MID_RATE'] = 1 - f['FG3_RATE']
    
    # Rebounding
    f['OREB_RATE'] = oreb / (reb + eps)
    
    # Defensive
    f['STOCKS_p36'] = f.get('STL_p36', 0) + f.get('BLK_p36', 0)
    f['STL_BLK_RATIO'] = stl / (blk + eps)
    
    # Physical
    f['BMI'] = weight / ((height/100)**2 + eps)
    f['WINGSPAN_RATIO'] = wingspan / (height + eps)
    f['SIZE_SCORE'] = (height - 170) / 35 + (weight - 70) / 50
    
    # Role indicators
    f['SCORER'] = 1 if f.get('PTS_p36', 0) > 15 else 0
    f['PLAYMAKER'] = 1 if f.get('AST_p36', 0) > 5 else 0
    f['RIM_PROTECTOR'] = 1 if f.get('BLK_p36', 0) > 1.5 else 0
    f['REBOUNDER'] = 1 if f.get('REB_p36', 0) > 8 else 0
    
    # Usage
    f['SHOT_VOLUME'] = f.get('FGA_p36', 0)
    f['USAGE_PROXY'] = (fga + 0.44*fta + tov) / (mins + eps)
    f['EXPERIENCE'] = gp / 82.0
    
    # Build list
    return [f.get(feat, 0) for feat in feature_list]

def predict_all(user_inputs, scaler, features, attr_models, badge_models):
    """Predict all attributes and badges."""
    feature_values = build_features(user_inputs, features)
    
    if scaler is not None:
        X = scaler.transform([feature_values])
    else:
        X = [feature_values]
        
    X_df = pd.DataFrame(X, columns=features)
    
    attr_preds = {}
    for attr, model in attr_models.items():
        try:
            pred = model.predict(X_df)[0]
            attr_preds[attr] =int(round(np.clip(pred, 25, 99)))
        except:
            pass
            
    badge_preds = {}
    for badge, model in badge_models.items():
        try:
            pred = model.predict(X_df)[0]
            tier_idx = int(round(np.clip(pred, 0, 4)))
            if tier_idx > 0:
                badge_preds[badge] = BADGE_MAP[tier_idx]
        except:
            pass
            
    return attr_preds, badge_preds

# -----------------------------------------------------------------------------
# 3. SIDEBAR & INPUTS (Layout Reverted - No Collapsing)
# -----------------------------------------------------------------------------
scaler, features, attr_models, badge_models = load_models()

with st.sidebar:
    st.header("🏀 Player Stats Input")
    
    # Physical
    st.markdown("### 🧬 Physical")
    height = st.number_input("Height (cm)", 165, 230, 198)
    weight = st.number_input("Weight (kg)", 60, 150, 98)
    wingspan = st.number_input("Wingspan (cm)", 170, 250, 208)
    position = st.selectbox("Position", ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'G-F', 'F-C'])

    # Season
    st.markdown("### 📅 Season")
    gp = st.number_input("Games Played", 1, 82, 65)
    mins = st.number_input("Minutes", 0.0, 48.0, 32.0, step=0.5)

    # Scoring
    st.markdown("### 🎯 Scoring")
    pts = st.number_input("Points", 0.0, 60.0, 20.0, step=0.1)
    fgm = st.number_input("FG Made", 0.0, 25.0, 8.0, step=0.1)
    fga = st.number_input("FG Attempted", 0.0, 50.0, 18.0, step=0.1)
    fg3m = st.number_input("3PM", 0.0, 15.0, 2.5, step=0.1)
    fg3a = st.number_input("3PA", 0.0, 30.0, 7.0, step=0.1)
    ftm = st.number_input("FT Made", 0.0, 25.0, 4.0, step=0.1)
    fta = st.number_input("FT Attempted", 0.0, 30.0, 5.0, step=0.1)

    # Playmaking & Defense
    st.markdown("### ⛹️ Playmaking & Def")
    ast = st.number_input("Assists", 0.0, 20.0, 5.0, step=0.1)
    tov = st.number_input("Turnovers", 0.0, 10.0, 2.0, step=0.1)
    oreb = st.number_input("Off Reb", 0.0, 10.0, 1.0, step=0.1)
    dreb = st.number_input("Def Reb", 0.0, 20.0, 4.0, step=0.1)
    stl = st.number_input("Steals", 0.0, 10.0, 1.2, step=0.1)
    blk = st.number_input("Blocks", 0.0, 10.0, 0.8, step=0.1)
    pf = st.number_input("Fouls", 0.0, 6.0, 2.5, step=0.1)

    predict_btn = st.button("🚀 Predict Ratings", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. MAIN CONTENT
# -----------------------------------------------------------------------------
st.title("🏀 NBA 2K26 Rating Predictor")
st.markdown("### Attribute & Badge Projections")

if predict_btn and attr_models:
    # 1. Prepare Inputs
    user_inputs = {
        'height': height, 'weight': weight, 'wingspan': wingspan,
        'position': position, 'gp': gp, 'min': mins, 'pts': pts,
        'fgm': fgm, 'fga': fga, 'fg3m': fg3m, 'fg3a': fg3a,
        'ftm': ftm, 'fta': fta, 'oreb': oreb, 'dreb': dreb,
        'ast': ast, 'tov': tov, 'stl': stl, 'blk': blk, 'pf': pf
    }
    
    # 2. Run Prediction with full feature engineering
    attr_preds, badge_preds = predict_all(user_inputs, scaler, features, attr_models, badge_models)
    
    # 3. Display Results
    
    # --- DISCLAIMER (Restored Text) ---
    st.warning("⚠️ **Note:** These predictions are approximations based on basic stats. Some attributes (especially, defensive and athletic) are difficult to capture from box scores alone. Use these as a starting point and adjust based on your own judgment.")

    # --- TABS FOR ATTRIBUTES ---
    tabs = st.tabs(list(ATTRIBUTES.keys()))
    
    for i, (category, attrs) in enumerate(ATTRIBUTES.items()):
        with tabs[i]:
            cols = st.columns(4)
            for j, attr in enumerate(attrs):
                val = attr_preds.get(attr, 70)
                
                # Color logic
                color = "#aaa"
                if val >= 90: color = "#2ECC71" # Pink
                elif val >= 80: color = "#3498DB" # Blue
                elif val >= 70: color = "#F1C40F" # Gold
                
                with cols[j % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{attr}</div>
                        <div class="metric-value" style="color: {color}">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # --- BADGES SECTION ---
    st.markdown("---")
    st.markdown("### 🏅 Predicted Badges")
    
    if badge_preds:
         # Sort badges: HOF -> Gold -> Silver -> Bronze
        tier_order = {'HOF': 0, 'Gold': 1, 'Silver': 2, 'Bronze': 3}
        tier_colors = {'HOF': 'badge-hof', 'Gold': 'badge-gold', 'Silver': 'badge-silver', 'Bronze': 'badge-bronze'}
        
        sorted_badges = sorted(badge_preds.items(), key=lambda x: tier_order.get(x[1], 99))
        
        badge_html = ""
        for badge, tier in sorted_badges:
            badge_html += f'<span class="badge-item {tier_colors[tier]}">{badge} <small>({tier})</small></span>'
        
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.info("No badges predicted for this statline.")

elif not attr_models:
    st.error("⚠️ Models not found! Please ensure 'models/v7' contains the valid model files.")

else:
    st.markdown("""
    Welcome! Enter player stats in the **sidebar** to generate projected NBA 2K attributes.
    This tool uses Machine Learning trained on real NBA stats and NBA 2K26 player attributes.
    """)

# -----------------------------------------------------------------------------
# 5. SUPPORT (Bottom of Main Page)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="support-box">
    <h3>☕ Support the Project</h3>
    <p>If this tool helps you with your roster edits or content creation, consider buying me a coffee! It helps keep me motivated and models updating.</p>
    <p><strong>Note:</strong> Future updates will improve accuracy, and include hot zones, tendencies, etc... using advanced nba stats to provide even better prediction!</p>
    <a href="https://buymeacoffee.com/basabu" target="_blank">☕ Buy me a Coffee</a>
</div>
""", unsafe_allow_html=True)

