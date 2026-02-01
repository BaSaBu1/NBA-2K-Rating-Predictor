"""
NBA 2K Rating Predictor - Streamlit App v7
Predicts player attributes and badges from real NBA stats.
Trained on 1,253 NBA players with Avg MAE 2.32
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Config ---
st.set_page_config(
    layout="wide", 
    page_title="NBA 2K Rating Predictor",
    page_icon="🏀"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 8px;
        background-color: #1A1C23;
        border: 1px solid #333;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 22px; font-weight: 800; margin: 0; }
    .metric-label {
        font-size: 11px; color: #888; text-transform: uppercase;
        letter-spacing: 0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    h3 { font-size: 18px !important; color: #E1E1E1 !important; margin-bottom: 5px !important; }
    h4 { font-size: 14px !important; color: #ff4b4b !important; text-transform: uppercase; margin-top: 5px !important; border-bottom: 1px solid #333; }
    
    /* Badge styles */
    .badge-hof { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
    .badge-gold { background: linear-gradient(135deg, #DAA520, #B8860B); color: #000; }
    .badge-silver { background: linear-gradient(135deg, #C0C0C0, #A9A9A9); color: #000; }
    .badge-bronze { background: linear-gradient(135deg, #CD7F32, #8B4513); color: #fff; }
    .badge-item {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 15px;
        font-size: 13px;
        font-weight: 600;
    }
    
    .support-box {
        background: linear-gradient(135deg, #1a1c23, #2d2d3d);
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .support-box a {
        color: #ff4b4b !important;
        text-decoration: none;
        font-weight: 600;
    }
    
    .stNumberInput input { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models/v7"

# Attribute groups for display
GROUPS = {
    "Outside Scoring": ['Close Shot', 'Mid-Range Shot', 'Three-Point Shot', 'Free Throw', 'Shot IQ', 'Offensive Consistency'],
    "Inside Scoring": ['Layup', 'Standing Dunk', 'Driving Dunk', 'Post Hook', 'Post Fade', 'Post Control', 'Draw Foul', 'Hands'],
    "Playmaking": ['Pass Accuracy', 'Ball Handle', 'Speed with Ball', 'Pass IQ', 'Pass Vision'],
    "Defense": ['Interior Defense', 'Perimeter Defense', 'Steal', 'Block', 'Help Defense IQ', 'Pass Perception', 'Defensive Consistency'],
    "Athleticism": ['Speed', 'Agility', 'Strength', 'Vertical', 'Stamina', 'Hustle', 'Overall Durability'],
    "Rebounding": ['Offensive Rebound', 'Defensive Rebound']
}

# All attributes
ALL_ATTRIBUTES = [attr for group in GROUPS.values() for attr in group]

# All badges
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

BADGE_MAP = {0: None, 1: 'Bronze', 2: 'Silver', 3: 'Gold', 4: 'HOF'}

@st.cache_resource
def load_models():
    """Load all models, scaler, and feature list."""
    try:
        scaler = joblib.load(f'{MODELS_DIR}/scaler.pkl')
        features = joblib.load(f'{MODELS_DIR}/features.pkl')
    except:
        scaler = None
        features = []
    
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
    """Build feature vector matching training features."""
    f = {}
    eps = 0.001
    
    # Basic stats from user
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
    
    # Basic columns
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
    f['USG_PCT'] = 0.2  # Default estimate
    f['PACE'] = 100  # Default estimate
    
    # Position encoding
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'G-F', 'F-C', 'F-G', 'C-F']
    for p in positions:
        f[f'POS_{p}'] = 1 if p in pos else 0
    
    f['IS_GUARD'] = 1 if 'G' in pos else 0
    f['IS_FORWARD'] = 1 if 'F' in pos else 0
    f['IS_CENTER'] = 1 if pos == 'C' else 0
    
    # Era (assume modern)
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
    
    # Build feature vector in correct order
    feature_values = []
    for feat in feature_list:
        feature_values.append(f.get(feat, 0))
    
    return feature_values

def predict_all(user_inputs, scaler, features, attr_models, badge_models):
    """Predict all attributes and badges."""
    # Build features
    feature_values = build_features(user_inputs, features)
    
    # Scale
    if scaler is not None:
        X = scaler.transform([feature_values])
    else:
        X = [feature_values]
    
    X_df = pd.DataFrame(X, columns=features)
    
    # Predict attributes
    attr_preds = {}
    for attr, model in attr_models.items():
        try:
            pred = model.predict(X_df)[0]
            pred = int(round(np.clip(pred, 25, 99)))
            attr_preds[attr] = pred
        except:
            attr_preds[attr] = 70
    
    # Predict badges
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

def render_metric_card(label, value):
    """Render a styled metric card."""
    color = "#bbb"
    if isinstance(value, (int, float)):
        if value >= 90: color = "#2ecc71"
        elif value >= 80: color = "#f1c40f"
        elif value >= 70: color = "#bbb"
        else: color = "#e74c3c"
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-label" title="{label}">{label}</div>
        </div>
    """, unsafe_allow_html=True)

def render_badges(badge_preds):
    """Render badges sorted by tier (HOF first)."""
    tier_order = {'HOF': 0, 'Gold': 1, 'Silver': 2, 'Bronze': 3}
    tier_class = {'HOF': 'badge-hof', 'Gold': 'badge-gold', 'Silver': 'badge-silver', 'Bronze': 'badge-bronze'}
    
    sorted_badges = sorted(badge_preds.items(), key=lambda x: tier_order.get(x[1], 99))
    
    if not sorted_badges:
        st.caption("No badges predicted")
        return
    
    html = ""
    for badge, tier in sorted_badges:
        css_class = tier_class.get(tier, '')
        html += f'<span class="badge-item {css_class}">{badge}</span>'
    
    st.markdown(html, unsafe_allow_html=True)

def main():
    st.title("🏀 NBA 2K Rating Predictor")
    st.caption("Predict player attributes and badges from real NBA stats • Trained on 1,253 NBA players")
    
    # Load models
    scaler, features, attr_models, badge_models = load_models()
    
    if not attr_models:
        st.error("❌ Models not found. Please run trainer_v7.py first.")
        return
    
    # --- Input Section ---
    with st.expander("📊 Enter Player Stats", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Physical**")
            height = st.number_input("Height (cm)", 165, 230, 198, help="Player height in centimeters")
            weight = st.number_input("Weight (kg)", 60, 150, 98, help="Player weight in kilograms")
            wingspan = st.number_input("Wingspan (cm)", 170, 250, 208, help="Player wingspan in centimeters")
            position = st.selectbox("Position", ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'G-F', 'F-C'])
        
        with col2:
            st.markdown("**Season Totals**")
            gp = st.number_input("Games Played", 1, 82, 65)
            mins = st.number_input("Minutes", 0.0, 3500.0, 1800.0)
            pts = st.number_input("Points", 0.0, 3000.0, 850.0)
            ast = st.number_input("Assists", 0.0, 1000.0, 180.0)
            tov = st.number_input("Turnovers", 0.0, 400.0, 95.0)
        
        with col3:
            st.markdown("**Shooting**")
            fgm = st.number_input("FG Made", 0.0, 1200.0, 320.0)
            fga = st.number_input("FG Attempted", 0.0, 2000.0, 680.0)
            fg3m = st.number_input("3PT Made", 0.0, 400.0, 85.0)
            fg3a = st.number_input("3PT Attempted", 0.0, 800.0, 230.0)
            ftm = st.number_input("FT Made", 0.0, 800.0, 125.0)
            fta = st.number_input("FT Attempted", 0.0, 1000.0, 165.0)
        
        with col4:
            st.markdown("**Rebounds & Defense**")
            oreb = st.number_input("Off Rebounds", 0.0, 400.0, 45.0)
            dreb = st.number_input("Def Rebounds", 0.0, 1000.0, 215.0)
            stl = st.number_input("Steals", 0.0, 200.0, 55.0)
            blk = st.number_input("Blocks", 0.0, 300.0, 35.0)
            pf = st.number_input("Personal Fouls", 0.0, 300.0, 130.0)
    
    # Build user inputs dict
    user_inputs = {
        'height': height, 'weight': weight, 'wingspan': wingspan,
        'position': position, 'gp': gp, 'min': mins, 'pts': pts,
        'fgm': fgm, 'fga': fga, 'fg3m': fg3m, 'fg3a': fg3a,
        'ftm': ftm, 'fta': fta, 'oreb': oreb, 'dreb': dreb,
        'ast': ast, 'tov': tov, 'stl': stl, 'blk': blk, 'pf': pf
    }
    
    # Predict button
    if st.button("🎮 Predict Ratings", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            attr_preds, badge_preds = predict_all(user_inputs, scaler, features, attr_models, badge_models)
        
        # --- Display Results ---
        st.markdown("---")
        
        # Overall rating estimate
        overall = int(np.mean(list(attr_preds.values())))
        st.markdown(f"### Estimated Overall: **{overall}**")
        
        # Attribute groups
        for group_name, attrs in GROUPS.items():
            st.markdown(f"#### {group_name}")
            cols = st.columns(len(attrs))
            for i, attr in enumerate(attrs):
                with cols[i]:
                    val = attr_preds.get(attr, 70)
                    render_metric_card(attr, val)
        
        # Badges
        st.markdown("---")
        st.markdown("### 🏅 Predicted Badges")
        
        if badge_preds:
            render_badges(badge_preds)
            st.caption(f"Total: {len(badge_preds)} badges")
        else:
            st.caption("No badges predicted for this player")
    
    # --- Support Section ---
    st.markdown("---")
    st.markdown("""
    <div class="support-box">
        <h4 style="margin-top: 0; color: #E1E1E1;">☕ Support This Project</h4>
        <p style="color: #aaa; font-size: 14px; margin-bottom: 10px;">
            Love this tool? Your support helps me keep improving it!<br>
            <strong>If this project gets enough support</strong>, I'll add more features like:
        </p>
        <ul style="color: #888; font-size: 13px; text-align: left; max-width: 400px; margin: 0 auto 15px;">
            <li>Player comparison mode</li>
            <li>Team building recommendations</li>
            <li>MyCareer build optimizer</li>
            <li>More accurate badge predictions</li>
        </ul>
        <a href="https://buymeacoffee.com/yourusername" target="_blank" 
           style="background: #FFDD00; color: #000; padding: 10px 25px; border-radius: 5px; 
                  text-decoration: none; font-weight: bold; display: inline-block;">
            ☕ Buy Me a Coffee
        </a>
        <p style="color: #666; font-size: 11px; margin-top: 10px;">
            Built with ❤️ using 1,253 real NBA player stats
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("v7.0 • Avg MAE: 2.32 • Trained on NBA players from multiple eras")

if __name__ == "__main__":
    main()

