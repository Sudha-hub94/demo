import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 1. Page Configuration
st.set_page_config(page_title="The Illusion of Data", layout="wide")

# Modern CSS Injection
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 { color: #1e293b; font-family: 'Inter', sans-serif; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# 2. Hero Section
st.title("📊 The Anscombe Dashboard")
st.write("### Exploring the paradox where identical stats lead to different visuals.")

# 3. Data Loading
@st.cache_data
def load_data():
    try:
        # Tries to load your excel file
        df = pd.read_excel('anscombes blog.xlsx').iloc[:11]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except:
        # Fallback data (Anscombe's Quartet)
        data = {
            'x1':[10,8,13,9,11,14,6,4,12,7,5], 'y1':[8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
            'x2':[10,8,13,9,11,14,6,4,12,7,5], 'y2':[9.14,8.14,8.74,8.77,9.26,8.1,6.13,3.1,9.13,7.26,4.74],
            'x3':[10,8,13,9,11,14,6,4,12,7,5], 'y3':[7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73],
            'x4':[8,8,8,8,8,8,8,19,8,8,8], 'y4':[6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.5,5.56,7.91,6.89]
        }
        return pd.DataFrame(data)

df = load_data()

# 4. Sidebar Navigation
with st.sidebar:
    st.header("Settings")
    dataset = st.selectbox("Select Perspective", ["Dataset I", "Dataset II", "Dataset III", "Dataset IV"])
    st.markdown("---")
    st.caption("Developed for Data Storytelling Blog")

# Mapping Data to Columns
mapping = {
    "Dataset I": (0, 1, '#3b82f6', "A Standard Linear Trend"),
    "Dataset II": (2, 3, '#10b981', "A Hidden Quadratic Curve"),
    "Dataset III": (4, 5, '#f59e0b', "The Influence of an Outlier"),
    "Dataset IV": (6, 7, '#ef4444', "A Single Lever Point")
}
x_idx, y_idx, theme_color, description = mapping[dataset]
x, y = df.iloc[:, x_idx], df.iloc[:, y_idx]

# 5. Dashboard Layout
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Statistical Calculation")
    slope, intercept, r_val, p, std = stats.linregress(x, y)
    
    st.metric("Correlation (r)", f"{r_val:.3f}")
    st.metric("Avg of Y", f"{y.mean():.2f}")
    st.metric("Variance", f"{x.var():.1f}")
    
    st.write("**Mathematical Model:**")
    st.latex(f"y = {intercept:.2f} + {slope:.2f}x")

with col_right:
    st.subheader(description)
    
    # Sophisticated Visualization
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # FIXED: Changed 'linewidth' to 'linewidths' in scatter_kws to fix the TypeError
    sns.regplot(
        x=x, y=y, ax=ax, ci=None,
        scatter_kws={
            's':200, 
            'color':theme_color, 
            'edgecolor':'white', 
            'linewidths':1.5, # This was the fix!
            'alpha':0.9
        },
        line_kws={
            'color':'#334155', 
            'linestyle':'--', 
            'linewidth':2
        }
    )
    
    ax.set_xlim(2, 20); ax.set_ylim(2, 14)
    ax.set_xlabel("X Variable", fontsize=10, fontweight='bold')
    ax.set_ylabel("Y Variable", fontsize=10, fontweight='bold')
    
    st.pyplot(fig)

st.divider()
st.markdown(f"**Conclusion:** In **{dataset}**, the statistics suggest a simple linear path. However, visualization reveals that the truth is actually **{description.lower()}**.")

