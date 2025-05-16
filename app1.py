import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Food Delivery Time Prediction", 
    layout="wide",
    page_icon="🛵"
)
st.markdown("""
    <style>
    /* Brighter background gradient */
    .stApp {
        background: linear-gradient(to right, #fffde7, #e0f7fa);
        background-attachment: fixed;
    }

    /* Brighter text and clean font */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        color: #1a1a1a;
    }

    /* Light and vivid card styling */
    .stMarkdown, .stDataFrame, .stSlider, .stSelectbox, .stButton, .stForm {
        background-color: #ffffffee;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0px 6px 16px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)


# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Food_Delivery_Times.csv")
        df = df.drop('Order_ID', axis=1)
        df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop()

# Define pipelines
try:
    continuous = ['Distance_km', 'Preparation_Time_min']
    nominal = ['Weather', 'Time_of_Day', 'Vehicle_Type']
    ordinal = ['Traffic_Level']
    orders = [['Low', 'Medium', 'High']]

    p1 = Pipeline([
        ('scaling', MinMaxScaler()),
        ('transform', PowerTransformer())
    ])

    p2 = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    p3 = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(categories=orders))
    ])

    preprocess = ColumnTransformer([
        ('continuous', p1, continuous),
        ('nominal', p2, nominal),
        ('ordinal', p3, ordinal)
    ])

    # Optimized model
    model = Pipeline([
        ('preprocess', preprocess),
        ('regressor', KNeighborsRegressor(
            algorithm='auto', 
            n_neighbors=28, 
            p=2, 
            weights='distance'
        ))
    ])

    # Train model
    X = df.drop('Delivery_Time_min', axis=1)
    y = df['Delivery_Time_min']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=43
    )
    model.fit(x_train, y_train)

except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a page", 
    [
        "Dataset & Model Information", 
        "Factors Affecting Delivery Time", 
        "Predict Delivery Time"
    ]
)

# Page 1: Dataset & Model Information
if page == "Dataset & Model Information":
    st.title("📊 Dataset & Model Information")
    
    st.subheader("About the Dataset")
    st.write("""
    This dataset contains historical information about food delivery orders including:
    - Delivery distance
    - Weather conditions
    - Traffic levels
    - Time of day
    - Vehicle types
    - Preparation times
    - Courier experience
    """)
    
    with st.expander("View Sample Data"):
        st.dataframe(df.head())
    
    with st.expander("Dataset Statistics"):
        st.write(df.describe())
    
    st.subheader("🔧 Model Information")
    st.write("""
    **Algorithm:** K-Nearest Neighbors Regressor  
    **Optimized Parameters:**
    - Neighbors: 28
    - Weighting: Distance-based
    - Distance Metric: Euclidean
    """)
    
    st.write(f"**Model Performance (R² Score):** {r2_score(y_test, model.predict(x_test)):.2f}")
    
    st.subheader("⚙️ Data Processing")
    st.write("""
    1. **Numerical Features:** Scaled and normalized  
    2. **Categorical Features:** One-hot encoded  
    3. **Ordinal Features:** Rank-encoded  
    """)

# Page 2: Factors Affecting Delivery Time
elif page == "Factors Affecting Delivery Time":
    st.title("📈 Factors Affecting Delivery Time")
    st.write("Understand how different parameters influence food delivery times")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Distance")
        st.markdown("""
        - **<2 km:** ⚡ Very fast delivery
        - **2-5 km:** 🚶‍♂️ Average delivery time  
        - **>5 km:** 🐢 Significantly longer
        """, unsafe_allow_html=True)
        
        st.subheader("⛅ Weather Conditions")
        st.markdown("""
        - **☀️ Clear/💨 Windy:** Minimal impact
        - **🌫️ Foggy:** Slight delays  
        - **🌧️ Rainy/❄️ Snowy:** Significant delays
        """, unsafe_allow_html=True)
        
        st.subheader("⏱️ Preparation Time")
        st.markdown("""
        - **<15 min:** ⚡ Delivery starts sooner  
        - **15-30 min:** 🕒 Standard wait  
        - **>30 min:** 🐢 Delivery starts later
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🚦 Traffic Level")
        st.markdown("""
        - **🟢 Low:** Fastest delivery  
        - **🟡 Medium:** Some delays  
        - **🔴 High:** Significant delays
        """, unsafe_allow_html=True)
        
        st.subheader("🌇 Time of Day")
        st.markdown("""
        - **🌅 Morning/🌞 Afternoon:** Normal  
        - **🌆 Evening:** Rush hour delays  
        - **🌃 Night:** Less traffic but slower restaurants
        """, unsafe_allow_html=True)
        
        st.subheader("🚗 Vehicle Type")
        st.markdown("""
        - **🚲 Bike:** Fast in traffic, slow long distances  
        - **🛵 Scooter:** Good balance  
        - **🚗 Car:** Best for long distances
        """, unsafe_allow_html=True)
        
        st.subheader("👨‍🍳 Courier Experience")
        st.markdown("""
        - **🆕 <1 yr:** May take longer  
        - **👨‍💼 1-3 yrs:** Efficient routes  
        - **👴 >3 yrs:** Most efficient
        """, unsafe_allow_html=True)
    
    st.subheader("🚀 Typical Delivery Scenarios")
    st.markdown("""
    - **⚡ Fastest (15-25 min):** Short distance + good weather + low traffic  
    - **⏱️ Average (30-45 min):** Medium distance + normal conditions  
    - **🐢 Slowest (>60 min):** Long distance + bad weather + heavy traffic
    """, unsafe_allow_html=True)
    with st.expander("📊 Detailed Factor Analysis", expanded=True):
        try:
            factor = st.selectbox("Select factor to analyze", 
                                 ["Distance", "Weather", "Time of Day", "Courier Experience"])
            
            if factor == "Distance":
                st.line_chart(df.groupby('Distance_km')['Delivery_Time_min'].mean())
            elif factor == "Weather":
                st.bar_chart(df.groupby('Weather')['Delivery_Time_min'].mean())
            elif factor == "Time of Day":
                st.bar_chart(df.groupby('Time_of_Day')['Delivery_Time_min'].mean())
            elif factor == "Courier Experience":
                st.line_chart(df.groupby('Courier_Experience_yrs')['Delivery_Time_min'].mean())
        except Exception as e:
            st.error(f"Couldn't generate chart: {str(e)}")



elif page == "Predict Delivery Time":
    st.title("⏳ Predict Delivery Time")
    
    with st.form("delivery_input"):
        col1, col2 = st.columns(2)
        
        with col1:
            Distance_km = st.slider('Distance (km)', 0.1, 20.0, 5.0, 0.1)
            Weather = st.selectbox('Weather', ['Clear', 'Windy', 'Foggy', 'Rainy', 'Snowy'])
            Traffic_Level = st.selectbox('Traffic Level', ['Low', 'Medium', 'High'])
            Time_of_Day = st.selectbox('Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
            
        with col2:
            Vehicle_Type = st.selectbox('Vehicle Type', ['Bike', 'Scooter', 'Car'])
            Preparation_Time_min = st.slider('Preparation Time (min)', 1, 60, 20)
            Courier_Experience_yrs = st.slider('Courier Experience (yrs)', 0, 30, 2)
        
        submitted = st.form_submit_button("Predict Delivery Time")
    
    if submitted:
        try:
            input_data = pd.DataFrame({
                'Distance_km': [Distance_km],
                'Weather': [Weather],
                'Traffic_Level': [Traffic_Level],
                'Time_of_Day': [Time_of_Day],
                'Vehicle_Type': [Vehicle_Type],
                'Preparation_Time_min': [Preparation_Time_min],
                'Courier_Experience_yrs': [Courier_Experience_yrs]
            })
            
            prediction = model.predict(input_data)[0]
            
            # Determine delivery speed category
            if prediction <= 30:
                speed_category = "Fast"
                color = "green"
                icon = "⚡"
            elif prediction <= 45:
                speed_category = "Average"
                color = "orange"
                icon = "⏱️"
            else:
                speed_category = "Slow"
                color = "red"
                icon = "🐢"
            
            st.markdown(f"""
            <div style='border-left: 5px solid {color}; padding: 10px; border-radius: 5px'>
                <h3 style='color: {color}'>
                    {icon} {speed_category} Delivery: {prediction:.0f} minutes
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🔍 Key Factors Affecting This Prediction")
        
            if prediction < 30:
                st.markdown("⚡ **This is a fast delivery!** Main contributing factors:")
                factors = []
                if Distance_km < 2: factors.append(f"- 🏃‍♂️ Very short distance ({Distance_km}km)")
                if Traffic_Level == 'Low': factors.append("- 🚦 Light traffic conditions")
                if Weather in ['Clear', 'Windy']: factors.append(f"- ☀️ Good weather ({Weather})")
                if Preparation_Time_min < 15: factors.append(f"- ⚡ Quick preparation ({Preparation_Time_min}min)")
                if Courier_Experience_yrs >= 3: factors.append(f"- 👴 Veteran courier ({Courier_Experience_yrs}yrs)")
                
                for factor in factors:
                    st.write(factor)
                    
            elif 30 <= prediction < 45:
                st.markdown("⏱️ **This is an average delivery time.** Main factors:")
                st.write(f"- 📏 Medium distance ({Distance_km}km)")
                st.write(f"- 🚦 {Traffic_Level} traffic")
                st.write(f"- ⛅ {Weather} weather")
                st.write(f"- ⏱️ {Preparation_Time_min}min preparation")
                
            else:
                st.markdown("🐢 **This delivery will take longer than average.** Main reasons:")
                factors = []
                if Distance_km > 5: factors.append(f"- 📏 Long distance ({Distance_km}km)")
                if Traffic_Level == 'High': factors.append("- 🚦 Heavy traffic")
                if Weather in ['Rainy', 'Snowy']: factors.append(f"- ⛈️ Bad weather ({Weather})")
                if Preparation_Time_min > 30: factors.append(f"⏳ Long preparation ({Preparation_Time_min}min)")
                if Courier_Experience_yrs < 1: factors.append(f"- 🆕 New courier ({Courier_Experience_yrs}yrs)")
                
                for factor in factors:
                    st.write(factor)
            
            if prediction > 40:
                st.subheader("💡 Suggestions for Faster Delivery")
                suggestions = []
                if Distance_km > 5: suggestions.append("- 🏠 Order from closer restaurants")
                if Traffic_Level == 'High': suggestions.append("- 🕒 Try off-peak hours")
                if Weather in ['Rainy', 'Snowy']: suggestions.append("- ☔ Weather delays are unavoidable")
                if Preparation_Time_min > 30: suggestions.append("- 🍳 Choose restaurants with faster prep times")
                
                for suggestion in suggestions:
                    st.info(suggestion)

            # Business Suggestions Based on Speed Category
            st.subheader(f"📈 {speed_category} Delivery Recommendations")
            
            if speed_category == "Fast":
                with st.expander("🚀 Capitalize on Fast Delivery", expanded=True):
                    st.markdown("""
                    **Marketing Opportunities:**
                    - Highlight these restaurants/conditions as "Express Delivery" options
                    - Create promotional campaigns around fastest delivery times
                    - Offer "Speed Challenge" discounts for maintaining fast times
                    
                    **Operational Excellence:**
                    - Recognize and reward top-performing couriers
                    - Document best practices from these scenarios
                    - Analyze patterns to replicate success
                    
                    **Customer Engagement:**
                    - Prompt for reviews during peak satisfaction
                    - Offer loyalty points for repeat fast deliveries
                    - Feature in "Fastest Deliveries" leaderboard
                    """)
            
            elif speed_category == "Average":
                with st.expander("🛠️ Improve to Fast Delivery", expanded=True):
                    st.markdown("""
                    **Quick Wins:**
                    - Identify 1-2 factors adding the most time (see analysis below)
                    - Test small routing or dispatch adjustments
                    - Partner with restaurants to shave 5 mins off prep time
                    
                    **Customer Communication:**
                    - Set accurate expectations upfront
                    - Offer slight over-delivery ("Your food will arrive by X:XX")
                    - Provide optional upgrades to faster delivery
                    
                    **Performance Tracking:**
                    - Monitor these middle-performing deliveries closely
                    - A/B test different improvement strategies
                    - Focus on consistency improvements
                    """)
            
            else:  # Slow delivery
                with st.expander("🛑 Urgent Improvement Needed", expanded=True):
                    st.markdown("""
                    **Immediate Actions:**
                    - Flag for manual dispatch oversight
                    - Assign to top-performing couriers
                    - Provide customer with compensation options
                    
                    **Root Cause Analysis:**
                    - Investigate primary delay factors (see below)
                    - Review restaurant prep time SLA compliance
                    - Evaluate traffic/weather routing protocols
                    
                    **Customer Recovery:**
                    - Proactive delay notifications
                    - Offer discounts/free items
                    - Provide real-time tracking updates
                    """)
            
            # Detailed factor analysis
            with st.expander("🔍 Why This Delivery is " + speed_category, expanded=True):
                factors = []
                if Distance_km > 5:
                    factors.append(f"- 📏 Long distance ({Distance_km}km)")
                if Traffic_Level == 'High':
                    factors.append("- 🚦 Heavy traffic")
                if Weather in ['Rainy', 'Snowy']:
                    factors.append(f"- ⛈️ Bad weather ({Weather})")
                if Preparation_Time_min > 30:
                    factors.append(f"- ⏳ Long prep time ({Preparation_Time_min}min)")
                if Courier_Experience_yrs < 1:
                    factors.append(f"- 🆕 New courier")
                
                if factors:
                    st.markdown("#### Main Contributing Factors:")
                    st.markdown("\n".join(factors))
                    
                    if speed_category == "Fast":
                        st.success("These factors are well-optimized in this scenario")
                    elif speed_category == "Average":
                        st.warning("Improving 1-2 factors could reach Fast delivery")
                    else:
                        st.error("Multiple factors need attention for improvement")
                else:
                    st.markdown("- ⚡ All factors optimized for fast delivery")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# # ... (rest of the code remains the same) ...




# ------------------------------------------



