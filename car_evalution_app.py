import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ==============================
# 🎯 Load and Preprocess Data
# ==============================
car_data = fetch_ucirepo(id=19)
X = car_data.data.features
y = car_data.data.targets

df = pd.concat([X, y], axis=1)

# Convert doors and persons to numeric
df['doors'] = df['doors'].replace({'5more': 5}).astype(int)
df['persons'] = df['persons'].replace({'more': 6}).astype(int)

# Encode categorical columns
buying_map = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
maint_map = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
lug_boot_map = {'small': 0, 'med': 1, 'big': 2}
safety_map = {'low': 0, 'med': 1, 'high': 2}
class_map = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

df['buying'] = df['buying'].map(buying_map)
df['maint'] = df['maint'].map(maint_map)
df['lug_boot'] = df['lug_boot'].map(lug_boot_map)
df['safety'] = df['safety'].map(safety_map)
df['class'] = df['class'].map(class_map)

X = df.drop('class', axis=1)
y = df['class']

# Train final model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)

# ==============================
# 🎨 Streamlit UI
# ==============================
st.title("🚗 Car Evaluation Classifier")
st.write("Predict car acceptability based on buying, maintenance, safety, and more!")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Enter Car Details")

buying = st.sidebar.selectbox("Buying Price", ('low', 'med', 'high', 'vhigh'))
maint = st.sidebar.selectbox("Maintenance Cost", ('low', 'med', 'high', 'vhigh'))
doors = st.sidebar.selectbox("Number of Doors", (2, 3, 4, 5))
persons = st.sidebar.selectbox("Persons Capacity", (2, 4, 6))
lug_boot = st.sidebar.selectbox("Luggage Boot Size", ('small', 'med', 'big'))
safety = st.sidebar.selectbox("Safety Level", ('low', 'med', 'high'))

# Convert input to numeric form
input_data = pd.DataFrame({
    'buying': [buying_map[buying]],
    'maint': [maint_map[maint]],
    'doors': [doors],
    'persons': [persons],
    'lug_boot': [lug_boot_map[lug_boot]],
    'safety': [safety_map[safety]]
})

# Prediction
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    inv_class_map = {v: k for k, v in class_map.items()}
    result = inv_class_map[pred]
    st.success(f"### 🚘 Car Acceptability: {result.upper()}")

    if result == "unacc":
        st.warning("❌ This car is unacceptable based on given parameters.")
    elif result == "acc":
        st.info("✅ This car is acceptable.")
    elif result == "good":
        st.success("🌟 This car is good choice!")
    else:
        st.balloons()
        st.success("🏆 This car is very good choice!")
