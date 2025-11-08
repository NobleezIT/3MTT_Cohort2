# **INTERACTIVE VISUALIZATION & DASHBOARDS IN PYTHON USING PLOTLY & DASH**

### **Course Learning Goal**

By the end of this module, students should be able to:

* Create interactive plots using Plotly.
* Build dashboards using Dash entirely in Python.
* Add filters and dynamic updates using callbacks.
* Use 3D charts to explore multi-variable relationships in complex datasets.

---

## **1. Library Setup and Dataset Loading**

```python
import pandas as pd

df = pd.read_csv("heart_attack_south_africa.csv")
df.head()
```

This reads the dataset into a pandas DataFrame. We will be using the following key columns:

| Column                  | Description                  |
| ----------------------- | ---------------------------- |
| Age                     | Age of the patient           |
| Cholesterol_Level       | Cholesterol level (mg/dl)    |
| Blood_Pressure_Systolic | Systolic blood pressure      |
| Heart_Attack_Outcome    | 0 = No heart attack, 1 = Yes |
| Diabetes_Status         | Yes / No indicator           |
| Smoking_Status          | Yes / No indicator           |

This dataset simulates heart disease across population samples.

---

## **2. Interactive 3D Scatter Plot (Exploring Relationships)**

This visualization allows us to compare three continuous risk factors at the same time.

```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter3d(
    x=df['Age'],
    y=df['Cholesterol_Level'],
    z=df['Blood_Pressure_Systolic'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['Heart_Attack_Outcome'],
        colorscale='Viridis',
        opacity=0.8
    ),
    text=[
        f"Diabetes: {d}, Smoking: {s}"
        for d, s in zip(df['Diabetes_Status'], df['Smoking_Status'])
    ]
))

fig.update_layout(
    title="3D Scatter: Age vs Cholesterol vs Systolic Blood Pressure",
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Cholesterol Level (mg/dl)',
        zaxis_title='Systolic BP'
    )
)

fig.show()
```

### **Explanation**

* We use `go.Scatter3d` to create a 3D scatter plot.
* The `x`, `y`, and `z` axes are mapped to Age, Cholesterol, and Blood Pressure.
* Marker color is based on heart attack outcome so we visually compare risk.
* Hovering the cursor shows medical context (diabetes and smoking).

This visualization allows students to see cluster tendencies among high-risk patients.

---

## **3. 3D Surface Plot of Predicted Heart Attack Risk**

Here we build a simple machine learning model and visualize how **risk probability changes** across age and cholesterol values.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np
import plotly.graph_objects as go

X = df[['Age', 'Cholesterol_Level']]
y = df['Heart_Attack_Outcome']

model = LogisticRegression().fit(X, y)

age_range = np.linspace(df['Age'].min(), df['Age'].max(), 50)
chol_range = np.linspace(df['Cholesterol_Level'].min(), df['Cholesterol_Level'].max(), 50)
age_grid, chol_grid = np.meshgrid(age_range, chol_range)

grid_points = np.c_[age_grid.ravel(), chol_grid.ravel()]
risk_prob = model.predict_proba(grid_points)[:, 1].reshape(age_grid.shape)

fig_surface = go.Figure(data=go.Surface(
    x=age_grid,
    y=chol_grid,
    z=risk_prob,
    colorscale='Reds'
))

fig_surface.update_layout(
    title="3D Surface Plot: Predicted Heart Attack Risk",
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Cholesterol Level',
        zaxis_title='Risk Probability'
    )
)

fig_surface.show()
```

### Explanation

* Logistic regression estimates the probability that a patient has a heart attack.
* A meshgrid generates every possible combination of Age and Cholesterol in the dataset range.
* The plot shows how risk increases in smooth upward regions of the surface.
* Red peaks represent high estimated risk.

---

## **4. Clustering and 3D Cluster Visualization**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

features = ['Age', 'Cholesterol_Level', 'Blood_Pressure_Systolic', 'Obesity_Index']
X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_cluster = X.copy()
df_cluster['Cluster'] = clusters
df_cluster['Patient_ID'] = df['Patient_ID']

fig_cluster = go.Figure(data=go.Scatter3d(
    x=df_cluster['Age'],
    y=df_cluster['Cholesterol_Level'],
    z=df_cluster['Blood_Pressure_Systolic'],
    mode='markers',
    marker=dict(
        size=5,
        color=df_cluster['Cluster'],
        colorscale='Viridis',
        opacity=0.8
    ),
    text=[f"Patient ID: {pid}, Obesity: {oi}" for pid, oi in zip(df_cluster['Patient_ID'], df_cluster['Obesity_Index'])]
))

fig_cluster.update_layout(
    title="3D K-Means Clustering of Heart Attack Risk Factors",
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Cholesterol Level',
        zaxis_title='Systolic BP'
    )
)

fig_cluster.show()
```

### Explanation

* KMeans groups similar patients based on four health metrics.
* Each cluster represents a distinct health risk group.
* The scatter plot color visually separates the clusters.

---

# **5. Building the Dash Interactive Dashboard**

```python
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go

app = Dash(__name__)

app.layout = html.Div([
    html.H1("3D Heart Risk Explorer"),

    dcc.Dropdown(
        id='smoking-filter',
        options=[{'label': s, 'value': s} for s in df['Smoking_Status'].unique()],
        value=df['Smoking_Status'].unique()[0],
        clearable=False
    ),

    dcc.Graph(id='scatter-3d')
])

@app.callback(
    Output('scatter-3d', 'figure'),
    Input('smoking-filter', 'value')
)
def update_3d(sc_filter):
    filtered = df[df['Smoking_Status'] == sc_filter]

    fig = go.Figure(data=go.Scatter3d(
        x=filtered['Age'],
        y=filtered['Cholesterol_Level'],
        z=filtered['Blood_Pressure_Systolic'],
        mode='markers',
        marker=dict(
            size=5,
            color=filtered['Heart_Attack_Outcome'],
            colorscale='Viridis',
            opacity=0.8
        )
    ))
    fig.update_layout(title=f"3D Risk Plot for Smoking Status: {sc_filter}")
    return fig

app.run_server(debug=True)
```

### Explanation

* The dropdown allows the user to filter by smoking status.
* The callback listens for user selection and updates the plot.
* No page refresh required. Everything responds automatically.

---

# **6. Full Filtering Dashboard with Download Button**

```python
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Heart Attack Risk Filter Dashboard"),

    html.Label("Age Range"),
    dcc.RangeSlider(
        id='age-range',
        min=int(df['Age'].min()),
        max=int(df['Age'].max()),
        value=[int(df['Age'].min()), int(df['Age'].max())]
    ),

    html.Label("Select Gender"),
    dcc.Dropdown(
        id='gender-filter',
        options=[{'label': g, 'value': g} for g in df['Gender'].unique()],
        value=df['Gender'].unique()[0]
    ),

    html.Label("Smoking Status"),
    dcc.RadioItems(
        id='smoking-filter-2',
        options=[{'label': s, 'value': s} for s in df['Smoking_Status'].unique()],
        value=df['Smoking_Status'].unique()[0]
    ),

    html.Button("Download Data", id='download-btn'),
    dcc.Download(id='download-data'),

    dcc.Graph(id='bar-chart')
])

@app.callback(
    Output('bar-chart', 'figure'),
    Input('age-range', 'value'),
    Input('gender-filter', 'value'),
    Input('smoking-filter-2', 'value')
)
def update_chart(age_range, gender, smoking):
    filtered = df[
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['Gender'] == gender) &
        (df['Smoking_Status'] == smoking)
    ]
    counts = filtered['Heart_Attack_Outcome'].value_counts().sort_index()

    fig = go.Figure(data=go.Bar(
        x=['No Attack', 'Yes Attack'],
        y=counts.values
    ))
    fig.update_layout(title="Heart Attack Outcomes under Selected Filters")
    return fig

@app.callback(
    Output('download-data', 'data'),
    Input('download-btn', 'n_clicks'),
    State('age-range', 'value'),
    State('gender-filter', 'value'),
    State('smoking-filter-2', 'value')
)
def download_data(n, age_range, gender, smoking):
    if not n:
        return
    filtered = df[
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['Gender'] == gender) &
        (df['Smoking_Status'] == smoking)
    ]
    return dcc.send_data_frame(filtered.to_csv, "filtered_data.csv", index=False)

app.run_server(debug=True)
```

---

## **This completes:**

✔ Interactive 3D Visualization
✔ Predictive Surface Model
✔ Clustering Visualization
✔ Dashboard Construction
✔ Filter Controls and Callbacks
✔ Downloadable Data Export Feature

