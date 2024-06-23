import streamlit as st
from PIL import Image
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static

st.set_page_config(layout="wide")

def predict_image(pil_image):
    # Convert the PIL image to a NumPy array
    image_array = np.array(pil_image)

    # Convert RGB to BGR (OpenCV expects BGR format)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Load the Model
    model = YOLO('best (7).pt')

    # Dictionary to decode predictions
    decoding_of_predictions = {
        0: 'damagedcommercialbuilding',
        1: 'damagedresidentialbuilding',
        2: 'undamagedcommercialbuilding',
        3: 'undamagedresidentialbuilding'
    }

    # Define a dictionary to count the number of each class
    class_count = {
        'damagedcommercialbuilding': 0,
        'damagedresidentialbuilding': 0,
        'undamagedcommercialbuilding': 0,
        'undamagedresidentialbuilding': 0
    }

    # Define a color dictionary for different classes
    color_dict = {
        'damagedcommercialbuilding': (0, 0, 255),  # Red
        'damagedresidentialbuilding': (0, 255, 0),  # Green
        'undamagedcommercialbuilding': (255, 0, 0),  # Blue
        'undamagedresidentialbuilding': (255, 255, 0)  # Cyan
    }

    # Assume 'model.predict' returns the prediction results similar to the structure you provided
    results = model.predict(image_bgr, save=True, iou=0.5, save_txt=False, conf=0.2)

    for r in results:
        conf_list = r.boxes.conf.cpu().numpy().tolist()
        clss_list = r.boxes.cls.cpu().numpy().tolist()
        original_list = clss_list
        updated_list = [decoding_of_predictions[int(element)] for element in original_list]

    bounding_boxes = r.boxes.xyxy.cpu().numpy()
    confidences = conf_list
    class_names = updated_list

    # Check if bounding boxes, confidences and class names match
    if len(bounding_boxes) != len(confidences) or len(bounding_boxes) != len(class_names):
        st.error("Error: Number of bounding boxes, confidences, and class names should be the same.")
        return None
    else:
        # Draw bounding boxes on the image
        for i in range(len(bounding_boxes)):
            left, top, right, bottom = bounding_boxes[i]
            class_name = class_names[i]
            confidence = confidences[i]

            # Draw the rectangle with class-specific color
            color = color_dict[class_name]
            cv2.rectangle(image_bgr, (int(left), int(top)), (int(right), int(bottom)), color, 2)

            # Increment the class count
            class_count[class_name] += 1

        # Convert the image back to RGB format for display
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb, class_count.get("damagedcommercialbuilding"), class_count.get("damagedresidentialbuilding"), class_count.get("undamagedcommercialbuilding"), class_count.get("undamagedresidentialbuilding")

# Define the MCDA functions
def normalize_scores(df, criteria):
    for criterion in criteria:
        min_value = df[criterion].min()
        max_value = df[criterion].max()
        df[criterion + "_normalized"] = (df[criterion] - min_value) / (max_value - min_value)
    return df

def compute_weighted_scores(df, criteria_weights):
    for criterion, weight in criteria_weights.items():
        df[criterion + "_weighted"] = df[criterion + "_normalized"] * weight
    return df

def aggregate_scores(df, criteria_weights):
    df["priority_score"] = df[[criterion + "_weighted" for criterion in criteria_weights]].sum(axis=1)
    return df

def mcda(df, criteria_weights):
    criteria = list(criteria_weights.keys())

    # Normalize the scores
    df = normalize_scores(df, criteria)

    # Compute the weighted scores
    df = compute_weighted_scores(df, criteria_weights)

    # Aggregate the scores
    df = aggregate_scores(df, criteria_weights)

    # Rank the areas based on the aggregated priority score
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="dense")

    return df.sort_values(by="priority_score", ascending=False)


# Initialize session state variables
if 'button1_clicked' not in st.session_state:
    st.session_state.button1_clicked = True

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Sidebar with title and buttons
st.sidebar.image("erosion.png", width=100)
st.sidebar.title("Coastal Resilience Management System ")


if st.sidebar.button("Emergency Response") or st.session_state.button1_clicked:
    st.session_state.button1_clicked = True
    st.session_state.button2_clicked = False

if st.sidebar.button("Disaster Planning  "):
    st.session_state.button1_clicked = False
    st.session_state.button2_clicked = True

# Button 1 content
if st.session_state.button1_clicked:
    st.title("Rapid Disaster Management 🚨")
    uploaded_file = st.file_uploader("Upload a Post Cyclone Satellite Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file:
        if st.button('Validate'):
            image = Image.open(st.session_state.uploaded_file)
            with st.spinner('Predicting...'):
                image2, a, b, c, d = predict_image(image)

            st.write("")
            st.subheader("Result:")
            col1, col2 = st.columns(2)
            col1.image(image, caption='Before Detection', use_column_width=True)
            col2.image(image2, caption='After Detection', use_column_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label=":violet_background[Damaged Residential]", value=int(a))
            col2.metric(label=":green_background[Undamaged Residential]", value=int(b))
            col3.metric(label=":red_background[Damaged Commercial]", value=int(c))
            col4.metric(label=":orange_background[Undamaged Commercial]", value=int(d))

            st.markdown("---")
            # st.subheader("Imagine we have additional dataset we can Have a Decision Support System")

            # Sample data for 10 areas near Puerto Rico
            data = {
                "area": [f"Area {i}" for i in range(1, 11)],
                "latitude": [18.2208, 18.3391, 18.4655, 18.2108, 18.4511, 18.2950, 18.2375, 18.3500, 18.3600, 18.4325],
                "longitude": [-66.5901, -66.2303, -66.1057, -65.8302, -66.0047, -66.6215, -66.5913, -66.0754, -66.1075,
                              -66.5153],
                "severity_of_damage": [4, 3, 5, 2, 4, 1, 3, 2, 5, 4],
                "number_of_affected_individuals": [100, 80, 150, 60, 120, 30, 90, 70, 160, 110],
                "vulnerability_of_population": [0.8, 0.6, 0.9, 0.4, 0.7, 0.2, 0.5, 0.3, 0.9, 0.8],
                "accessibility": [3, 4, 2, 5, 3, 4, 2, 5, 3, 4],
                "availability_of_resources": [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]
            }

            df = pd.DataFrame(data)

            # Define the criteria weights
            criteria_weights = {
                "severity_of_damage": 0.4,
                "number_of_affected_individuals": 0.3,
                "vulnerability_of_population": 0.1,
                "accessibility": 0.1,
                "availability_of_resources": 0.1
            }

            # Run the MCDA function
            result = mcda(df, criteria_weights)

            # Ensure numerical sorting of areas
            result['area'] = pd.Categorical(result['area'], categories=[f"Area {i}" for i in range(1, 11)],
                                            ordered=True)
            result = result.sort_values('area')

            # Streamlit app
            st.title("MCDA for Coastal Disaster Response 🚑🚒")

            # Visualizations for each criterion
            # st.subheader("Criteria Visualizations")

            criteria_columns = ["severity_of_damage", "number_of_affected_individuals", "vulnerability_of_population",
                                "accessibility", "availability_of_resources"]
            criteria_titles = ["Severity of Damage", "Number of Affected Individuals", "Vulnerability of Population",
                               "Accessibility", "Availability of Resources"]

            # Define colors for each visualization
            colors = ["#FF6347", "#4682B4", "#32CD32", "#FFD700", "#BF40BF"]

            # Create 3-column, 2-row layout for the criteria visualizations
            # First row: 3 columns
            columns = st.columns(3)

            with columns[0]:
                i=0
                fig = px.bar(result, x='area', y=criteria_columns[i], title=criteria_titles[i],
                             color_discrete_sequence=[colors[i]])
                fig.update_layout(title='Severity of Damage',
                                  xaxis_title='Area',
                                  yaxis_title='Severity of Damage',
                                  plot_bgcolor='#3b3b3b',
                                  paper_bgcolor='#3b3b3b')
                st.plotly_chart(fig)

            with columns[1]:
                i=1
                fig = px.bar(result, x='area', y=criteria_columns[i], title=criteria_titles[i],
                             color_discrete_sequence=[colors[i]])
                fig.update_layout(title='Number of Affected Individuals',
                                  xaxis_title='Area',
                                  yaxis_title='Number of Affected Individuals',
                                  plot_bgcolor='#3b3b3b',
                                  paper_bgcolor='#3b3b3b')
                st.plotly_chart(fig)

            with columns[2]:
                i=2
                fig = px.bar(result, x='area', y=criteria_columns[i], title=criteria_titles[i],
                             color_discrete_sequence=[colors[i]])
                fig.update_layout(title='Vulnerability of Population',
                                  xaxis_title='Area',
                                  yaxis_title='Vulnerability of Population',
                                  plot_bgcolor='#3b3b3b',
                                  paper_bgcolor='#3b3b3b')
                st.plotly_chart(fig)

            # Second row: 2 columns
            columns = st.columns(2)
            # for i in range(3, 5):
            # with columns[i - 3]:
            with columns[0]:
                fig = px.line(result, x='area', y=criteria_columns[3], title=criteria_titles[3],
                                line_shape='linear', markers=True, color_discrete_sequence=[colors[3]])
                fig.update_layout(title='Accessibility',
                                xaxis_title='Area',
                                yaxis_title='Accessibility',
                                plot_bgcolor='#3b3b3b',
                                paper_bgcolor='#3b3b3b')
                st.plotly_chart(fig)
            with columns[1]:
                fig = px.line(result, x='area', y=criteria_columns[4], title=criteria_titles[4],
                                line_shape='linear', markers=True, color_discrete_sequence=[colors[4]])
                fig.update_layout(title='Availability of Resources',
                                xaxis_title='Area',
                                yaxis_title='Availability of Resources',
                                plot_bgcolor='#3b3b3b',
                                paper_bgcolor='#3b3b3b')
                st.plotly_chart(fig)

            # Expander for explanation
            with st.expander("Expand to view the explanation of Priority Score Calculation"):
                st.markdown("""
                The priority score for each area is calculated using a Multi-Criteria Decision Analysis (MCDA) approach. 
                Here are the steps involved:

                1. **Normalization of Scores**: 
                   Each criterion score (severity of damage, number of affected individuals, vulnerability of population,
                   accessibility, availability of resources) is normalized between 0 and 1 to ensure comparability across different scales.
                   Formula: 
                   \[
                   X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}
                   \]
                   where \( X \) is the original score, \( X_{min} \) is the minimum score for the criterion, and \( X_{max} \) is the maximum score for the criterion.

                2. **Weighted Scores Calculation**: 
                   Each normalized criterion score is multiplied by its respective weight:
                   Formula: 
                   \[
                   X_{weighted} = X_{normalized} \times W
                   \]
                   where \( W \) is the weight for the criterion.The weights for each criterion are as follows:
                   - Severity of damage: **0.4**
                   - Number of affected individuals: **0.3**
                   - Vulnerability of population: **0.2**
                   - Accessibility: **0.05**
                   - Availability of resources: **0.05**
            
                3. **Aggregation of Scores**: 
                   The weighted scores for each criterion are summed up to obtain the priority score for each area.
                   Formula: 
                   \[
                   \text{Priority Score} = \sum (X_{weighted})
                   \]

                4. **Ranking**: 
                   The areas are ranked based on their priority scores in descending order to identify the areas 
                   that require immediate attention.
                """)

            with st.expander("Expand to view the detailed result"):
                # result.rename(columns={'severity_of_damage': 'Severity of Damage',
                #                        'number_of_affected_individuals': 'Number of Affected Individuals',
                #                        'vulnerability_of_population': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                        'C': 'Column 3',
                #                                     }, inplace=True)
                st.dataframe(result)

            # Create a bar chart for the priority scores
            st.subheader("Priority Scores by Area")
            # fig = px.bar(result, x='area', y='priority_score', color='priority_rank', title="Priority Scores", color_continuous_scale=px.colors.sequential.Viridis)
            # st.plotly_chart(fig)

            # Priority scores as a line plot
            # st.subheader("Priority Scores by Area (Highest to Lowest)")
            # Sort the DataFrame by priority_score (highest to lowest)
            df = result.sort_values(by='priority_score', ascending=False)

            fig = px.line(df, x='area', y='priority_score', title="Priority Scores by Area (Highest to Lowest)",
                          labels={'area': 'Area', 'priority_score': 'Priority Score'}, line_shape='linear',
                          markers=True)
            fig.update_layout(title='Priority Scores by Area',
                              xaxis_title='Area',
                              yaxis_title='Priority Score',
                              plot_bgcolor='#3b3b3b',
                              paper_bgcolor='#3b3b3b')

            st.plotly_chart(fig)

            # 2-column layout for map and table
            st.subheader("Priority Table and Map 📍🗺️")
            col1, col2 = st.columns([1.5, 3])  # Adjust the ratio here to make the map column larger

            # Left column: table
            with col1:

                priority_table = result[['area', 'priority_rank']]
                st.dataframe(priority_table)

            # Right column: map
            with col2:
                # Initialize the map centered around Puerto Rico
                m = folium.Map(location=[18.2208, -66.5901], zoom_start=8)


                # Define marker colors based on priority rank
                def get_marker_color(rank):
                    if rank <= 3:
                        return "red"
                    elif rank <= 6:
                        return "blue"
                    else:
                        return "green"


                # Add markers for each area
                for idx, row in result.iterrows():
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"{row['area']} - Priority Rank: {row['priority_rank']:.0f}",
                        tooltip=row["area"],
                        icon=folium.Icon(color=get_marker_color(row["priority_rank"]))
                    ).add_to(m)

                # Display the map
                folium_static(m)



# Button 2 content
if st.session_state.button2_clicked:
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import folium
    from streamlit_folium import folium_static
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Sample data for multiple areas
    np.random.seed(0)
    areas = [f"Area {i}" for i in range(1, 11)]
    timestamps = pd.date_range(start='2024-01-01', periods=20, freq='D')  # Reduced to 20 daily data points

    data = []
    for area in areas:
        for timestamp in timestamps:
            data.append({
                'area': area,
                'timestamp': timestamp,
                'sea_level': np.random.normal(5, 0.5),
                'humidity': np.random.uniform(60, 200),
                'rainfall': np.random.uniform(0, 20),
                'wind_speed': np.random.uniform(0, 30),
                'wind_direction': np.random.uniform(0, 360),
                'atmospheric_pressure': np.random.normal(1013, 10),
                'temperature': np.random.normal(25, 5),
            })

    df = pd.DataFrame(data)

    # Streamlit app
    st.title("Coastal Disaster Response Monitoring Dashboard 📊")

    # Define titles for the visualizations
    criteria_titles = ["Sea Level", "Humidity", "Rainfall", "Wind Speed and Direction", "Atmospheric Pressure",
                       "Temperature"]

    # Select area
    selected_area = st.selectbox("Select Area", areas)
    area_df = df[df['area'] == selected_area]

    # Create 3-column, 2-row layout for the visualizations
    columns = st.columns(3)

    # First row: 3 columns
    with columns[0]:
        fig = px.scatter(area_df, x='timestamp', y='sea_level', title="Sea Level", color_discrete_sequence=["#FF6347"])
        fig.update_layout(title='Sea Level',
                          xaxis_title='Time',
                          yaxis_title='Sea Level',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

    with columns[1]:
        fig = px.histogram(area_df, x='humidity', nbins=10, title="Humidity", color_discrete_sequence=["#4682B4"])
        fig.update_layout(title='Humidity',
                          xaxis_title='Time',
                          yaxis_title='Humidity',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

    with columns[2]:
        fig = px.bar(area_df, x='timestamp', y='rainfall', title="Rainfall", color_discrete_sequence=["#32CD32"])
        fig.update_layout(title='Rainfall',
                          xaxis_title='Time',
                          yaxis_title='Rainfall',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

    # Second row: 3 columns
    columns = st.columns(3)

    with columns[0]:
        fig = px.line(area_df, x='timestamp', y='wind_speed', title="Wind Speed", color_discrete_sequence=["#FFD700"])
        fig.update_layout(title='Wind Speed',
                          xaxis_title='Time',
                          yaxis_title='Wind Speed',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

        # wind_dir_fig = px.scatter_polar(area_df, r='wind_speed', theta='wind_direction', title="Wind Direction",
        #                                 color_discrete_sequence=["#4B0082"])
        # fig.update_layout(title='Wind Direction',
        #                   xaxis_title='Area',
        #                   yaxis_title='Combined Score',
        #                   plot_bgcolor='#3b3b3b',
        #                   paper_bgcolor='#3b3b3b')
        # st.plotly_chart(wind_dir_fig)

    with columns[1]:
        fig = px.box(area_df, y='atmospheric_pressure', title="Atmospheric Pressure",
                     color_discrete_sequence=["#FF4500"])
        fig.update_layout(title='Atmospheric Pressure',

                          yaxis_title='Atmospheric Pressure',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

    with columns[2]:
        fig = px.area(area_df, x='timestamp', y='temperature', title="Temperature", color_discrete_sequence=["#2E8B57"])
        fig.update_layout(title='Temperature',
                          xaxis_title='Time',
                          yaxis_title='Temperature',
                          plot_bgcolor='#3b3b3b',
                          paper_bgcolor='#3b3b3b')
        st.plotly_chart(fig)

    # Generate dummy historical data
    np.random.seed(0)
    areas = [f"Area {i}" for i in range(1, 11)]
    data = []
    for area in areas:
        for _ in range(50):  # 50 data points per area
            sea_level = np.random.normal(5, 0.5)
            humidity = np.random.uniform(60, 100)
            rainfall = np.random.uniform(0, 20)
            wind_speed = np.random.uniform(0, 30)
            wind_direction = np.random.uniform(0, 360)
            atmospheric_pressure = np.random.normal(1013, 10)
            temperature = np.random.normal(25, 5)
            storm = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% chance of storm

            data.append({
                'area': area,
                'sea_level': sea_level,
                'humidity': humidity,
                'rainfall': rainfall,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'atmospheric_pressure': atmospheric_pressure,
                'temperature': temperature,
                'storm': storm
            })

    df = pd.DataFrame(data)

    # Split data into features and target
    X = df.drop(columns=['storm', 'area'])
    y = df['storm']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict probabilities for each area
    area_predictions = df.groupby('area').apply(
        lambda x: model.predict_proba(x.drop(columns=['storm', 'area']))[:, 1].mean()).reset_index()
    area_predictions.columns = ['area', 'storm_probability']

    # Sort areas by storm probability
    area_predictions = area_predictions.sort_values(by='storm_probability', ascending=False)

    st.markdown("---")
    # Streamlit app
    st.title("Storm Prediction 🌪️ ")

    col1, col2 = st.columns([1.5, 3])
    with col1:
        # Display a table of storm probabilities
        st.subheader("Storm Risk by Area")
        st.dataframe(area_predictions)

    with col2:
        # Map visualization
        st.subheader("Storm Risk Map")
        map_center = [18.2208, -66.5901]  # Centered around Puerto Rico
        m = folium.Map(location=map_center, zoom_start=8)


        # Define marker colors based on storm probability
        def get_marker_color(probability):
            if probability > 0.7:
                return "red"
            elif probability > 0.4:
                return "orange"
            else:
                return "green"


        # Add markers for each area
        for idx, row in area_predictions.iterrows():
            folium.Marker(
                location=[map_center[0] + np.random.uniform(-0.5, 0.5), map_center[1] + np.random.uniform(-0.5, 0.5)],
                # Randomize location for demo
                popup=f"{row['area']} - Storm Probability: {row['storm_probability']:.2f}",
                tooltip=row["area"],
                icon=folium.Icon(color=get_marker_color(row["storm_probability"]))
            ).add_to(m)

        # Display the map
        folium_static(m)

    # Generate dummy historical data for beach erosion
    np.random.seed(0)
    areas = [f"Area {i}" for i in range(1, 11)]
    data = []
    for area in areas:
        for _ in range(50):  # 50 data points per area
            wave_height = np.random.uniform(0, 5)  # meters
            tidal_range = np.random.uniform(0, 4)  # meters
            storm_frequency = np.random.uniform(0, 10)  # storms per year
            sediment_supply = np.random.uniform(0, 20)  # cubic meters per year
            erosion = np.random.uniform(0, 10)  # meters of coastline erosion

            data.append({
                'area': area,
                'wave_height': wave_height,
                'tidal_range': tidal_range,
                'storm_frequency': storm_frequency,
                'sediment_supply': sediment_supply,
                'erosion': erosion
            })

    df = pd.DataFrame(data)

    # Split data into features and target
    X = df.drop(columns=['erosion', 'area'])
    y = df['erosion']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Predict erosion amounts for each area
    area_erosion_predictions = df.groupby('area').apply(
        lambda x: model.predict(x.drop(columns=['erosion', 'area'])).mean()).reset_index()
    area_erosion_predictions.columns = ['area', 'predicted_erosion']

    # Ensure numerical sorting of areas
    area_erosion_predictions['area'] = pd.Categorical(area_erosion_predictions['area'],
                                                      categories=[f"Area {i}" for i in range(1, 21)], ordered=True)
    area_erosion_predictions = area_erosion_predictions.sort_values('area')

    st.markdown("---")

    # Streamlit app
    st.title("Beach Erosion Prediction🌊")

    # Visualize coastline before and after erosion
    st.subheader("Coastline Before and After Erosion")
    selected_area = st.selectbox("Select an Area", areas)

    # Generate dummy coastline data for selected area
    np.random.seed(0)
    before_erosion = np.random.normal(0, 1, 100).cumsum()
    after_erosion = before_erosion - area_erosion_predictions.loc[
        area_erosion_predictions['area'] == selected_area, 'predicted_erosion'].values[0]

    # Create a line chart for coastline visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=before_erosion, mode='lines', name='Before Erosion'))
    fig.add_trace(go.Scatter(y=after_erosion, mode='lines', name='After Erosion'))
    fig.update_layout(title=f'Coastline Before and After Erosion ({selected_area})',
                      xaxis_title='Distance Along Coastline',
                      yaxis_title='Elevation (meters)')
    st.plotly_chart(fig)

    priority_data = {
        "area": [f"Area {i}" for i in range(1, 21)],
        "priority_score": np.random.uniform(0, 1, 20),
        "economic_value": np.random.uniform(1, 10, 20),
        "population_density": np.random.uniform(1, 10, 20)
    }
    priority_df = pd.DataFrame(priority_data)
    #
    # # Merge priority and erosion data
    combined_df = pd.merge(priority_df, area_erosion_predictions, on='area')
    combined_df['combined_score'] = combined_df['priority_score'] + combined_df['predicted_erosion']
    #
    # Rank areas based on combined score
    combined_df['combined_rank'] = combined_df['combined_score'].rank(ascending=False, method='dense')

    # Display a table of combined priority and erosion scores
    st.subheader("Combined Priority and Erosion Scores by Area")
    st.dataframe(combined_df[['area', 'priority_score', 'predicted_erosion', 'combined_score', 'combined_rank']])

    # Create a bar chart for combined scores
    st.subheader("Combined Priority and Erosion Scores")
    combined_df = combined_df.sort_values(by='combined_score', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=combined_df['area'], y=combined_df['combined_score'],
                         marker_color='blue'))
    fig.update_layout(title='Combined Priority and Erosion Scores by Area',
                      xaxis_title='Area',
                      yaxis_title='Combined Score')
    st.plotly_chart(fig)

    # Summarization text box
    highest_impact_area = combined_df.loc[combined_df['combined_rank'] == 1, 'area'].values[0]
    highest_impact_economic_value = combined_df.loc[combined_df['combined_rank'] == 1, 'economic_value'].values[0]
    highest_impact_population_density = combined_df.loc[combined_df['combined_rank'] == 1, 'population_density'].values[
        0]

    summary_text = f"""
    **Highest Impact Area: {highest_impact_area}**

    This area has the highest combined priority and erosion score, indicating it is the most highly potentially impacted by coastal erosion and disaster risks. Special attention and proper planning are required for this area due to its high economic value and population density.

    - **Economic Value**: {highest_impact_economic_value:.2f}
    - **Population Density**: {highest_impact_population_density:.2f}
    """

    st.markdown("---")
    st.title("💬 Summary Chatbot 🤖")
    st.markdown(summary_text)

# Adding multiple empty lines using a loop
for _ in range(10):
    st.sidebar.write("")

st.sidebar.markdown("---")  # Horizontal line to separate the footer
st.sidebar.caption("Developed by Francis and Sylvia")
