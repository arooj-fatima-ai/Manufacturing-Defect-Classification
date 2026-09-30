import joblib
import pandas as pd
import plotly.graph_objects as graph_objects
import streamlit as st

st.set_page_config(page_title="Defect Risk Predictor", page_icon="🏭", layout="wide")

trained_model = joblib.load("models/defect_classifier_model_logreg.pkl")
feature_scaler = joblib.load("models/feature_scaler.pkl")
production_line_encoder = joblib.load("models/production_line_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.markdown("""
<style>
.main-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0rem;}
.sub-title {color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem;}
.result-card {padding: 1.5rem; border-radius: 14px; text-align: center; color: white;}
.result-card h2 {margin: 0; font-size: 1.6rem;}
.result-card p {margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏭 Manufacturing Defect Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Enter a unit\'s process readings to check whether it is likely to be '
    'defective before it leaves the line.</p>',
    unsafe_allow_html=True
)

input_column, result_column = st.columns([1, 1.2], gap="large")

with input_column:
    st.subheader("Process Readings")

    selected_production_line = st.selectbox(
        "Production Line", options=list(production_line_encoder.classes_)
    )

    process_temperature_input = st.slider(
        "Process Temperature (°C)", min_value=120.0, max_value=240.0, value=180.0, step=0.5
    )
    process_pressure_input = st.slider(
        "Process Pressure (bar)", min_value=2.0, max_value=8.0, value=5.0, step=0.05
    )
    cycle_time_input = st.slider(
        "Cycle Time (seconds)", min_value=20.0, max_value=70.0, value=45.0, step=0.5
    )
    material_thickness_input = st.slider(
        "Material Thickness (mm)", min_value=1.5, max_value=4.5, value=3.0, step=0.05
    )
    machine_vibration_input = st.slider(
        "Machine Vibration Level", min_value=0.0, max_value=5.0, value=0.8, step=0.05
    )
    operator_experience_input = st.slider(
        "Operator Experience (years)", min_value=0, max_value=20, value=5
    )
    inspection_score_input = st.slider(
        "Inspection Score (0-100)", min_value=0.0, max_value=100.0, value=85.0, step=0.5
    )

    run_prediction = st.button("Check Defect Risk", type="primary", use_container_width=True)

with result_column:
    st.subheader("Prediction")

    if run_prediction:
        production_line_encoded_value = production_line_encoder.transform([selected_production_line])[0]

        input_row = pd.DataFrame([{
            "process_temperature_celsius": process_temperature_input,
            "process_pressure_bar": process_pressure_input,
            "cycle_time_seconds": cycle_time_input,
            "material_thickness_mm": material_thickness_input,
            "machine_vibration_level": machine_vibration_input,
            "operator_experience_years": operator_experience_input,
            "inspection_score": inspection_score_input,
            "production_line_encoded": production_line_encoded_value,
        }])[feature_columns]

        input_row_scaled = feature_scaler.transform(input_row)
        predicted_class = trained_model.predict(input_row_scaled)[0]
        predicted_probability = trained_model.predict_proba(input_row_scaled)[0][1]
        defect_risk_percentage = round(predicted_probability * 100, 1)

        if predicted_class == 1:
            result_label = "Likely Defective"
            result_color = "#dc2626"
            result_message = "Process readings are outside the safe range — recommend a manual inspection before shipping."
        else:
            result_label = "Likely Good Unit"
            result_color = "#16a34a"
            result_message = "Process readings are within the range typically seen in non-defective units."

        st.markdown(
            f"""
            <div class="result-card" style="background-color: {result_color};">
                <h2>{result_label}</h2>
                <p>{result_message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        gauge_figure = graph_objects.Figure(graph_objects.Indicator(
            mode="gauge+number",
            value=defect_risk_percentage,
            number={"suffix": "%"},
            title={"text": "Predicted Defect Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result_color},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
            }
        ))
        gauge_figure.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge_figure, use_container_width=True)

        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric(
            "Pressure Deviation", f"{round(abs(process_pressure_input - 5.0), 2)} bar from target"
        )
        metric_col_2.metric(
            "Thickness Deviation", f"{round(abs(material_thickness_input - 3.0), 2)} mm from target"
        )

        with st.expander("See how far each reading is from the safe target"):
            deviation_chart = graph_objects.Figure()
            deviation_chart.add_trace(graph_objects.Bar(
                x=["Temperature (°C)", "Pressure (bar x10)", "Thickness (mm x10)", "Vibration"],
                y=[
                    round(abs(process_temperature_input - 180), 1),
                    round(abs(process_pressure_input - 5.0) * 10, 1),
                    round(abs(material_thickness_input - 3.0) * 10, 1),
                    machine_vibration_input
                ],
                marker_color=["#2563eb", "#f97316", "#a855f7", "#dc2626"]
            ))
            deviation_chart.update_layout(
                height=300, title="Deviation from Target Process Settings", margin=dict(t=40, b=10)
            )
            st.plotly_chart(deviation_chart, use_container_width=True)
            st.caption(
                "The further pressure and material thickness drift from their target values, and the "
                "higher the vibration, the more likely the model is to flag the unit as defective."
            )
    else:
        st.info("Fill in the process readings on the left and click **Check Defect Risk** to see the prediction.")

st.divider()
st.caption(
    "Model: Logistic Regression Classifier · Trained on a simulated manufacturing-process dataset · "
    "Learn Depth Academy — Track 1 Capstone"
)
