import numpy as np
import pandas as pd

random_seed = 42
number_of_units = 6000
rng = np.random.default_rng(random_seed)

production_lines = ["Line_A", "Line_B", "Line_C"]
production_line_choice = rng.choice(production_lines, size=number_of_units)

process_temperature_celsius = np.round(rng.normal(loc=180, scale=12, size=number_of_units), 1)
process_pressure_bar = np.round(rng.normal(loc=5.0, scale=0.6, size=number_of_units), 2)
cycle_time_seconds = np.round(rng.normal(loc=45, scale=6, size=number_of_units), 1)
material_thickness_mm = np.round(rng.normal(loc=3.0, scale=0.25, size=number_of_units), 2)
machine_vibration_level = np.round(np.clip(rng.gamma(shape=2.0, scale=0.4, size=number_of_units), 0, None), 2)
operator_experience_years = rng.integers(low=0, high=20, size=number_of_units)
inspection_score = np.round(np.clip(rng.normal(loc=85, scale=8, size=number_of_units), 0, 100), 1)

temperature_deviation = np.abs(process_temperature_celsius - 180)
pressure_deviation = np.abs(process_pressure_bar - 5.0)
thickness_deviation = np.abs(material_thickness_mm - 3.0)

defect_score = (
    0.08 * temperature_deviation
    + 3.0 * pressure_deviation
    + 0.03 * np.abs(cycle_time_seconds - 45)
    + 6.0 * thickness_deviation
    + 1.2 * machine_vibration_level
    - 0.05 * operator_experience_years
    - 0.06 * (inspection_score - 85)
    + rng.normal(loc=0.0, scale=1.8, size=number_of_units)
)

defect_threshold = np.percentile(defect_score, 78)
is_defective = (defect_score > defect_threshold).astype(int)

label_noise_mask = rng.random(number_of_units) < 0.03
is_defective[label_noise_mask] = 1 - is_defective[label_noise_mask]

manufacturing_dataframe = pd.DataFrame({
    "unit_id": [f"UNIT_{i:05d}" for i in range(number_of_units)],
    "production_line": production_line_choice,
    "process_temperature_celsius": process_temperature_celsius,
    "process_pressure_bar": process_pressure_bar,
    "cycle_time_seconds": cycle_time_seconds,
    "material_thickness_mm": material_thickness_mm,
    "machine_vibration_level": machine_vibration_level,
    "operator_experience_years": operator_experience_years,
    "inspection_score": inspection_score,
    "is_defective": is_defective,
})

missing_value_positions = rng.choice(manufacturing_dataframe.index, size=50, replace=False)
manufacturing_dataframe.loc[missing_value_positions, "inspection_score"] = np.nan

duplicate_rows = manufacturing_dataframe.sample(n=22, random_state=random_seed)
manufacturing_dataframe = pd.concat([manufacturing_dataframe, duplicate_rows], ignore_index=True)

manufacturing_dataframe.to_csv("/home/claude/project3/data/manufacturing_defects.csv", index=False)
print("Dataset saved:", manufacturing_dataframe.shape)
print(manufacturing_dataframe["is_defective"].value_counts(normalize=True))
