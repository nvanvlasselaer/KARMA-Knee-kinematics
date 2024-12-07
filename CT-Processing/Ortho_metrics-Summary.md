

### **1. Setup and Dependencies**
- **Imported Libraries**:
  - **`transforms3d._gohlketransforms`**: Used for computing Euler angles from transformation matrices.
  - **`scipy.signal`**: Provides smoothing filters like Savitzky-Golay for time-series data.
  - **`numpy`**: Handles matrix operations and numerical computations.
  - **`matplotlib.pyplot`**: Used for 3D plotting and visualization.
- **Purpose**: These libraries enable advanced computations, interpolation, smoothing, and visualization for motion and kinematic data.

---

### **2. Core Functionality: `Compute_Orthopedic_Metrics`**
This is the main function of the script, responsible for calculating biomechanical metrics for knee joint analysis, specifically for the femur, tibia, and patella. Its sub-steps include:

---

#### **A. Input Parameters**
The function takes the following inputs:
- **Subjects and conditions**: Lists of participants and experimental conditions.
- **Tests and sides**: Specific tests (e.g., gait analysis) and body sides (left/right).
- **Angle thresholds**: Used for segmenting and normalizing data.
- **Input folder**: Path to the folder containing biomechanical data.

---

#### **B. Data Initialization**
- **Zero Initialization**:
  - Arrays are initialized for storing biomechanical metrics like translation, rotation, and morphological parameters.
  - For example, `X21_Tf_all` stores tibiofemoral joint rotations across time steps, and `Patella_translation_delta_X_all` stores patella translations.

---

#### **C. Data Processing**
- **Reading Input Data**:
  - Reads `.csv` files for femur, tibia, and patella coordinates (`Femur_mypts.csv`, etc.).
  - Handles missing values using NumPy's `genfromtxt`.
- **Coordinate System Transformation**:
  - Converts global coordinates (GCS) into a local coordinate system (LCS) for each bone.
  - Defines reference frames dynamically for femur, tibia, and patella based on anatomical landmarks.
  - Uses rotation matrices (`Dynfem_R`, `Dyntib_R`, `Dynpat_R`) to apply these transformations.

---

#### **D. Joint Kinematics and Morphological Metrics**
- **Kinematic Metrics**:
  - Computes Cardan angles (Euler angles) for:
    - **Tibiofemoral Joint** (flexion/extension, abduction/adduction, internal/external rotation).
    - **Patellofemoral Joint**.
- **Morphological Metrics**:
  - **TTTG Distance**: The distance between the tibial tuberosity and trochlear groove.
  - **Bisect Offset (BO)**: Percentage offset of the patella's bisecting axis.
  - **Lateral Tilt (alpha)**: Angle between the femur and patella planes.

---

#### **E. Interpolation and Smoothing**
- **Interpolation**:
  - Resamples all time-series data to a consistent length (e.g., 100 points) for normalization.
  - Uses `np.interp` to align data for comparison across participants.
- **Smoothing**:
  - Applies Savitzky-Golay filtering (`scipy.signal.savgol_filter`) to smooth noisy kinematic data.

---

#### **F. Data Normalization**
- **Zero Offset**:
  - Normalizes data to start from zero for easier interpretation.
  - For example, `X21_Tf_norm = X21_Tf_norm - X21_Tf_norm[0]`.

---

#### **G. Output Metrics**
- Aggregates and computes:
  - **Means and Standard Deviations**: For joint angles, translations, and morphological metrics.
  - **Confidence Intervals (CI)**: 95% CI for mean values across participants.

---

### **3. Visualization**
- Contains commented-out sections for 3D plotting:
  - Visualizes the anatomical points of interest (e.g., femoral epicondyles, tibial condyles, and patellar landmarks).

---

### **4. Applications**
This script is tailored for studies involving:
- **Dynamic joint analysis**: Capturing and interpreting joint kinematics during movement (e.g., gait).
- **Morphological analysis**: Quantifying anatomical alignment and relationships.
- **Biomechanical experiments**: Testing hypotheses about joint mechanics and alignment in conditions like patellar instability or ligament injuries.