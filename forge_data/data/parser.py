import numpy as np
import open3d as o3d
import pandas as pd


def process_linescanner_file(file):
    """
    Load raw linescanner file & process into pandas dataframe.
    """

    file_version = get_linescanner_file_version(file)

    if file_version == "ue-csv-0.1.0":
        df = parse_ue_csv_0_1_0(file)
    elif file_version == "ue-csv-0.1.1":
        df = parse_ue_csv_0_1_1(file)

    return df


def process_load_stroke_file(file):
    """
    TODO Docstring
    """

    file_version = get_load_stroke_file_version(file)

    if file_version == "LS-csv-0.1.0" or file_version == "LS-csv-0.1.1":
        df = parse_ls_csv_0_1_0(file)
    else:
        raise ValueError(f"Unsuppoted file version detected in file: {file}")

    return df


def process_obj_file(file):
    """
    Process an OBJ file and save its mesh data to the databases.
    """
    mesh = o3d.io.read_triangle_mesh(str(file))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if faces is None:
        print(f"Warning: No triangular faces found in {file}. Skipping.")
        return

    return vertices, faces


def get_folder_structure_version(path):
    """
    This seems to change often, determine the layout of raw files for parsing.
    """
    pass


def get_linescanner_file_version(file):
    """
    Determine which version / format the csv is so we know what dataframe ops to do.
    """

    if not str(file).endswith(".csv"):
        return None

    try:
        with open(file) as f:
            line1 = f.readline()
            line2 = f.readline()

        if not line1 or not line2:
            return None

        row0 = line1.strip().split(",")
        row1 = line2.strip().split(",")

        if (
            row0[0] == "Part Temperature (C)"
            and row0[1] == "Time Unix (ms):"
            and row1[1] == "A Axis Angle (deg):"
            and row0[3] == "X values (mm):"
            and row1[3] == "Z values (mm):"
        ):
            return "ue-csv-0.1.0"
        elif (
            row0[0] == "Scan offset X:"
            and row0[2] == "Part Temperature (C)"
            and row0[4] == "Time Unix (ms):"
            and row0[6] == "X values (mm):"
            and row1[0] == "A Axis Angle (deg):"
            and row1[2] == "Z values (mm):"
        ):
            return "ue-csv-0.1.1"

    except (OSError, UnicodeDecodeError):
        return None

    return None


def get_load_stroke_file_version(file):
    """
    Peeks at CSV header to determine version/format
    """

    EXPECTED_HEADERS_V0_1_0 = [
        "Time Unix (ms)",
        "Position (mm)",
        "Force (kN)",
        "Live Velocity (mm/s)",
        "Thermal Cam Temp (c)",
        "Pyrometer Temp (c)",
        "Press Target Position (mm)",
        "Press Target Velocity (mm/s)",
        "X",
        "Y",
        "Z",
        "A",
        "Ram Side Tool Number",
        "Static Side Tool Number",
        "Hit Number",
    ]
    EXPECTED_HEADERS_V0_1_1 = [
        "Time Unix (ms)",
        "Position (mm)",
        "Force (kN)",
        "Live Velocity (mm/s)",
        "Thermal Cam Temp (c)",
        "Pyrometer Temp (c)",
        "Press Target Position (mm)",
        "Press Target Velocity (mm/s)",
        "X",
        "Y",
        "Z",
        "A",
        "X pos referenced to target part butt",
        "Ram Side Tool Number",
        "Static Side Tool Number",
        "Hit Number",
    ]
    if not str(file).endswith(".csv"):
        return None

    try:
        with open(file) as f:
            first_line = f.readline()

        headers = first_line.strip().split(",")

        # 3. Compare the lists
        if headers == EXPECTED_HEADERS_V0_1_0:
            return "LS-csv-0.1.0"
        elif headers == EXPECTED_HEADERS_V0_1_1:
            return "LS-csv-0.1.1"

    except (OSError, UnicodeDecodeError):
        return None

    return None


def parse_ue_csv_0_1_0(file):
    df = pd.read_csv(file, header=None)

    # Assemble into data format
    df_even = df.iloc[::2].reset_index(drop=True)
    df_odd = df.iloc[1::2].reset_index(drop=True)
    timestamps = df_even.iloc[:, 2]
    temps = df_odd.iloc[:, 0]
    x_values = df_even.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)
    z_values = df_odd.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)
    a_axis_angles = df_odd.iloc[:, 2]

    df = pd.DataFrame({
        "timestamps_ms": timestamps,
        "temperature_C": temps,
        "x_mm": x_values,
        "z_mm": z_values,
        "a_axis_deg": a_axis_angles,
    })

    return df


def parse_ue_csv_0_1_1(file):
    df = pd.read_csv(file, header=None)

    # Assemble into data format
    df_even = df.iloc[::2].reset_index(drop=True)
    df_odd = df.iloc[1::2].reset_index(drop=True)
    timestamps = df_even.iloc[:, 5]
    temps = df_even.iloc[:, 3]
    x_values = df_even.iloc[:, 7:].iloc[:, :1280].astype(float).apply(lambda row: row.tolist(), axis=1)
    z_values = df_odd.iloc[:, 3:].iloc[:, :1280].astype(float).apply(lambda row: row.tolist(), axis=1)
    a_axis_angles = df_odd.iloc[:, 1]
    scan_offset_x = df_even.iloc[:, 1]

    df = pd.DataFrame({
        "timestamps_ms": timestamps,
        "temperature_C": temps,
        "x_mm": x_values,
        "z_mm": z_values,
        "a_axis_deg": a_axis_angles,
        "scan_offset_x": scan_offset_x,
    })
    # print(df)
    return df


def parse_ls_csv_0_1_0(file):
    # The first row is the header
    df = pd.read_csv(file, header=0)
    return df
