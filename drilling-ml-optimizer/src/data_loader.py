import pandas as pd


def load_data(drilling_path, material_path):
    """
    Load and merge drilling + material datasets
    """

    drilling_df = pd.read_csv(drilling_path)
    material_df = pd.read_csv(material_path)

    merged_df = drilling_df.merge(material_df, on="Material")

    return merged_df


def load_material_properties(material_path):
    """
    Load material properties separately
    """

    return pd.read_csv(material_path)