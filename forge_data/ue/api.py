from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forge_data.ue.core import Recon


def mesh_from_dataframe(df: pd.DataFrame, args=None) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    recon = Recon(ReconConfig(error_comp=False))
    recon.preprocess(df)
    recon.process()
    pcd = recon.post_process()
    mesh = recon.pcd_to_mesh(pcd)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    return vertices, faces


@dataclass
class ReconConfig:
    default_axis_angle_vector: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 286.8])
    default_optimization_bounds: tuple[tuple[float, float], ...] = ((0.8, 1), (-0.05, 0.05), (-0.05, 0.05), (280, 290))
    error_comp: bool = False
