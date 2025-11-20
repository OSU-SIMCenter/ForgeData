import pyvista as pv


def pv_pcd_visualization(pcd_list):
    """
    Visualize one or more numpy pointclouds in pyvista, each with a different color.
    Accepts a single pointcloud (Nx3) or a list of pointclouds.
    Adds grid and axis distance labels.
    """
    if not isinstance(pcd_list, list):
        pcd_list = [pcd_list]

    colors = ["blue", "red", "green"]
    pl = pv.Plotter()
    for i, pcd in enumerate(pcd_list):
        mesh = pv.PolyData(pcd)
        color = colors[i % len(colors)]
        pl.add_mesh(mesh, style="points", opacity=1, color=color, point_size=5)
    pl.add_floor()
    pl.show_bounds(xtitle="X", ytitle="Y", ztitle="Z")
    pl.add_axes_at_origin()
    pl.show()
