import pytest
import numpy as np
import yabplot as yab
import pyvista as pv

# tell PyVista to run in "off-screen" mode so it doesn't try to open a real window
pv.OFF_SCREEN = True

def test_version():
    """Check that the package has a version string."""
    assert yab.__version__ is not None

def test_none_returns_empty_dict():
    """
    Unit test: Verify that passing None disables the background mesh 
    by correctly returning an empty dictionary.
    """
    result = yab.mesh.load_bmesh(None)
    assert result == {}

def test_dict_passthrough():
    """
    Unit test: Verify that custom dictionary keys for hemispheres 
    are properly sanitized to strict 'L' and 'R' keys.
    """
    mesh_l = pv.Sphere()
    mesh_r = pv.Cube()
    mesh_other = pv.Cone()
    d = {'left': mesh_l, 'RIGHT': mesh_r, 'other': mesh_other}
    result = yab.mesh.load_bmesh(d)
    expected = {'L': mesh_l, 'R': mesh_r, 'other': mesh_other}
    assert result == expected

def test_polydata_wrapped_in_both():
    """
    Unit test: Verify that passing a single PyVista mesh (whole brain) 
    safely wraps it in a dictionary with the 'both' key.
    """
    mesh = pv.Sphere()
    result = yab.mesh.load_bmesh(mesh)
    assert 'both' in result
    assert result['both'] is mesh

def test_plotter_instantiation():
    """
    Smoke test: Can we create a Plotter without crashing?
    This verifies VTK and PyVista are correctly linked to the system display.
    """
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show()
    plotter.close()

def test_plot_cortical():
    """
    Integration test: Downloads 'aparc' and plots it.
    """
    yab.plot_cortical(atlas='aparc', display_type=None)

def test_plot_subcortical():
    """
    Integration test: Downloads 'aseg' and plots it.
    """
    yab.plot_subcortical(atlas='aseg', display_type=None)

def test_plot_tracts():
    """
    Integration test: Downloads 'xtract_tiny' and plots it.
    """
    yab.plot_tracts(atlas='xtract_tiny', display_type=None)

def test_plot_vertexwise():
    """
    Integration test: plot_vertexwise with synthetic sphere meshes.
    """
    lh = pv.Sphere()
    rh = pv.Sphere()
    lh['Data'] = np.random.rand(lh.n_points)
    rh['Data'] = np.random.rand(rh.n_points)
    yab.plot_vertexwise(lh, rh, display_type=None)
