import bornagain as ba

BEAM_INTENSITY = 1e+13


def get_simulation(detector=None):
    """
    Simulation of the GISAXS experiment
    Source: https://github.com/Photon-AI-Research/PhaseRetrieval/tree/master/OneStepPhasing/GISAXS_Siegen/InverseNet
    """
    if detector is None:
        detector = {
            "y_bins": 512,
            "z_bins": 1024,
            "psize": 0.05,
            "y": 215,
            "z": 413,
            "wavelength":0.14073,
            "alpha_i":0.64
        }
    wl = detector["wavelength"]
    ai = detector["alpha_i"]

    sdd = 1277.0  # Sample-detector-distance in millimeter
    nbins_y = detector["y_bins"]  # Number of pixel in the y-direction of the detector
    nbins_z = detector["z_bins"]  # Number of pixel in the z-direction of the detector
    psize = detector["psize"]  # Dimension (width and height) of one pixel in millimeter
    direct_beam_pixel_y = detector["y"]  # Y-component of the position of the direct beam in pixel
    direct_beam_pixel_z = detector["z"]  # Z-component of the position of the direct beam in pixel

    # create detector
    detector = ba.RectangularDetector(nbins_y, nbins_y * psize, nbins_z, nbins_z * psize)
    detector.setPerpendicularToReflectedBeam(sdd, direct_beam_pixel_y * psize, direct_beam_pixel_z * psize)

    simulation = ba.GISASSimulation()
    simulation.setDetector(detector)
    simulation.setDetectorResolutionFunction(ba.ResolutionFunction2DGaussian(0.02, 0.02))
    simulation.setBeamParameters(wl*ba.nm, ai*ba.deg, 0.0*ba.deg)
    simulation.setBeamIntensity(BEAM_INTENSITY)

    simulation.getOptions().setUseAvgMaterials(True)
    simulation.getOptions().setIncludeSpecular(False)
    background = ba.ConstantBackground(6e+01)
    simulation.setBackground(background)

    return simulation
