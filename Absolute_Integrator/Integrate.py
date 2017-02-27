import numpy as np

def Integrate(img, points, maxRadius='Auto'):
    """Given an image a set of points and a maximum outer radius,
    this function integrates the voronoi cell surround each point.

    Parameters
    ----------
    img: np.array
        assumed to be 2D in the first instance.
    points : list
        Detailed list of the x and y coordinates of each point of
        interest within the image.
    max_radius: {'Auto'} int
        A maximum outer radius for each Voronoi Cell.
        If a pixel exceeds this radius it will not be included in the cell.
        This allows analysis of a surface and particles.
        If 'max_radius' is left as 'Auto' then it will be set to the largest
        dimension in the image.

    Returns
    -------
    integrated_points: list
        A list of integrated intensities the same length as the imput points.
    img: (opptional) np.array
        Intensity record showing where each integrated intensity came from.

    **Note: Should try and make sure this also works with 3D or 4D np.array
    such that spectrum images could be integrated in the same way.

    """
    #Setting max_radius to the width of the image, if none is set.

    if maxRadius='Auto':
        maxRadius = max(img.shape)

    pointRecord = np.zeros_like(img)
    distanceLog = np.zeros_like(points[0])
    inegratedIntensity = np.zeros_like(distanceLog)
    intensityRecord = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #For every pixel the distance to all points must be calculated.
            distance_log = (points[0]-i)^2 + (points[1]-j)^2)^0.5

            # Next for that pixel the minimum distance to and point should be
            # checked and discarded if too large:
            distMin = np.min(distance_log)
            minIndex = np.argmin(distance_log)

            if distMin >= maxRadius:
                pointRecord[i][j] = 0
            else:
                pointRecord[i][j] = minIndex + 1

    for i in range(points.shape[0]):
        currentMask = (pointRecord == i+1)
        currentFeature = curentMask * img
        integratedIntensity[i] = sum(sum(currentFeature))
        intensityRecord += currentFeature

    return (integratedIntensity, intensityRecord)
