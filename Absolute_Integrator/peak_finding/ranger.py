import numpy as np
import scipy.signal
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import white_tophat
from scipy.ndimage.filters import gaussian_filter
import math
from Absolute_Integrator.peak_finding.UI import RoiPoint

# dictionary describing options available to tune this algorithm
options = {
    "best_size":{"purpose":"The estimate of the peak size, in pixels.  If 'auto', attempts to determine automatically.  Otherwise, this should be an integer.",
                 "default":"auto"},
    "refine_positions":{"purpose":"Improve peak location accuracy to sub-pixel accuracy.",
                        "default":False},
    "sensitivity_threshold":{"purpose":"TODO",
                             "default":0.34},
    "start_search":{"purpose":"TODO",
                    "default":3},
    "end_search":{"purpose":"TODO",
                  "default":"auto"},
    "progress_object":{"purpose":"Object used to present a progress bar to the user.  For definition, see UI_interface folder.",
                       "default":None},
}

def normalise_dynamic_range(image):
    image -= image.min()
    image /= image.max()
    return image


    return image - gaussian_filter(image, filter_width)

def get_data_shape(image):
    """ Returns data shape as (columns, rows).
    Note that this is opposite of standard Numpy/Hyperspy notation.  Presumably,
    this is to help Lewys keep X and Y in order because Matlab uses column-major indexing.
    """
    im_dim = image.shape[::-1]
    m, n = im_dim
    return m, n

def get_trial_size(image,
                  sensitivity_threshold=33,
                  start_search=3,
                  end_search='auto'):

    """ Autotmatically find best-size for feature spacing.

    Best size is found by interatively trying every possible trial_size and
    calculating how many features this results in. For a crystalline image there
    will be a plateau in the graph of trial_size vs number of features detected
    the plateau loaction is calculated and returned as the best_size.

    Parameters
    ----------
    image:
    sensitivity_threshold:
    start_search:
    end_search: {'Auto'}

    Returns
    -------
    best_size: int
    For use in the full peak finding routine."""

    big = get_end_search(image, end_search)
    k = np.zeros(1, (int(round(big-3/2)) +1))
    size_axis = range(3, big, 2)
    for trialSize in range(3, len(k), 2):
        peaks = feature_find(image, best_size=trialSize)
        total_features = len(peaks[0])
        k[(trialSize-1)/2] = total_features
        k_diff = gradient(k, 2)
        print(k_diff)

        if trialSize >= 15: #value of 15 gives minimum 5 unique points for quartic fitting
            fittedGradient = np.polyfit(size_axis[1:((trialSize-5)/2)],
                                        math.log(-1*(k_diff[1:(trialSize-5)/2])),
                                        4)
            poly = np.polyval(fittedGradient, size_axis[1:((trialSize-5)/2)])
            (gradientLock, Index) = min((v, i) for i, v in enumerate(poly))
            #Some conditional statements to end the search.
            if (math.log(-k_diff[0][int((trialSize-5)/2)]) > gradientLock*np.log(10)
                    and (trialSize-1)/2 > 1.25*Index):
                print('Optimum feature spacing determined at',
                       size_axis[0][Index],
                       'px. Total number of atoms is',
                       k[Index])
                break

        elif trialSize >= 13:#value of 13 gives minimum 4 unique points for cubic fitting
            fittedGradient = np.polyfit(size_axis[1:((trialSize-5)/2)],
                                        math.log(-1*(k_diff[1:(trialSize-5)/2])),
                                        3)
            poly = np.polyval(fittedGradient, size_axis[1:((trialSize-5)/2)])
            (gradientLock, Index) = min((v, i) for i, v in enumerate(poly))
            #Some conditional statements to end the search.
            if (math.log(-k_diff[0][int((trialSize-5)/2)]) > gradientLock*np.log(10)
                    and (trialSize-1)/2 > 1.25*Index):
                    print('Optimum feature spacing determined at',
                        size_axis[Index],
                        'px. Total number of atoms is',
                        k[Index])
                    break

        elif trialSize >= 11:#value of 11 gives minimum 3 unique points for quadratic fitting
            fittedGradient = np.polyfit(size_axis[1:((trialSize-5)/2)],
                                        math.log(-1*(k_diff[1:(trialSize-5)/2])),
                                        2)
            poly = np.polyval(fittedGradient, size_axis[1:((trialSize-5)/2)])
            (gradientLock, Index) = min((v, i) for i, v in enumerate(poly))
            #Some conditional statements to end the search.
            if (math.log(-k_diff[0][int((trialSize-5)/2)]) > gradientLock*np.log(10)
                    and (trialSize-1)/2 > 1.25*Index):
                    print('Optimum feature spacing determined at',
                        size_axis[Index],
                        'px. Total number of atoms is',
                        k[Index])
                    break

        else:
                #This 'else' statement allows a gradient based match to be
                #calculated when insufficient unique points exist for quadratic
                #fitting method above, but does not have the ability to teminate
                #the loop.
            Index = n.index(max(k_diff))

    best_size = size_axis[Index]
    if best_size == big:
        warnings.warn('Upper test-box size limit reached. Considering \
        increasing upper limit and running again.')

    return best_size

def get_end_search(image, end_search="auto"):
    im_dim = image.shape
    if end_search== "auto":
        return 2 * np.floor(( float(np.min(im_dim)) / 8) / 2) - 1
    else:
        return end_search

def fit_block(block, base_axis):
    A = np.vstack([base_axis**2 , base_axis , np.ones(base_axis.size)]).T
    h_profile = np.sum(block, axis=0)
    v_profile = np.sum(block, axis=1)
    log1 = np.log(h_profile)
    log2 = np.log(v_profile)

    if len(A) != len(log1):
        log1a = log1[:(len(A))]
        Aa = A[:(len(log1))]
        solution_h = np.linalg.lstsq(Aa, log1a)[0]
    else:
        solution_h = np.linalg.lstsq(A, log1)[0]

    if len(A) != len(log1):
        log2a = log2[:(len(A))]
        Aa = A[:(len(log2))]
        solution_v = np.linalg.lstsq(Aa, log2a)[0]
    else:
        solution_v = np.linalg.lstsq(A, log2)[0]

    y = -solution_v[1]/solution_v[0]/2.0
    x = -solution_h[1]/solution_h[0]/2.0
    height = ( h_profile.max() + v_profile.max() ) / 2.0
    spread = np.sqrt((np.abs(solution_h[0])+np.abs(solution_v[0])) / 4.0)

    return y, x, height, spread

# Feature identification section:
def filter_peaks(normalized_heights, spread, offset_radii, trial_size, sensitivity_threshold):

    # Normalise distances and heights:
    normalized_heights[normalized_heights < 0] = 0  # Forbid negative (concave) Gaussians.
    spread /= trial_size
    spread[spread > math.sqrt(2)] = math.sqrt(2)
    spread[spread == 0] = math.sqrt(2)
    offset_radii = offset_radii / trial_size
    offset_radii[offset_radii == 0] = 0.001  # Remove zeros values to prevent division error later.

    # Create search metric and screen impossible peaks:
    search_record = normalized_heights / offset_radii
    search_record /= 100.0
    search_record[search_record > 1] = 1
    search_record[spread < 0.5] = 0       # Invalidates negative Gaussian widths.
    search_record[spread > 1] = 0          # Invalidates Gaussian widths greater than a feature spacing.
    search_record[offset_radii > 1] = 0    # Invalidates Gaussian widths greater than a feature spacing.
    kernel = int(np.round(trial_size/3))
    if kernel % 2 == 0:
        kernel += 1
    search_record = scipy.signal.medfilt2d(search_record, kernel)  # Median filter to strip impossibly local false-positive features.
    search_record[search_record < sensitivity_threshold ] = 0   # Collapse improbable features to zero likelyhood.
    search_record[search_record >= sensitivity_threshold ] = 1  # Round likelyhood of genuine features to unity.

    # Erode regions of likely features down to points.
    search_record = binary_erosion(search_record, iterations=-1 )
    y, x = np.where(search_record==1)
    return np.vstack((y,x)).T  # Extract the locations of the identified features.


def feature_find(image,
              best_size,
              sensitivity_threshold=33,
              start_search=3,
              end_search="auto",
              progress_object=None):
    """
    A one-line summary needed to explain what function does.

    Several sentances providing extended description.

    Parameters
    ----------
    image: np.array
    Peak_find assumes a dark-field image where features are white and
    background is black.
    *Note: If you wish to use this function on bright-field images simply
    invert the image before using the function.
    best_size :  int
    An odd integer 3 or larger which is smaller than the width of the image.
    If this is unknown the get_trial_size() function needs to be run to
    determine the best feature spacing.
    sensitivity_threshold :
    start_search :
    end_search :
    progress_object :

    Returns
    -------
    list: x, y coordinates of peak location.

    Examples
    --------

    """

    # Removes slowly varying background from image to simplify Gaussian fitting.
    input_offset = white_tophat(image, 2*trial_size)
    # image dimension sizes, used for loop through image pixels
    m, n = get_data_shape(image)
    #print (m, n)
    big = get_end_search(image, end_search)

    # Create blank arrays.
    heights        = np.empty(image.shape, dtype=np.float32)
    spreads         = np.empty(image.shape, dtype=np.float32)
    xs              = np.empty(image.shape, dtype=np.float32)
    ys              = np.empty(image.shape, dtype=np.float32)


    # Half of the trial size, equivalent to the border that will not be inspected.
    test_box_padding = int(( trial_size - 1 ) / 2.)

    # Coordinate set for X and Y fitting.
    base_axis = np.arange(-test_box_padding, test_box_padding+1., dtype=np.float32)
    # Followed by the restoration progress bar:
    if progress_object is not None:
        progress_object.set_title("Identifying Image Peaks...")
        progress_object.set_position(0)

    for i in range( test_box_padding, n - ( test_box_padding)):
        currentStrip = input_offset[ i - test_box_padding : i + (test_box_padding + 1)]
        for j in range(test_box_padding + 1, m - ( test_box_padding + 1)):
            I = currentStrip[:, j - test_box_padding : j + test_box_padding + 1]
            y, x, height, spread = fit_block(I, base_axis)
            ys[i, j] = y
            xs[i, j] = x
            heights[i, j] = height
            spreads[i, j] = spread

            if progress_object is not None:
                percentage_refined = (((trial_size-3.)/2.) / ((big-1.)/2.)) +  (((i-test_box_padding) / (m - 2*test_box_padding)) / (((big-1)/2)))  # Progress metric when using a looping peak-finding waitbar.
                progress_object.set_position(percentage_refined)
    # normalize peak heights
    heights = heights / ( np.max(input_offset) - np.min(input_offset) )
    # normalize fitted Gaussian widths
    spreads = spreads / trial_size
    offset_radii = np.sqrt(ys**2 + xs**2)  # Calculate offset radii.

    return filter_peaks(heights, spreads, offset_radii, trial_size, sensitivity_threshold)

def peak_find(image,
                best_size="auto",
                refine_positions=False,
                sensitivity_threshold=33,
                start_search=3,
                end_search="auto",
                progress_object=None):
    """
    Full wrapper ranger function which estimates best_size for feature spacings,
    carryings out peak-finding, peak refinement and allows addition or removal
    of stray peaks.

    Parameters
    ----------
    image: np.array
    Peak_find assumes a dark-field image where features are white and
    background is black.
    *Note: If you wish to use this function on bright-field images simply
    invert the image before using the function.
    best_size :  {Auto} int
    An odd integer 3 or larger which is smaller than the width of the image.
    If this is unknown the get_trial_size() function needs to be run to
    determine the best feature spacing.
    refine_position : bool
        ddf
    sensitivity_threshold :
    start_search :
    end_search :
    progress_object :

    Returns
    -------
    list: x, y coordinates of peak location.

    Examples
    --------

    """
    #run through trial_size function to estimate feature spacing.
    if best_size =="auto":
        best_size = get_trial_size(image,
                                    sensitivity_threshold,
                                    start_search,
                                    end_search)

    #find peaks to integer pixel accuracy.
    peaks = feature_find(image, best_size,
                        sensitivity_threshold, start_search,
                        end_search, progress_object)

    #user interaction to add missing peaks or remove excess peaks.

    #if refine_positions is used the sub-pixel refinement routine is performed.
    if refine_positions == False:
        return peaks
    else:
        return peaks
