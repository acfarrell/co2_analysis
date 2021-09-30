import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors
import astropy.units as u
import cv2

# set plotting style
plt.style.reload_library()
plt.style.use("ucla_presentations")


def beam_size(beam, pix_size=None):
    """
    Calculates the beam spot parameters of a laser spot
    from the 2D intensity data

    Arguments:
        beam: np.ndarray, 2D array holding data for each value
        pix_size: size of each data pixel using units from astropy.units
                  default is 80 micron
    Returns:
        centroid_x: x-position of the beam center in units of pix_size
        centroid_y: y-position of the beam center
        dx: width of spot in x
        dy: width of spot in y
        phi: ellipse tilt from horizontal axis in radians
    """
    # Set maximum number of iterations before quitting
    max_iter = 25

    # Remove the background
    beam_clean, _ = remove_corner_background(beam, corner_ratio=.1)
    # Get initial beam parameters using entire image
    x_center, y_center, r_x, r_y, phi = calculate_parameters(beam_clean)

    # iterate to improve fits
    for _iteration in range(1,max_iter):
        # Save the last value of each parameter
        old_x, old_y, old_r_x, old_r_y = x_center, y_center, r_x, r_y
        # Set a new rectangular region to analyze within
        beam_masked = better_box(beam_clean, x_center, y_center, r_x, r_y, phi)

        # Recalculate the beam parameters in this region
        x_center, y_center, r_x, r_y, phi = calculate_parameters(beam_masked)

        # Check if we're close enough to the last estimate
        if (
            abs(x_center - old_x) < 1
            and abs(y_center - old_y) < 1
            and abs(r_x - old_r_x) < 1
            and abs(r_y - old_r_y) < 1
        ):
            break
        
        
    # establish scale and correct label
    if pix_size is None:
        scale = 1 * u.dimensionless_unscaled
    else:
        scale = pix_size

    x_center *= scale
    y_center *= scale
    r_x *= scale
    r_y *= scale
    phi = phi.to(u.deg)
    return x_center, y_center, r_x, r_y, phi


def calculate_parameters(beam):
    """
    Determines the centroid position, widths, and tilt of the beam
    These are the definitions of each parameter, beam_size then optimizes
    by applying masks to the image to filter out noise

    Arguments:
        beam: np.ndarray, 2D array holding data for each value

    Returns:
        centroid_x: x-position of the beam center in units of pix_size
        centroid_y: y-position of the beam center
        dx: width of spot in x
        dy: width of spot in y
        phi: ellipse tilt from horizontal axis in radians
    """
    # get the size of the image
    h, w = beam.shape

    tot = np.sum(beam, dtype=np.float)  # sum of all values in the image

    # Make pseudo axes for the image
    x = np.arange(w, dtype=np.float)
    y = np.arange(h, dtype=np.float)

    # Calculate centroid positions
    x_center = np.sum(np.dot(beam, x)) / tot
    y_center = np.sum(np.dot(beam.T, y)) / tot

    # Calculate variance (sigma squared) in each direction
    x_res = x - x_center  # residual distance in x from the center
    y_res = y - y_center  # residual distance in y from the center

    x_var = np.sum(np.dot(beam, x_res** 2)) / tot
    y_var = np.sum(np.dot(beam.T, y_res** 2)) / tot
    xy_var = np.dot(np.dot(beam.T, y_res), x_res) / tot
    
    if x_var == y_var:
        discriminant = np.abs(2 * xy_var)
        phi = np.sign(xy_var) * np.pi / 4
    else:
        discriminant = np.sign(x_var - y_var) * np.sqrt((x_var - y_var) ** 2 + 4 * xy_var ** 2)
        phi = np.arctan(2 * xy_var / (x_var - y_var)) / 2.

    # Calculate spot radii
    r_x = np.sqrt(2 * (x_var + y_var + discriminant))
    r_y = np.sqrt(2 * (x_var + y_var - discriminant))

    phi *= -u.rad

    return x_center, y_center, r_x, r_y, phi


def remove_corner_background(beam, corner_ratio=0.05):
    """
    Estimate the background signal in the data
    by looking in the corners of the image and remove it

    Arguments:
        beam: 2D array of intensety data
        corner_ratio: float, definition of the corner size
                      relative to the size of the image

    Returns:
        beam_clean: np.ndarray with corner background removed
    """
    if corner_ratio == 0:
        return 0, 0  # take care of trivial case

    h, w = beam.shape
    h_idx = int(h * corner_ratio)
    w_idx = int(w * corner_ratio)

    # Make a boolean mask of each corner
    mask = np.full_like(beam, False, dtype=np.bool)
    mask[:h_idx, :w_idx] = True
    mask[:h_idx, -w_idx:] = True
    mask[-h_idx:, :w_idx] = True
    mask[-h_idx:, -w_idx:] = True

    # Return mean and std of data only inside the corners
    beam_corners = np.ma.masked_array(beam, ~mask)
    bg_mean = np.mean(beam_corners)
    bg_std = np.std(beam_corners)
    n = 3

    # Calculate background threshold and remove it from data
    threshold = int(bg_mean + n * bg_std)
    beam_clean = np.copy(beam)
    np.place(beam_clean, beam_clean < threshold, threshold)
    beam_clean -= threshold
    return beam_clean, bg_mean

def better_box(beam, x_center, y_center, r_x, r_y, phi, mask_scale=2):
    """
    Define a rotated box around the spot aligned with the ellipse axes, and mask
    the image within this box
    """
    rectangle = np.full_like(beam, 0, dtype=np.float)

    h, w = beam.shape

    box_h = mask_scale * r_y
    box_w = mask_scale * r_x

    top = min(h, int(y_center + box_h))
    bottom = max(0, int(y_center - box_h))
    left = max(0, int(x_center - box_w))
    right = min(w, int(x_center + box_w))

    rectangle[bottom:top, left:right] = 1
    mask = rotate(rectangle, phi, (x_center, y_center))
    #plt.imshow(mask, cmap='plasma')
    #plt.show()
    beam_masked = np.copy(beam)

    # Set all points outside the box to zero    
    beam_masked[mask < 1] = 0

    return beam_masked

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle.to(u.deg).value, scale)
    rotated = cv2.warpAffine(image, rotation_matrix, (w,h))
    return rotated

def rotate_point(x, y, x0=0, y0=0, angle=0):

    x_rot = x0 + np.cos(angle) * (x - x0) - np.sin(angle) * (y - y0)
    y_rot = y0 + np.sin(angle) * (x - x0) + np.cos(angle) * (y - y0)

    return x_rot, y_rot

def ellipse(x_center, y_center, r_x, r_y, phi, npoints=200):
    theta = np.linspace(0, 2*np.pi, npoints)

    a = r_x * np.cos(theta)
    b = r_y * np.sin(theta)

    x_points = x_center + a*np.cos(phi) - b*np.sin(phi)
    y_points = y_center - a*np.sin(phi) - b*np.cos(phi)

    return x_points, y_points

def crosshairs(x_center, y_center, r_x, r_y, phi, npoints=100, box_scale=2):
    a = r_x * box_scale
    b = r_y * box_scale

    a_x = np.array([x_center - a, x_center + a])
    a_y = np.array([y_center for x in a_x])
    
    b_y = np.array([y_center - b, y_center + b])
    b_x = np.array([x_center for y in b_y])

    a_x_rot, a_y_rot = rotate_point(a_x, a_y, x_center, y_center, -phi)
    b_x_rot, b_y_rot = rotate_point(b_x, b_y, x_center, y_center, -phi)

    a_line = (a_x_rot, a_y_rot)
    b_line = (b_x_rot, b_y_rot)
    return a_line, b_line

def line_out(beam, points_x, points_y, npoints=100):
    """
    Takes the indices of two points and returns the profile of the image along that line
    """
    x1, x2 = points_x 
    y1, y2 = points_y
    x, y = np.linspace(x1, x2, npoints).astype(int), np.linspace(y1, y2, npoints).astype(int)

    h, w = beam.shape

    # Define an axis along the line centered at its midpoint
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r = np.linspace(-line_length / 2, line_length / 2,npoints)


    # Crop to within image boundaries
    start_idx = max(np.searchsorted(x, 0, side='right'), np.searchsorted(y, 0, side='right'))
    end_idx = min(np.searchsorted(x, w, side='left'), np.searchsorted(y, h, side='left'))

    x_indices = x[start_idx:end_idx]
    y_indices = y[start_idx:end_idx]
    r = r[start_idx:end_idx]


    # Extract the values along the line, using cubic interpolation
    line_out_data = beam[y_indices, x_indices]
    return r, line_out_data


def coordinate_to_index(x,y, extent, scale = 1):
    """Return the pixel center of an index given the extent of the axes"""
    left, right, bottom, top = extent

    x_shift = left
    y_shift = top

    shifted_x = x - x_shift
    shifted_y = y - y_shift

    # undo pixel scaling and convert to integers
    x_idx = (shifted_x / scale).astype(int)
    y_idx = (shifted_y / scale).astype(int)
    
    return x_idx, y_idx

def spot_plot(beam, pix_size = None, cmap='plasma'):
    plt.rc('axes', labelsize=8, titlesize=10)
    plt.rc('legend', fontsize=8, handlelength=1)
    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6)

    x_center, y_center, r_x, r_y, phi = beam_size(beam, pix_size)

    print(f"The center of the beam ellipse is at ({x_center:.0f}, {y_center:.0f})")
    print(f"The ellipse radius (closest to horizontal) is {r_x:.0f}")
    print(f"The ellipse radius (closest to vertical) is {r_y:.0f}")
    print(f"The ellipse is rotated {phi:.0f} ccw from horizontal")

    if pix_size is None:
        scale = 1
        unit = "pixels"
        label = "Distance from Center (pixels)"
    else:
        scale = pix_size.value
        unit = pix_size.unit
        label = "Distance from Center (%s)" % unit
    

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8,8))
    # original image
    img1 = ax1.imshow(beam, cmap=cmap, vmin=0, vmax = np.max(beam))
    ax1.set_xlabel('Position (pixels)')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title("Original Image")
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1)

    h, w = beam.shape * pix_size

    beam_clean, bg_mean = remove_corner_background(beam)
    extent = np.array([-x_center.value, (w - x_center).value, (h - y_center).value, -y_center.value])

    # Scaled image with background removed
    img2 = ax2.imshow(beam_clean, extent = extent, cmap=cmap, vmin=0, vmax = np.max(beam))
    ax2.set_xlim(ax2.get_xlim())
    ax2.set_ylim(ax2.get_ylim())
    ax2.set_xlabel(label)
    ax2.set_ylabel(label)
    ax2.set_title("Spot Check")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax2)

    # Draw bounding ellipse
    spot_bounds_x, spot_bounds_y = ellipse(0, 0, r_x, r_y,phi)
    ax2.plot(spot_bounds_x, spot_bounds_y, ':', color="white", lw=1)

    # Draw semimajor and minor axes
    (a_x, a_y), (b_x, b_y) = crosshairs(0, 0, r_x.value, r_y.value,phi)
    ax2.plot(a_x, a_y, 'o:', color="white", lw=1, markersize = 1)
    ax2.plot(b_x, b_y, 'o:', color="white", lw=1, markersize = 1)
    
    # Draw lineout of intensity along each axis

    # Determine which axis is semimajor and semiminor
    if r_x >= r_y:
        a_ax = ax3
        b_ax = ax4
    else:
        b_ax = ax3
        a_ax = ax4

    # Get profiles
    
    r_a, lineout_a = line_out(beam, *coordinate_to_index(a_x, a_y, extent, scale) )
    r_b, lineout_b = line_out(beam, *coordinate_to_index(b_x, b_y, extent, scale) )

    r_a *= scale
    r_b *= scale
    
    # Calculate Gaussian pulse curves
    sig_a = r_x.value
    A = np.sqrt(2 / np.pi) / sig_a * np.sum(lineout_a - bg_mean) * abs(r_a[1] - r_a[0])
    gaussian_a = A*np.exp(-2 * (r_a/sig_a)**2) + bg_mean
    baseline_a = A* np.exp(-2) + bg_mean


    sig_b = r_y.value
    B = np.sqrt(2 / np.pi) / sig_b * np.sum(lineout_b - bg_mean) * abs(r_b[1] - r_b[0])
    gaussian_b = B*np.exp(-2 * (r_b/sig_b)**2) + bg_mean
    baseline_b = B* np.exp(-2) + bg_mean

    style = dict(size=8, ha='left', va='top')
    arrows =  {'arrowstyle': '<->', 'color':'grey', 'lw':0.5, 'shrinkA':0.05, 'shrinkB':0.05}

    # make sure each profile shares the same y-scale    
    max_I = max(max(max(gaussian_a), max(gaussian_b)), max(max(lineout_a), max(lineout_b)))
    padding = 0.025 * max_I
    a_ax.set_ylim(-padding, max_I + padding)
    b_ax.set_ylim(-padding, max_I + padding)

    a_ax.plot(r_a, gaussian_a,':', label="Gaussian")
    a_ax.annotate('', (0, baseline_a), (sig_a, baseline_a),
            arrowprops=arrows, **style)
    a_ax.text(0.05 * sig_a, 0.75 * baseline_a, f'r$_x$ = {r_x:.0f}', **style)
    a_ax.plot(r_a, lineout_a, label="Profile")

    b_ax.plot(r_b, gaussian_b,':', label="Gaussian")
    b_ax.annotate('', (0, baseline_b), (sig_b, baseline_b),
            arrowprops=arrows, **style)
    b_ax.text(0.05 * sig_b, 0.8 * baseline_b, f'r$_y$ = {r_y:.0f}', **style)
    b_ax.plot(r_b, lineout_b, label="Profile")

    ax3.axvline(color="grey",lw=.5, ls='--')
    ax3.set_title("Semimajor Profile")
    ax3.set_ylabel("Pixel Intensity Along Axis")
    ax3.set_xlabel(label)
    ax3.legend()


    ax4.axvline(color="grey",lw=.5, ls='--')
    ax4.set_title("Semiminor Profile")
    ax4.set_ylabel("Pixel Intensity Along Axis")
    ax4.set_xlabel(label)
    ax4.legend()

    plt.show()

if __name__ == "__main__":

    directory = "/home/oods/Research/AE98-99/CO2 Profiles/AE98/"
    fname = directory + "CO2_profile_data.txt"

    test = np.loadtxt(fname)

    spot_plot(test, pix_size = 2 * u.um)

   
