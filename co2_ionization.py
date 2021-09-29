#%%
"""
1. Reads CO2 beam profile data from BeamGage data files and calibrates intensity profiles to given beam energy and pulse length
2. Analyzes beam spot to calculate spot size and Rayleigh length
3. Uses beam parameters to calculate theoretical intensity profile along the propagation axis
4. Calculates the ionization contour at each point in the rz-plane for Helium using ADK ionization rates
"""
from dataclasses import InitVar, dataclass, field

import h5py as h5
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
import astropy.constants as const
from astropy.constants import c, m_e, e, hbar, alpha, eps0
e = e.to(u.C)
import spot_analysis
#import laserbeamsize as lbs
from tqdm import tqdm as progressbar

# set plotting style
plt.style.reload_library()
plt.style.use('ucla_presentations')

@dataclass
class CO2Profile:
    fname: InitVar[str] = None
    energy: float = 1.0 * u.mJ
    pulse_length: float = 1.0 * u.ps
    focus_scale: float = 1.0
    lambda_0: float = 9.6 * u.um
    width: int = None
    height: int = None
    x: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    intensity: np.ndarray = field(repr=False, default=None)


    def __post_init__(self, fname):
        """
        Pull data from BeamGage files (.bgData files were renamed to have .h5 file extension)
        This data structure is documented in the BeamGage User Manual
        """
        f = h5.File(fname, "r")
        raw_data = np.array(f["BG_DATA"]["1"]["DATA"], dtype=float)

        width = int(f["BG_DATA"]["1"]["RAWFRAME"]["WIDTH"][0])
        height = int(f["BG_DATA"]["1"]["RAWFRAME"]["HEIGHT"][0])

        # Get pixel size on camera and scale to pixel size at the focus
        pix_scale_x = f["BG_DATA"]["1"]["RAWFRAME"]["PIXELSCALEXUM"][0] * self.focus_scale * u.um
        pix_scale_y = f["BG_DATA"]["1"]["RAWFRAME"]["PIXELSCALEYUM"][0] * self.focus_scale * u.um
        self.pix_size = pix_scale_x
        pix_area = pix_scale_x * pix_scale_y
        
        # reshape raw data into a 2D array (matrix where each element is a pixel intensity)
        self.raw_image = np.reshape(raw_data, (width, height))

        # Initialize intensity to unscaled raw data
        self.intensity = self.raw_image * u.dimensionless_unscaled

        # Define x and y axes using pixel size
        self.x = np.arange(width) * self.pix_size
        self.y = np.arange(height) * self.pix_size

        # Save the image size
        self.width = width
        self.height = height


    def calibrate_intensity(self, intensity_scale = None):
        """
        Set the intensity scale for the beam profile, default normalizes to total beam energy
        """
        if intensity_scale is None:
            intensity_scale = (self.energy / self.pulse_length / self.pix_size**2).to(u.W / u.cm**2)
        # normalize raw data to sum to 1
        norm = np.sum(self.raw_image)
        norm_data = self.raw_image/norm

        # rescale normalized data to given beam energy
        self.intensity = norm_data * intensity_scale

    def analyze_spot(self):
        spot_analysis.spot_plot(self.intensity.value, pix_size = self.pix_size)

        x, y, r_x, r_y, phi = spot_analysis.beam_size(self.intensity.value, pix_size = self.pix_size)

        self.centroid_x = x
        self.centroid_y = y

        self.w_0 = (r_x + r_y)/2
        self.z_0 = self.w_0**2 * np.pi / self.lambda_0.value

        print(f"The spot size is {self.w_0:0.0f}")


    def w(self, z):
        """ Beam spot size as a function of distance from focus """
        w = self.w_0 * np.sqrt(1+(z/self.z_0)**2)
        return w

    def long_intensity(self, r, z):
        """Intensity in r,z (longitudinal) plane given by Gaussian beam propagation
        r: distance from optical axis
        z: distance from beam focus
        """
        I = (self.w_0 / self.w(z) * np.exp(- r**2/ self.w(z)**2))**2
        return I

    def plot_rz(self):
        N = 50
        z = np.linspace(-2*self.z_0, 2*self.z_0, N)
        r = np.linspace(-2*self.w_0, 2*self.w_0, N)

        X, Y = np.meshgrid(r, z)

        I = self.long_intensity(X,Y)
        norm = np.max(I)
        norm_I = I/norm
        # rescale normalized data to given beam energy
        intensity_factor = np.max(self.intensity)
        self.prop_I = norm_I * intensity_factor

        fig, ax = plt.subplots()

        colors = ax.contourf(X,Y, self.prop_I, cmap="plasma")

        cbar = fig.colorbar(colors, ax=ax)
        cbar.set_label("Intensity (W/cm$^2$)")
        ax.set_xlabel("r (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title("Theoretical Intensity Along Propagation")

    def image(self):
        """
        Display original image loaded from data file
        """
        fig, ax = plt.subplots()
        img = ax.imshow(self.raw_image, cmap='plasma')
        #cbar = fig.colorbar(colors, ax=ax)
        #cbar.set_label("Intensity (W/cm$^2$)")
        ax.set_xlabel("Position (pixels)")
        ax.set_ylabel("Position (pixels)")
        ax.set_title("Raw CO2 Intensity Profile")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)

    def plot(self):
        #lbs.beam_size_plot(self.intensity.value, pixel_size= 0.08, units='mm')
        spot_analysis.spot_plot(self.intensity.value, pix_size = self.pix_size)

        #fig, ax = plt.subplots()
        #X, Y = np.meshgrid(self.x.to(u.mm).value, self.y.to(u.mm).value)

        #colors = ax.pcolormesh(X,Y, self.intensity, cmap="plasma")

        #cbar = fig.colorbar(colors, ax=ax)
        #cbar.set_label("Intensity (W/cm$^2$)")
        #ax.set_xlabel("x (mm)")
        #ax.set_ylabel("y (mm)")
        #ax.set_title("Measured CO2 Intensity Profile")


def ADKRate(Z, E_I, I, polarization="linear"):
    """Ammosov-Delone-Krainov ionization rate.
    Z: charge state of the resulting ion
    E_I: ionization potential 
    I: Intensity
    polarization: laser polarization in ['linear' (default), 'circular']
    
    returns: ionization rate [unit: 1/s]
    """
    pol = polarization
    if pol not in ["linear", "circular"]:
        raise NotImplementedError(
            "Cannot interpret polarization='{}'.\n".format(pol) +
            "So far, the only implemented options are: " +
            "['linear', 'circular']"
            )

    # Normalize values to unscaled atomic units:
    E_scale = (m_e**2 * e**5 / (hbar**4 * (4 * np.pi * eps0)**3)).to(u.V/u.m)
    I_scale =  (m_e**4 / (8 * np.pi * alpha * hbar**9) * e**12 / (4 * np.pi * eps0)**6).to(u.W/ u.cm**2)
    Energy_scale = (m_e * e**4 / (hbar**2 * (4 * np.pi * eps0)**2)).to(u.eV)
    t_scale = (hbar**3 * (4 * np.pi * eps0)**2 / m_e / e**4).to(u.s)


    I_norm = ((I / I_scale).to(u.dimensionless_unscaled)).value
    I_norm = np.where(I_norm<0, 0, I_norm)

    E_I_norm = (E_I / Energy_scale).to(u.dimensionless_unscaled).value
    F = np.sqrt(I_norm) # dimensionless field strength


    # Calculate effective priinciple quantum number
    nEff = Z / np.sqrt(2. * E_I_norm)

    D = ((4. * Z**3.) / (F * nEff**4.))**nEff
    rate = (F * D**2.) / (8. * np.pi * Z) \
        * np.exp(-(2. * Z**3.) / (3. * nEff**3. * F))

    if pol == 'linear':
        rate = rate* np.sqrt((3. * nEff**3. * F) / (np.pi * Z**3.))

    # set nan values due to near-zero field strengths to zero
    rate = (np.nan_to_num(rate)/t_scale).to(1/u.s)

    return rate

if __name__ == "__main__":

    directory = "/home/oods/Research/AE98-99/CO2 Profiles/AE98/"
    fname = directory + "2021-07-30 001 regen focus.h5"
    #fname = directory + "2021-08-05 007 regen focus.h5"

    beam = CO2Profile(fname, energy = 200 * u.mJ, pulse_length = 2 * u.ps)
    np.savetxt(directory+'CO2_profile_data.txt', beam.intensity.value)
    beam.calibrate_intensity()
    beam.analyze_spot()
    #beam.plot_rz()
    #plt.show()

    h, w = beam.intensity.shape
    x_idx = int(.25 * w)
    y_idx = int(.25 * h)
    intensity_cropped = beam.intensity[y_idx:h - y_idx, x_idx:w - x_idx]

    # Helium charge states and ionization energies
    Z_max = 1 #proton number for Hydrogen
    Z = np.arange(1, Z_max + 1, dtype=int)
    ionization_energies = np.array([13.6]) * u.eV

    intensity_fwhm = 2.e-12  # s
    intensity_sigma = intensity_fwhm / (2. * np.sqrt(2. * np.log(2)))  # s

    N_x = len(intensity_cropped[0])
    N_y = len(intensity_cropped)

    t_res = 3
    time = np.linspace(-2e-12, 2e-12, t_res)
    pulse_shape = np.exp(- .5 * time**2 / intensity_sigma**2)

    state_populations = np.zeros([Z_max+1, t_res+1, N_y,N_x])

    for i in progressbar(range(len(intensity_cropped))):
        for j in range(len(intensity_cropped[0])):
            I = intensity_cropped[i][j]
                    
            rate_matrix = np.zeros([len(ionization_energies), t_res])



            for k, cs in enumerate(Z):
                rate_matrix[k,:] = ADKRate(cs, ionization_energies[k], I * pulse_shape)

            # transition matrix
            trans_mat_base = np.diag(np.ones([Z_max + 1]))
            trans_mat_before = trans_mat_base
            # preparation of the transition matrix: Markov absorbing state CS = 10
            trans_mat_base[Z_max, Z_max] = 1

            # prepare initial state
            initState = np.zeros([Z_max + 1])
            # all atoms are initially unionized
            initState[0] = 1

            # prepare expected charge distribution array
            charge_dist = np.zeros([Z_max + 1, t_res + 1])
            # manipulate last entry for loop
            charge_dist[:, -1] = initState

            # time step of the Markov process
            dt = (time[-1] - time[0]) / t_res

            # loop over steps
            for t in np.arange(t_res):
                # calculate the transition matrix of this step
                trans_mat = trans_mat_base
                for k, cs in enumerate(Z):
                    # probability to stay bound
                    trans_mat[k, k] = np.exp(-rate_matrix[k, t] * dt)
                    # probability to ionize
                    trans_mat[k + 1, k] = 1. - np.exp(-rate_matrix[k, t] * dt)

                # Markov step
                charge_dist[:, t] = np.dot(charge_dist[:, t - 1], trans_mat.T)
            state_populations[:,:,i,j] = charge_dist

    #%%

    # clip populations at 100% ionization
    population_percent = np.where(state_populations>1, 1,state_populations) * 100

    fig, ax = plt.subplots()
    x = np.arange(N_x)
    y = np.arange(N_y)
    X, Y = np.meshgrid(x,y)
    ##ax.plot(time, state_populations[1,:-1,50,50])
    #colors = ax.contourf(X, Y, population_percent[0,-2,:,:], cmap="plasma")

    #cbar = fig.colorbar(colors, ax=ax)
    #cbar.set_label("Ion Population (\%)")
    #ax.set_xlabel("x (mm)")
    #ax.set_ylabel("y (mm)")
    #ax.set_title("H Population After Pulse")
    #plt.show()

    # Make color axis of intensity
    #ax.plot(time, state_populations[1,:-1,50,50])
    colors = ax.contourf(X, Y, population_percent[1,-2,:,:], cmap="plasma")

    cbar = fig.colorbar(colors, ax=ax)
    cbar.set_label("Ion Population (\%)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("H+ Population After Pulse")


    plt.show()
    plt.close()
