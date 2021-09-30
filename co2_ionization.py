#%%
"""
1. Reads CO2 beam profile data from BeamGage data files and calibrates intensity profiles to given beam energy and pulse length
2. Analyzes beam spot to calculate spot size and Rayleigh length
3. Uses beam parameters to calculate theoretical intensity profile along the propagation axis
4. Calculates the ionization contour at each point in the rz-plane for Helium using ADK ionization rates
"""
from dataclasses import InitVar, dataclass, field

import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.constants import m_e, e, hbar, alpha, eps0
e = e.to(u.C)

import spot_analysis

# set plotting style
plt.style.reload_library()
plt.style.use('ucla_presentations')

@dataclass
class CO2Profile:
    fname: InitVar[str] = None
    energy: float = 1.0 * u.mJ
    pulse_length: float = 1.0 * u.ps
    magnification: float = 1.0
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
        pix_scale_x = f["BG_DATA"]["1"]["RAWFRAME"]["PIXELSCALEXUM"][0] / self.magnification * u.um
        pix_scale_y = f["BG_DATA"]["1"]["RAWFRAME"]["PIXELSCALEYUM"][0] / self.magnification * u.um
        self.pix_size = pix_scale_x
        
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

    def remove_background(self, intensity):
        intensity = intensity.to(u.W / u.cm**2)

        intensity_clean, _ = spot_analysis.remove_corner_background(intensity.value)

        return intensity_clean * u.W / u.cm**2

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
        N = 100
        z = np.linspace(-7*self.z_0, 7*self.z_0, N)
        r = np.linspace(-5*self.w_0, 5*self.w_0, N)

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

        ax.set_xlabel("Position (pixels)")
        ax.set_ylabel("Position (pixels)")
        ax.set_title("Raw CO2 Intensity Profile")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)

    def plot(self):
        spot_analysis.spot_plot(self.intensity.value, pix_size = self.pix_size)


def ADKRate(I, Z, E_I):
    """Ammosov-Delone-Krainov ionization rate.
    Arguments:
        I: Intensity
        Z: charge state of the resulting ion
        E_I: ionization potential 
    
    Returns: ionization rate in 1/s
    """
    # Normalize values to unscaled atomic units:
    E_scale = (m_e**2 * e**5 / (hbar**4 * (4 * np.pi * eps0)**3)).to(u.V/u.m)
    I_scale =  (m_e**4 / (8 * np.pi * alpha * hbar**9) * e**12 / (4 * np.pi * eps0)**6).to(u.W/ u.cm**2)
    Energy_scale = (m_e * e**4 / (hbar**2 * (4 * np.pi * eps0)**2)).to(u.eV)
    t_scale = (hbar**3 * (4 * np.pi * eps0)**2 / m_e / e**4).to(u.s)

    I_norm = ((I / I_scale).to(u.dimensionless_unscaled)).value
    I_norm = np.where(I_norm<0, 0, I_norm)

    E_I_norm = (E_I / Energy_scale).to(u.dimensionless_unscaled).value
    F = np.sqrt(I_norm) # dimensionless field strength


    # Calculate effective principle quantum number
    nEff = Z / np.sqrt(2. * E_I_norm)
    
    with np.errstate(all='ignore'):
        D = ((4. * Z**3.) / (F * nEff**4.))**nEff
        rate = (F * D**2.) / (8. * np.pi * Z) \
            * np.exp(-(2. * Z**3.) / (3. * nEff**3. * F))

        # Include coefficient for linear polarization
        rate *= np.sqrt((3. * nEff**3. * F) / (np.pi * Z**3.))

    # get rid of any nan values from very small fields and revert to mks units
    rate = (np.nan_to_num(rate)/t_scale).to(1/u.s)

    return rate

def ion_yield(intensity, dt, charge_state = 2, ionization_energy = 13.6 * u.eV):
    """
    Returns percent of atoms ionized to given charge state in a field of given intensity
    Default values set for hydrogen atoms.
    """
    # Get ADK ionization rate in 1/s
    ionization_rate = ADKRate(intensity, charge_state, ionization_energy)

    # Calculate probability of ionization in time dt
    ionization_probability = 1 - np.exp(-ionization_rate * dt)
    percent_ionized = ionization_probability * 100
    
    # Clip percentages above 100 and below zero
    percent_ionized = np.where(percent_ionized > 100, 100, percent_ionized) 
    percent_ionized = np.where(percent_ionized < 0, 0, percent_ionized) 
    
    return percent_ionized


def plot_ionization(x, y, intensity, ion_population, longitudinal=False, ncontours = 8):
    if longitudinal:
        x_label = 'r (um)'
        y_label = 'z (um)'
    else:
        x_label = 'x (um)'
        y_label = 'y (um)'

    X, Y = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))

    colors1 = ax1.contourf(X, Y, intensity, cmap="plasma", vmin=0, vmax=np.max(intensity).value)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title("Beam Profile")

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(colors1, cax=cax1)
    cbar1.set_label("Intensity (W/cm$^2$)")

    colors2 = ax2.contourf(X, Y, ion_population, ncontours, cmap="plasma", vmin=0, vmax=100)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title("H+ Population After Pulse")

    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(colors2, cax=cax2)
    cbar2.set_label("Ion Population (\%)")

    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == "__main__":

    directory = "/home/oods/Research/AE98-99/CO2 Profiles/AE98/"
    fname = directory + "2021-08-05 007 regen focus.h5"

    pulse_length = 2 * u.ps
    beam = CO2Profile(fname, energy = 200 * u.mJ, pulse_length = 2 * u.ps, magnification = 50.)
    np.savetxt(directory+'CO2_profile_data.txt', beam.intensity.value)
    beam.calibrate_intensity()
    beam.analyze_spot()

    # Hydrogen charge states and ionization energies
    Z_max = 1 #proton number for Hydrogen
    Z = np.arange(1, Z_max + 1, dtype=int)
    ionization_energies = np.array([13.6]) * u.eV

    ion_population = ion_yield(beam.intensity, pulse_length)
    plot_ionization(beam.x, beam.y, beam.intensity, ion_population)
