import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import schur, solve_continuous_lyapunov


PI = np.pi
HBAR = 1.054_571_817e-34
C0 = 299_792_458.0
IDLER = 0
SIGNAL = 1
PUMP = 2
MODE_NAMES = ["idler", "signal", "pump"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def omega_from_lambda(lambda_m):
    return 2.0 * PI * C0 / np.asarray(lambda_m, dtype=float)


def lambda_from_omega(omega):
    return 2.0 * PI * C0 / np.asarray(omega, dtype=float)


def sinc_unscaled(x):
    """sin(x)/x with the convention sinc(0)=1."""
    return np.sinc(np.asarray(x) / PI)


def complex_to_real(z):
    z = np.asarray(z, dtype=np.complex128)
    out = np.empty(2 * len(z), dtype=float)
    out[0::2] = np.real(z)
    out[1::2] = np.imag(z)
    return out


def real_to_complex(x):
    x = np.asarray(x, dtype=float)
    return x[0::2] + 1j * x[1::2]


def numerical_jacobian(fun, x0, eps=1e-3):
    """Fourth-order central-difference Jacobian (error O(eps^4), which keeps the
    neutral phase-mode eigenvalue of an OPO at the 1e-5 s^-1 level)."""
    x0 = np.asarray(x0, dtype=float)
    f0 = np.asarray(fun(x0), dtype=float)
    J = np.zeros((f0.size, x0.size), dtype=float)
    for j in range(x0.size):
        dx = np.zeros_like(x0)
        dx[j] = eps * max(1.0, abs(x0[j]))
        h = dx[j]
        f1 = np.asarray(fun(x0 + dx), dtype=float) - np.asarray(fun(x0 - dx), dtype=float)
        f2 = np.asarray(fun(x0 + 2 * dx), dtype=float) - np.asarray(fun(x0 - 2 * dx), dtype=float)
        J[:, j] = (8.0 * f1 - f2) / (12.0 * h)
    return J


# -----------------------------------------------------------------------------
# PPLN dispersion
# -----------------------------------------------------------------------------
# We use the Jundt extraordinary-index Sellmeier equation for congruent LiNbO3.
# This is standard for type-0 / eee PPLN modeling. If the actual crystal is
# MgO:PPLN or stoichiometric PPLN, replace this model with the correct fit.
# -----------------------------------------------------------------------------

def ln_ne_jundt(lambda_m, temperature_C=25.0):
    lam_um = 1e6 * np.asarray(lambda_m, dtype=float)
    f = (temperature_C - 24.5) * (temperature_C + 570.82)
    n2 = (
        5.35583
        + 4.629e-7 * f
        + (0.100473 + 3.862e-8 * f)
        / (lam_um**2 - (0.20692 - 0.89e-8 * f) ** 2)
        + (100.0 + 2.657e-5 * f)
        / (lam_um**2 - 11.34927**2)
        - 1.5334e-2 * lam_um**2
    )
    return np.sqrt(n2)


def ln_group_index_jundt(lambda_m, temperature_C=25.0):
    """Group index from the Jundt extraordinary-index Sellmeier fit."""
    omega = omega_from_lambda(lambda_m)
    domega = np.maximum(np.abs(omega) * 1e-5, 1.0)
    n0 = ln_ne_jundt(lambda_m, temperature_C=temperature_C)
    n_plus = ln_ne_jundt(lambda_from_omega(omega + domega), temperature_C=temperature_C)
    n_minus = ln_ne_jundt(lambda_from_omega(omega - domega), temperature_C=temperature_C)
    return n0 + omega * (n_plus - n_minus) / (2.0 * domega)


def qpm_period_for_target_mismatch(
    omega_carriers,
    photon_numbers_center,
    direct_delta_k_pull,
    target_delta_k,
    temperature_C=25.0,
):
    """Choose the QPM period for a target mismatch at a chosen mean occupation."""
    omegas = np.asarray(omega_carriers, dtype=float)
    lambdas = lambda_from_omega(omegas)
    n = ln_ne_jundt(lambdas, temperature_C=temperature_C)
    k_lin = n * omegas / C0
    k_nl = direct_delta_k_pull.T @ np.asarray(photon_numbers_center, dtype=float)
    k_total = k_lin + k_nl
    delta_k_without_grating = k_total[PUMP] - k_total[SIGNAL] - k_total[IDLER]
    k_qpm = delta_k_without_grating - target_delta_k
    if k_qpm <= 0.0:
        raise ValueError("Target mismatch gives a nonpositive first-order QPM grating wavevector")
    return 2.0 * PI / k_qpm


# -----------------------------------------------------------------------------
# Platform and model parameters
# -----------------------------------------------------------------------------

@dataclass
class SystemParams:
    platform_name: str
    chi2_material: str
    chi3_material: str

    # Bare cavity resonance frequencies [rad/s]
    omega_bare: np.ndarray

    # Reference rotating-frame/carrier frequencies [rad/s]. The pump is fixed
    # by its drive. A seeded signal is fixed by its seed; for an unseeded signal,
    # signal_frequency_offset is solved and the idler shifts oppositely so the
    # three carriers continue to obey exact frequency-sum conservation.
    omega_drive: np.ndarray

    # Total linear amplitude decay rates [1/s] (intrinsic plus external).
    kappa: np.ndarray

    # Coherent drive amplitudes in sqrt(photons/s)
    drive_amplitudes: np.ndarray

    # Total fractional cavity-resonance pulls per photon from the Kerr media.
    # Entry [ell, m] means photons in mode ell shift the resonance of mode m by
    # omega_bare[m] * cavity_pull[ell, m] per photon (units: 1/photon).
    cavity_pull: np.ndarray

    # Optional two-photon absorption coefficients [1/(s photon)] in a simple mean-field model
    tpa: np.ndarray

    # Parametric coupling scale. This is kept phenomenological because the exact
    # cavity normalization depends on mode overlaps and coupling conventions.
    g0: complex

    # PPLN phase-matching parameters
    crystal_length: float
    poling_period: float
    temperature_C: float = 25.0

    # Direct intensity-dependent propagation-constant shifts inside PPLN,
    # evaluated at the reference carriers. Matrix indices are [source photon
    # mode, affected optical mode]; units are 1/m per photon.
    direct_delta_k_pull: np.ndarray = None

    # External coupling rates [1/s]. None assumes all linear damping is through
    # modeled input/output ports; set separately to distinguish intrinsic loss.
    kappa_external: np.ndarray = None

    # Numerically centered model coefficients. Detuning and phase mismatch are
    # specified at reference_photon_numbers, so the RHS only evaluates small
    # changes rather than subtracting optical-scale frequencies or wavevectors.
    reference_photon_numbers: np.ndarray = None
    detuning_at_reference: np.ndarray = None  # reference carrier minus resonance [rad/s]
    kerr_rate: np.ndarray = None  # [source, affected], rad/s per photon
    delta_k_at_reference: float = 0.0  # 1/m
    q_dk: np.ndarray = None  # 1/m per photon, [source]
    phase_match_offset_x: float = 0.5

    # Isolate the two intensity-to-gain paths for controlled comparisons.
    use_cavity_kerr: bool = True
    use_phase_match_kerr: bool = True

    # OPO configuration. The idler input remains exactly zero in both branches.
    wavelength_regime: str = "nondegenerate"
    signal_seeded: bool = True
    signal_seed_power_W: float = 1e-3
    signal_frequency_offset: float = 0.0  # rad/s relative to the reference frame
    pump_input_power_W: float = 1.0


@dataclass
class ExperimentalPlatform:
    name: str
    chi2_material: str
    chi3_material: str
    chi3_n2_m2_per_W: float


def platform_ppln_tantala():
    return ExperimentalPlatform(
        name="PPLN chi2/chi3 + tantala Kerr section",
        chi2_material="congruent PPLN (type-0)",
        chi3_material="Ta2O5 and congruent LiNbO3 Kerr pulls; LiNbO3 also shifts PPLN delta-k",
        chi3_n2_m2_per_W=6.2e-19,
    )


def n2_to_kerr_matrices(
    n2_self_m2_per_W,
    omega_modes,
    n_group_modes,
    mode_volume_m3,
    section_fraction=1.0,
    xpm_factor=2.0,
):
    """Estimate fractional cavity pulls and local delta-k pulls per photon.

    The one-photon intensity is approximated as I_1 = hbar*omega*c/(n_g*V_eff).
    For each source mode ell and affected mode m, use delta-n_m = n2[m,ell]*I_ell;
    then delta-omega_m/omega_m ~= -delta-n_m/n_g,m over the specified cavity
    section. The local propagation-constant shift is delta-k_m = omega_m*delta-n_m/c.

    Off-diagonal n2 values use xpm_factor times the geometric mean of the two
    self-n2 values. This is a scalar co-polarized approximation, not a substitute
    for the frequency-dependent LiNbO3 chi3 tensor and actual modal overlaps.
    """
    n2_self = np.asarray(n2_self_m2_per_W, dtype=float)
    omega = np.asarray(omega_modes, dtype=float)
    n_group = np.asarray(n_group_modes, dtype=float)
    volume = np.broadcast_to(np.asarray(mode_volume_m3, dtype=float), omega.shape)

    if not (n2_self.shape == omega.shape == n_group.shape):
        raise ValueError("n2_self, omega_modes, and n_group_modes must have matching shapes")
    if np.any(n2_self < 0.0) or np.any(n_group <= 0.0) or np.any(volume <= 0.0):
        raise ValueError("n2_self must be nonnegative and group indices/volumes positive")

    n2_cross = np.sqrt(np.outer(n2_self, n2_self))
    xpm = np.full(n2_cross.shape, float(xpm_factor))
    np.fill_diagonal(xpm, 1.0)
    n2_cross *= xpm

    intensity_per_photon = HBAR * omega * C0 / (n_group * volume)
    fractional_cavity_pull = np.zeros((len(omega), len(omega)), dtype=float)
    delta_k_pull = np.zeros_like(fractional_cavity_pull)
    for source in range(len(omega)):
        for affected in range(len(omega)):
            n2_ml = n2_cross[affected, source]
            fractional_cavity_pull[source, affected] = (
                -n2_ml
                * intensity_per_photon[source]
                * section_fraction
                / n_group[affected]
            )
            delta_k_pull[source, affected] = (
                omega[affected] / C0 * n2_ml * intensity_per_photon[source]
            )
    return fractional_cavity_pull, delta_k_pull


def default_system(
    phase_match_offset_x=0.5,
    wavelength_regime="nondegenerate",
    signal_seeded=True,
    signal_seed_power_W=1e-3,
    pump_input_power_W=1.0,
    reference_photon_numbers=None,
    use_cavity_kerr=True,
    use_phase_match_kerr=True,
    n2_ln_electronic=None,
):
    platform = platform_ppln_tantala()

    # The pump is fixed at 525 nm. The two presets select the signal carrier;
    # the idler carrier follows from omega_p = omega_s + omega_i.
    signal_wavelengths = {
        "quasi-degenerate": 1030e-9,
        "nondegenerate": 700e-9,
    }
    if wavelength_regime not in signal_wavelengths:
        raise ValueError("wavelength_regime must be 'quasi-degenerate' or 'nondegenerate'")
    if signal_seed_power_W < 0.0 or pump_input_power_W < 0.0:
        raise ValueError("Input powers must be nonnegative")
    wavelength_regime = str(wavelength_regime)
    lambda_p = 525e-9
    lambda_s = signal_wavelengths[wavelength_regime]
    omega_p = omega_from_lambda(lambda_p)
    omega_s = omega_from_lambda(lambda_s)
    omega_i = omega_p - omega_s
    omega_carrier = np.array([omega_i, omega_s, omega_p], dtype=float)

    temperature_C = 80.0
    crystal_length = 1e-2

    # |s_in|^2 is photon flux. The idler is never seeded; signal and pump
    # input powers are explicit configuration parameters.
    s_p = np.sqrt(pump_input_power_W / (HBAR * omega_p))
    s_s = (
        np.sqrt(signal_seed_power_W / (HBAR * omega_s))
        if signal_seeded and signal_seed_power_W > 0.0 else 0.0
    )
    signal_seeded = bool(signal_seeded and signal_seed_power_W > 0.0)
    kappa = np.array([1e8, 1e8, 1e8], dtype=float)

    # First-pass hybrid-cavity geometry: PPLN occupies the non-tantala fraction.
    # Replace these effective volumes/overlaps with simulated mode data.
    mode_area = (1.0e-6) ** 2
    tantala_section_fraction = 0.25
    ppln_section_fraction = 1.0 - tantala_section_fraction
    cavity_length = crystal_length / ppln_section_fraction
    mode_volume = mode_area * cavity_length

    # Tantala contribution to cavity resonance pulls. This estimate assumes
    # co-polarized modes and a scalar XPM/SPM ratio of 2.
    n_group_tantala = np.full(3, 2.05, dtype=float)
    n2_tantala = np.full(3, platform.chi3_n2_m2_per_W, dtype=float)
    tantala_cavity_pull, _ = n2_to_kerr_matrices(
        n2_self_m2_per_W=n2_tantala,
        omega_modes=omega_carrier,
        n_group_modes=n_group_tantala,
        mode_volume_m3=mode_volume,
        section_fraction=tantala_section_fraction,
        xpm_factor=2.0,
    )

    # Preliminary extraordinary electronic-LN n2 values. The equal signal/idler
    # entries are a placeholder, especially for the 700/2100-nm preset; supply a
    # wavelength-resolved array for quantitative predictions. Raman and its noise
    # are omitted, and scalar cross-frequency XPM remains an approximation.
    if n2_ln_electronic is None:
        n2_ln_electronic = np.array([2.2e-19, 2.2e-19, 5.9e-19], dtype=float)
    else:
        n2_ln_electronic = np.asarray(n2_ln_electronic, dtype=float)
    if n2_ln_electronic.shape != (3,) or np.any(n2_ln_electronic < 0.0):
        raise ValueError("n2_ln_electronic must be three nonnegative values [idler, signal, pump]")
    lambda_modes = lambda_from_omega(omega_carrier)
    n_group_ln = ln_group_index_jundt(lambda_modes, temperature_C=temperature_C)
    ppln_cavity_pull, direct_delta_k_pull = n2_to_kerr_matrices(
        n2_self_m2_per_W=n2_ln_electronic,
        omega_modes=omega_carrier,
        n_group_modes=n_group_ln,
        mode_volume_m3=mode_volume,
        section_fraction=ppln_section_fraction,
        xpm_factor=2.0,
    )
    cavity_pull = tantala_cavity_pull + ppln_cavity_pull

    # Center all Kerr-shifted resonances at these trial occupations. The solver
    # must still be tuned to make them the actual steady-state photon numbers.
    if reference_photon_numbers is None:
        center_photon_numbers = np.array([1e6, 1e6, 1e8], dtype=float)
    else:
        center_photon_numbers = np.asarray(reference_photon_numbers, dtype=float)
        if center_photon_numbers.shape != (3,) or np.any(~np.isfinite(center_photon_numbers)) or np.any(center_photon_numbers < 0.0):
            raise ValueError("reference_photon_numbers must be three finite nonnegative values [idler, signal, pump]")
    center_fractional_shifts = cavity_pull.T @ center_photon_numbers
    omega_bare = omega_carrier / (1.0 + center_fractional_shifts)
    # The affine Kerr rates reproduce the original resonance-pull model, but
    # are evaluated around the chosen reference occupation in the RHS.
    kerr_rate = cavity_pull * omega_bare[None, :]

    # x0 = delta-k * L / 2 at the reference occupation. This is a scan
    # parameter, not an asserted optimum; the useful point trades slope against
    # remaining parametric gain.
    target_phase_match_offset_x = float(phase_match_offset_x)
    if not 0.0 <= target_phase_match_offset_x < PI:
        raise ValueError("phase_match_offset_x must lie in [0, pi)")
    target_delta_k = 2.0 * target_phase_match_offset_x / crystal_length
    q_dk = (
        direct_delta_k_pull[:, PUMP]
        - direct_delta_k_pull[:, SIGNAL]
        - direct_delta_k_pull[:, IDLER]
    )
    poling_period = qpm_period_for_target_mismatch(
        omega_carriers=omega_carrier,
        photon_numbers_center=center_photon_numbers,
        direct_delta_k_pull=direct_delta_k_pull,
        target_delta_k=target_delta_k,
        temperature_C=temperature_C,
    )

    # TPA is left off in this first pass. It may matter for the 525-nm LN pump;
    # only the real electronic Kerr response is represented here [Bac12].
    tpa = np.zeros(3, dtype=float)

    # Effective cavity chi2 coupling remains phenomenological pending mode-overlap normalization.
    g0 = 3e3 + 0.0j

    return SystemParams(
        platform_name=platform.name,
        chi2_material=platform.chi2_material,
        chi3_material=platform.chi3_material,
        omega_bare=omega_bare,
        omega_drive=omega_carrier,
        kappa=kappa,
        drive_amplitudes=np.array([0.0, s_s, s_p], dtype=np.complex128),
        cavity_pull=cavity_pull,
        tpa=tpa,
        g0=g0,
        crystal_length=crystal_length,
        poling_period=poling_period,
        temperature_C=temperature_C,
        direct_delta_k_pull=direct_delta_k_pull,
        reference_photon_numbers=center_photon_numbers,
        detuning_at_reference=np.zeros(3, dtype=float),
        kerr_rate=kerr_rate,
        delta_k_at_reference=target_delta_k,
        q_dk=q_dk,
        phase_match_offset_x=target_phase_match_offset_x,
        use_cavity_kerr=bool(use_cavity_kerr),
        use_phase_match_kerr=bool(use_phase_match_kerr),
        wavelength_regime=wavelength_regime,
        signal_seeded=signal_seeded,
        signal_seed_power_W=float(signal_seed_power_W if signal_seeded else 0.0),
        signal_frequency_offset=0.0,
        pump_input_power_W=float(pump_input_power_W),
    )


# -----------------------------------------------------------------------------
# Nonlinear model pieces
# -----------------------------------------------------------------------------

def photon_numbers(alpha):
    alpha = np.asarray(alpha, dtype=np.complex128)
    return np.abs(alpha) ** 2


def detunings(alpha, system, signal_frequency_offset=None):
    """Carrier-minus-resonance detunings, centered at the design reference."""
    if system.reference_photon_numbers is None or system.detuning_at_reference is None:
        raise ValueError("SystemParams must define centered detuning coefficients")
    offset = (
        system.signal_frequency_offset
        if signal_frequency_offset is None else float(signal_frequency_offset)
    )
    carrier_shift = np.array([-offset, offset, 0.0], dtype=float)
    delta = np.asarray(system.detuning_at_reference, dtype=float).copy() + carrier_shift
    if system.use_cavity_kerr:
        delta -= system.kerr_rate.T @ (
            photon_numbers(alpha) - np.asarray(system.reference_photon_numbers, dtype=float)
        )
    return delta


def effective_resonance_omegas(alpha, system, signal_frequency_offset=None):
    # Compute from small detunings, not by subtracting optical-scale frequencies.
    carriers = carrier_frequencies_for_phase_matching(system, signal_frequency_offset)
    return carriers - detunings(alpha, system, signal_frequency_offset)


def tpa_losses(alpha, system):
    n_total = np.sum(photon_numbers(alpha))
    return system.tpa * n_total


def external_coupling_rates(system):
    if system.kappa_external is None:
        return np.asarray(system.kappa, dtype=float)
    kappa_external = np.asarray(system.kappa_external, dtype=float)
    kappa_total = np.asarray(system.kappa, dtype=float)
    if kappa_external.shape != kappa_total.shape:
        raise ValueError("kappa_external and kappa must have matching shapes")
    tolerance = 100.0 * np.finfo(float).eps * max(float(np.max(kappa_total)), 1.0)
    if np.any(kappa_external < 0.0) or np.any(kappa_external > kappa_total + tolerance):
        raise ValueError("Each external coupling rate must lie between zero and total kappa")
    return kappa_external


def carrier_frequencies_for_phase_matching(system, signal_frequency_offset=None):
    """Actual carriers; free-running signal and idler shift oppositely."""
    omegas = np.asarray(system.omega_drive, dtype=float).copy()
    frame_mismatch = omegas[PUMP] - omegas[SIGNAL] - omegas[IDLER]
    scale = max(float(np.max(np.abs(omegas))), 1.0)
    roundoff_tolerance = 100.0 * np.finfo(float).eps * scale
    if abs(frame_mismatch) > roundoff_tolerance:
        raise ValueError("Reference carriers must obey omega_p = omega_s + omega_i")
    offset = (
        system.signal_frequency_offset
        if signal_frequency_offset is None else float(signal_frequency_offset)
    )
    omegas[SIGNAL] += offset
    omegas[IDLER] -= offset
    return omegas


def delta_k_frequency_correction(system, signal_frequency_offset):
    """Evaluate the Sellmeier mismatch change relative to the reference carriers.

    Subtract nearby wavevectors, not the three large optical wavevectors. This
    captures group-velocity mismatch in the unseeded frequency-selection solve.
    """
    offset = float(signal_frequency_offset)
    if offset == 0.0:
        return 0.0
    omega_ref = np.asarray(system.omega_drive, dtype=float)

    def k_ln(omega):
        wavelength = lambda_from_omega(omega)
        return ln_ne_jundt(wavelength, temperature_C=system.temperature_C) * omega / C0

    delta_k_s = k_ln(omega_ref[SIGNAL] + offset) - k_ln(omega_ref[SIGNAL])
    delta_k_i = k_ln(omega_ref[IDLER] - offset) - k_ln(omega_ref[IDLER])
    return float(-delta_k_s - delta_k_i)


def phase_mismatch(alpha, system, signal_frequency_offset=None):
    """Centered PPLN mismatch, including nonlinear index and carrier-frequency shifts."""
    if system.reference_photon_numbers is None or system.q_dk is None:
        raise ValueError("SystemParams must define centered phase-mismatch coefficients")
    offset = (
        system.signal_frequency_offset
        if signal_frequency_offset is None else float(signal_frequency_offset)
    )
    delta_k = float(system.delta_k_at_reference)
    delta_k += delta_k_frequency_correction(system, offset)
    if system.use_phase_match_kerr:
        delta_k += float(
            np.asarray(system.q_dk, dtype=float)
            @ (photon_numbers(alpha) - np.asarray(system.reference_photon_numbers, dtype=float))
        )
    return delta_k


def chi2_eff(alpha, system, signal_frequency_offset=None):
    """Single-pass coupling referenced to the crystal entrance plane.

    Dropping the exponential alone is not generally a harmless convention
    change: mode amplitudes and coherent-drive phases must be transformed with
    the reference plane, especially when delta-k depends on intensity.
    """
    delta_k = phase_mismatch(alpha, system, signal_frequency_offset)
    L = system.crystal_length
    return system.g0 * sinc_unscaled(0.5 * delta_k * L) * np.exp(0.5j * delta_k * L)


def intensity_channel_sensitivities(alpha, system):
    """Return local A/B sensitivities per intracavity photon.

    Channel A is cavity Kerr pulling; channel B is the local PPLN phase-match
    shift. The cavity phase sensitivity is included because the cavity response
    is steepest in phase at resonance, even where its magnitude slope vanishes.
    """
    n = photon_numbers(alpha)
    detuning = detunings(alpha, system)
    kappa_eff = system.kappa + tpa_losses(alpha, system)

    if system.use_cavity_kerr:
        kerr = np.asarray(system.kerr_rate, dtype=float).T
    else:
        kerr = np.zeros((len(n), len(n)), dtype=float)
    d_detuning_dn = -kerr
    d_normalized_detuning_dn = d_detuning_dn / kappa_eff[:, None]
    denom = kappa_eff[:, None] ** 2 + detuning[:, None] ** 2
    d_cavity_phase_dn = -kappa_eff[:, None] * kerr / denom
    d_log_cavity_amplitude_dn = detuning[:, None] * kerr / denom

    if system.use_phase_match_kerr:
        q_dk = np.asarray(system.q_dk, dtype=float)
    else:
        q_dk = np.zeros_like(n)
    d_x_dn = 0.5 * system.crystal_length * q_dk
    x = 0.5 * system.crystal_length * phase_mismatch(alpha, system)
    if abs(x) < 1e-5:
        d_log_sinc_dx = -x / 3.0 - x**3 / 45.0 - 2.0 * x**5 / 945.0
    elif abs(sinc_unscaled(x)) < 1e-12:
        d_log_sinc_dx = np.nan
    else:
        d_log_sinc_dx = 1.0 / np.tan(x) - 1.0 / x
    d_log_chi2_dn = d_log_sinc_dx * d_x_dn

    return {
        "d_detuning_d_photon": d_detuning_dn,
        "d_detuning_over_kappa_d_photon": d_normalized_detuning_dn,
        "d_cavity_phase_d_photon": d_cavity_phase_dn,
        "d_log_cavity_amplitude_d_photon": d_log_cavity_amplitude_dn,
        "d_phase_match_x_d_photon": d_x_dn,
        "d_log_abs_chi2_d_photon": d_log_chi2_dn,
        "phase_match_x": x,
    }


def rhs_complex(alpha, system, signal_frequency_offset=None):
    alpha = np.asarray(alpha, dtype=np.complex128)
    ai, a_s, ap = alpha

    kappa_eff = system.kappa + tpa_losses(alpha, system)

    # Seeded branch: carrier frequencies are drive-locked. Unseeded branch:
    # signal_frequency_offset is solved, with the idler shifted oppositely.
    # Cavity Kerr enters through detuning and local PPLN Kerr through phase mismatch.
    detuning = detunings(alpha, system, signal_frequency_offset)
    g = chi2_eff(alpha, system, signal_frequency_offset)

    drive_terms = np.sqrt(2.0 * external_coupling_rates(system)) * system.drive_amplitudes

    return np.array(
        [
            (1j * detuning[IDLER] - kappa_eff[IDLER]) * ai + g * ap * np.conj(a_s) + drive_terms[IDLER],
            (1j * detuning[SIGNAL] - kappa_eff[SIGNAL]) * a_s + g * ap * np.conj(ai) + drive_terms[SIGNAL],
            (1j * detuning[PUMP] - kappa_eff[PUMP]) * ap - np.conj(g) * ai * a_s + drive_terms[PUMP],
        ],
        dtype=np.complex128,
    )


def relative_fixed_point_residual(alpha, system, signal_frequency_offset=None):
    """Residual norm scaled by the magnitudes of the steady-state flow terms."""
    alpha = np.asarray(alpha, dtype=np.complex128)
    kappa_eff = system.kappa + tpa_losses(alpha, system)
    detuning = detunings(alpha, system, signal_frequency_offset)
    g = chi2_eff(alpha, system, signal_frequency_offset)
    drive_terms = np.sqrt(2.0 * external_coupling_rates(system)) * system.drive_amplitudes
    linear_flow = (1j * detuning - kappa_eff) * alpha
    chi2_flow = np.array(
        [
            g * alpha[PUMP] * np.conj(alpha[SIGNAL]),
            g * alpha[PUMP] * np.conj(alpha[IDLER]),
            -np.conj(g) * alpha[IDLER] * alpha[SIGNAL],
        ],
        dtype=np.complex128,
    )
    scale = np.linalg.norm(linear_flow) + np.linalg.norm(chi2_flow) + np.linalg.norm(drive_terms)
    return float(
        np.linalg.norm(rhs_complex(alpha, system, signal_frequency_offset)) / max(scale, 1.0)
    )


def drive_amplitudes_for_target_state(alpha_target, system, signal_seeded=None):
    """Find pump and optional signal drives for a specified complex target state.

    The idler drive is constrained to zero. Photon numbers alone do not fix the
    field phases, so the caller must supply all three complex amplitudes. For an
    unseeded signal, the target must also satisfy its zero-input steady-state equation.
    """
    alpha_target = np.asarray(alpha_target, dtype=np.complex128)
    if alpha_target.shape != (3,):
        raise ValueError("alpha_target must contain complex idler, signal, and pump amplitudes")
    if signal_seeded is None:
        signal_seeded = bool(system.signal_seeded)
    kappa_ext = external_coupling_rates(system)
    drive_scale = np.sqrt(2.0 * kappa_ext)
    current_drive_terms = drive_scale * np.asarray(system.drive_amplitudes, dtype=np.complex128)
    required_drive_terms = -(rhs_complex(alpha_target, system) - current_drive_terms)
    zero_port_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(required_drive_terms))))
    if abs(required_drive_terms[IDLER]) > zero_port_tolerance:
        raise ValueError("Target state is incompatible with an unseeded idler at these parameters")
    if not signal_seeded and abs(required_drive_terms[SIGNAL]) > zero_port_tolerance:
        raise ValueError("Target state requires a signal seed, but signal_seeded=False")

    result = np.zeros(3, dtype=np.complex128)
    allowed = [SIGNAL, PUMP] if signal_seeded else [PUMP]
    for mode in allowed:
        if drive_scale[mode] == 0.0:
            if abs(required_drive_terms[mode]) > zero_port_tolerance:
                raise ValueError(f"Target state requires a drive through uncoupled mode {MODE_NAMES[mode]}")
        else:
            result[mode] = required_drive_terms[mode] / drive_scale[mode]
    return result


def design_reference_state(system, adjust_mode=PUMP):
    """Design drives around a reference point; the forward solver remains free.

    With a seed, the zero-idler-input equation fixes one design occupation.
    Without a seed, the generated-mode equations also require the signal/idler
    loss ratio to match. The cavity and QPM centers are updated consistently;
    the subsequent IVP and LSQ do not constrain any occupation to this point.
    """
    if abs(system.drive_amplitudes[IDLER]) > 0.0:
        raise ValueError("The idler input must remain unseeded")
    if not system.signal_seeded and abs(system.drive_amplitudes[SIGNAL]) > 0.0:
        raise ValueError("An unseeded design requires exactly zero signal input")
    if np.any(np.asarray(system.tpa, dtype=float) != 0.0):
        raise NotImplementedError("Target-state design currently assumes zero TPA")
    if adjust_mode not in (IDLER, SIGNAL, PUMP):
        raise ValueError("adjust_mode must be IDLER, SIGNAL, or PUMP")

    target = np.asarray(system.reference_photon_numbers, dtype=float).copy()
    if target.shape != (3,) or np.any(~np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("reference_photon_numbers must be three finite nonnegative values")
    if np.any(target[np.arange(3) != adjust_mode] <= 0.0):
        raise ValueError("The two fixed occupations must be positive")

    gain = abs(system.g0) * abs(sinc_unscaled(system.phase_match_offset_x))
    if gain == 0.0 or system.kappa[IDLER] <= 0.0 or system.kappa[SIGNAL] <= 0.0:
        raise ValueError("Nonzero parametric gain and positive signal/idler losses are required")
    if system.signal_seeded:
        if adjust_mode == IDLER:
            target[IDLER] = gain**2 * target[SIGNAL] * target[PUMP] / system.kappa[IDLER]**2
        elif adjust_mode == SIGNAL:
            target[SIGNAL] = target[IDLER] * system.kappa[IDLER]**2 / (gain**2 * target[PUMP])
        else:
            target[PUMP] = target[IDLER] * system.kappa[IDLER]**2 / (gain**2 * target[SIGNAL])
    else:
        if adjust_mode != PUMP:
            raise ValueError("For an unseeded OPO, adjust_mode must be PUMP")
        required_ratio = system.kappa[IDLER] / system.kappa[SIGNAL]
        if not np.isclose(target[SIGNAL] / target[IDLER], required_ratio, rtol=1e-8):
            raise ValueError("Unseeded target requires n_s/n_i = kappa_i/kappa_s")
        target[PUMP] = system.kappa[SIGNAL] * system.kappa[IDLER] / gain**2

    # The reference occupations define the designed resonance and mismatch
    # center; update both consistently rather than changing only RHS offsets.
    fractional_shift = system.cavity_pull.T @ target
    if np.any(1.0 + fractional_shift <= 0.0):
        raise ValueError("Reference Kerr pulls imply a nonpositive bare resonance")
    system.omega_bare = system.omega_drive / (1.0 + fractional_shift)
    system.kerr_rate = system.cavity_pull * system.omega_bare[None, :]
    system.reference_photon_numbers = target
    system.detuning_at_reference = np.zeros(3, dtype=float)
    system.signal_frequency_offset = 0.0
    system.delta_k_at_reference = (
        2.0 * system.phase_match_offset_x / system.crystal_length
    )
    system.poling_period = qpm_period_for_target_mismatch(
        omega_carriers=system.omega_drive,
        photon_numbers_center=target,
        direct_delta_k_pull=system.direct_delta_k_pull,
        target_delta_k=system.delta_k_at_reference,
        temperature_C=system.temperature_C,
    )

    trial = np.sqrt(target).astype(np.complex128)
    g = chi2_eff(trial, system)
    delta_i = detunings(trial, system)[IDLER]
    alpha_target = np.array(
        [
            g * trial[PUMP] * np.conj(trial[SIGNAL])
            / (system.kappa[IDLER] - 1j * delta_i),
            trial[SIGNAL],
            trial[PUMP],
        ],
        dtype=np.complex128,
    )
    if not np.isclose(abs(alpha_target[IDLER])**2, target[IDLER], rtol=1e-8):
        raise RuntimeError("Reference occupations failed the zero-idler-drive balance")

    drives = drive_amplitudes_for_target_state(
        alpha_target, system, signal_seeded=system.signal_seeded
    )
    system.drive_amplitudes = drives
    system.signal_seed_power_W = float(
        HBAR * system.omega_drive[SIGNAL] * abs(drives[SIGNAL])**2
        if system.signal_seeded else 0.0
    )
    system.pump_input_power_W = float(
        HBAR * system.omega_drive[PUMP] * abs(drives[PUMP])**2
    )
    if relative_fixed_point_residual(alpha_target, system) > 1e-8:
        raise RuntimeError("Designed reference state does not satisfy the full steady-state equations")
    return alpha_target


def rhs_real(t, x, system, signal_frequency_offset=None):
    alpha = real_to_complex(x)
    return complex_to_real(rhs_complex(alpha, system, signal_frequency_offset))


# -----------------------------------------------------------------------------
# Steady-state solvers for seeded and free-running branches
# -----------------------------------------------------------------------------

@dataclass
class SteadyStateResult:
    alpha_ss: np.ndarray
    residual_norm: float
    relative_residual_norm: float
    transient_state: np.ndarray
    converged: bool
    signal_frequency_offset_rad_s: float = 0.0
    frequency_determined: bool = True
    oscillating: bool = True
    parametric_growth_rate_s_inv: float = np.nan
    frequency_offset_bound_hit: bool = False
    transient_converged: bool = True
    transient_time_s: float = 0.0


def _solver_scales(system, alpha=None):
    """Reference amplitude and decay rate for dimensionless numerical solves."""
    n_scale = 1.0
    if system.reference_photon_numbers is not None:
        n_scale = max(n_scale, float(np.max(system.reference_photon_numbers)))
    if alpha is not None:
        n_scale = max(n_scale, float(np.max(photon_numbers(alpha))))
    amplitude_scale = np.sqrt(n_scale)
    rate_scale = max(float(np.max(system.kappa)), 1.0)
    return amplitude_scale, rate_scale


def integrate_to_near_steady_state(system, alpha0=None, t_final=5e-7, method="BDF"):
    if alpha0 is None:
        alpha0 = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    amplitude_scale, rate_scale = _solver_scales(system, alpha0)
    u0 = complex_to_real(np.asarray(alpha0, dtype=np.complex128) / amplitude_scale)
    sol = solve_ivp(
        fun=lambda tau, u: rhs_real(0.0, u * amplitude_scale, system)
        / (rate_scale * amplitude_scale),
        t_span=(0.0, rate_scale * t_final),
        y0=u0,
        method=method,
        rtol=1e-9,
        atol=1e-12,
    )
    # Restore the public result to physical seconds and field amplitudes.
    sol.t = sol.t / rate_scale
    sol.y = sol.y * amplitude_scale
    alpha_t = np.vstack([real_to_complex(sol.y[:, j]) for j in range(sol.y.shape[1])])
    return real_to_complex(sol.y[:, -1]), sol, alpha_t


def integrate_until_photon_numbers_settle(
    system, alpha0, t_block=5e-7, tolerance=1e-7, max_time=None, method="BDF"
):
    """Integrate in growing blocks until all mode occupations stop changing."""
    if t_block <= 0.0 or tolerance <= 0.0:
        raise ValueError("t_block and tolerance must be positive")
    if max_time is None:
        max_time = 100.0 * t_block
    if max_time < t_block:
        raise ValueError("max_time must be at least t_block")

    state = np.asarray(alpha0, dtype=np.complex128).copy()
    elapsed = 0.0
    duration = t_block
    previous_numbers = None
    while elapsed < max_time:
        duration = min(duration, max_time - elapsed)
        state, _, alpha_t = integrate_to_near_steady_state(
            system, alpha0=state, t_final=duration, method=method
        )
        elapsed += duration
        numbers = photon_numbers(alpha_t)
        tail = numbers[max(0, int(0.75 * len(numbers))):]
        scale = np.maximum(np.max(tail, axis=0), 1.0)
        tail_spread = np.max(tail, axis=0) - np.min(tail, axis=0)
        if previous_numbers is not None:
            block_change = np.abs(numbers[-1] - previous_numbers) / scale
            if np.max(tail_spread / scale) <= tolerance and np.max(block_change) <= tolerance:
                return state, True, elapsed
        previous_numbers = numbers[-1].copy()
        duration *= 2.0
    return state, False, elapsed


def solve_drive_locked_steady_state(system, alpha0=None, t_final=5e-7):
    if not system.signal_seeded:
        raise ValueError("Use solve_unseeded_steady_state when the signal input is zero")
    if abs(system.drive_amplitudes[IDLER]) > 0.0:
        raise ValueError("The idler input must remain unseeded")
    system.signal_frequency_offset = 0.0
    if alpha0 is None:
        alpha0 = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    else:
        alpha0 = 0.97 * np.asarray(alpha0, dtype=np.complex128)
    transient_alpha, transient_converged, transient_time = integrate_until_photon_numbers_settle(
        system, alpha0=alpha0, t_block=t_final
    )
    amplitude_scale, rate_scale = _solver_scales(system, transient_alpha)

    def residual(u):
        return rhs_real(0.0, u * amplitude_scale, system) / (rate_scale * amplitude_scale)

    lsq = least_squares(
        residual,
        complex_to_real(transient_alpha / amplitude_scale),
        x_scale=1.0,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=20000,
    )

    alpha_ss = real_to_complex(lsq.x) * amplitude_scale
    res_norm = np.linalg.norm(rhs_complex(alpha_ss, system))
    relative_res_norm = relative_fixed_point_residual(alpha_ss, system)
    converged = bool(
        lsq.success and relative_res_norm <= 1e-8 and transient_converged
    )
    return SteadyStateResult(
        alpha_ss=alpha_ss,
        residual_norm=res_norm,
        relative_residual_norm=relative_res_norm,
        transient_state=transient_alpha,
        converged=converged,
        signal_frequency_offset_rad_s=0.0,
        frequency_determined=True,
        oscillating=bool(np.abs(alpha_ss[SIGNAL]) > 0.0 and np.abs(alpha_ss[IDLER]) > 0.0),
        transient_converged=transient_converged,
        transient_time_s=transient_time,
    )


def _pump_only_branches(system):
    """Find pump-only Kerr roots from the scalar cubic balance equation.

    These roots guide the frequency search only. The final state is obtained
    from the full three-mode IVP and then refined with least squares.
    """
    kappa_ext = external_coupling_rates(system)
    pump_drive = np.sqrt(2.0 * kappa_ext[PUMP]) * system.drive_amplitudes[PUMP]
    drive_power = float(abs(pump_drive) ** 2)
    kappa_p = float(system.kappa[PUMP])
    if kappa_p <= 0.0:
        raise ValueError("Pump amplitude decay rate must be positive")
    if drive_power == 0.0:
        return [0.0 + 0.0j]

    n_upper = drive_power / kappa_p**2
    alpha_zero = np.zeros(3, dtype=np.complex128)
    alpha_one = alpha_zero.copy()
    alpha_one[PUMP] = 1.0
    delta_zero = float(detunings(alpha_zero, system, 0.0)[PUMP])
    delta_slope = float(detunings(alpha_one, system, 0.0)[PUMP] - delta_zero)
    tpa_slope = float(system.tpa[PUMP])

    rate_scale = max(
        kappa_p, abs(delta_zero), abs(delta_slope) * n_upper,
        abs(tpa_slope) * n_upper, 1.0,
    )
    n_scale = max(n_upper, 1.0)
    k = kappa_p / rate_scale
    d = delta_zero / rate_scale
    q = delta_slope * n_scale / rate_scale
    t = tpa_slope * n_scale / rate_scale
    drive_scaled = drive_power / (n_scale * rate_scale**2)
    coefficients = np.trim_zeros(
        np.array([t**2 + q**2, 2.0 * (k * t + d * q), k**2 + d**2, -drive_scaled]),
        trim="f",
    )
    scaled_roots = np.roots(coefficients)

    def balance(n_p):
        alpha = np.zeros(3, dtype=np.complex128)
        alpha[PUMP] = np.sqrt(max(float(n_p), 0.0))
        kappa_eff = system.kappa + tpa_losses(alpha, system)
        delta_p = detunings(alpha, system, 0.0)[PUMP]
        return float(n_p * (kappa_eff[PUMP]**2 + delta_p**2) - drive_power)

    roots = []
    for root in scaled_roots:
        if abs(root.imag) > 1e-8 * max(1.0, abs(root.real)) or root.real < 0.0:
            continue
        n_p = float(root.real * n_scale)
        if abs(balance(n_p)) <= 1e-7 * max(drive_power, 1.0):
            roots.append(n_p)

    unique_roots = []
    for n_p in sorted(roots):
        if not unique_roots or abs(n_p - unique_roots[-1]) > 1e-8 * max(n_p, 1.0):
            unique_roots.append(n_p)

    branches = []
    for n_p in unique_roots:
        alpha_test = np.zeros(3, dtype=np.complex128)
        alpha_test[PUMP] = np.sqrt(n_p)
        kappa_eff = system.kappa + tpa_losses(alpha_test, system)
        delta_p = detunings(alpha_test, system, 0.0)[PUMP]
        alpha_p = pump_drive / (kappa_eff[PUMP] - 1j * delta_p)

        branches.append(alpha_p)
    return branches


def _unseeded_small_signal_growth(system, alpha_p, signal_frequency_offset):
    """Growth rate of the signal/idler pair about a pump-only solution."""
    alpha = np.array([0.0j, 0.0j, alpha_p], dtype=np.complex128)
    kappa_eff = system.kappa + tpa_losses(alpha, system)
    delta = detunings(alpha, system, signal_frequency_offset)
    g = chi2_eff(alpha, system, signal_frequency_offset)
    M = np.array(
        [
            [-kappa_eff[SIGNAL] + 1j * delta[SIGNAL], g * alpha_p],
            [np.conj(g * alpha_p), -kappa_eff[IDLER] - 1j * delta[IDLER]],
        ],
        dtype=np.complex128,
    )
    return float(np.max(np.real(np.linalg.eigvals(M))))


def _maximize_unseeded_small_signal_growth(
    system, alpha_p, max_frequency_offset, initial_offset=None
):
    offsets = np.linspace(-max_frequency_offset, max_frequency_offset, 801)
    if initial_offset is not None and abs(initial_offset) <= max_frequency_offset:
        offsets = np.unique(np.append(offsets, float(initial_offset)))
    growth = np.array(
        [_unseeded_small_signal_growth(system, alpha_p, offset) for offset in offsets]
    )
    index = int(np.argmax(growth))
    best_offset = float(offsets[index])
    best_growth = float(growth[index])
    bound_hit = index == 0 or index == len(offsets) - 1
    if not bound_hit:
        result = minimize_scalar(
            lambda offset: -_unseeded_small_signal_growth(system, alpha_p, offset),
            bounds=(offsets[index - 1], offsets[index + 1]),
            method="bounded",
            options={"xatol": max_frequency_offset * 1e-10},
        )
        refined_offset = float(result.x)
        refined_growth = -float(result.fun)
        if refined_growth >= best_growth:
            best_offset, best_growth = refined_offset, refined_growth
    return best_growth, best_offset, bound_hit


def solve_unseeded_steady_state(
    system,
    alpha0=None,
    frequency_offset_initial=0.0,
    max_frequency_offset=None,
    t_final=5e-7,
):
    """Integrate the unseeded OPO, then refine the evolved state with LSQ.

    Pump-only Kerr roots guide the frequency search only. All three modal
    occupations evolve freely in the IVP; the reference values are initial scales,
    not constraints on the least-squares solution.
    """
    if system.signal_seeded:
        raise ValueError("Set signal_seeded=False before solving an unseeded OPO")
    if np.any(np.abs(system.drive_amplitudes[[IDLER, SIGNAL]]) > 0.0):
        raise ValueError("Unseeded branch requires exactly zero signal and idler input amplitudes")

    rate_scale = max(float(np.max(system.kappa)), 1.0)
    if max_frequency_offset is None:
        max_frequency_offset = 20.0 * rate_scale
    max_frequency_offset = min(
        float(max_frequency_offset),
        0.1 * min(float(system.omega_drive[IDLER]), float(system.omega_drive[SIGNAL])),
    )
    if max_frequency_offset <= 0.0:
        raise ValueError("max_frequency_offset must be positive")

    if alpha0 is None:
        alpha0 = np.sqrt(np.maximum(np.asarray(system.reference_photon_numbers, dtype=float), 1.0))
        alpha0 = alpha0.astype(np.complex128)
    else:
        alpha0 = np.asarray(alpha0, dtype=np.complex128).copy()
    if alpha0.shape != (3,):
        raise ValueError("alpha0 must contain idler, signal, and pump amplitudes")

    # Initial values only: perturb the design point so it is not accepted merely
    # because it was constructed as an exact algebraic root.
    alpha0[IDLER] *= 0.93
    alpha0[SIGNAL] *= 1.07
    alpha0[PUMP] *= 0.97

    pump_branches = _pump_only_branches(system)
    if not pump_branches:
        raise RuntimeError("Could not find a self-consistent pump-only solution")
    threshold_candidates = []
    for alpha_p in pump_branches:
        growth, offset, bound_hit = _maximize_unseeded_small_signal_growth(
            system, alpha_p, max_frequency_offset, initial_offset=frequency_offset_initial
        )
        threshold_candidates.append((growth, offset, bound_hit))
    growth, best_offset, bound_hit = max(threshold_candidates, key=lambda item: item[0])

    ivp_offset = best_offset if growth > 1e-9 * rate_scale else 0.0
    system.signal_frequency_offset = ivp_offset
    transient_alpha, transient_converged, transient_time = integrate_until_photon_numbers_settle(
        system, alpha0=alpha0, t_block=t_final
    )
    def has_nonzero_pair(alpha):
        field_scale, _ = _solver_scales(system, alpha)
        tolerance = max(1e-12, 1e-10 * field_scale)
        return bool(
            abs(alpha[SIGNAL]) > tolerance and abs(alpha[IDLER]) > tolerance
        )

    oscillating_guess = has_nonzero_pair(transient_alpha)

    if oscillating_guess:
        # Remove only the free U(1) phase; no occupation is fixed.
        signal_phase = np.angle(transient_alpha[SIGNAL])
        alpha_gauge = transient_alpha.copy()
        alpha_gauge[SIGNAL] *= np.exp(-1j * signal_phase)
        alpha_gauge[IDLER] *= np.exp(1j * signal_phase)
        amplitude_scale, rate_scale = _solver_scales(system, alpha_gauge)

        def pack(alpha, offset):
            return np.array(
                [
                    alpha[IDLER].real / amplitude_scale,
                    alpha[IDLER].imag / amplitude_scale,
                    alpha[SIGNAL].real / amplitude_scale,
                    alpha[PUMP].real / amplitude_scale,
                    alpha[PUMP].imag / amplitude_scale,
                    offset / rate_scale,
                ],
                dtype=float,
            )

        def unpack(u):
            alpha = np.array(
                [
                    (u[0] + 1j * u[1]) * amplitude_scale,
                    u[2] * amplitude_scale,
                    (u[3] + 1j * u[4]) * amplitude_scale,
                ],
                dtype=np.complex128,
            )
            return alpha, float(u[5] * rate_scale)

        def residual(u):
            alpha, offset = unpack(u)
            return rhs_real(0.0, complex_to_real(alpha), system, offset) / (
                rate_scale * amplitude_scale
            )

        lower = np.array(
            [-np.inf, -np.inf, 0.0, -np.inf, -np.inf,
             -max_frequency_offset / rate_scale]
        )
        upper = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf,
             max_frequency_offset / rate_scale]
        )
        u0 = pack(alpha_gauge, ivp_offset)
        u0[2] = max(u0[2], np.finfo(float).eps)
        u0[5] = np.clip(u0[5], lower[5] + 1e-12, upper[5] - 1e-12)
        lsq = least_squares(
            residual, u0, bounds=(lower, upper), x_scale=1.0,
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=40000,
        )
        alpha_ss, offset = unpack(lsq.x)
        system.signal_frequency_offset = offset
        oscillating = has_nonzero_pair(alpha_ss)
        if not oscillating:
            system.signal_frequency_offset = 0.0
            amplitude_scale, rate_scale = _solver_scales(system, alpha_ss)
            lsq = least_squares(
                lambda u: rhs_real(0.0, u * amplitude_scale, system)
                / (rate_scale * amplitude_scale),
                complex_to_real(alpha_ss / amplitude_scale),
                x_scale=1.0, ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=20000,
            )
            alpha_ss = real_to_complex(lsq.x) * amplitude_scale
            offset = 0.0
    else:
        system.signal_frequency_offset = 0.0
        amplitude_scale, rate_scale = _solver_scales(system, transient_alpha)
        lsq = least_squares(
            lambda u: rhs_real(0.0, u * amplitude_scale, system)
            / (rate_scale * amplitude_scale),
            complex_to_real(transient_alpha / amplitude_scale),
            x_scale=1.0, ftol=1e-12, xtol=1e-12, gtol=1e-12,
            max_nfev=20000,
        )
        alpha_ss = real_to_complex(lsq.x) * amplitude_scale
        offset = 0.0
        oscillating = has_nonzero_pair(alpha_ss)

    frequency_determined = bool(oscillating)
    frequency_offset = float(offset if oscillating else 0.0)
    system.signal_frequency_offset = frequency_offset
    relative_residual = relative_fixed_point_residual(alpha_ss, system, frequency_offset)
    residual_norm = float(np.linalg.norm(rhs_complex(alpha_ss, system, frequency_offset)))
    converged = bool(
        lsq.success and relative_residual <= 1e-8 and transient_converged
    )
    final_offset_bound_hit = bool(
        oscillating and abs(frequency_offset) >= max_frequency_offset * (1.0 - 1e-5)
    )
    return SteadyStateResult(
        alpha_ss=alpha_ss,
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual,
        transient_state=transient_alpha,
        converged=converged,
        signal_frequency_offset_rad_s=frequency_offset,
        frequency_determined=frequency_determined,
        oscillating=oscillating,
        parametric_growth_rate_s_inv=growth,
        frequency_offset_bound_hit=final_offset_bound_hit,
        transient_converged=transient_converged,
        transient_time_s=transient_time,
    )


def solve_steady_state(
    system, alpha0=None, t_final=5e-7, max_frequency_offset=None
):
    """Dispatch to seeded, drive-locked or unseeded, frequency-selected branch."""
    if system.signal_seeded:
        return solve_drive_locked_steady_state(system, alpha0=alpha0, t_final=t_final)
    return solve_unseeded_steady_state(
        system, alpha0=alpha0, max_frequency_offset=max_frequency_offset,
        t_final=t_final,
    )


# -----------------------------------------------------------------------------
# Linearized covariance and spectra around the selected rotating-frame branch
# -----------------------------------------------------------------------------

@dataclass
class LinearizedResult:
    drift_matrix: np.ndarray
    diffusion_matrix: np.ndarray
    covariance: np.ndarray
    stable: bool
    marginally_stable: bool
    spectral_valid: bool
    fixed_point_valid: bool
    covariance_valid: bool
    covariance_physical: bool
    covariance_symplectic_eigenvalues: np.ndarray
    number_observable_covariance: np.ndarray
    number_observable_covariance_valid: bool
    fixed_point_relative_residual: float
    eigvals: np.ndarray


def diffusion_matrix_real(alpha_ss, system):
    # Vacuum Markov diffusion for canonical quadratures X=sqrt(2) Re(a),
    # Y=sqrt(2) Im(a). The real/imag drift matrix is unchanged by this uniform
    # coordinate scaling. Nonlinear-loss noise remains a first-pass model.
    kappa_eff = system.kappa + tpa_losses(alpha_ss, system)
    D = np.zeros((6, 6), dtype=float)
    for m in range(3):
        D[2 * m : 2 * m + 2, 2 * m : 2 * m + 2] = kappa_eff[m] * np.eye(2)
    return D


def symplectic_eigenvalues(covariance):
    """Symplectic eigenvalues for interleaved canonical quadratures."""
    V = np.asarray(covariance, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2:
        raise ValueError("Covariance must be an even-dimensional square matrix")
    n_modes = V.shape[0] // 2
    Omega = np.zeros_like(V)
    for m in range(n_modes):
        Omega[2 * m, 2 * m + 1] = 1.0
        Omega[2 * m + 1, 2 * m] = -1.0
    eigvals = np.linalg.eigvals(1j * Omega @ V)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    if np.max(np.abs(np.imag(eigvals))) > 1e-8 * scale:
        return np.full(n_modes, np.nan)
    return np.sort(np.abs(np.real(eigvals)))[::2]


def linearize_about_steady_state(alpha_ss, system):
    x_ss = complex_to_real(alpha_ss)
    amplitude_scale, rate_scale = _solver_scales(system, alpha_ss)
    u_ss = x_ss / amplitude_scale
    A_scaled = numerical_jacobian(
        lambda u: rhs_real(0.0, u * amplitude_scale, system)
        / (rate_scale * amplitude_scale),
        u_ss,
    )
    A = rate_scale * A_scaled
    D = diffusion_matrix_real(alpha_ss, system)
    eigvals = np.linalg.eigvals(A)
    fixed_point_relative_residual = relative_fixed_point_residual(alpha_ss, system)
    fixed_point_valid = fixed_point_relative_residual <= 1e-8
    stability_tolerance = max(
        1e-9 * rate_scale,
        100.0 * np.finfo(float).eps * max(float(np.linalg.norm(A, 2)), 1.0),
    )
    real_parts = np.real(eigvals)
    stable = bool(fixed_point_valid and np.all(real_parts < -stability_tolerance))
    near_zero = np.abs(eigvals) <= 10.0 * stability_tolerance
    marginally_stable = bool(
        fixed_point_valid
        and np.count_nonzero(near_zero) == 1
        and np.all(real_parts[~near_zero] < -stability_tolerance)
    )
    # The full DC covariance is singular for a free phase, but spectra at
    # nonzero analysis frequency remain defined.
    spectral_valid = bool(fixed_point_valid and np.all(real_parts <= stability_tolerance))

    covariance_physical = False
    covariance_symplectic_eigenvalues = np.full(3, np.nan)
    number_observable_covariance = np.full_like(D, np.nan)
    number_observable_covariance_valid = False
    if stable:
        V = solve_continuous_lyapunov(A, -D)
        V = 0.5 * (V + V.T)
        if np.all(np.isfinite(V)):
            covariance_eigvals, covariance_eigvecs = np.linalg.eigh(V)
            covariance_scale = max(float(np.max(np.abs(covariance_eigvals))), 1.0)
            psd_tolerance = 1e-10 * covariance_scale
            covariance_psd = bool(np.min(covariance_eigvals) >= -psd_tolerance)
            if covariance_psd:
                covariance_eigvals = np.maximum(covariance_eigvals, 0.0)
                V = (covariance_eigvecs * covariance_eigvals) @ covariance_eigvecs.T
                V = 0.5 * (V + V.T)
                covariance_symplectic_eigenvalues = symplectic_eigenvalues(V)
                covariance_physical = bool(
                    np.all(np.isfinite(covariance_symplectic_eigenvalues))
                    and np.all(covariance_symplectic_eigenvalues >= 0.5 - 1e-8)
                )
                covariance_valid = covariance_psd and covariance_physical
            else:
                covariance_valid = False
        else:
            covariance_valid = False
        if not covariance_valid:
            V = np.full_like(D, np.nan)
        else:
            number_observable_covariance = V.copy()
            number_observable_covariance_valid = True
    else:
        covariance_valid = False
        V = np.full_like(D, np.nan)

        # For a single neutral phase mode, solve the Lyapunov equation only on
        # the stable invariant subspace. Photon-number gradients must be
        # orthogonal to the discarded Goldstone direction; then their variances
        # are finite even though the full phase-referenced covariance is not.
        if marginally_stable:
            _, Z, stable_dimension = schur(
                A,
                output="real",
                sort=lambda wr, wi: wr < -stability_tolerance,
            )
            if stable_dimension == A.shape[0] - 1:
                Q = Z[:, :stable_dimension]
                # Use the left projection dual to the stable Schur basis and
                # the right Goldstone vector. An orthogonal projection alone is
                # wrong when phase diffusion is non-normal and feeds stable modes.
                _, _, vh = np.linalg.svd(A)
                null_direction = vh[-1, :]
                coordinate_basis = np.column_stack((Q, null_direction))
                if np.linalg.cond(coordinate_basis) < 1e10:
                    left_coordinates = np.linalg.solve(
                        coordinate_basis, np.eye(A.shape[0])
                    )[:stable_dimension, :]
                    A_stable = left_coordinates @ A @ Q
                    D_stable = left_coordinates @ D @ left_coordinates.T
                    V_stable = solve_continuous_lyapunov(A_stable, -D_stable)
                    V_projected = Q @ V_stable @ Q.T
                    V_projected = 0.5 * (V_projected + V_projected.T)
                else:
                    V_projected = np.full_like(D, np.nan)

                gradients = np.zeros((3, 6), dtype=float)
                for mode in range(3):
                    a = alpha_ss[mode]
                    gradients[mode, 2 * mode] = np.sqrt(2.0) * np.real(a)
                    gradients[mode, 2 * mode + 1] = np.sqrt(2.0) * np.imag(a)
                projection_error = np.linalg.norm(gradients @ null_direction)
                projection_scale = max(float(np.linalg.norm(gradients)), 1.0)
                if np.all(np.isfinite(V_projected)):
                    projected_eigs = np.linalg.eigvalsh(V_projected)
                    projected_scale = max(float(np.max(np.abs(projected_eigs))), 1.0)
                else:
                    projected_eigs = np.array([np.nan])
                    projected_scale = 1.0
                null_residual = np.linalg.norm(A @ null_direction) / max(
                    float(np.linalg.norm(A, 2)), 1.0
                )
                if (
                    np.all(np.isfinite(V_projected))
                    and null_residual <= 1e-8
                    and projection_error <= 1e-8 * projection_scale
                    and np.min(projected_eigs) >= -1e-9 * projected_scale
                ):
                    projected_eigs, projected_vecs = np.linalg.eigh(V_projected)
                    projected_eigs = np.maximum(projected_eigs, 0.0)
                    number_observable_covariance = (
                        projected_vecs * projected_eigs
                    ) @ projected_vecs.T
                    number_observable_covariance = 0.5 * (
                        number_observable_covariance + number_observable_covariance.T
                    )
                    number_observable_covariance_valid = True

    return LinearizedResult(
        drift_matrix=A,
        diffusion_matrix=D,
        covariance=V,
        stable=stable,
        marginally_stable=marginally_stable,
        spectral_valid=spectral_valid,
        fixed_point_valid=fixed_point_valid,
        covariance_valid=covariance_valid,
        covariance_physical=covariance_physical,
        covariance_symplectic_eigenvalues=covariance_symplectic_eigenvalues,
        number_observable_covariance=number_observable_covariance,
        number_observable_covariance_valid=number_observable_covariance_valid,
        fixed_point_relative_residual=fixed_point_relative_residual,
        eigvals=eigvals,
    )


def spectral_covariance(drift_matrix, diffusion_matrix, omega_noise):
    if omega_noise <= 0.0:
        raise ValueError("Use a strictly positive analysis frequency; DC may be singular")
    I = np.eye(drift_matrix.shape[0])
    G = np.linalg.solve(1j * omega_noise * I - drift_matrix, I)
    return G @ diffusion_matrix @ G.conj().T


# -----------------------------------------------------------------------------
# Noise observables
# -----------------------------------------------------------------------------

def number_gradient(alpha_ss, mode_index):
    g = np.zeros(6, dtype=float)
    a = alpha_ss[mode_index]
    g[2 * mode_index] = np.sqrt(2.0) * np.real(a)
    g[2 * mode_index + 1] = np.sqrt(2.0) * np.imag(a)
    return g


def number_variance(alpha_ss, covariance, mode_index):
    if covariance is None or not np.all(np.isfinite(covariance)):
        return np.nan
    g = number_gradient(alpha_ss, mode_index)
    return float(g @ covariance @ g)


def fano_factor(alpha_ss, covariance, mode_index):
    nbar = float(np.abs(alpha_ss[mode_index]) ** 2)
    if nbar <= 0.0:
        return np.nan
    return number_variance(alpha_ss, covariance, mode_index) / nbar


def shot_noise_normalized_pair_variance(alpha_ss, covariance, mode_a, mode_b, sign=-1):
    if covariance is None or not np.all(np.isfinite(covariance)):
        return np.nan
    ga = number_gradient(alpha_ss, mode_a)
    gb = number_gradient(alpha_ss, mode_b)
    g = ga + sign * gb
    numerator = float(g @ covariance @ g)
    denominator = float(np.abs(alpha_ss[mode_a]) ** 2 + np.abs(alpha_ss[mode_b]) ** 2)
    if denominator <= 0.0:
        return np.nan
    return numerator / denominator


def number_noise_spectrum(alpha_ss, linearized, mode_index, omega_noise):
    if not linearized.spectral_valid:
        return np.nan
    Sx = spectral_covariance(
        linearized.drift_matrix,
        linearized.diffusion_matrix,
        omega_noise,
    )
    g = number_gradient(alpha_ss, mode_index)
    return np.real(g @ Sx @ g)


def pair_noise_spectrum(alpha_ss, linearized, mode_a, mode_b, omega_noise, sign=-1):
    if not linearized.spectral_valid:
        return np.nan
    Sx = spectral_covariance(
        linearized.drift_matrix,
        linearized.diffusion_matrix,
        omega_noise,
    )
    g = number_gradient(alpha_ss, mode_a) + sign * number_gradient(alpha_ss, mode_b)
    return np.real(g @ Sx @ g)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def summary_dict(system, steady_state, linearized=None):
    alpha_ss = steady_state.alpha_ss
    sensitivities = intensity_channel_sensitivities(alpha_ss, system)
    chi2_reference = abs(system.g0)
    out = {
        "platform": system.platform_name,
        "chi2_material": system.chi2_material,
        "chi3_material": system.chi3_material,
        "wavelength_regime": system.wavelength_regime,
        "signal_seeded": bool(system.signal_seeded),
        "signal_seed_power_W": float(system.signal_seed_power_W),
        "signal_seed_power_W_actual": float(
            HBAR * system.omega_drive[SIGNAL] * abs(system.drive_amplitudes[SIGNAL])**2
        ),
        "idler_seeded": bool(abs(system.drive_amplitudes[IDLER]) > 0.0),
        "signal_frequency_offset_Hz": float(system.signal_frequency_offset / (2.0 * PI)),
        "pump_input_power_W": float(system.pump_input_power_W),
        "pump_input_power_W_actual": float(
            HBAR * system.omega_drive[PUMP] * abs(system.drive_amplitudes[PUMP]) ** 2
        ),
        "oscillating": bool(steady_state.oscillating),
        "frequency_determined": bool(steady_state.frequency_determined),
        "parametric_growth_rate_s_inv": float(steady_state.parametric_growth_rate_s_inv),
        "frequency_offset_bound_hit": bool(steady_state.frequency_offset_bound_hit),
        "converged": steady_state.converged,
        "transient_converged": steady_state.transient_converged,
        "transient_time_s": steady_state.transient_time_s,
        "residual_norm": steady_state.residual_norm,
        "relative_residual_norm": steady_state.relative_residual_norm,
        "photon_numbers": {MODE_NAMES[m]: float(np.abs(alpha_ss[m]) ** 2) for m in range(3)},
        "reference_photon_numbers": {
            MODE_NAMES[m]: float(system.reference_photon_numbers[m]) for m in range(3)
        },
        "effective_resonance_Hz": {
            MODE_NAMES[m]: float(effective_resonance_omegas(alpha_ss, system)[m] / (2.0 * PI))
            for m in range(3)
        },
        "carrier_frame_Hz": {
            MODE_NAMES[m]: float(carrier_frequencies_for_phase_matching(system)[m] / (2.0 * PI))
            for m in range(3)
        },
        "carrier_wavelength_nm": {
            MODE_NAMES[m]: float(1e9 * lambda_from_omega(carrier_frequencies_for_phase_matching(system)[m]))
            for m in range(3)
        },
        "delta_k_per_m": float(phase_mismatch(alpha_ss, system)),
        "phase_match_x_reference": float(system.phase_match_offset_x),
        "phase_match_x": float(sensitivities["phase_match_x"]),
        "chi2_eff": chi2_eff(alpha_ss, system),
        "chi2_gain_fraction": float(abs(chi2_eff(alpha_ss, system)) / chi2_reference)
        if chi2_reference > 0.0 else np.nan,
        "intensity_channels": {
            "cavity_kerr_A": bool(system.use_cavity_kerr),
            "phase_match_kerr_B": bool(system.use_phase_match_kerr),
        },
        "channel_sensitivities_per_photon": {
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in sensitivities.items()
        },
    }
    if linearized is not None:
        number_covariance = (
            linearized.number_observable_covariance
            if linearized.number_observable_covariance_valid else linearized.covariance
        )
        out["stable_linearization"] = linearized.stable
        out["marginally_stable_phase_mode"] = linearized.marginally_stable
        out["spectral_valid"] = linearized.spectral_valid
        out["fixed_point_valid_for_linearization"] = linearized.fixed_point_valid
        out["covariance_valid"] = linearized.covariance_valid
        out["covariance_physical"] = linearized.covariance_physical
        out["number_observable_covariance_valid"] = linearized.number_observable_covariance_valid
        out["covariance_symplectic_eigenvalues"] = linearized.covariance_symplectic_eigenvalues.tolist()
        out["fixed_point_relative_residual"] = linearized.fixed_point_relative_residual
        out["max_real_eigenvalue_s_inv"] = float(np.max(np.real(linearized.eigvals)))
        out["fano_factors"] = {
            MODE_NAMES[m]: float(fano_factor(alpha_ss, number_covariance, m)) for m in range(3)
        }
        out["pair_noise"] = {
            "signal_minus_idler": float(
                shot_noise_normalized_pair_variance(alpha_ss, number_covariance, SIGNAL, IDLER, sign=-1)
            ),
            "signal_plus_idler": float(
                shot_noise_normalized_pair_variance(alpha_ss, number_covariance, SIGNAL, IDLER, sign=+1)
            ),
        }
    return out


def scan_phase_match_offsets(
    x_values,
    wavelength_regime="nondegenerate",
    signal_seeded=True,
    signal_seed_power_W=1e-3,
    pump_input_power_W=1.0,
    reference_photon_numbers=None,
):
    """Solve and report the actual steady state over requested first-lobe x0 values."""
    rows = []
    for x0 in x_values:
        system = default_system(
            phase_match_offset_x=float(x0),
            wavelength_regime=wavelength_regime,
            signal_seeded=signal_seeded,
            signal_seed_power_W=signal_seed_power_W,
            pump_input_power_W=pump_input_power_W,
            reference_photon_numbers=reference_photon_numbers,
        )
        steady = solve_steady_state(system)
        linearized = linearize_about_steady_state(steady.alpha_ss, system)
        rows.append(summary_dict(system, steady, linearized))
    return rows


def compare_intensity_channels(
    phase_match_offset_x=0.5,
    wavelength_regime="nondegenerate",
    signal_seeded=True,
    signal_seed_power_W=1e-3,
    pump_input_power_W=1.0,
    reference_photon_numbers=None,
):
    """Compare self-consistent runs with cavity Kerr A and phase-match Kerr B toggled."""
    rows = []
    for use_a in (False, True):
        for use_b in (False, True):
            system = default_system(
                phase_match_offset_x=phase_match_offset_x,
                wavelength_regime=wavelength_regime,
                signal_seeded=signal_seeded,
                signal_seed_power_W=signal_seed_power_W,
                pump_input_power_W=pump_input_power_W,
                reference_photon_numbers=reference_photon_numbers,
            )
            system.use_cavity_kerr = use_a
            system.use_phase_match_kerr = use_b
            steady = solve_steady_state(system)
            linearized = linearize_about_steady_state(steady.alpha_ss, system)
            rows.append(summary_dict(system, steady, linearized))
    return rows


def pretty_print_summary(summary):
    print("platform:", summary["platform"])
    print("chi2:", summary["chi2_material"])
    print("chi3:", summary["chi3_material"])
    print("wavelength regime:", summary["wavelength_regime"])
    print("signal seeded:", summary["signal_seeded"])
    print("signal seed power configured [W]:", summary["signal_seed_power_W"])
    print("signal seed power from drive amplitude [W]:", summary["signal_seed_power_W_actual"])
    print("idler seeded:", summary["idler_seeded"])
    print("pump input power configured [W]:", summary["pump_input_power_W"])
    print("pump input power from drive amplitude [W]:", summary["pump_input_power_W_actual"])
    print("free-running signal offset [Hz]:", summary["signal_frequency_offset_Hz"])
    print("frequency determined:", summary["frequency_determined"])
    print("pump-only signal/idler growth rate [1/s]:", summary["parametric_growth_rate_s_inv"])
    print("frequency search bound hit:", summary["frequency_offset_bound_hit"])
    print("oscillating:", summary["oscillating"])
    print("converged:", summary["converged"])
    print("transient occupations converged:", summary["transient_converged"])
    print("transient time [s]:", summary["transient_time_s"])
    print("residual norm:", summary["residual_norm"])
    print("relative residual norm:", summary["relative_residual_norm"])
    print("photon numbers:")
    for k, v in summary["photon_numbers"].items():
        print(f"  {k:>6s}: {v:.6e}")
    print("reference photon numbers:")
    for k, v in summary["reference_photon_numbers"].items():
        print(f"  {k:>6s}: {v:.6e}")
    print("carrier frequencies [Hz]:")
    for k, v in summary["carrier_frame_Hz"].items():
        print(f"  {k:>6s}: {v:.6e}")
    print("carrier wavelengths [nm]:")
    for k, v in summary["carrier_wavelength_nm"].items():
        print(f"  {k:>6s}: {v:.6f}")
    print("effective cavity frequencies [Hz]:")
    for k, v in summary["effective_resonance_Hz"].items():
        print(f"  {k:>6s}: {v:.6e}")
    print("delta_k [1/m]:", summary["delta_k_per_m"])
    print("phase-match x at reference:", summary["phase_match_x_reference"])
    print("phase-match x at solved state:", summary["phase_match_x"])
    print("chi2 gain fraction:", summary["chi2_gain_fraction"])
    print("intensity channels:", summary["intensity_channels"])
    print("channel sensitivities per photon:")
    for key, value in summary["channel_sensitivities_per_photon"].items():
        print(f"  {key}: {value}")
    print("chi2_eff:", summary["chi2_eff"])
    if "stable_linearization" in summary:
        print("strictly stable linearization:", summary["stable_linearization"])
        print("marginal OPO phase mode:", summary["marginally_stable_phase_mode"])
        print("spectral analysis valid at Omega > 0:", summary["spectral_valid"])
        print("fixed point valid for linearization:", summary["fixed_point_valid_for_linearization"])
        print("covariance valid:", summary["covariance_valid"])
        print("full state covariance physical:", summary["covariance_physical"])
        print("number-observable covariance available:", summary["number_observable_covariance_valid"])
        print("covariance symplectic eigenvalues:", summary["covariance_symplectic_eigenvalues"])
        print("linearization relative residual:", summary["fixed_point_relative_residual"])
        print("max real drift eigenvalue [1/s]:", summary["max_real_eigenvalue_s_inv"])
        print("Fano factors:")
        for k, v in summary["fano_factors"].items():
            print(f"  {k:>6s}: {v:.6e}")
        print("pair noise / shot-noise level:")
        for k, v in summary["pair_noise"].items():
            print(f"  {k:>18s}: {v:.6e}")


# -----------------------------------------------------------------------------
# Oscillating operating point: direct design, branch solve and attractor tests
# -----------------------------------------------------------------------------
# Kerr bistability of the pump makes a cold start settle on the low pump-only
# branch. The oscillating branch is therefore constructed directly. Pick the
# signal occupation, the effective detunings at that state and the pump
# detuning; the pump (and seed) drives that make it an exact fixed point follow
# algebraically, and the pump must sit above its Kerr-shifted threshold.

def design_oscillating_state(
    system, n_signal=2e6, pump_detuning_over_kappa=2.0,
    pair_detuning_over_kappa=0.0, seed_deficit=0.0, seed_offset_over_kappa=0.0,
):
    """Re-center `system` on an oscillating fixed point and set its drives.

    n_signal: signal photon number of the target state.
    pump_detuning_over_kappa: effective pump detuning at the state, in units of kappa_p.
        Positive values put the pump on the Kerr-stable side of its resonance.
    pair_detuning_over_kappa: common effective signal detuning at the state (the idler
        detuning follows so that a zero-signal-input solution exists).
    seed_deficit, seed_offset_over_kappa: both 0 gives the free-running state (zero signal
        input). A deficit in (0, 1) lowers the pump occupation to (1 - deficit) of the
        clamped value; an offset w moves the signal and idler detunings to D + w and D - w
        (units of kappa). Either way a signal seed supplies the balance, which gives the
        frequency-locked (seeded) branch. An offset with zero deficit is injection locking
        of the free-running oscillator at unchanged pump power.
    Returns the target complex amplitudes [idler, signal, pump].
    """
    if np.any(np.asarray(system.tpa, dtype=float) != 0.0):
        raise NotImplementedError("Operating-point design assumes zero TPA")
    if not 0.0 <= seed_deficit < 1.0 or n_signal <= 0.0:
        raise ValueError("Need n_signal > 0 and 0 <= seed_deficit < 1")
    kap = np.asarray(system.kappa, dtype=float)
    d_mean = float(pair_detuning_over_kappa) * kap[SIGNAL]
    d_off = float(seed_offset_over_kappa) * kap[SIGNAL]
    d_s = d_mean + d_off
    d_i = (d_mean - d_off) * kap[IDLER] / kap[SIGNAL]
    d_p = float(pump_detuning_over_kappa) * kap[PUMP]

    x0 = float(system.phase_match_offset_x)
    gain = abs(system.g0) * float(sinc_unscaled(x0))
    n_pump = (1.0 - seed_deficit) * (kap[SIGNAL] * kap[IDLER] + d_mean**2) / gain**2
    a_p, a_s = np.sqrt(n_pump), np.sqrt(n_signal)
    g_phase = system.g0 * np.exp(1j * x0) / abs(system.g0)
    a_i = gain * g_phase * a_p * a_s / (kap[IDLER] - 1j * d_i)
    target = np.array([a_i, a_s, a_p], dtype=np.complex128)
    n_target = photon_numbers(target)

    # Re-center resonances and the QPM period on the target occupations.
    system.omega_bare = system.omega_drive / (1.0 + system.cavity_pull.T @ n_target)
    system.kerr_rate = system.cavity_pull * system.omega_bare[None, :]
    system.reference_photon_numbers = n_target
    system.detuning_at_reference = np.array([d_i, d_s, d_p], dtype=float)
    system.signal_frequency_offset = 0.0
    system.delta_k_at_reference = 2.0 * x0 / system.crystal_length
    system.poling_period = qpm_period_for_target_mismatch(
        omega_carriers=system.omega_drive, photon_numbers_center=n_target,
        direct_delta_k_pull=system.direct_delta_k_pull,
        target_delta_k=system.delta_k_at_reference, temperature_C=system.temperature_C,
    )
    seeded = bool(seed_deficit > 0.0 or seed_offset_over_kappa != 0.0)
    system.signal_seeded = seeded
    system.drive_amplitudes = np.zeros(3, dtype=np.complex128)
    system.drive_amplitudes = drive_amplitudes_for_target_state(target, system, signal_seeded=seeded)
    system.signal_seed_power_W = float(
        HBAR * system.omega_drive[SIGNAL] * abs(system.drive_amplitudes[SIGNAL]) ** 2
    )
    system.pump_input_power_W = float(
        HBAR * system.omega_drive[PUMP] * abs(system.drive_amplitudes[PUMP]) ** 2
    )
    if relative_fixed_point_residual(target, system) > 1e-8:
        raise RuntimeError("Designed oscillating state is not a fixed point")
    return target


def solve_oscillating_branch(system, alpha_guess):
    """Solve the full fixed-point equations directly from `alpha_guess`.

    Seeded: six real unknowns. Free-running: the global signal/idler phase is gauge-fixed
    (signal amplitude real) and the signal frequency offset is solved with the amplitudes.
    No time integration is used.
    """
    alpha_guess = np.asarray(alpha_guess, dtype=np.complex128)
    amplitude_scale, rate_scale = _solver_scales(system, alpha_guess)
    scale = rate_scale * amplitude_scale
    free = not system.signal_seeded
    if free:
        gauge = alpha_guess[SIGNAL] / abs(alpha_guess[SIGNAL])
        alpha_guess = np.array([alpha_guess[IDLER] * gauge, abs(alpha_guess[SIGNAL]), alpha_guess[PUMP]])

        def unpack(u):
            alpha = np.array([u[0] + 1j * u[1], u[2], u[3] + 1j * u[4]]) * amplitude_scale
            return alpha, float(u[5]) * rate_scale

        u0 = np.array([alpha_guess[IDLER].real, alpha_guess[IDLER].imag, alpha_guess[SIGNAL].real,
                       alpha_guess[PUMP].real, alpha_guess[PUMP].imag, system.signal_frequency_offset / rate_scale])
        u0[:5] /= amplitude_scale
    else:
        def unpack(u):
            return real_to_complex(u[:6] * amplitude_scale), 0.0

        u0 = complex_to_real(alpha_guess / amplitude_scale)

    def residual(u):
        alpha, offset = unpack(u)
        return rhs_real(0.0, complex_to_real(alpha), system, offset) / scale

    lsq = least_squares(residual, u0, x_scale=1.0, ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
    alpha_ss, offset = unpack(lsq.x)
    if abs(offset) <= 1e-9 * rate_scale:
        offset = 0.0  # solver noise on a state that sits exactly on the reference carriers
    system.signal_frequency_offset = offset
    rel = relative_fixed_point_residual(alpha_ss, system, offset)
    n = photon_numbers(alpha_ss)
    return SteadyStateResult(
        alpha_ss=alpha_ss,
        residual_norm=float(np.linalg.norm(rhs_complex(alpha_ss, system, offset))),
        relative_residual_norm=rel,
        transient_state=alpha_guess,
        converged=bool(rel <= 1e-8),
        signal_frequency_offset_rad_s=offset,
        frequency_determined=True,
        oscillating=bool(n[SIGNAL] > 0.0 and n[IDLER] > 0.0),
        parametric_growth_rate_s_inv=np.nan,
        transient_converged=True,
        transient_time_s=0.0,
    )


def pump_only_branches(system, signal_frequency_offset=0.0):
    """All real pump-only fixed points (idler = signal = 0), sorted by pump photon number.

    Solves the Kerr cubic n [kappa^2 + (A + B n)^2] = |drive|^2 and polishes the roots.
    """
    kap = float(system.kappa[PUMP])
    drive = np.sqrt(2.0 * external_coupling_rates(system)[PUMP]) * system.drive_amplitudes[PUMP]
    d2 = float(abs(drive) ** 2)
    zero = np.zeros(3, dtype=np.complex128)
    unit = zero.copy()
    unit[PUMP] = 1.0
    A = float(detunings(zero, system, signal_frequency_offset)[PUMP])
    # Kerr slope taken from the matrix: differencing detunings (|A| ~ 4e9) loses 8 digits.
    B = -float(system.kerr_rate[PUMP, PUMP]) if system.use_cavity_kerr else 0.0
    f = lambda n: n * (kap**2 + (A + B * n) ** 2) - d2
    df = lambda n: kap**2 + (A + B * n) ** 2 + 2.0 * B * n * (A + B * n)
    rate = max(kap, abs(A))
    roots = np.roots([B**2, 2 * A * B, A**2 + kap**2, -d2]) if B != 0.0 else np.array([d2 / (kap**2 + A**2)])
    out = []
    for r in roots:
        if abs(r.imag) > 1e-6 * max(1.0, abs(r.real)) or r.real < 0.0:
            continue
        n = float(r.real)
        for _ in range(30):
            step = f(n) / df(n) if df(n) != 0.0 else 0.0
            n -= step
            if abs(step) <= 1e-14 * max(n, 1.0):
                break
        if n >= 0.0 and abs(f(n)) <= 1e-9 * max(d2, rate**2):
            if not out or abs(n - out[-1]) > 1e-9 * max(n, 1.0):
                out.append(n)
    states = []
    for n in sorted(out):
        a = zero.copy()
        a[PUMP] = np.sqrt(n)
        delta_p = detunings(a, system, signal_frequency_offset)[PUMP]
        a[PUMP] = drive / (kap - 1j * delta_p)
        states.append(a)
    return states


def pump_only_pair_growth(system, alpha_pump_only, max_frequency_offset=None):
    """Largest signal/idler growth rate [1/s] about a pump-only state, maximized over offset."""
    if max_frequency_offset is None:
        max_frequency_offset = 20.0 * float(np.max(system.kappa))
    growth, offset, bound = _maximize_unseeded_small_signal_growth(
        system, alpha_pump_only[PUMP], max_frequency_offset
    )
    return growth, offset


def integrate_trajectory(system, alpha0, t_final, n_samples=400, pump_detuning_ramp=None, rtol=1e-9):
    """Integrate the full three-mode dynamics and return (t [s], alpha(t) with shape (n, 3)).

    pump_detuning_ramp: optional callable t -> pump detuning at the reference [rad/s]
    (laser sweep). The system is restored afterwards. Samples are log-spaced in time.
    """
    amplitude_scale, rate_scale = _solver_scales(system, alpha0)
    saved = system.detuning_at_reference.copy()

    def fun(tau, u):
        if pump_detuning_ramp is not None:
            system.detuning_at_reference[PUMP] = pump_detuning_ramp(tau / rate_scale)
        return rhs_real(0.0, u * amplitude_scale, system) / (rate_scale * amplitude_scale)

    t_eval = np.concatenate([[0.0], np.geomspace(t_final * 1e-4, t_final, n_samples - 1)])
    try:
        sol = solve_ivp(
            fun, (0.0, rate_scale * t_final),
            complex_to_real(np.asarray(alpha0, dtype=np.complex128) / amplitude_scale),
            method="LSODA", t_eval=t_eval * rate_scale, rtol=rtol, atol=1e-12,
        )
    finally:
        system.detuning_at_reference[:] = saved
    alpha_t = np.vstack([real_to_complex(sol.y[:, j] * amplitude_scale) for j in range(sol.y.shape[1])])
    return sol.t / rate_scale, alpha_t


def attractor_checks(system, alpha_ss, t_final=1e-3, perturbation=0.05, seed_photons=1.0, rng_seed=1):
    """Integrate from (a) a perturbed steady state and (b) a small seed on the pump-only
    upper branch, and (c) from a cold start with all modes at the seed level.

    Returns a dict of photon-number trajectories and final states. The cold start is a
    diagnostic: Kerr bistability can trap it on the low pump-only branch.
    """
    rng = np.random.default_rng(rng_seed)
    n_ss = photon_numbers(alpha_ss)
    pert = 1.0 + perturbation * np.exp(2j * PI * rng.random(3)) * np.array([1.0, 1.0, 1.0])
    runs = {"perturbed": np.asarray(alpha_ss) * pert}
    branches = pump_only_branches(system, system.signal_frequency_offset)
    upper = branches[-1] if branches else np.zeros(3, dtype=np.complex128)
    seed = np.sqrt(seed_photons) * np.exp(2j * PI * rng.random(3))
    runs["small_seed_on_pump_branch"] = np.array([seed[0], seed[1], upper[PUMP]])
    runs["cold_start"] = seed.astype(np.complex128)
    out = {"n_steady": n_ss, "pump_only_branches_n": [float(abs(b[PUMP]) ** 2) for b in branches]}
    for name, a0 in runs.items():
        t, a_t = integrate_trajectory(system, a0, t_final)
        n_t = photon_numbers(a_t)
        out[name] = {"t": t, "n": n_t, "n_final": n_t[-1],
                     "rel_diff_signal_idler": float(np.max(np.abs(n_t[-1, :2] - n_ss[:2]) / n_ss[:2]))}
    return out


# -----------------------------------------------------------------------------
# All fixed points: interval branch and bound over a rigorous bounded region
# -----------------------------------------------------------------------------
# With phases eliminated, the fixed-point equations depend only on the photon
# numbers (free-running: n_s, n_p after solving the neutral pair conditions;
# seeded: n_i, n_s, n_p). Interval arithmetic on the polynomial form of these
# equations discards every box that cannot contain a root, so the survivors
# cover all roots inside the region 0 <= n <= bound, where the bound follows
# from photon-flux balance (derived in `fixed_point_bounds`). The survivors are
# polished by Newton's method and then by the full six-variable solve.

class _Iv:
    """Vectorized closed interval [lo, hi] with the few operations needed here."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)

    @staticmethod
    def _c(o):
        return o if isinstance(o, _Iv) else _Iv(o)

    def __add__(self, o):
        o = _Iv._c(o)
        return _Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return _Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _Iv._c(o)
        return _Iv(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        return _Iv._c(o) - self

    def __mul__(self, o):
        o = _Iv._c(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return _Iv(np.minimum.reduce(p), np.maximum.reduce(p))

    __rmul__ = __mul__

    def sq(self):
        a, b = self.lo ** 2, self.hi ** 2
        lo = np.where((self.lo <= 0.0) & (self.hi >= 0.0), 0.0, np.minimum(a, b))
        return _Iv(lo, np.maximum(a, b))

    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))


def _sinc2(v):
    v = np.asarray(v, dtype=float)
    small = np.abs(v) < 1e-6
    return np.where(small, 1.0 - v * v / 3.0, (np.sin(v) / np.where(small, 1.0, v)) ** 2)


_SINC_CRIT = None


def _sinc2_interval(x):
    """Exact range of sinc^2 over [x.lo, x.hi] (critical points: tan v = v, and zeros k*pi)."""
    global _SINC_CRIT
    f1, f2 = _sinc2(x.lo), _sinc2(x.hi)
    lo, hi = np.minimum(f1, f2), np.maximum(f1, f2)
    inside0 = (x.lo <= 0.0) & (x.hi >= 0.0)
    hi = np.where(inside0, 1.0, hi)
    wide = (np.maximum(np.abs(x.lo), np.abs(x.hi)) >= 3.0)
    if np.any(wide):
        if _SINC_CRIT is None:
            crit = []
            for k in range(1, 40):
                lo_, hi_ = k * PI + 1e-9, (k + 0.5) * PI - 1e-9
                f = lambda v: np.tan(v) - v
                a, b = lo_, hi_
                for _ in range(200):
                    m = 0.5 * (a + b)
                    if f(m) < 0.0:
                        a = m
                    else:
                        b = m
                crit.append(0.5 * (a + b))
            _SINC_CRIT = (np.array(crit), PI * np.arange(1, 41))
        crit, zeros = _SINC_CRIT
        w = np.nonzero(wide)[0] if lo.ndim else np.array([0])
        L_, H_ = np.atleast_1d(x.lo)[w], np.atleast_1d(x.hi)[w]
        lo_w, hi_w = np.atleast_1d(lo)[w].copy(), np.atleast_1d(hi)[w].copy()
        for c in crit:
            for s in (c, -c):
                m = (L_ <= s) & (s <= H_)
                hi_w = np.where(m, np.maximum(hi_w, _sinc2(s)), hi_w)
        for z in zeros:
            for s in (z, -z):
                m = (L_ <= s) & (s <= H_)
                lo_w = np.where(m, 0.0, lo_w)
        beyond = np.maximum(np.abs(L_), np.abs(H_)) > crit[-1]
        hi_w = np.where(beyond, 1.0, hi_w)
        lo_w = np.where(beyond, 0.0, lo_w)
        lo, hi = np.atleast_1d(lo).copy(), np.atleast_1d(hi).copy()
        lo[w], hi[w] = lo_w, hi_w
    return _Iv(lo, hi)


class FixedPointProblem:
    """Reduced fixed-point equations of one system, in interval or point arithmetic.

    Free-running (no signal drive): unknowns (n_s, n_p). The neutral pair conditions give
    n_i = (kappa_s/kappa_i) n_s, a common detuning ratio t = delta_s/kappa_s = delta_i/kappa_i
    that is linear in the photon numbers, and the carrier offset omega. Two equations remain,
    the pair threshold condition and the pump balance.
    Seeded: unknowns (n_i, n_s, n_p) and three modulus equations.
    """

    def __init__(self, system):
        if np.any(np.asarray(system.tpa, dtype=float) != 0.0):
            raise NotImplementedError("fixed-point enumeration assumes zero TPA")
        self.system = system
        s = system
        self.kap = np.asarray(s.kappa, dtype=float)
        self.K = np.asarray(s.kerr_rate, dtype=float) if s.use_cavity_kerr else np.zeros((3, 3))
        self.q = np.asarray(s.q_dk, dtype=float) if s.use_phase_match_kerr else np.zeros(3)
        self.nref = np.asarray(s.reference_photon_numbers, dtype=float)
        self.d0 = np.asarray(s.detuning_at_reference, dtype=float).copy()
        self.L = float(s.crystal_length)
        self.g2 = abs(complex(s.g0)) ** 2
        self.drive = np.sqrt(2.0 * external_coupling_rates(s)) * np.asarray(s.drive_amplitudes, dtype=complex)
        self.free = bool(abs(self.drive[SIGNAL]) == 0.0)
        self.omega0 = 0.0 if self.free else float(s.signal_frequency_offset)
        self.dk_ref = float(s.delta_k_at_reference)
        self.n_unknown = 2 if self.free else 3
        # frequency-dependent part of delta-k: cubic fit (free-running unknown omega); exact at the root afterwards
        om = np.linspace(-6e10, 6e10, 241)
        corr = np.array([delta_k_frequency_correction(s, w) for w in om])
        self.corr_poly = np.polyfit(om / 1e10, corr, 5)
        self.corr_err = float(np.max(np.abs(np.polyval(self.corr_poly, om / 1e10) - corr)))
        self.bounds = self._bounds()

    # ---- rigorous region -------------------------------------------------------------------
    def _bounds(self):
        kp, ks, ki = self.kap[PUMP], self.kap[SIGNAL], self.kap[IDLER]
        d = abs(self.drive[PUMP])
        np_max = d ** 2 / kp ** 2                      # |a_p| (kappa_p) <= |drive|
        ni_max = d ** 2 / (4.0 * kp * ki)              # kappa_i n_i = X <= |d| sqrt(n_p) - kappa_p n_p
        if self.free:
            ns_max = ni_max * ki / ks                  # n_i = (kappa_s/kappa_i) n_s
        else:
            sg = abs(self.drive[SIGNAL])               # kappa_s n_s = X + Re(s a_s*) <= kappa_i n_i + |s| sqrt(n_s)
            ns_max = ((sg + np.sqrt(sg ** 2 + 4.0 * ks * ki * ni_max)) / (2.0 * ks)) ** 2
        return dict(idler=ni_max, signal=ns_max, pump=np_max)

    def upper(self):
        b = self.bounds
        return np.array([b["signal"], b["pump"]]) if self.free else np.array([b["idler"], b["signal"], b["pump"]])

    # ---- residuals -------------------------------------------------------------------------
    def _delta(self, N, omega):
        """Detunings (interval or point); N = [n_i, n_s, n_p]; omega may be an interval or float."""
        dn = [N[m] - self.nref[m] for m in range(3)]
        out = []
        for j in range(3):
            acc = None
            for l in range(3):
                term = dn[l] * (-self.K[l, j])
                acc = term if acc is None else acc + term
            base = self.d0[j] + (-omega if j == IDLER else omega if j == SIGNAL else 0.0)
            out.append(acc + base)
        return out

    def _G(self, N, omega):
        acc = None
        for l in range(3):
            term = (N[l] - self.nref[l]) * self.q[l]
            acc = term if acc is None else acc + term
        shift = self.dk_ref
        if self.free:
            w = omega * 1e-10
            poly = None
            for c in self.corr_poly:
                poly = _Iv(c) if poly is None else poly * w + c
            shift = shift + poly + _Iv(-2 * self.corr_err, 2 * self.corr_err)
        else:
            shift = shift + delta_k_frequency_correction(self.system, self.omega0)
        x = (acc + shift) * (0.5 * self.L)
        return _sinc2_interval(x) * self.g2

    def residuals(self, lo, hi):
        """Interval residuals and term scales for boxes lo, hi of shape (m, n_unknown)."""
        ki, ks, kp = self.kap[IDLER], self.kap[SIGNAL], self.kap[PUMP]
        d2 = abs(self.drive[PUMP]) ** 2
        if self.free:
            Ns, Np = _Iv(lo[:, 0], hi[:, 0]), _Iv(lo[:, 1], hi[:, 1])
            N = [Ns * (ks / ki), Ns, Np]
            # t from delta_s + delta_i = (kappa_s + kappa_i) t with the omega-independent sum
            dsum = None
            for l in range(3):
                term = (N[l] - self.nref[l]) * (-(self.K[l, SIGNAL] + self.K[l, IDLER]))
                dsum = term if dsum is None else dsum + term
            t = (dsum + (self.d0[SIGNAL] + self.d0[IDLER])) * (1.0 / (ks + ki))
            dn_s = None
            for l in range(3):
                term = (N[l] - self.nref[l]) * (-self.K[l, SIGNAL])
                dn_s = term if dn_s is None else dn_s + term
            omega = t * ks - self.d0[SIGNAL] - dn_s                        # delta_s = d0_s + omega + dn_s part
            G = self._G(N, omega)
            dp = self._delta(N, omega)[PUMP]
            t2 = t.sq()
            e2 = Np * G - (t2 + 1.0) * (ks * ki)
            re = (t * dp * (-1.0) + kp) * ki + G * Ns
            im = (dp + t * kp) * (-ki)
            e4 = Np * (re.sq() + im.sq()) - (t2 + 1.0) * (d2 * ki ** 2)
            scales = [(Np * G).mag() + (t2 + 1.0).mag() * ks * ki,
                      (Np * (re.sq() + im.sq())).mag() + (t2 + 1.0).mag() * d2 * ki ** 2]
            return [e2, e4], scales
        Ni, Ns, Np = (_Iv(lo[:, k], hi[:, k]) for k in range(3))
        N = [Ni, Ns, Np]
        dl = self._delta(N, self.omega0)
        G = self._G(N, self.omega0)
        di, ds, dp = dl[IDLER], dl[SIGNAL], dl[PUMP]
        mi = di.sq() + ki ** 2
        e2 = Ni * mi - G * Np * Ns
        re1 = ds * di + ks * ki - G * Np
        im1 = di * ks - ds * ki
        e1 = Ns * (re1.sq() + im1.sq()) - mi * abs(self.drive[SIGNAL]) ** 2
        re3 = G * Ns + kp * ki - dp * di
        im3 = (di * kp + dp * ki) * (-1.0)
        e3 = Np * (re3.sq() + im3.sq()) - mi * d2
        scales = [(Ni * mi).mag() + (G * Np * Ns).mag(),
                  (Ns * (re1.sq() + im1.sq())).mag() + (mi * abs(self.drive[SIGNAL]) ** 2).mag(),
                  (Np * (re3.sq() + im3.sq())).mag() + (mi * d2).mag()]
        return [e2, e1, e3], scales

    def normalized(self, u):
        """Point residuals normalized by their term scales, for Newton. u shape (m, n_unknown)."""
        res, sc = self.residuals(u, u)
        return np.stack([r.lo / np.maximum(s, 1e-300) for r, s in zip(res, sc)], axis=1)

    # ---- amplitudes from photon numbers -----------------------------------------------------
    def amplitudes(self, n):
        """Complex amplitudes [a_i, a_s, a_p], carrier offset omega for one root of photon numbers n."""
        ki, ks, kp = self.kap[IDLER], self.kap[SIGNAL], self.kap[PUMP]
        n = np.asarray(n, dtype=float)
        if self.free:
            N = [ks / ki * n[0], n[0], n[1]]
            dn = np.array(N) - self.nref
            t = (self.d0[SIGNAL] + self.d0[IDLER] - (self.K[:, SIGNAL] + self.K[:, IDLER]) @ dn) / (ks + ki)
            omega = ks * t - self.d0[SIGNAL] + self.K[:, SIGNAL] @ dn
            deltas = self.d0 + np.array([-omega, omega, 0.0]) - self.K.T @ dn
        else:
            N = list(n)
            omega = self.omega0
            dn = np.array(N) - self.nref
            deltas = self.d0 + np.array([-omega, omega, 0.0]) - self.K.T @ dn
        probe = np.sqrt(np.array(N, dtype=complex))
        g = chi2_eff(probe, self.system, omega)
        G = abs(g) ** 2
        if self.free:
            a_s = np.sqrt(N[SIGNAL])
        else:
            a_s = self.drive[SIGNAL] / ((ks - 1j * deltas[SIGNAL]) - G * N[PUMP] / (ki + 1j * deltas[IDLER]))
        a_p = self.drive[PUMP] / (kp - 1j * deltas[PUMP] + G * N[SIGNAL] / (ki - 1j * deltas[IDLER]))
        a_i = g * a_p * np.conj(a_s) / (ki - 1j * deltas[IDLER])
        return np.array([a_i, a_s, a_p], dtype=complex), float(omega)


def fixed_point_bounds(system):
    """Rigorous photon-number bounds [idler, signal, pump] containing every fixed point.

    Photon-flux balance: kappa_i n_i = X, kappa_s n_s = X + Re(s a_s*), kappa_p n_p + X = Re(d a_p*) with
    X = Re(g a_p a_s* a_i*). Hence n_p <= |d|^2/kappa_p^2 and X <= |d|^2/(4 kappa_p).
    """
    return FixedPointProblem(system).bounds


def _bnb_boxes(problem, per_decade=5, floor=1e-6, rel_tol=2e-3, chunk=400_000, max_boxes=30_000_000):
    """Interval branch and bound. Returns (lo, hi) arrays of surviving boxes of relative width <= rel_tol."""
    ub = problem.upper() * 1.0000001
    nd = problem.n_unknown
    edges = []
    for u in ub:
        dec = max(np.log10(u / floor), 1.0)
        e = np.concatenate([[0.0], np.geomspace(floor, u, int(np.ceil(dec * per_decade)) + 1)])
        edges.append(e)
    grids = np.meshgrid(*[np.arange(len(e) - 1) for e in edges], indexing="ij")
    lo = np.stack([edges[k][g.ravel()] for k, g in enumerate(grids)], axis=1)
    hi = np.stack([edges[k][g.ravel() + 1] for k, g in enumerate(grids)], axis=1)

    def keep(lo, hi):
        out = np.ones(len(lo), dtype=bool)
        for s in range(0, len(lo), chunk):
            res, sc = problem.residuals(lo[s:s + chunk], hi[s:s + chunk])
            m = np.ones(min(chunk, len(lo) - s), dtype=bool)
            for r, c in zip(res, sc):
                tol = 1e-11 * c
                m &= (r.lo <= tol) & (r.hi >= -tol)
            out[s:s + chunk] = m
        return out

    def done_dims(lo, hi):
        w = hi - lo
        return (w <= rel_tol * hi) | (w <= 2.0 * floor)

    m = keep(lo, hi)
    lo, hi = lo[m], hi[m]
    for it in range(80):
        dn = done_dims(lo, hi)
        if np.all(dn):
            break
        for j in range(nd):
            todo = ~dn[:, j]
            if not np.any(todo):
                continue
            a, b = lo[todo], hi[todo]
            mid = np.where(a > 0.0, np.sqrt(a * b), 0.5 * b)
            lo_new, hi_new = a.copy(), b.copy()
            hi_new[:, j] = mid[:, j]
            lo2, hi2 = a.copy(), b.copy()
            lo2[:, j] = mid[:, j]
            lo = np.concatenate([lo[~todo], lo_new, lo2])
            hi = np.concatenate([hi[~todo], hi_new, hi2])
            m = keep(lo, hi)
            lo, hi = lo[m], hi[m]
            dn = done_dims(lo, hi)
            if len(lo) > max_boxes:
                raise RuntimeError("interval search exceeded the box budget")
    return lo, hi


def _newton_batch(problem, u0, iters=60):
    """Batch Newton on normalized residuals in log photon numbers. Returns (u, max |F|)."""
    v = np.log(np.maximum(u0, 1e-12))
    h = 1e-6
    n = v.shape[1]
    for _ in range(iters):
        F = problem.normalized(np.exp(v))
        J = np.empty((len(v), n, n))
        for k in range(n):
            dv = np.zeros(n)
            dv[k] = h
            J[:, :, k] = (problem.normalized(np.exp(v + dv)) - problem.normalized(np.exp(v - dv))) / (2 * h)
        try:
            step = np.linalg.solve(J, -F[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(v)
            for i in range(len(v)):
                step[i] = np.linalg.lstsq(J[i], -F[i], rcond=None)[0]
        big = np.max(np.abs(step), axis=1, keepdims=True)
        step *= np.minimum(1.0, 0.5 / np.maximum(big, 1e-300))
        v = v + step
        v = np.clip(v, np.log(1e-9), np.log(1e14))
        if np.max(np.abs(step)) < 1e-14:
            break
    return np.exp(v), np.max(np.abs(problem.normalized(np.exp(v))), axis=1)


@dataclass
class FixedPoint:
    alpha: np.ndarray
    offset: float
    kind: str                 # "pump_only" or "oscillating"
    rel_residual: float

    @property
    def n(self):
        return photon_numbers(self.alpha)


def _equivalent_system(system, offset):
    """Copy of `system` carrying the signal offset as a plain parameter."""
    import copy
    s = copy.deepcopy(system)
    s.signal_frequency_offset = float(offset)
    return s


def zero_offset_equivalent(system, offset):
    """System with signal_frequency_offset = 0 and the offset folded into detunings and delta-k."""
    s = _equivalent_system(system, 0.0)
    s.detuning_at_reference = np.asarray(system.detuning_at_reference, dtype=float) + np.array([-offset, offset, 0.0])
    s.delta_k_at_reference = float(system.delta_k_at_reference) + delta_k_frequency_correction(system, offset)
    return s


def enumerate_fixed_points(system, rel_tol=2e-3, per_decade=5, floor=1e-6, return_info=False):
    """All fixed points of the rate equations with 0 <= n <= fixed_point_bounds(system).

    Pump-only roots come from the exact Kerr cubic (free-running only; with a signal seed
    the signal cannot vanish). Oscillating roots come from interval branch and bound on the
    reduced equations, Newton polish, and the full six-variable solve. Roots are returned
    sorted by signal photon number. Residuals are relative (relative_fixed_point_residual).
    """
    prob = FixedPointProblem(system)
    info = dict(bounds=prob.bounds, free_running=prob.free)
    roots = []
    if prob.free:
        for a in pump_only_branches(system, 0.0):
            roots.append(FixedPoint(alpha=a, offset=0.0, kind="pump_only",
                                    rel_residual=relative_fixed_point_residual(a, system, 0.0)))
    lo, hi = _bnb_boxes(prob, per_decade=per_decade, floor=floor, rel_tol=rel_tol)
    info["surviving_boxes"] = int(len(lo))
    cand = []
    if len(lo):
        centers = np.where(lo > 0.0, np.sqrt(np.maximum(lo, 1e-300) * hi), 0.5 * hi)
        keys = np.round(np.log(np.maximum(centers, 1e-9)) / (4 * rel_tol)).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        starts = np.maximum(centers[idx], 1e-6)
        u, F = _newton_batch(prob, starts)
        ok = (F < 1e-9) & np.all(u > floor, axis=1) & np.all(u <= prob.upper() * 1.001, axis=1)
        cand = u[ok]
        info["clusters_before_newton"] = int(len(idx))
        info["newton_converged"] = int(ok.sum())
    seen = []
    for u in cand:
        if any(np.all(np.abs(u - w) <= 1e-6 * np.maximum(u, w)) for w in seen):
            continue
        seen.append(u)
    info["distinct_after_newton"] = len(seen)
    for u in seen:
        a0, om = prob.amplitudes(u)
        sysc = _equivalent_system(system, om if prob.free else system.signal_frequency_offset)
        try:
            st = solve_oscillating_branch(sysc, a0)
        except Exception:
            continue
        a, off = st.alpha_ss, st.signal_frequency_offset_rad_s
        rel = relative_fixed_point_residual(a, system, off)
        if rel > 1e-8 or photon_numbers(a)[SIGNAL] < 1e-4:
            continue
        if any(r.kind == "oscillating" and np.all(np.abs(r.n - photon_numbers(a)) <= 1e-6 * np.maximum(r.n, photon_numbers(a)))
               and abs(r.offset - off) <= 1e-6 * max(abs(off), 1e6) for r in roots):
            continue
        roots.append(FixedPoint(alpha=a, offset=float(off), kind="oscillating", rel_residual=rel))
    roots.sort(key=lambda r: (r.n[SIGNAL], r.n[PUMP]))
    info["n_roots"] = len(roots)
    return (roots, info) if return_info else roots


def classify_fixed_point(system, root, zero_tol_over_kappa=1e-6, stab_tol_over_kappa=1e-4):
    """Jacobian eigenvalues (finite-difference, carriers fixed at the root's offset) and stability label.

    Labels: "stable" (every non-neutral eigenvalue has Re < -stab_tol kappa; free-running allows exactly
    one neutral phase mode), "unstable" (every non-neutral eigenvalue has Re > 0), "saddle" (mixed),
    "marginal" (some eigenvalue within stab_tol of the imaginary axis other than the allowed phase mode).
    """
    sysc = _equivalent_system(system, root.offset)
    lin = linearize_about_steady_state(root.alpha, sysc)
    ev = np.linalg.eigvals(lin.drift_matrix)
    ev = ev[np.argsort(-ev.real)]
    kappa = float(np.max(system.kappa))
    free = not system.signal_seeded
    neutral = np.abs(ev) < zero_tol_over_kappa * kappa
    allowed = 1 if (free and root.kind == "oscillating") else 0
    rest = ev[~neutral] if neutral.sum() == allowed else ev
    tol = stab_tol_over_kappa * kappa
    n_pos = int(np.sum(rest.real > tol))
    n_marg = int(np.sum(np.abs(rest.real) <= tol))
    if neutral.sum() != allowed:
        label = "marginal"
    elif n_marg:
        label = "marginal"
    elif n_pos == 0:
        label = "stable"
    elif n_pos == len(rest):
        label = "unstable"
    else:
        label = "saddle"
    return dict(label=label, eigvals=ev, n_unstable=n_pos, n_neutral=int(neutral.sum()),
                max_re_nonneutral=float(np.max(rest.real)) if len(rest) else float("nan"),
                drift_matrix=lin.drift_matrix, linearized=lin)


def is_strongly_oscillating(root, n_min=1e3, ratio_min=1e-3):
    n = root.n
    return bool(root.kind == "oscillating" and n[SIGNAL] >= n_min and n[IDLER] >= n_min
                and n[SIGNAL] >= ratio_min * n[PUMP] and n[IDLER] >= ratio_min * n[PUMP])


def select_stable_oscillating_root(system, roots=None, classes=None, n_signal_target=None):
    """Pick the stable root with signal and idler >= 1e3 photons and >= 1e-3 of the pump number.

    If several qualify, the one with signal photon number closest (in log) to n_signal_target
    (largest signal if no target). Returns (root, classification, index) or None.
    """
    roots = enumerate_fixed_points(system) if roots is None else roots
    classes = [classify_fixed_point(system, r) for r in roots] if classes is None else classes
    good = [i for i, (r, c) in enumerate(zip(roots, classes)) if c["label"] == "stable" and is_strongly_oscillating(r)]
    if not good:
        return None
    if n_signal_target is None:
        i = max(good, key=lambda k: roots[k].n[SIGNAL])
    else:
        i = min(good, key=lambda k: abs(np.log(roots[k].n[SIGNAL] / n_signal_target)))
    return roots[i], classes[i], i


def print_root_table(system, roots, classes, chosen=None):
    print("fixed points (every root with 0 <= n <= flux-balance bound):")
    print(f"  {'#':>2s} {'kind':>10s} {'idler':>11s} {'signal':>11s} {'pump':>11s} {'rel.res':>8s} "
          f"{'class':>9s} {'max Re (1/s)':>13s}")
    for k, (r, c) in enumerate(zip(roots, classes)):
        n = r.n
        mark = "  <- selected" if chosen == k else ""
        print(f"  {k:>2d} {r.kind:>10s} {n[IDLER]:11.4e} {n[SIGNAL]:11.4e} {n[PUMP]:11.4e} {r.rel_residual:8.1e} "
              f"{c['label']:>9s} {c['max_re_nonneutral']:13.4e}{mark}")


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------

def kerr_shifted_threshold_report(system, steady):
    """Pump drive versus the Kerr-shifted threshold, and growth on the pump-only branch."""
    branches = pump_only_branches(system, system.signal_frequency_offset)
    upper = branches[-1]
    offset = float(system.signal_frequency_offset)
    growth = _unseeded_small_signal_growth(system, upper[PUMP], offset)
    g = abs(chi2_eff(steady.alpha_ss, system, system.signal_frequency_offset))
    kap = float(system.kappa[SIGNAL] * system.kappa[IDLER]) ** 0.5
    return {
        "pump_only_branches_photons": [float(abs(b[PUMP]) ** 2) for b in branches],
        "pump_only_upper_photons": float(abs(upper[PUMP]) ** 2),
        "pump_photons_at_state": float(abs(steady.alpha_ss[PUMP]) ** 2),
        "threshold_photons_kappa2_over_g2": float(kap**2 / g**2),
        "pump_only_upper_over_threshold": float(abs(upper[PUMP]) ** 2 * g**2 / kap**2),
        "pump_only_pair_growth_rate_s_inv": float(growth),  # in the carrier frame of the state
        "pump_only_pair_growth_offset_rad_s": float(offset),
    }


def run_example(
    wavelength_regime="nondegenerate",
    signal_seeded=True,
    signal_seed_power_W=1e-3,
    pump_input_power_W=1.0,
    reference_photon_numbers=None,
    phase_match_offset_x=0.5,
    max_frequency_offset=None,
    use_cavity_kerr=True,
    use_phase_match_kerr=True,
    retune_reference_state=False,
    adjust_reference_mode=PUMP,
    operating_point=True,
    n_signal=2e6,
    pump_detuning_over_kappa=2.0,
    pair_detuning_over_kappa=-0.1,
    seed_deficit=0.0,
    seed_offset_over_kappa=0.1,
):
    """Run the default case.

    operating_point=True (default) builds the oscillating branch directly: signal
    occupation n_signal, effective pump detuning pump_detuning_over_kappa * kappa at the
    state, and (if seeded) a signal seed that injection-locks the oscillator at an offset
    seed_offset_over_kappa * kappa (optionally with the pump seed_deficit below the clamped value). The drives follow from the state, the fixed point is solved
    directly, and the pump-only branch it must out-compete is reported.
    operating_point=False keeps the legacy path (transient from a perturbed design point,
    which settles on the low pump-only branch because of pump Kerr bistability).
    """
    if operating_point:
        signal_seed_power_W = 0.0 if not signal_seeded else signal_seed_power_W
    system = default_system(
        wavelength_regime=wavelength_regime,
        signal_seeded=signal_seeded,
        signal_seed_power_W=signal_seed_power_W,
        pump_input_power_W=pump_input_power_W,
        reference_photon_numbers=reference_photon_numbers,
        phase_match_offset_x=phase_match_offset_x,
        use_cavity_kerr=use_cavity_kerr,
        use_phase_match_kerr=use_phase_match_kerr,
    )

    lambda_i = lambda_from_omega(carrier_frequencies_for_phase_matching(system)[IDLER])
    print("idler wavelength [um] =", 1e6 * lambda_i)

    if operating_point:
        target = design_oscillating_state(
            system, n_signal=n_signal,
            pump_detuning_over_kappa=pump_detuning_over_kappa,
            pair_detuning_over_kappa=pair_detuning_over_kappa,
            seed_deficit=seed_deficit if signal_seeded else 0.0,
            seed_offset_over_kappa=seed_offset_over_kappa if signal_seeded else 0.0,
        )
        print("QPM period [um]       =", 1e6 * system.poling_period)
        # Enumerate every fixed point in the flux-balance box, classify them by the Jacobian
        # eigenvalues and select a stable, strongly oscillating one (closest to the design target).
        roots, fp_info = enumerate_fixed_points(system, return_info=True)
        classes = [classify_fixed_point(system, r) for r in roots]
        sel = select_stable_oscillating_root(
            system, roots, classes, n_signal_target=float(photon_numbers(target)[SIGNAL]))
        if sel is None:
            print_root_table(system, roots, classes)
            raise RuntimeError("no stable oscillating fixed point (signal and idler >= 1e3 photons)")
        root, _, k_sel = sel
        print(f"operating point: all-roots search, {len(roots)} fixed points in the flux-balance box "
              f"({sum(r.kind == 'oscillating' for r in roots)} oscillating)")
        print_root_table(system, roots, classes, chosen=k_sel)
        system.signal_frequency_offset = float(root.offset)
        steady = SteadyStateResult(
            alpha_ss=root.alpha,
            residual_norm=float(np.linalg.norm(rhs_complex(root.alpha, system, root.offset))),
            relative_residual_norm=float(root.rel_residual),
            transient_state=np.asarray(target, dtype=np.complex128),
            converged=bool(root.rel_residual <= 1e-8),
            signal_frequency_offset_rad_s=float(root.offset),
            frequency_determined=True,
            oscillating=True,
            parametric_growth_rate_s_inv=np.nan,
            transient_converged=True,
            transient_time_s=0.0,
        )
        report = kerr_shifted_threshold_report(system, steady)
        steady.parametric_growth_rate_s_inv = report["pump_only_pair_growth_rate_s_inv"]
        print("pump-only fixed points [photons]:", report["pump_only_branches_photons"])
        print("pump-only upper branch / threshold (kappa^2/|g|^2):",
              report["pump_only_upper_over_threshold"])
    else:
        alpha0 = None
        if retune_reference_state:
            alpha0 = design_reference_state(system, adjust_mode=adjust_reference_mode)
        print("QPM period [um]       =", 1e6 * system.poling_period)
        steady = solve_steady_state(
            system, alpha0=alpha0, max_frequency_offset=max_frequency_offset
        )
    lin = linearize_about_steady_state(steady.alpha_ss, system)
    summary = summary_dict(system, steady, lin)
    pretty_print_summary(summary)
    if operating_point:
        print("drift eigenvalues [1/s]:")
        for ev in sorted(lin.eigvals, key=lambda z: -z.real):
            print(f"  {ev.real:+.6e} {ev.imag:+.6e}j")

    return system, steady, lin
