from dataclasses import asdict, dataclass
import math
from rocketcea.cea_obj import CEA_Obj
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# isp_units            = 'sec',         # N-s/kg, m/s, km/s
# cstar_units          = 'm/sec',      # m/s
# pressure_units       = 'Bar',        # MPa, KPa, Pa, Bar, Atm, Torr
# temperature_units    = 'K',        # K, C, F
# sonic_velocity_units = 'ft/sec',      # m/s
# enthalpy_units       = 'BTU/lbm',     # J/g, kJ/kg, J/kg, kcal/kg, cal/g
# density_units        = 'lbm/cuft',    # g/cc, sg, kg/m^3
# specific_heat_units  = 'BTU/lbm degR' # kJ/kg-K, cal/g-C, J/kg-K
# viscosity_units      = 'millipoise'   # lbf-sec/sqin, lbf-sec/sqft, lbm/ft-sec, poise, centipoise
# thermal_cond_units   = 'mcal/cm-K-s'  # millical/cm-degK-sec, BTU/hr-ft-degF, BTU/s-in-degF,
#                                       #  cal/s-cm-degC, W/cm-degC

INCHES_TO_METERS = 25.4 / 1000
MM_TO_METERS = 1 / 1000
BAR_TO_PA = 100000
PA_TO_PSI = 0.000145038
M_TO_FT = 3.28084
FT_TO_M = 1 / M_TO_FT
KELVIN_TO_RANKINE = 1.8
RANKINE_TO_KELVIN = 1 / KELVIN_TO_RANKINE

FROZEN = 0

G_MPSPS = 9.80655
CHAMBER_PRESSURES_BAR = np.arange(10, 20, 3)
MIXURE_RATIOS = np.arange(1, 2, 0.1)
THRUST_N = np.arange(10, 15, 1)
L_STAR_M = 40 * INCHES_TO_METERS
CHAMBER_DIAMETER_M = 20 * MM_TO_METERS

EXIT_PRESSURE_BAR = 1


@dataclass
class EngineParameters:
    chamber_pressure: float = 0
    injector_pressure: float = 0
    area_ratio: float = 0
    mixture_ratio: float = 0
    thrust: float = 0

    chamber_temp: float = 0
    throat_temp: float = 0
    exit_temp: float = 0

    mdot_fuel: float = 0
    mdot_ox: float = 0

    chamber_volume: float = 0
    chamber_diam: float = 0
    chamber_len: float = 0
    throat_diam: float = 0
    exit_diam: float = 0

    exit_pressure: float = 0

    c_star: float = 0
    throat_isp_sec: float = 0
    exit_isp_sec: float = 0


def design_engine(
    pc_pa, throat_isp_sec, thrust_n, cstar_mps, area_ratio, mixture_ratio
):

    # V_throat [m/s] = Isp(throat) * g
    effective_exhaust_vel = throat_isp_sec * G_MPSPS

    # m-dot [kg/s] = thrust [kg m/s/s] / isp [m/s]
    m_dot_kgps = thrust_n / effective_exhaust_vel

    # throat area [m^2] = mdot [kg/s] * c* [m/s] / P_c [kg*s/s/m]
    throat_area = m_dot_kgps * cstar_mps / pc_pa

    # exit area = throat area * expansion ratio
    exit_area = throat_area * area_ratio

    # area = pi d^2 / 4 so d = sqrt(4 * area / pi)
    throat_diameter = math.sqrt(4 * throat_area / math.pi)
    exit_diameter = math.sqrt(4 * exit_area / math.pi)

    # L-star is defined as typical ratios of chamber volume to nozzle sonic throat cross section
    # Chamber volume [m^3] = L-star [m] * throat area [m^2]
    chamber_volume_m3 = L_STAR_M * throat_area

    # chamber volume / chamber cross section = chamber length
    # where chamber cross section = pi d^2/4 (area of circle)
    chamber_cross_section_m2 = math.pi * CHAMBER_DIAMETER_M**2 / 4
    chamber_length = chamber_volume_m3 / chamber_cross_section_m2

    # Need a mdot-ox and mdot-fuel such the ox/fuel = mixture ratio and the m_dot=ox+fuel
    # mdot-ox = mdot-fuel*mixtureratio
    # mdot = mdot-fuel * (1 + mixtureratio)
    # mdot-fuel = mdot / (1 + mixtureratio)
    # mdot-ox = mdot - mdot-fuel
    mdot_fuel = m_dot_kgps / (1 + mixture_ratio)
    mdot_ox = m_dot_kgps - mdot_fuel

    return (
        mdot_fuel,
        mdot_ox,
        chamber_volume_m3,
        CHAMBER_DIAMETER_M,
        chamber_length,
        throat_diameter,
        exit_diameter,
    )


cea = CEA_Obj(propName="", oxName="O2", fuelName="RP-1")

# , cstar_units="m/sec", pressure_units="Bar", isp_units="sec"

results = {}
result_list = []

for pc_bar in CHAMBER_PRESSURES_BAR:
    results[pc_bar] = {}
    for mr in MIXURE_RATIOS:
        results[pc_bar][mr] = {}
        for thrust in THRUST_N:
            results[pc_bar][mr][thrust] = {}
            pc_psi = pc_bar * BAR_TO_PA * PA_TO_PSI
            cstar_mps = cea.get_Cstar(pc_psi, mr) * FT_TO_M
            isp_sec = cea.get_Throat_Isp(pc_psi, mr, FROZEN)

            # Determine an area ratio for our Pc/Pexit
            area_ratio = cea.get_eps_at_PcOvPe(
                pc_psi, mr, (pc_bar / EXIT_PRESSURE_BAR), frozen=FROZEN
            )

            # Get exit isp
            (exit_isp_sec, mode) = cea.estimate_Ambient_Isp(pc_psi, mr, area_ratio, FROZEN)

            (
                mdot_fuel,
                mdot_ox,
                chamber_volume,
                chamber_diam,
                chamber_length,
                throat_diameter,
                exit_diameter,
            ) = design_engine(
                pc_bar * BAR_TO_PA, isp_sec, thrust, cstar_mps, area_ratio, mr
            )

            # in Rankine
            (t_chamber, t_throat, t_exit) = cea.get_Temperatures(
                pc_psi, mr, area_ratio, FROZEN
            )

            # contraction ratio = cross-section of chamber/cross-section of throat = (pi d_c^2/4)/(pi d_t^2/4) = d_c^2/d_t^2
            p_inj_psi = (
                cea.get_Pinj_over_Pcomb(
                    pc_psi, mr, fac_CR=chamber_diam**2 / throat_diameter**2
                )
                * pc_psi
            )

            r = EngineParameters(
                pc_psi,
                p_inj_psi,
                area_ratio,
                mr,
                thrust,
                t_chamber * RANKINE_TO_KELVIN,
                t_throat * RANKINE_TO_KELVIN,
                t_exit * RANKINE_TO_KELVIN,
                mdot_fuel,
                mdot_ox,
                chamber_volume,
                chamber_diam,
                chamber_length,
                throat_diameter,
                exit_diameter,
                EXIT_PRESSURE_BAR * BAR_TO_PA * PA_TO_PSI,
                cstar_mps,
                isp_sec, exit_isp_sec
            )
            results[pc_bar][mr][thrust] = r
            result_list.append(asdict(r))


if False:
    # want a graph of temperature vs mixture ratio for a given chamber pressure
    for pressure_bar in results:
        exit_temp = []
        throat_temp = []
        chamber_temp = []
        p_inj = []
        mrs = []
        for mr in results[pressure_bar]:
            engine: EngineParameters = results[pressure_bar][mr][THRUST_N[0]]
            mrs.append(mr)
            exit_temp.append(engine.exit_temp)
            throat_temp.append(engine.throat_temp)
            chamber_temp.append(engine.chamber_temp)

        plt.figure()
        plt.title(f"Pressure: {pressure_bar} bar")
        plt.plot(mrs, exit_temp, label=f"Exit temp vs mr, Pc={pressure_bar} bar")
        plt.plot(mrs, throat_temp, label=f"Throat temp vs mr, Pc={pressure_bar} bar")
        plt.plot(mrs, chamber_temp, label=f"Chamber temp vs mr, Pc={pressure_bar} bar")
        plt.xlabel("Mixture ratio, O:F")
        plt.ylabel("Temperature, K")
        plt.legend()

        # plt.figure()
        # plt.title(f"Pressure: {pressure_bar} bar")
        # plt.scatter(mrs, p_inj, label=f"Injector pressure (psi), Pc={pressure_bar * BAR_TO_PA * PA_TO_PSI} psi")
        # plt.legend()

    plt.show()

if True:
    df = pd.DataFrame(result_list)
    print(df)
    df.to_csv("results.csv")
