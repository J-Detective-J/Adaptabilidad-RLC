"""
Simulación de un sistema RLC en serie - verificación de sus estados
Sistemas Complejos - Taller: Adaptabilidad

Circuito RLC serie:

    L*di/dt + R*i + q/C = 0
    dq/dt = i

Representación en espacio de estados, con x = [q, i]^T:

    dx/dt = A*x

    A = [[0, 1],
         [-1/(L*C), -R/L]]

Los estados del sistema son:
    x = [q, i]

donde:
    q = carga del capacitor
    i = corriente del circuito

El comportamiento depende de la razón de amortiguamiento:

    omega0 = sqrt(1/(L*C))
    zeta   = (R/2)*sqrt(C/L)

    zeta = 0      -> No amortiguado
    0 < zeta < 1  -> Subamortiguado
    zeta = 1      -> Críticamente amortiguado
    zeta > 1      -> Sobreamortiguado
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ==============================================================
# CONFIGURACIÓN DE MATPLOTLIB
# ==============================================================

plt.style.use("dark_background")


# ==============================================================
# 1. PARÁMETROS FIJOS DEL CIRCUITO
# ==============================================================

L = 1.0        # Henrios
C = 0.25       # Faradios
Q0 = 1.0       # Carga inicial (Coulombs)
I0 = 0.0       # Corriente inicial (Amperios)

# Frecuencia natural
OMEGA0 = np.sqrt(1 / (L * C))

# Resistencia crítica:
# zeta = (R/2)*sqrt(C/L)
# Para zeta = 1:
# Rcrit = 2*sqrt(L/C)
R_CRITICO = 2 * np.sqrt(L / C)


print(f"omega0 = {OMEGA0:.3f} rad/s")
print(f"R crítico (zeta=1) = {R_CRITICO:.3f} ohm")


# ==============================================================
# 2. CARPETA DE RESULTADOS
# ==============================================================

# Obtiene la carpeta donde está ubicado este archivo .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Crea una carpeta "resultados_rlc" junto al programa
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_rlc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nResultados guardados en:")
print(OUTPUT_DIR)


# ==============================================================
# 3. MODELO EN ESPACIO DE ESTADOS
# ==============================================================

def state_matrix(R, L, C):
    """
    Construye la matriz A del sistema RLC.
    """

    return np.array([
        [0, 1],
        [-1 / (L * C), -R / L]
    ], dtype=float)


def rlc_derivatives(t, x, R, L, C):
    """
    Ecuaciones diferenciales del sistema RLC.

    x[0] = q
    x[1] = i
    """

    A = state_matrix(R, L, C)

    return A @ x


# ==============================================================
# 4. SIMULACIÓN
# ==============================================================

def simulate(R, L, C, x0, t_max=15, n_steps=3000):
    """
    Simula el sistema RLC utilizando solve_ivp.
    """

    t_eval = np.linspace(0, t_max, n_steps)

    sol = solve_ivp(
        rlc_derivatives,
        [0, t_max],
        x0,
        t_eval=t_eval,
        args=(R, L, C),
        method="RK45",
        rtol=1e-9,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(
            f"Error durante la simulación: {sol.message}"
        )

    # q(t) = sol.y[0]
    # i(t) = sol.y[1]

    return sol.t, sol.y[0], sol.y[1]


# ==============================================================
# 5. RAZÓN DE AMORTIGUAMIENTO
# ==============================================================

def damping_ratio(R, L, C):
    """
    Calcula la razón de amortiguamiento zeta.
    """

    return (R / 2) * np.sqrt(C / L)


# ==============================================================
# 6. CLASIFICACIÓN DEL RÉGIMEN
# ==============================================================

def classify_regime(zeta):

    # Se utiliza np.isclose() para evitar problemas
    # de precisión con números de punto flotante.

    if np.isclose(zeta, 0):
        return "No amortiguado"

    elif zeta < 1:
        return "Subamortiguado"

    elif np.isclose(zeta, 1):
        return "Críticamente amortiguado"

    else:
        return "Sobreamortiguado"


# ==============================================================
# 7. VERIFICACIÓN DE LOS ESTADOS
# ==============================================================

def verify_states(R, L, C):

    A = state_matrix(R, L, C)

    # Calculamos los valores propios de A
    eigvals = np.linalg.eigvals(A)

    # Calculamos zeta
    zeta = damping_ratio(R, L, C)

    # Clasificamos el comportamiento
    regime = classify_regime(zeta)

    return eigvals, zeta, regime


# ==============================================================
# 8. ESCENARIOS
# ==============================================================

SCENARIOS = {
    "No amortiguado (R=0)": 0.0,

    "Subamortiguado (R=0.5*Rc)": 0.5 * R_CRITICO,

    "Críticamente amortiguado (R=Rc)": R_CRITICO,

    "Sobreamortiguado (R=2.5*Rc)": 2.5 * R_CRITICO,
}


COLORS = [
    "#00e5ff",
    "#6ee7a0",
    "#ffb454",
    "#ff6b81"
]


# ==============================================================
# 9. EJECUTAR TODOS LOS ESCENARIOS
# ==============================================================

def run_all():

    results = {}

    # ----------------------------------------------------------
    # FIGURA 1: RESPUESTA TEMPORAL
    # ----------------------------------------------------------

    fig_time = plt.figure(figsize=(15, 9))

    # ----------------------------------------------------------
    # FIGURA 2: RETRATO DE FASE
    # ----------------------------------------------------------

    fig_phase = plt.figure(figsize=(7, 7))

    ax_phase = fig_phase.add_subplot(1, 1, 1)

    # ----------------------------------------------------------
    # SIMULAR LOS CUATRO REGÍMENES
    # ----------------------------------------------------------

    for idx, (name, R) in enumerate(SCENARIOS.items()):

        # Estado inicial
        x0 = [Q0, I0]

        # Simulación
        t, q, i = simulate(
            R,
            L,
            C,
            x0
        )

        # Voltaje en el capacitor
        vC = q / C

        # Verificación mediante valores propios
        eigvals, zeta, regime = verify_states(
            R,
            L,
            C
        )

        # Guardar resultados
        results[name] = {
            "R": R,
            "zeta": zeta,
            "regime": regime,
            "eigvals": eigvals
        }

        # ------------------------------------------------------
        # GRÁFICA TEMPORAL
        # ------------------------------------------------------

        ax1 = fig_time.add_subplot(
            2,
            2,
            idx + 1
        )

        # Voltaje del capacitor
        ax1.plot(
            t,
            vC,
            color=COLORS[idx],
            label="vC(t)"
        )

        # Corriente
        ax1.plot(
            t,
            i,
            color=COLORS[idx],
            alpha=0.45,
            linestyle="--",
            label="i(t)"
        )

        # Título
        ax1.set_title(
            f"{name}\n"
            f"ζ={zeta:.2f} | "
            f"λ={np.round(eigvals, 2)}",
            fontsize=9.5
        )

        ax1.set_xlabel("Tiempo (s)")
        ax1.set_ylabel("Amplitud")

        ax1.legend(
            fontsize=8,
            loc="upper right"
        )

        ax1.axhline(
            0,
            color="white",
            alpha=0.15,
            lw=0.8
        )

        ax1.grid(
            alpha=0.1
        )

        # ------------------------------------------------------
        # RETRATO DE FASE
        # ------------------------------------------------------

        ax_phase.plot(
            q,
            i,
            color=COLORS[idx],
            label=name,
            lw=1.4
        )

    # ==========================================================
    # GUARDAR FIGURA TEMPORAL
    # ==========================================================

    fig_time.tight_layout()

    temporal_path = os.path.join(
        OUTPUT_DIR,
        "rlc_regimenes_temporales.png"
    )

    fig_time.savefig(
        temporal_path,
        dpi=150,
        facecolor="black"
    )

    plt.close(fig_time)

    # ==========================================================
    # CONFIGURAR RETRATO DE FASE
    # ==========================================================

    ax_phase.scatter(
        [Q0],
        [I0],
        color="white",
        zorder=5,
        s=40,
        label="Estado inicial"
    )

    ax_phase.scatter(
        [0],
        [0],
        color="white",
        marker="x",
        zorder=5,
        s=60,
        label="Equilibrio (0,0)"
    )

    ax_phase.set_xlabel(
        "Carga q (C)"
    )

    ax_phase.set_ylabel(
        "Corriente i (A)"
    )

    ax_phase.set_title(
        "Retrato de fase: espacio de estados (q, i)"
    )

    ax_phase.legend(
        fontsize=8
    )

    ax_phase.axhline(
        0,
        color="white",
        alpha=0.15,
        lw=0.8
    )

    ax_phase.axvline(
        0,
        color="white",
        alpha=0.15,
        lw=0.8
    )

    ax_phase.grid(
        alpha=0.1
    )

    fig_phase.tight_layout()

    phase_path = os.path.join(
        OUTPUT_DIR,
        "rlc_retrato_fase.png"
    )

    fig_phase.savefig(
        phase_path,
        dpi=150,
        facecolor="black"
    )

    plt.close(fig_phase)

    # ==========================================================
    # FIGURA 3: EIGENVALORES
    # ==========================================================

    fig_eig, ax_eig = plt.subplots(
        figsize=(7, 6)
    )

    for idx, (name, R) in enumerate(
        SCENARIOS.items()
    ):

        eigvals = results[name]["eigvals"]

        ax_eig.scatter(
            eigvals.real,
            eigvals.imag,
            color=COLORS[idx],
            s=70,
            label=name,
            zorder=5
        )

    # Ejes
    ax_eig.axhline(
        0,
        color="white",
        alpha=0.2,
        lw=0.8
    )

    ax_eig.axvline(
        0,
        color="white",
        alpha=0.2,
        lw=0.8
    )

    ax_eig.set_xlabel(
        "Parte real"
    )

    ax_eig.set_ylabel(
        "Parte imaginaria"
    )

    ax_eig.set_title(
        "Eigenvalores de la matriz de estados A"
    )

    ax_eig.legend(
        fontsize=7.5,
        loc="upper left"
    )

    ax_eig.grid(
        alpha=0.1
    )

    fig_eig.tight_layout()

    eig_path = os.path.join(
        OUTPUT_DIR,
        "rlc_eigenvalores.png"
    )

    fig_eig.savefig(
        eig_path,
        dpi=150,
        facecolor="black"
    )

    plt.close(fig_eig)

    # ==========================================================
    # MOSTRAR UBICACIÓN DE LOS ARCHIVOS
    # ==========================================================

    print("\n=== ARCHIVOS GENERADOS ===")

    print(
        f"1. Regímenes temporales:\n   {temporal_path}"
    )

    print(
        f"2. Retrato de fase:\n   {phase_path}"
    )

    print(
        f"3. Eigenvalores:\n   {eig_path}"
    )

    return results


# ==============================================================
# 10. PROGRAMA PRINCIPAL
# ==============================================================

if __name__ == "__main__":

    results = run_all()

    print(
        "\n=== VERIFICACIÓN DE LOS ESTADOS "
        "(eigenvalores de A) ===\n"
    )

    print(
        f"{'Escenario':<38}"
        f"{'R (ohm)':<12}"
        f"{'zeta':<10}"
        f"{'Eigenvalores':<30}"
        f"{'Régimen'}"
    )

    print("-" * 115)

    for name, r in results.items():

        # Convertir eigenvalores a texto
        ev_str = ", ".join(
            f"{e:.2f}"
            for e in r["eigvals"]
        )

        print(
            f"{name:<38}"
            f"{r['R']:<12.3f}"
            f"{r['zeta']:<10.2f}"
            f"{ev_str:<30}"
            f"{r['regime']}"
        )

    print("\nSimulación terminada correctamente.")