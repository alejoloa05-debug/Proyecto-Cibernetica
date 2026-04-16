"""
Simulacion Cibernetica Animada: Dinamica de Discusion de Pareja
===============================================================
Version terminal — muestra el proceso turno a turno en tiempo real
con graficas animadas usando matplotlib.

Uso:
    python simulacion_animada.py                  # parametros por defecto
    python simulacion_animada.py --velocidad 0.3  # mas lento
    python simulacion_animada.py --escenario 2    # escenario predefinido (1-6)
    python simulacion_animada.py --ayuda

Universidad Sergio Arboleda — Analisis Cibernetico
"""

import argparse
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from typing import Literal
from collections import Counter, deque


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION VISUAL
# ─────────────────────────────────────────────────────────────────────────────

PALETA = {
    "fondo":      "#0d0d0d",
    "superficie": "#161616",
    "borde":      "#2a2a2a",
    "agente_a":   "#9C89F0",
    "agente_b":   "#F28C5E",
    "estres":     "#F5C242",
    "escalada":   "#E05252",
    "resolucion": "#4CAF82",
    "ciclo":      "#9C89F0",
    "texto":      "#cccccc",
    "texto_sec":  "#666666",
}

plt.rcParams.update({
    "figure.facecolor":  PALETA["fondo"],
    "axes.facecolor":    PALETA["superficie"],
    "axes.edgecolor":    PALETA["borde"],
    "axes.labelcolor":   PALETA["texto"],
    "axes.titlecolor":   PALETA["texto"],
    "xtick.color":       PALETA["texto_sec"],
    "ytick.color":       PALETA["texto_sec"],
    "grid.color":        PALETA["borde"],
    "grid.linewidth":    0.5,
    "text.color":        PALETA["texto"],
    "legend.facecolor":  PALETA["superficie"],
    "legend.edgecolor":  PALETA["borde"],
    "font.family":       "DejaVu Sans",
})

ACCIONES_REACTIVAS = [
    "Eleva el tono", "Lanza una critica", "Se pone a la defensiva",
    "Interrumpe", "Dice algo hiriente", "Sube la voz", "Hace un reproche",
]
ACCIONES_CALMANTES = [
    "Escucha", "Intenta explicarse", "Baja el tono", "Pide espacio",
    "Cede parcialmente", "Propone un tema neutro", "Guarda silencio",
]
DESENCADENANTES = {"Leve": 8, "Moderado": 22, "Fuerte": 40}
Resultado = Literal["escalada", "resolucion", "ciclo"]


# ─────────────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Agente:
    nombre:      str
    umbral:      float = 50.0
    reactividad: float = 5.0
    calma:       float = 5.0
    tension:     float = 0.0

    def sobre_umbral(self, th: float) -> bool:
        return self.tension > th

    def regular(self) -> None:
        self.tension = max(0.0, self.tension - self.calma * 0.15)


@dataclass
class FactoresExternos:
    estres:   float = 0.0
    terceros: int   = 0

    def umbral_efectivo(self, base: float) -> float:
        adj = 10.0 if self.terceros == 1 else (-8.0 if self.terceros == -1 else 0.0)
        return max(5.0, base - self.estres + adj)


@dataclass
class EventoTurno:
    turno:     int
    hablante:  str
    accion:    str
    delta:     float
    tension_a: float
    tension_b: float
    ruido:     bool = False


# ─────────────────────────────────────────────────────────────────────────────
# SIMULADOR PASO A PASO
# ─────────────────────────────────────────────────────────────────────────────

class SimuladorPasoAPaso:
    """
    Generador que produce un EventoTurno por llamada a next().
    Permite integrar facilmente con bucles de animacion.
    """
    MAX_TURNOS = 60
    PROB_RUIDO = 0.05

    def __init__(self, a: Agente, b: Agente, f: FactoresExternos,
                 trigger: str = "moderado", seed: int | None = None):
        self.a = a; self.b = b; self.f = f; self.trigger = trigger
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        base = DESENCADENANTES.get(trigger, 15)
        self.a.tension = base + random.uniform(0, 5)
        self.b.tension = base * 0.7 + random.uniform(0, 5)
        self.th_a = f.umbral_efectivo(a.umbral)
        self.th_b = f.umbral_efectivo(b.umbral)
        self.turno = 0
        self.snaps: list[str] = []
        self.terminado = False
        self.resultado: Resultado | None = None

    def siguiente(self) -> EventoTurno | None:
        """Avanza un turno. Devuelve None si ya termino."""
        if self.terminado or self.turno >= self.MAX_TURNOS:
            self.terminado = True
            if self.resultado is None:
                self.resultado = "ciclo"
            return None

        self.turno += 1
        n = self.turno
        hablante, receptor = (self.a, self.b) if n % 2 == 1 else (self.b, self.a)
        th_ef = self.th_a if hablante is self.a else self.th_b

        ruido = False
        if hablante.sobre_umbral(th_ef):
            delta  = hablante.reactividad * (1.5 + random.uniform(0, 1.2))
            accion = random.choice(ACCIONES_REACTIVAS)
        else:
            delta  = -hablante.calma * (0.4 + random.uniform(0, 0.6))
            accion = random.choice(ACCIONES_CALMANTES)

        receptor.tension = max(0.0, min(100.0, receptor.tension + delta))
        hablante.regular()

        if random.random() < self.PROB_RUIDO:
            receptor.tension = min(100.0, receptor.tension + random.uniform(5, 13))
            ruido = True

        self.snaps.append(f"{round(self.a.tension)},{round(self.b.tension)}")

        ev = EventoTurno(
            turno     = n,
            hablante  = hablante.nombre,
            accion    = accion + (" [malinterpretado]" if ruido else ""),
            delta     = round(delta, 2),
            tension_a = round(self.a.tension, 2),
            tension_b = round(self.b.tension, 2),
            ruido     = ruido,
        )

        if self.a.tension <= 5 and self.b.tension <= 5:
            self.resultado = "resolucion"; self.terminado = True
        elif self.a.tension >= 95 or self.b.tension >= 95:
            self.resultado = "escalada"; self.terminado = True
        elif (len(self.snaps) >= 7
              and all(self.snaps.count(v) > 1 for v in self.snaps[-3:])
              and n > 10):
            self.resultado = "ciclo"; self.terminado = True

        return ev


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZADOR ANIMADO
# ─────────────────────────────────────────────────────────────────────────────

class VisualizadorAnimado:
    """
    Muestra el proceso de simulacion turno a turno con:
      - Barras de tension en tiempo real
      - Grafica de trayectoria acumulada
      - Espacio de fases acumulado
      - Log de acciones (ultimas 8)
      - Indicador de estado del sistema
    """

    LOG_MAX = 8

    def __init__(self, sim: SimuladorPasoAPaso, velocidad: float = 0.18,
                 titulo: str = "Simulacion Cibernetica"):
        self.sim = sim
        self.velocidad = velocidad
        self.titulo = titulo
        self.hist_a: list[float] = []
        self.hist_b: list[float] = []
        self.turnos: list[int]   = []
        self.log: deque[str]     = deque(maxlen=self.LOG_MAX)
        self._construir_figura()

    def _construir_figura(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor(PALETA["fondo"])

        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            hspace=0.45, wspace=0.35,
            left=0.06, right=0.97,
            top=0.88, bottom=0.07,
        )

        self.ax_barras   = self.fig.add_subplot(gs[0, :2])   # barras tension
        self.ax_trayect  = self.fig.add_subplot(gs[1:, :2])  # trayectoria
        self.ax_fases    = self.fig.add_subplot(gs[1, 2])    # espacio fases
        self.ax_log      = self.fig.add_subplot(gs[2, 2])    # log acciones
        self.ax_estado   = self.fig.add_subplot(gs[0, 2])    # estado sistema

        for ax in [self.ax_barras, self.ax_trayect, self.ax_fases,
                   self.ax_log, self.ax_estado]:
            ax.set_facecolor(PALETA["superficie"])
            for sp in ax.spines.values():
                sp.set_color(PALETA["borde"])
                sp.set_linewidth(0.5)

        self._init_barras()
        self._init_trayectoria()
        self._init_fases()
        self._init_log()
        self._init_estado()

        self.fig.suptitle(
            self.titulo, fontsize=15, fontweight="bold",
            color=PALETA["texto"], y=0.95,
        )

    # ── Inicializacion de paneles ─────────────────────────────────────────────

    def _init_barras(self):
        ax = self.ax_barras
        ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["  Persona B", "  Persona A"], fontsize=10)
        ax.set_xlabel("Nivel de tension", fontsize=9)
        ax.set_title("Tension actual", fontsize=10, pad=6)
        ax.grid(True, axis="x", alpha=0.3)
        ax.axvline(self.sim.th_a, color=PALETA["agente_a"],
                   lw=0.8, ls=":", alpha=0.7, label=f"Umbral A ({self.sim.th_a:.0f})")
        ax.axvline(self.sim.th_b, color=PALETA["agente_b"],
                   lw=0.8, ls=":", alpha=0.7, label=f"Umbral B ({self.sim.th_b:.0f})")
        ax.legend(fontsize=8, loc="upper right")

        self.barra_a = ax.barh(1, 0, height=0.45,
                                color=PALETA["agente_a"], alpha=0.85)[0]
        self.barra_b = ax.barh(0, 0, height=0.45,
                                color=PALETA["agente_b"], alpha=0.85)[0]
        self.txt_a = ax.text(2, 1, "0", va="center", fontsize=10,
                              color=PALETA["agente_a"], fontweight="bold")
        self.txt_b = ax.text(2, 0, "0", va="center", fontsize=10,
                              color=PALETA["agente_b"], fontweight="bold")

    def _init_trayectoria(self):
        ax = self.ax_trayect
        ax.set_xlim(0, 62); ax.set_ylim(0, 108)
        ax.set_xlabel("Turno", fontsize=9)
        ax.set_ylabel("Nivel de tension", fontsize=9)
        ax.set_title("Trayectoria de tension (acumulada)", fontsize=10, pad=6)
        ax.grid(True, alpha=0.25)
        ax.axhline(self.sim.th_a, color=PALETA["agente_a"],
                   lw=0.8, ls=":", alpha=0.6)
        ax.axhline(self.sim.th_b, color=PALETA["agente_b"],
                   lw=0.8, ls=":", alpha=0.6)
        self.linea_a, = ax.plot([], [], color=PALETA["agente_a"],
                                 lw=2, label="Persona A")
        self.linea_b, = ax.plot([], [], color=PALETA["agente_b"],
                                 lw=2, ls="--", label="Persona B")
        self.fill_a = None
        self.fill_b = None
        ax.legend(fontsize=9, loc="upper right")

    def _init_fases(self):
        ax = self.ax_fases
        ax.set_xlim(0, 108); ax.set_ylim(0, 108)
        ax.set_xlabel("Tension A", fontsize=8)
        ax.set_ylabel("Tension B", fontsize=8)
        ax.set_title("Espacio de fases", fontsize=10, pad=6)
        ax.grid(True, alpha=0.2)
        ax.plot([0, 100], [0, 100], ls=":", color=PALETA["texto_sec"],
                lw=0.6, alpha=0.5)
        ax.axhline(self.sim.th_b, color=PALETA["agente_b"],
                   lw=0.6, ls=":", alpha=0.4)
        ax.axvline(self.sim.th_a, color=PALETA["agente_a"],
                   lw=0.6, ls=":", alpha=0.4)
        self.tray_fases, = ax.plot([], [], color="#9C89F0", lw=1.2, alpha=0.6)
        self.punto_fases, = ax.plot([], [], "o", color="white",
                                     ms=6, zorder=5)

    def _init_log(self):
        ax = self.ax_log
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Registro de acciones", fontsize=10, pad=6)
        ax.axis("off")
        self.log_texts = [
            ax.text(0.04, 1 - (i + 0.5) / self.LOG_MAX,
                    "", fontsize=8, va="center",
                    color=PALETA["texto_sec"],
                    transform=ax.transAxes)
            for i in range(self.LOG_MAX)
        ]

    def _init_estado(self):
        ax = self.ax_estado
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Estado del sistema", fontsize=10, pad=6)
        ax.axis("off")
        self.txt_turno    = ax.text(0.5, 0.82, "Turno 0",
                                     ha="center", fontsize=13, fontweight="bold",
                                     color=PALETA["texto"], transform=ax.transAxes)
        self.txt_resultado = ax.text(0.5, 0.55, "En curso",
                                      ha="center", fontsize=11,
                                      color=PALETA["texto_sec"], transform=ax.transAxes)
        self.txt_ta       = ax.text(0.5, 0.32, "A: 0",
                                     ha="center", fontsize=10,
                                     color=PALETA["agente_a"], transform=ax.transAxes)
        self.txt_tb       = ax.text(0.5, 0.16, "B: 0",
                                     ha="center", fontsize=10,
                                     color=PALETA["agente_b"], transform=ax.transAxes)

    # ── Actualizacion por turno ────────────────────────────────────────────────

    def _actualizar(self, ev: EventoTurno):
        self.hist_a.append(ev.tension_a)
        self.hist_b.append(ev.tension_b)
        self.turnos.append(ev.turno)

        # Barras
        self.barra_a.set_width(ev.tension_a)
        self.barra_b.set_width(ev.tension_b)
        self.txt_a.set_text(f"{ev.tension_a:.1f}")
        self.txt_b.set_text(f"{ev.tension_b:.1f}")
        color_hab = PALETA["agente_a"] if ev.hablante == "A" else PALETA["agente_b"]
        barra_act = self.barra_a if ev.hablante == "A" else self.barra_b
        barra_act.set_alpha(1.0)
        otra = self.barra_b if ev.hablante == "A" else self.barra_a
        otra.set_alpha(0.55)

        # Color de barra segun estado
        th_ef_a = self.sim.th_a; th_ef_b = self.sim.th_b
        self.barra_a.set_color(PALETA["escalada"] if ev.tension_a > th_ef_a
                                else PALETA["agente_a"])
        self.barra_b.set_color(PALETA["escalada"] if ev.tension_b > th_ef_b
                                else PALETA["agente_b"])

        # Trayectoria
        self.linea_a.set_data(self.turnos, self.hist_a)
        self.linea_b.set_data(self.turnos, self.hist_b)
        if self.fill_a: self.fill_a.remove()
        if self.fill_b: self.fill_b.remove()
        self.fill_a = self.ax_trayect.fill_between(
            self.turnos, self.hist_a, alpha=0.06, color=PALETA["agente_a"])
        self.fill_b = self.ax_trayect.fill_between(
            self.turnos, self.hist_b, alpha=0.06, color=PALETA["agente_b"])

        # Espacio de fases
        self.tray_fases.set_data(self.hist_a, self.hist_b)
        self.punto_fases.set_data([ev.tension_a], [ev.tension_b])

        # Log
        signo = "+" if ev.delta > 0 else ""
        tag_ruido = " [ruido]" if ev.ruido else ""
        linea = (f"T{ev.turno:02d} {ev.hablante}: {ev.accion[:28]}"
                 f"  {signo}{ev.delta:.1f}{tag_ruido}")
        self.log.appendleft(linea)
        for i, txt in enumerate(self.log_texts):
            if i < len(self.log):
                entrada = list(self.log)[i]
                c = color_hab if i == 0 else PALETA["texto_sec"]
                txt.set_text(entrada)
                txt.set_color(c)
            else:
                txt.set_text("")

        # Estado
        self.txt_turno.set_text(f"Turno {ev.turno}")
        self.txt_ta.set_text(f"Tension A: {ev.tension_a:.1f}")
        self.txt_tb.set_text(f"Tension B: {ev.tension_b:.1f}")

        if self.sim.terminado and self.sim.resultado:
            col = PALETA[self.sim.resultado]
            self.txt_resultado.set_text(self.sim.resultado.upper())
            self.txt_resultado.set_color(col)
            self.fig.suptitle(
                f"{self.titulo}  —  {self.sim.resultado.upper()} en turno {ev.turno}",
                fontsize=15, fontweight="bold", color=col, y=0.95,
            )

        self.fig.canvas.draw_idle()
        plt.pause(self.velocidad)

    # ── Bucle principal ────────────────────────────────────────────────────────

    def correr(self):
        plt.ion()
        plt.show()

        while True:
            ev = self.sim.siguiente()
            if ev is None:
                break
            self._actualizar(ev)

        # Pausa final para ver el resultado
        plt.ioff()
        col_final = PALETA.get(self.sim.resultado or "neutro", PALETA["texto"])
        self.fig.suptitle(
            f"{self.titulo}  —  RESULTADO: {(self.sim.resultado or '?').upper()}"
            f"  (turno {self.sim.turno})",
            fontsize=15, fontweight="bold", color=col_final, y=0.95,
        )
        self.fig.canvas.draw()
        plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
# ESCENARIOS PREDEFINIDOS
# ─────────────────────────────────────────────────────────────────────────────

ESCENARIOS_PREDEFINIDOS = {
    1: {
        "nombre":  "Umbrales altos / Baja reactividad",
        "th_a": 70, "th_b": 70, "re_a": 2, "re_b": 2,
        "cal_a": 7, "cal_b": 7, "estres": 0, "terceros": 0, "trigger": "leve",
    },
    2: {
        "nombre":  "Umbrales bajos / Alta reactividad",
        "th_a": 25, "th_b": 25, "re_a": 9, "re_b": 9,
        "cal_a": 3, "cal_b": 3, "estres": 0, "terceros": 0, "trigger": "moderado",
    },
    3: {
        "nombre":  "Asimetrico: A muy reactivo, B muy calmo",
        "th_a": 22, "th_b": 70, "re_a": 9, "re_b": 2,
        "cal_a": 3, "cal_b": 8, "estres": 0, "terceros": 0, "trigger": "moderado",
    },
    4: {
        "nombre":  "Estres externo alto",
        "th_a": 50, "th_b": 50, "re_a": 5, "re_b": 5,
        "cal_a": 5, "cal_b": 5, "estres": 25, "terceros": 0, "trigger": "leve",
    },
    5: {
        "nombre":  "Terceros presentes (inhiben escalada)",
        "th_a": 50, "th_b": 50, "re_a": 6, "re_b": 6,
        "cal_a": 4, "cal_b": 4, "estres": 0, "terceros": 1, "trigger": "moderado",
    },
    6: {
        "nombre":  "Desencadenante fuerte",
        "th_a": 50, "th_b": 50, "re_a": 5, "re_b": 5,
        "cal_a": 5, "cal_b": 5, "estres": 0, "terceros": 0, "trigger": "fuerte",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulacion Cibernetica Animada — Discusion de Pareja",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--escenario", type=int, choices=range(1, 7), default=None,
                   help="Escenario predefinido (1-6). Omitir para usar parametros manuales.")
    p.add_argument("--th_a",     type=float, default=50,       help="Umbral A (default: 50)")
    p.add_argument("--th_b",     type=float, default=50,       help="Umbral B (default: 50)")
    p.add_argument("--re_a",     type=float, default=5,        help="Reactividad A (default: 5)")
    p.add_argument("--re_b",     type=float, default=5,        help="Reactividad B (default: 5)")
    p.add_argument("--cal_a",    type=float, default=5,        help="Calma A (default: 5)")
    p.add_argument("--cal_b",    type=float, default=5,        help="Calma B (default: 5)")
    p.add_argument("--estres",   type=float, default=0,        help="Estres externo 0-30 (default: 0)")
    p.add_argument("--terceros", type=int,   default=0,        help="-1/0/1 (default: 0)")
    p.add_argument("--trigger",  type=str,   default="moderado",
                   choices=["leve", "moderado", "fuerte"],     help="Intensidad inicial (default: moderado)")
    p.add_argument("--velocidad",type=float, default=0.18,     help="Pausa entre turnos en seg (default: 0.18)")
    p.add_argument("--seed",     type=int,   default=None,     help="Semilla aleatoria para reproducibilidad")
    p.add_argument("--ayuda",    action="store_true",
                   help="Muestra los escenarios disponibles y sale.")
    return p


def main():
    parser = construir_parser()
    args   = parser.parse_args()

    if args.ayuda:
        print("\n  Escenarios predefinidos disponibles:")
        print("  " + "─" * 50)
        for k, v in ESCENARIOS_PREDEFINIDOS.items():
            print(f"  {k}. {v['nombre']}")
        print()
        print("  Ejemplo: python simulacion_animada.py --escenario 3 --velocidad 0.25")
        print("  Ejemplo: python simulacion_animada.py --th_a 30 --re_a 8 --estres 15\n")
        sys.exit(0)

    # Seleccionar parametros
    if args.escenario:
        cfg    = ESCENARIOS_PREDEFINIDOS[args.escenario]
        th_a   = cfg["th_a"];   th_b  = cfg["th_b"]
        re_a   = cfg["re_a"];   re_b  = cfg["re_b"]
        cal_a  = cfg["cal_a"];  cal_b = cfg["cal_b"]
        estres = cfg["estres"]; terceros = cfg["terceros"]
        trigger = cfg["trigger"]
        nombre  = cfg["nombre"]
    else:
        th_a = args.th_a; th_b = args.th_b
        re_a = args.re_a; re_b = args.re_b
        cal_a = args.cal_a; cal_b = args.cal_b
        estres = args.estres; terceros = args.terceros
        trigger = args.trigger
        nombre = f"Personalizado | Umbral A:{th_a} B:{th_b} | React A:{re_a} B:{re_b}"

    print(f"\n  Iniciando simulacion: {nombre}")
    print(f"  Umbral A:{th_a}  B:{th_b} | Reactividad A:{re_a}  B:{re_b}"
          f" | Calma A:{cal_a}  B:{cal_b}")
    print(f"  Estres:{estres} | Terceros:{terceros} | Trigger:{trigger}"
          f" | Velocidad:{args.velocidad}s\n")

    a   = Agente("A", umbral=th_a, reactividad=re_a, calma=cal_a)
    b   = Agente("B", umbral=th_b, reactividad=re_b, calma=cal_b)
    f   = FactoresExternos(estres=estres, terceros=terceros)
    sim = SimuladorPasoAPaso(a, b, f, trigger=trigger, seed=args.seed)
    viz = VisualizadorAnimado(sim, velocidad=args.velocidad, titulo=nombre)
    viz.correr()

    print(f"\n  Resultado final: {(sim.resultado or '?').upper()}")
    print(f"  Turnos totales: {sim.turno}\n")


if __name__ == "__main__":
    main()
