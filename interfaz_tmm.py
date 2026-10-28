"""
Interfaz Gráfica para TMM (Transfer Matrix Method) y Ajuste Elipsométrico
Permite configurar el stack óptico dinámicamente (agregar, quitar, reordenar capas),
seleccionar modelos de dispersión (Cauchy, Bruggeman, etc.) o materiales tabulados,
ejecutar ajustes elipsométricos con PyTorch (Autorange), optimizar espesores para
celdas solares / capas antirreflectantes, y comparar stacks y mediciones experimentales
de elipsometría (Is, Ic) o espectrofotometría (R).
"""

import sys
import os
import re
import ast
import queue
import threading
import traceback
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Asegurar que el directorio de este script esté en sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Funciones_tmm as ftmm
from tmm_utils_Rodrigo import (
    cauchy_fn, constant_fn, brugg_fn, load_interp, load_interp_only_n, load_interp_in3,
    leer_datos_guardados, guardar_resultados_txt, guardar_indice_refraccion,
    transform_I_to_psi_delta, RT_with_cache, stack2tmm, interpolar_todo
)
from tmm_core import coh_tmm

# ==============================================================================
# ESTRUCTURA DE DATOS PARA LAS CAPAS
# ==============================================================================
@dataclass
class LayerConfig:
    name: str = "Layer"
    layer_type: str = "c"  # 'i' para medio infinito (aire o sustrato), 'c' para capa delgada
    material: str = "air"
    model_type: str = "none"  # 'none', 'cauchy', 'cauchy_absorbent', 'bruggeman'
    d_mode: str = "range"  # 'inf', 'fixed', 'range'
    d_val: float = 100.0
    d_min: float = 0.0
    d_max: float = 100.0
    # Parámetros Cauchy / Cauchy absorbente
    cauchy_a_bounds: Tuple[float, float] = (0.0, 4.0)
    cauchy_b_bounds: Tuple[float, float] = (-10.0, 40.0)
    cauchy_c_bounds: Tuple[float, float] = (-40.0, 10.0)
    cauchy_ka_bounds: Tuple[float, float] = (0.0, 4.0)
    cauchy_kb_bounds: Tuple[float, float] = (-4.0, 4.0)
    cauchy_kc_bounds: Tuple[float, float] = (-4.0, 4.0)
    # Parámetros Bruggeman
    brugg_f_mode: str = "range"  # 'range', 'fixed'
    brugg_f_bounds: Tuple[float, float] = (0.3, 0.8)
    brugg_f_fixed: float = 0.5
    brugg_host_material: str = "air"
    brugg_link_mode: str = "default"  # 'default' (adyacente inf), 'stack' (capa del stack), 'custom' (nuevo material)
    brugg_linked_layer: str = ""      # nombre de la capa del stack a vincular
    brugg_custom_mat_name: str = "material_denso"
    brugg_custom_model: str = "cauchy"  # 'cauchy', 'cauchy_absorbent'
    brugg_custom_a_bounds: Tuple[float, float] = (0.0, 4.0)
    brugg_custom_b_bounds: Tuple[float, float] = (-10.0, 40.0)
    brugg_custom_c_bounds: Tuple[float, float] = (-40.0, 10.0)
    brugg_custom_ka_bounds: Tuple[float, float] = (0.0, 4.0)
    brugg_custom_kb_bounds: Tuple[float, float] = (-4.0, 4.0)
    brugg_custom_kc_bounds: Tuple[float, float] = (-4.0, 4.0)
    # Límites de índice n
    n_limits_enabled: bool = False
    n_min: float = 1.0
    n_max: float = 2.5

    def get_layer_model(self) -> Optional[Dict[str, Any]]:
        """Construye el diccionario layer_model requerido por tmm_utils_Rodrigo."""
        if self.model_type == "none" or self.layer_type == "i":
            return None
        elif self.model_type == "cauchy":
            return {
                'model': 'cauchy',
                'bounds': [
                    list(self.cauchy_a_bounds),
                    list(self.cauchy_b_bounds),
                    list(self.cauchy_c_bounds)
                ]
            }
        elif self.model_type == "cauchy_absorbent":
            return {
                'model': 'cauchy_absorbent',
                'bounds': [
                    list(self.cauchy_a_bounds),
                    list(self.cauchy_b_bounds),
                    list(self.cauchy_c_bounds),
                    list(self.cauchy_ka_bounds),
                    list(self.cauchy_kb_bounds),
                    list(self.cauchy_kc_bounds)
                ]
            }
        elif self.model_type == "bruggeman":
            bm = {'model': 'bruggeman'}
            if self.brugg_f_mode == "range":
                bm['f_bounds'] = [list(self.brugg_f_bounds)]
            else:
                bm['f_air'] = float(self.brugg_f_fixed)

            # Configurar linked_material según brugg_link_mode
            if self.brugg_link_mode == "stack" and self.brugg_linked_layer:
                bm['linked_material'] = self.brugg_linked_layer
            elif self.brugg_link_mode == "custom":
                custom_dict = {
                    'name': self.brugg_custom_mat_name.strip() or f"{self.name}_base",
                    'model': self.brugg_custom_model
                }
                if self.brugg_custom_model == "cauchy":
                    custom_dict['bounds'] = [
                        list(self.brugg_custom_a_bounds),
                        list(self.brugg_custom_b_bounds),
                        list(self.brugg_custom_c_bounds)
                    ]
                else:  # cauchy_absorbent
                    custom_dict['bounds'] = [
                        list(self.brugg_custom_a_bounds),
                        list(self.brugg_custom_b_bounds),
                        list(self.brugg_custom_c_bounds),
                        list(self.brugg_custom_ka_bounds),
                        list(self.brugg_custom_kb_bounds),
                        list(self.brugg_custom_kc_bounds)
                    ]
                bm['linked_material'] = custom_dict
            else:
                bm['linked_material'] = None

            return bm
        return None

    def get_d_spec(self) -> Any:
        """Devuelve el valor o rango de espesor físico."""
        if self.layer_type == "i":
            return np.inf
        if self.d_mode == "fixed":
            return float(self.d_val)
        return (float(self.d_min), float(self.d_max))

    def get_n_limit_spec(self) -> Optional[Tuple[float, float]]:
        """Devuelve los límites de índice n o None."""
        if self.layer_type != "i" and self.model_type != "none" and self.n_limits_enabled:
            return (float(self.n_min), float(self.n_max))
        return None



def layers_to_tmm_stack(layers: List[LayerConfig]) -> List[List[Any]]:
    """Convierte una lista de LayerConfig al formato nativo de tmm: [[d, 'material', 'c'/'i'], ...]."""
    st = []
    for l in layers:
        if l.layer_type == 'i':
            d = np.inf
        elif l.d_mode == 'fixed':
            d = float(l.d_val)
        elif l.d_mode == 'range':
            d = float(l.d_val if (l.d_val and l.d_min <= l.d_val <= l.d_max) else (l.d_min + l.d_max) / 2.0)
        else:
            d = np.inf
        st.append([d, l.material, l.layer_type])
    return st


def format_stack_summary(layers: List[LayerConfig]) -> str:
    """Devuelve un resumen textual amigable del stack."""
    parts = []
    for l in layers:
        if l.layer_type == 'i':
            parts.append(f"{l.name} (∞)")
        else:
            d_val = f"{l.d_val:.1f}nm" if l.d_mode == 'fixed' else f"({l.d_min:.0f}-{l.d_max:.0f}nm)"
            parts.append(f"{l.name} [{d_val}]")
    return "  /  ".join(parts)


# ==============================================================================
# REDIRECCIÓN DE SALIDA DE CONSOLA THREAD-SAFE
# ==============================================================================
class QueueWriter:
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue

    def write(self, text: str):
        if text:
            self.log_queue.put(text)

    def flush(self):
        pass


# ==============================================================================
# DIÁLOGO DE CONFIGURACIÓN DE PARÁMETROS DEL MODELO ÓPTICO
# ==============================================================================
class ModelParamsDialog(tk.Toplevel):
    def __init__(self, parent, layer: LayerConfig, material_list: List[str], stack_layers: Optional[List[str]] = None):
        super().__init__(parent)
        self.layer = layer
        self.material_list = material_list
        self.stack_layers = [s for s in (stack_layers or []) if s != layer.name]
        self.result = False
        
        self.title(f"Parámetros del Modelo Óptico - {layer.name}")
        self.geometry("580x680")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self):
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vscrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        container = ttk.Frame(canvas, padding=15)
        
        container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=vscrollbar.set)
        
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="left", fill="both", expand=True)
        vscrollbar.pack(side="right", fill="y")

        info_frame = ttk.LabelFrame(container, text="Modelo Seleccionado", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_frame, text=f"Capa: {self.layer.name}", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Tipo de modelo: {self.layer.model_type}").pack(anchor=tk.W)

        if self.layer.model_type in ("cauchy", "cauchy_absorbent"):
            cauchy_frame = ttk.LabelFrame(container, text="Límites de Parámetros Cauchy: n(λ) = A + B/λ² + C/λ⁴", padding=10)
            cauchy_frame.pack(fill=tk.X, pady=(0, 10))

            row_a = ttk.Frame(cauchy_frame)
            row_a.pack(fill=tk.X, pady=3)
            ttk.Label(row_a, text="A (Offset):", width=18).pack(side=tk.LEFT)
            ttk.Label(row_a, text="Min:").pack(side=tk.LEFT)
            self.ent_a_min = ttk.Entry(row_a, width=8)
            self.ent_a_min.insert(0, str(self.layer.cauchy_a_bounds[0]))
            self.ent_a_min.pack(side=tk.LEFT, padx=5)
            ttk.Label(row_a, text="Max:").pack(side=tk.LEFT)
            self.ent_a_max = ttk.Entry(row_a, width=8)
            self.ent_a_max.insert(0, str(self.layer.cauchy_a_bounds[1]))
            self.ent_a_max.pack(side=tk.LEFT, padx=5)

            row_b = ttk.Frame(cauchy_frame)
            row_b.pack(fill=tk.X, pady=3)
            ttk.Label(row_b, text="B (Dispersión 1):", width=18).pack(side=tk.LEFT)
            ttk.Label(row_b, text="Min:").pack(side=tk.LEFT)
            self.ent_b_min = ttk.Entry(row_b, width=8)
            self.ent_b_min.insert(0, str(self.layer.cauchy_b_bounds[0]))
            self.ent_b_min.pack(side=tk.LEFT, padx=5)
            ttk.Label(row_b, text="Max:").pack(side=tk.LEFT)
            self.ent_b_max = ttk.Entry(row_b, width=8)
            self.ent_b_max.insert(0, str(self.layer.cauchy_b_bounds[1]))
            self.ent_b_max.pack(side=tk.LEFT, padx=5)

            row_c = ttk.Frame(cauchy_frame)
            row_c.pack(fill=tk.X, pady=3)
            ttk.Label(row_c, text="C (Dispersión 2):", width=18).pack(side=tk.LEFT)
            ttk.Label(row_c, text="Min:").pack(side=tk.LEFT)
            self.ent_c_min = ttk.Entry(row_c, width=8)
            self.ent_c_min.insert(0, str(self.layer.cauchy_c_bounds[0]))
            self.ent_c_min.pack(side=tk.LEFT, padx=5)
            ttk.Label(row_c, text="Max:").pack(side=tk.LEFT)
            self.ent_c_max = ttk.Entry(row_c, width=8)
            self.ent_c_max.insert(0, str(self.layer.cauchy_c_bounds[1]))
            self.ent_c_max.pack(side=tk.LEFT, padx=5)

        if self.layer.model_type == "cauchy_absorbent":
            abs_frame = ttk.LabelFrame(container, text="Límites de Parámetros de Extinción: k(λ) = D + E/λ² + F/λ⁴", padding=10)
            abs_frame.pack(fill=tk.X, pady=(0, 10))

            row_d = ttk.Frame(abs_frame)
            row_d.pack(fill=tk.X, pady=3)
            ttk.Label(row_d, text="D (k Offset):", width=18).pack(side=tk.LEFT)
            self.ent_ka_min = ttk.Entry(row_d, width=8)
            self.ent_ka_min.insert(0, str(self.layer.cauchy_ka_bounds[0]))
            self.ent_ka_min.pack(side=tk.LEFT, padx=5)
            self.ent_ka_max = ttk.Entry(row_d, width=8)
            self.ent_ka_max.insert(0, str(self.layer.cauchy_ka_bounds[1]))
            self.ent_ka_max.pack(side=tk.LEFT, padx=5)

        if self.layer.model_type == "bruggeman":
            brugg_frame = ttk.LabelFrame(container, text="Modelo Bruggeman EMA (Medio Efectivo)", padding=10)
            brugg_frame.pack(fill=tk.X, pady=(0, 10))

            # 1. Sección Vinculación de Material (linked_material)
            link_lf = ttk.LabelFrame(brugg_frame, text="Material Base / Capa Vinculada (linked_material)", padding=8)
            link_lf.pack(fill=tk.X, pady=(0, 10))

            self.var_link_mode = tk.StringVar(value=self.layer.brugg_link_mode)

            ttk.Radiobutton(
                link_lf, text="Capa inferior adyacente (Automático / Por defecto)",
                variable=self.var_link_mode, value="default", command=self._on_link_mode_change
            ).pack(anchor=tk.W, pady=2)

            ttk.Radiobutton(
                link_lf, text="Vincular a otra capa existente en el stack",
                variable=self.var_link_mode, value="stack", command=self._on_link_mode_change
            ).pack(anchor=tk.W, pady=2)

            self.frame_link_stack = ttk.Frame(link_lf, padding=(22, 2, 0, 4))
            ttk.Label(self.frame_link_stack, text="Capa del Stack:").pack(side=tk.LEFT)
            stack_vals = self.stack_layers if self.stack_layers else ["(No hay otras capas)"]
            self.cmb_stack_layer = ttk.Combobox(self.frame_link_stack, values=stack_vals, state="readonly", width=22)
            if self.layer.brugg_linked_layer in stack_vals:
                self.cmb_stack_layer.set(self.layer.brugg_linked_layer)
            elif stack_vals:
                self.cmb_stack_layer.set(stack_vals[0])
            self.cmb_stack_layer.pack(side=tk.LEFT, padx=8)

            ttk.Radiobutton(
                link_lf, text="Nuevo material denso a ajustar (no en el stack)",
                variable=self.var_link_mode, value="custom", command=self._on_link_mode_change
            ).pack(anchor=tk.W, pady=(4, 2))

            self.frame_link_custom = ttk.Frame(link_lf, padding=(22, 4, 0, 4))

            row_cn = ttk.Frame(self.frame_link_custom)
            row_cn.pack(fill=tk.X, pady=2)
            ttk.Label(row_cn, text="Nombre del Material:", width=18).pack(side=tk.LEFT)
            self.ent_cust_name = ttk.Entry(row_cn, width=18)
            self.ent_cust_name.insert(0, str(self.layer.brugg_custom_mat_name))
            self.ent_cust_name.pack(side=tk.LEFT, padx=5)

            row_cm = ttk.Frame(self.frame_link_custom)
            row_cm.pack(fill=tk.X, pady=2)
            ttk.Label(row_cm, text="Modelo Dispersión:", width=18).pack(side=tk.LEFT)
            self.cmb_cust_model = ttk.Combobox(row_cm, values=["cauchy", "cauchy_absorbent"], state="readonly", width=18)
            self.cmb_cust_model.set(self.layer.brugg_custom_model)
            self.cmb_cust_model.bind("<<ComboboxSelected>>", lambda e: self._on_cust_model_change())
            self.cmb_cust_model.pack(side=tk.LEFT, padx=5)

            # Bounds Cauchy para el nuevo material:
            self.frame_cust_cauchy = ttk.Frame(self.frame_link_custom)
            self.frame_cust_cauchy.pack(fill=tk.X, pady=3)

            row_ca = ttk.Frame(self.frame_cust_cauchy)
            row_ca.pack(fill=tk.X, pady=2)
            ttk.Label(row_ca, text="A (Offset):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_ca, text="Min:").pack(side=tk.LEFT)
            self.ent_ca_min = ttk.Entry(row_ca, width=7)
            self.ent_ca_min.insert(0, str(self.layer.brugg_custom_a_bounds[0]))
            self.ent_ca_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_ca, text="Max:").pack(side=tk.LEFT)
            self.ent_ca_max = ttk.Entry(row_ca, width=7)
            self.ent_ca_max.insert(0, str(self.layer.brugg_custom_a_bounds[1]))
            self.ent_ca_max.pack(side=tk.LEFT, padx=4)

            row_cb = ttk.Frame(self.frame_cust_cauchy)
            row_cb.pack(fill=tk.X, pady=2)
            ttk.Label(row_cb, text="B (Disp 1):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_cb, text="Min:").pack(side=tk.LEFT)
            self.ent_cb_min = ttk.Entry(row_cb, width=7)
            self.ent_cb_min.insert(0, str(self.layer.brugg_custom_b_bounds[0]))
            self.ent_cb_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_cb, text="Max:").pack(side=tk.LEFT)
            self.ent_cb_max = ttk.Entry(row_cb, width=7)
            self.ent_cb_max.insert(0, str(self.layer.brugg_custom_b_bounds[1]))
            self.ent_cb_max.pack(side=tk.LEFT, padx=4)

            row_cc = ttk.Frame(self.frame_cust_cauchy)
            row_cc.pack(fill=tk.X, pady=2)
            ttk.Label(row_cc, text="C (Disp 2):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_cc, text="Min:").pack(side=tk.LEFT)
            self.ent_cc_min = ttk.Entry(row_cc, width=7)
            self.ent_cc_min.insert(0, str(self.layer.brugg_custom_c_bounds[0]))
            self.ent_cc_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_cc, text="Max:").pack(side=tk.LEFT)
            self.ent_cc_max = ttk.Entry(row_cc, width=7)
            self.ent_cc_max.insert(0, str(self.layer.brugg_custom_c_bounds[1]))
            self.ent_cc_max.pack(side=tk.LEFT, padx=4)

            # Bounds de extinción (k) para cauchy_absorbent:
            self.frame_cust_abs = ttk.Frame(self.frame_link_custom)

            row_cka = ttk.Frame(self.frame_cust_abs)
            row_cka.pack(fill=tk.X, pady=2)
            ttk.Label(row_cka, text="D (k Offset):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_cka, text="Min:").pack(side=tk.LEFT)
            self.ent_cka_min = ttk.Entry(row_cka, width=7)
            self.ent_cka_min.insert(0, str(self.layer.brugg_custom_ka_bounds[0]))
            self.ent_cka_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_cka, text="Max:").pack(side=tk.LEFT)
            self.ent_cka_max = ttk.Entry(row_cka, width=7)
            self.ent_cka_max.insert(0, str(self.layer.brugg_custom_ka_bounds[1]))
            self.ent_cka_max.pack(side=tk.LEFT, padx=4)

            row_ckb = ttk.Frame(self.frame_cust_abs)
            row_ckb.pack(fill=tk.X, pady=2)
            ttk.Label(row_ckb, text="E (k Disp 1):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_ckb, text="Min:").pack(side=tk.LEFT)
            self.ent_ckb_min = ttk.Entry(row_ckb, width=7)
            self.ent_ckb_min.insert(0, str(self.layer.brugg_custom_kb_bounds[0]))
            self.ent_ckb_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_ckb, text="Max:").pack(side=tk.LEFT)
            self.ent_ckb_max = ttk.Entry(row_ckb, width=7)
            self.ent_ckb_max.insert(0, str(self.layer.brugg_custom_kb_bounds[1]))
            self.ent_ckb_max.pack(side=tk.LEFT, padx=4)

            row_ckc = ttk.Frame(self.frame_cust_abs)
            row_ckc.pack(fill=tk.X, pady=2)
            ttk.Label(row_ckc, text="F (k Disp 2):", width=14).pack(side=tk.LEFT)
            ttk.Label(row_ckc, text="Min:").pack(side=tk.LEFT)
            self.ent_ckc_min = ttk.Entry(row_ckc, width=7)
            self.ent_ckc_min.insert(0, str(self.layer.brugg_custom_kc_bounds[0]))
            self.ent_ckc_min.pack(side=tk.LEFT, padx=4)
            ttk.Label(row_ckc, text="Max:").pack(side=tk.LEFT)
            self.ent_ckc_max = ttk.Entry(row_ckc, width=7)
            self.ent_ckc_max.insert(0, str(self.layer.brugg_custom_kc_bounds[1]))
            self.ent_ckc_max.pack(side=tk.LEFT, padx=4)

            # Inicializar estados de los subframes
            self._on_link_mode_change()

            # 2. Sección Fracción de Inclusión de Aire
            f_lf = ttk.LabelFrame(brugg_frame, text="Fracción de Inclusión de Aire (f_air)", padding=8)
            f_lf.pack(fill=tk.X, pady=(0, 10))

            f_frame = ttk.Frame(f_lf)
            f_frame.pack(fill=tk.X, pady=2)

            self.var_f_mode = tk.StringVar(value=self.layer.brugg_f_mode)
            ttk.Radiobutton(f_frame, text="Optimizar Rango", variable=self.var_f_mode, value="range").pack(side=tk.LEFT)
            ttk.Radiobutton(f_frame, text="Fijo", variable=self.var_f_mode, value="fixed").pack(side=tk.LEFT, padx=15)

            range_f = ttk.Frame(f_lf)
            range_f.pack(fill=tk.X, pady=5)
            ttk.Label(range_f, text="f Min:").pack(side=tk.LEFT)
            self.ent_f_min = ttk.Entry(range_f, width=8)
            self.ent_f_min.insert(0, str(self.layer.brugg_f_bounds[0]))
            self.ent_f_min.pack(side=tk.LEFT, padx=5)
            ttk.Label(range_f, text="f Max:").pack(side=tk.LEFT)
            self.ent_f_max = ttk.Entry(range_f, width=8)
            self.ent_f_max.insert(0, str(self.layer.brugg_f_bounds[1]))
            self.ent_f_max.pack(side=tk.LEFT, padx=5)

            ttk.Label(range_f, text="f Fijo:").pack(side=tk.LEFT, padx=(15, 0))
            self.ent_f_fix = ttk.Entry(range_f, width=8)
            self.ent_f_fix.insert(0, str(self.layer.brugg_f_fixed))
            self.ent_f_fix.pack(side=tk.LEFT, padx=5)

            # 3. Inclusión Host
            host_frame = ttk.Frame(brugg_frame)
            host_frame.pack(fill=tk.X, pady=2)
            ttk.Label(host_frame, text="Material Inclusión (Host):").pack(side=tk.LEFT)
            self.cmb_host = ttk.Combobox(host_frame, values=self.material_list, state="readonly", width=18)
            self.cmb_host.set(self.layer.brugg_host_material if self.layer.brugg_host_material in self.material_list else "air")
            self.cmb_host.pack(side=tk.LEFT, padx=8)

        n_lim_frame = ttk.LabelFrame(container, text="Filtro de Índice de Refracción Real n(λ)", padding=10)
        n_lim_frame.pack(fill=tk.X, pady=(0, 10))

        self.var_n_enabled = tk.BooleanVar(value=self.layer.n_limits_enabled)
        chk_n = ttk.Checkbutton(n_lim_frame, text="Restringir límites de n (n_min ≤ n ≤ n_max)", variable=self.var_n_enabled)
        chk_n.pack(anchor=tk.W, pady=(0, 5))

        n_row = ttk.Frame(n_lim_frame)
        n_row.pack(fill=tk.X)
        ttk.Label(n_row, text="n_min:").pack(side=tk.LEFT)
        self.ent_n_min = ttk.Entry(n_row, width=8)
        self.ent_n_min.insert(0, str(self.layer.n_min))
        self.ent_n_min.pack(side=tk.LEFT, padx=5)
        ttk.Label(n_row, text="n_max:").pack(side=tk.LEFT, padx=(10, 0))
        self.ent_n_max = ttk.Entry(n_row, width=8)
        self.ent_n_max.insert(0, str(self.layer.n_max))
        self.ent_n_max.pack(side=tk.LEFT, padx=5)

        btn_bar = ttk.Frame(container)
        btn_bar.pack(fill=tk.X, pady=(15, 5))
        ttk.Button(btn_bar, text="Guardar Cambios", command=self._save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_bar, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

    def _on_link_mode_change(self):
        mode = self.var_link_mode.get()
        if mode == "stack":
            self.frame_link_stack.pack(fill=tk.X, pady=3)
            self.frame_link_custom.pack_forget()
        elif mode == "custom":
            self.frame_link_stack.pack_forget()
            self.frame_link_custom.pack(fill=tk.X, pady=3)
            self._on_cust_model_change()
        else:
            self.frame_link_stack.pack_forget()
            self.frame_link_custom.pack_forget()

    def _on_cust_model_change(self):
        m = self.cmb_cust_model.get()
        if m == "cauchy_absorbent":
            self.frame_cust_abs.pack(fill=tk.X, pady=2)
        else:
            self.frame_cust_abs.pack_forget()

    def _save_and_close(self):
        try:
            if self.layer.model_type in ("cauchy", "cauchy_absorbent"):
                a_min = float(self.ent_a_min.get())
                a_max = float(self.ent_a_max.get())
                b_min = float(self.ent_b_min.get())
                b_max = float(self.ent_b_max.get())
                c_min = float(self.ent_c_min.get())
                c_max = float(self.ent_c_max.get())
                self.layer.cauchy_a_bounds = (a_min, a_max)
                self.layer.cauchy_b_bounds = (b_min, b_max)
                self.layer.cauchy_c_bounds = (c_min, c_max)

            if self.layer.model_type == "cauchy_absorbent":
                ka_min = float(self.ent_ka_min.get())
                ka_max = float(self.ent_ka_max.get())
                self.layer.cauchy_ka_bounds = (ka_min, ka_max)

            if self.layer.model_type == "bruggeman":
                self.layer.brugg_host_material = self.cmb_host.get()
                self.layer.brugg_f_mode = self.var_f_mode.get()
                f_min = float(self.ent_f_min.get())
                f_max = float(self.ent_f_max.get())
                f_fix = float(self.ent_f_fix.get())
                self.layer.brugg_f_bounds = (f_min, f_max)
                self.layer.brugg_f_fixed = f_fix

                link_mode = self.var_link_mode.get()
                self.layer.brugg_link_mode = link_mode
                if link_mode == "stack":
                    self.layer.brugg_linked_layer = self.cmb_stack_layer.get().strip()
                elif link_mode == "custom":
                    self.layer.brugg_custom_mat_name = self.ent_cust_name.get().strip() or "material_denso"
                    self.layer.brugg_custom_model = self.cmb_cust_model.get()
                    ca_min = float(self.ent_ca_min.get())
                    ca_max = float(self.ent_ca_max.get())
                    cb_min = float(self.ent_cb_min.get())
                    cb_max = float(self.ent_cb_max.get())
                    cc_min = float(self.ent_cc_min.get())
                    cc_max = float(self.ent_cc_max.get())
                    self.layer.brugg_custom_a_bounds = (ca_min, ca_max)
                    self.layer.brugg_custom_b_bounds = (cb_min, cb_max)
                    self.layer.brugg_custom_c_bounds = (cc_min, cc_max)
                    if self.layer.brugg_custom_model == "cauchy_absorbent":
                        cka_min = float(self.ent_cka_min.get())
                        cka_max = float(self.ent_cka_max.get())
                        ckb_min = float(self.ent_ckb_min.get())
                        ckb_max = float(self.ent_ckb_max.get())
                        ckc_min = float(self.ent_ckc_min.get())
                        ckc_max = float(self.ent_ckc_max.get())
                        self.layer.brugg_custom_ka_bounds = (cka_min, cka_max)
                        self.layer.brugg_custom_kb_bounds = (ckb_min, ckb_max)
                        self.layer.brugg_custom_kc_bounds = (ckc_min, ckc_max)

            self.layer.n_limits_enabled = self.var_n_enabled.get()
            self.layer.n_min = float(self.ent_n_min.get())
            self.layer.n_max = float(self.ent_n_max.get())

            self.result = True
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Error de Valor", f"Por favor verifica que todos los campos sean números válidos:\n{e}", parent=self)


# ==============================================================================
# DIÁLOGO DE CONFIGURACIÓN DEL STACK 2 (PARA COMPARACIONES)
# ==============================================================================
class EditStack2Dialog(tk.Toplevel):
    def __init__(self, parent, layers: List[LayerConfig], material_list: List[str]):
        super().__init__(parent)
        self.layers = copy.deepcopy(layers)
        self.material_list = material_list
        self.result = False

        self.title("Configuración de Stack 2 (Secundario)")
        self.geometry("760x490")
        self.minsize(640, 380)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self):
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(toolbar, text="➕ Agregar", command=self._add_layer).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗑️ Eliminar", command=self._del_layer).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⬆️ Subir", command=lambda: self._move_layer(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⬇️ Bajar", command=lambda: self._move_layer(1)).pack(side=tk.LEFT, padx=2)

        ttk.Label(toolbar, text="Preset:").pack(side=tk.LEFT, padx=(10, 3))
        self.cmb_preset = ttk.Combobox(
            toolbar,
            values=[
                "Preset Celda TiO2 oxigeno (Al2O3 / TiO2 ox / SiO2 / Si)",
                "Preset Celda TiO2 aire (Al2O3 / TiO2 aire / SiO2 / Si)",
                "Preset Antirreflectante (MgF2 / Si)",
                "Preset Sustrato Puro (Si)"
            ],
            state="readonly", width=38
        )
        self.cmb_preset.current(0)
        self.cmb_preset.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Cargar", command=self._apply_preset).pack(side=tk.LEFT, padx=2)

        # Contenedor desplazable de capas
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.layers_frame = ttk.Frame(canvas)

        self.layers_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.layers_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.selected_idx: Optional[int] = None
        self._refresh_rows()

        btn_bar = ttk.Frame(self, padding=(12, 8))
        btn_bar.pack(fill=tk.X)
        ttk.Button(btn_bar, text="Guardar Stack 2", style="Accent.TButton", command=self._save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_bar, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

    def _refresh_rows(self):
        for w in self.layers_frame.winfo_children():
            w.destroy()

        headers = ["#", "Nombre", "Tipo", "Material", "Espesor d [nm]"]
        widths = [4, 18, 14, 20, 16]
        h_frame = ttk.Frame(self.layers_frame, padding=(4, 4))
        h_frame.pack(fill=tk.X)
        for col, (ht, w) in enumerate(zip(headers, widths)):
            ttk.Label(h_frame, text=ht, font=("Segoe UI", 9, "bold"), width=w).grid(row=0, column=col, padx=3, sticky=tk.W)

        for i, l in enumerate(self.layers):
            bg = "#e8f0fe" if i == self.selected_idx else ("#f8f9fa" if i % 2 == 0 else "#ffffff")
            rf = tk.Frame(self.layers_frame, bg=bg, padx=4, pady=3, relief=tk.SOLID, bd=1 if i == self.selected_idx else 0)
            rf.pack(fill=tk.X, pady=1)

            def select_fn(idx):
                return lambda e: self._select_row(idx)
            rf.bind("<Button-1>", select_fn(i))

            tk.Label(rf, text=f"#{i}", width=4, bg=bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=3)

            var_n = tk.StringVar(value=l.name)
            def on_n_ch(var, idx):
                def cb(*a):
                    self.layers[idx].name = var.get()
                return cb
            var_n.trace_add("write", on_n_ch(var_n, i))
            ttk.Entry(rf, textvariable=var_n, width=18).grid(row=0, column=1, padx=3)

            cmb_t = ttk.Combobox(rf, values=["Infinito (i)", "Capa Delgada (c)"], state="readonly", width=14)
            cmb_t.set("Infinito (i)" if l.layer_type == "i" else "Capa Delgada (c)")
            def on_t_ch(idx, c):
                self.layers[idx].layer_type = "i" if "Infinito" in c.get() else "c"
                self._refresh_rows()
            cmb_t.bind("<<ComboboxSelected>>", lambda e, idx=i, c=cmb_t: on_t_ch(idx, c))
            cmb_t.grid(row=0, column=2, padx=3)

            cmb_m = ttk.Combobox(rf, values=self.material_list, state="readonly", width=20)
            cmb_m.set(l.material if l.material in self.material_list else "air")
            def on_m_ch(idx, c):
                self.layers[idx].material = c.get()
            cmb_m.bind("<<ComboboxSelected>>", lambda e, idx=i, c=cmb_m: on_m_ch(idx, c))
            cmb_m.grid(row=0, column=3, padx=3)

            if l.layer_type == "i":
                tk.Label(rf, text="Infinito (∞)", bg=bg, font=("Segoe UI", 9, "italic")).grid(row=0, column=4, padx=3)
            else:
                ent_d = ttk.Entry(rf, width=14)
                ent_d.insert(0, f"{l.d_val:.1f}")
                def on_d_ch(e, idx=i, ent=ent_d):
                    try:
                        self.layers[idx].d_val = float(ent.get().strip())
                        self.layers[idx].d_mode = "fixed"
                    except ValueError:
                        pass
                ent_d.bind("<FocusOut>", on_d_ch)
                ent_d.bind("<Return>", on_d_ch)
                ent_d.grid(row=0, column=4, padx=3)

    def _select_row(self, idx: int):
        self.selected_idx = idx
        self._refresh_rows()

    def _add_layer(self):
        new_l = LayerConfig(name=f"Layer_{len(self.layers)}", layer_type="c", material="SiO2", d_mode="fixed", d_val=50.0)
        if len(self.layers) >= 2:
            self.layers.insert(len(self.layers) - 1, new_l)
        else:
            self.layers.append(new_l)
        self.selected_idx = len(self.layers) - 2
        self._refresh_rows()

    def _del_layer(self):
        if self.selected_idx is not None and len(self.layers) > 2:
            del self.layers[self.selected_idx]
            self.selected_idx = min(self.selected_idx, len(self.layers) - 1)
            self._refresh_rows()

    def _move_layer(self, delta: int):
        if self.selected_idx is not None:
            n_idx = self.selected_idx + delta
            if 0 <= n_idx < len(self.layers):
                self.layers[self.selected_idx], self.layers[n_idx] = self.layers[n_idx], self.layers[self.selected_idx]
                self.selected_idx = n_idx
                self._refresh_rows()

    def _apply_preset(self):
        p_idx = self.cmb_preset.current()
        if p_idx == 0:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", d_mode="inf"),
                LayerConfig(name="Al2O3_porosa", layer_type="c", material="Al2O3_porosa", d_mode="fixed", d_val=95.0),
                LayerConfig(name="TiO2_RTP_oxigeno", layer_type="c", material="TiO2_RTP_oxigeno", d_mode="fixed", d_val=33.5),
                LayerConfig(name="SiO2", layer_type="c", material="SiO2", d_mode="fixed", d_val=11.0),
                LayerConfig(name="Si", layer_type="i", material="Si", d_mode="inf")
            ]
        elif p_idx == 1:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", d_mode="inf"),
                LayerConfig(name="Al2O3_porosa", layer_type="c", material="Al2O3_porosa", d_mode="fixed", d_val=95.0),
                LayerConfig(name="TiO2_RTP_aire", layer_type="c", material="TiO2_RTP_aire", d_mode="fixed", d_val=33.5),
                LayerConfig(name="SiO2", layer_type="c", material="SiO2", d_mode="fixed", d_val=11.0),
                LayerConfig(name="Si", layer_type="i", material="Si", d_mode="inf")
            ]
        elif p_idx == 2:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", d_mode="inf"),
                LayerConfig(name="MgF2", layer_type="c", material="MgF2", d_mode="fixed", d_val=105.0),
                LayerConfig(name="Si", layer_type="i", material="Si", d_mode="inf")
            ]
        elif p_idx == 3:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", d_mode="inf"),
                LayerConfig(name="Si", layer_type="i", material="Si", d_mode="inf")
            ]
        self.selected_idx = None
        self._refresh_rows()

    def _save_and_close(self):
        self.result = True
        self.destroy()


# ==============================================================================
# CLASE PARA TOOLTIPS / MENSAJES DESPLEGABLES FLOTANTES
# ==============================================================================
class ToolTip:
    """Muestra un mensaje flotante (tooltip desplegable) al posar el mouse sobre un widget."""
    def __init__(self, widget, text: str, wraplength: int = 380, delay: int = 150):
        self.widget = widget
        self.text = text.strip()
        self.wraplength = wraplength
        self.delay = delay
        self.tip_window = None
        self._after_id = None
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")
        self.widget.bind("<Destroy>", self._on_destroy, add="+")

    def _on_enter(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _on_destroy(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self.tip_window or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass

        frame = tk.Frame(tw, bg="#1e293b", relief="solid", bd=1, padx=10, pady=8)
        frame.pack()
        lbl = tk.Label(
            frame,
            text=self.text,
            justify=tk.LEFT,
            bg="#1e293b",
            fg="#f8fafc",
            font=("Segoe UI", 9),
            wraplength=self.wraplength
        )
        lbl.pack()

        tw.update_idletasks()
        w = tw.winfo_width()
        h = tw.winfo_height()
        sw = tw.winfo_screenwidth()
        sh = tw.winfo_screenheight()
        if x + w > sw - 10:
            x = max(10, sw - w - 10)
        if y + h > sh - 10:
            y = max(10, self.widget.winfo_rooty() - h - 4)
        tw.geometry(f"+{x}+{y}")

    def _hide(self):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================
class TMMInterfaceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TMM Suite - Análisis Óptico y Ajuste Elipsométrico")
        self.root.geometry("1300x890")
        self.root.minsize(1050, 700)

        # Estado del stack óptico principal (Stack 1)
        self.layers: List[LayerConfig] = []
        self._init_default_stack()

        # Estado de Stack 2 (para comparaciones)
        self.layers_stack2: List[LayerConfig] = []
        self._init_default_stack2()

        # Cola para logs de consola
        self.log_queue = queue.Queue()
        self.queue_writer = QueueWriter(self.log_queue)

        # Hilo de optimización
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_requested = False

        # Resultados de elipsometría
        self.last_elips_res: Optional[Dict[str, Any]] = None
        # Resultados de reflectancia
        self.last_rt_res: Optional[Dict[str, Any]] = None
        # Curvas comparativas de reflectancia: [(label, lams, R_curve)]
        self.comparison_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
        # Rangos específicos de longitud de onda [nm] por material: {mat_name: (min, max)}
        self.custom_material_ranges: Dict[str, Tuple[float, float]] = {}

        self._setup_style()
        self._build_main_ui()
        self._check_log_queue()

    def _init_default_stack(self):
        """Carga el stack por defecto (Preset de Elipsometría Al2O3)."""
        self.layers = [
            LayerConfig(name="air", layer_type="i", material="air", model_type="none", d_mode="inf"),
            LayerConfig(
                name="Al2O3_poroso", layer_type="c", material="Al2O3", model_type="bruggeman",
                d_mode="range", d_min=200.0, d_max=500.0, brugg_f_bounds=(0.3, 0.8),
                n_limits_enabled=False
            ),
            LayerConfig(
                name="Al2O3_denso", layer_type="c", material="Al2O3", model_type="cauchy",
                d_mode="range", d_min=200.0, d_max=500.0,
                cauchy_a_bounds=(0.0, 4.0), cauchy_b_bounds=(-10.0, 40.0), cauchy_c_bounds=(-40.0, 10.0),
                n_limits_enabled=True, n_min=1.0, n_max=1.8
            ),
            LayerConfig(
                name="SiO2", layer_type="c", material="SiO2", model_type="cauchy",
                d_mode="range", d_min=0.0, d_max=15.0,
                cauchy_a_bounds=(0.0, 4.0), cauchy_b_bounds=(-25.0, 25.0), cauchy_c_bounds=(-25.0, 25.0),
                n_limits_enabled=True, n_min=1.0, n_max=1.6
            ),
            LayerConfig(name="Si", layer_type="i", material="Si", model_type="none", d_mode="inf")
        ]

    def _init_default_stack2(self):
        """Carga el Stack 2 por defecto (Celda TiO2/Al2O3)."""
        self.layers_stack2 = [
            LayerConfig(name="air", layer_type="i", material="air", model_type="none", d_mode="inf"),
            LayerConfig(name="Al2O3_porosa", layer_type="c", material="Al2O3_porosa", model_type="none", d_mode="fixed", d_val=95.04),
            LayerConfig(name="TiO2_RTP_oxigeno", layer_type="c", material="TiO2_RTP_oxigeno", model_type="none", d_mode="fixed", d_val=33.57),
            LayerConfig(name="SiO2", layer_type="c", material="SiO2", model_type="none", d_mode="fixed", d_val=11.0),
            LayerConfig(name="Si", layer_type="i", material="Si", model_type="none", d_mode="inf")
        ]

    def _setup_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.style.configure(".", font=("Segoe UI", 9))
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"))
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_main_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Barra superior
        top_bar = ttk.Frame(main_frame, padding=(15, 8))
        top_bar.pack(fill=tk.X)

        title_lbl = ttk.Label(top_bar, text="⚡ TMM Suite - Modelado Óptico, Ajuste Elipsométrico y Comparación", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        cuda_ok = torch.cuda.is_available()
        cuda_txt = f"🟢 GPU CUDA Activa ({torch.cuda.get_device_name(0)})" if cuda_ok else "🟡 Dispositivo: CPU (CUDA no detectado)"
        cuda_lbl = ttk.Label(top_bar, text=cuda_txt, font=("Segoe UI", 9, "italic"))
        cuda_lbl.pack(side=tk.RIGHT)

        # Notebook principal (Pestañas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        self.tab_stack = ttk.Frame(self.notebook, padding=10)
        self.tab_elips = ttk.Frame(self.notebook, padding=10)
        self.tab_refl = ttk.Frame(self.notebook, padding=10)
        self.tab_comp = ttk.Frame(self.notebook, padding=10)
        self.tab_mat = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_stack, text="  🧱 Stack Óptico (Capas)  ")
        self.notebook.add(self.tab_elips, text="  🔬 Ajuste Elipsométrico  ")
        self.notebook.add(self.tab_refl, text="  ☀️ Reflectancia y Jsc  ")
        self.notebook.add(self.tab_comp, text="  📊 Comparar Stacks / Medidas  ")
        self.notebook.add(self.tab_mat, text="  📚 Biblioteca de Materiales  ")

        self._build_tab_stack()
        self._build_tab_elips()
        self._build_tab_refl()
        self._build_tab_comp()
        self._build_tab_mat()

        # Panel inferior: Consola y Barra de Estado
        bot_frame = ttk.Frame(main_frame, padding=(10, 5))
        bot_frame.pack(fill=tk.X)

        self.console_visible = False
        self.btn_toggle_console = ttk.Button(bot_frame, text="▲ Mostrar Consola de Salida", command=self._toggle_console, width=25)
        self.btn_toggle_console.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(bot_frame, text="Listo para operar.", font=("Segoe UI", 9))
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        self.progress_bar = ttk.Progressbar(bot_frame, mode="indeterminate", length=200)

        self.console_frame = ttk.Frame(main_frame, padding=10)
        self.txt_console = tk.Text(self.console_frame, height=8, bg="#1e1e1e", fg="#00ff66", font=("Consolas", 9))
        scroll_console = ttk.Scrollbar(self.console_frame, orient=tk.VERTICAL, command=self.txt_console.yview)
        self.txt_console.configure(yscrollcommand=scroll_console.set)
        scroll_console.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _toggle_console(self):
        if self.console_visible:
            self.console_frame.pack_forget()
            self.btn_toggle_console.config(text="▲ Mostrar Consola de Salida")
            self.console_visible = False
        else:
            self.console_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
            self.btn_toggle_console.config(text="▼ Ocultar Consola de Salida")
            self.console_visible = True

    def _check_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.txt_console.insert(tk.END, msg)
            self.txt_console.see(tk.END)
        self.root.after(100, self._check_log_queue)

    def log(self, text: str):
        self.log_queue.put(text + "\n")

    # ==========================================================================
    # PESTAÑA 1: GESTIÓN DEL STACK ÓPTICO (CAPAS Y MATERIALES)
    # ==========================================================================
    def _build_tab_stack(self):
        toolbar = ttk.Frame(self.tab_stack)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(toolbar, text="➕ Agregar Capa", command=self._add_layer).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="🗑️ Eliminar Capa", command=self._delete_selected_layer).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="⬆️ Subir Capa", command=lambda: self._move_layer(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="⬇️ Bajar Capa", command=lambda: self._move_layer(1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="📑 Duplicar", command=self._duplicate_layer).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Presets:").pack(side=tk.LEFT, padx=(15, 3))
        self.cmb_preset = ttk.Combobox(
            toolbar,
            values=[
                "Preset 1: Elipsometría Al2O3 (poroso / denso / SiO2 / Si)",
                "Preset 2: Celda TiO2 + Al2O3 (porosa / TiO2 RTP / SiO2 / Si)",
                "Preset 3: Antirreflectante Simple (MgF2 / Si)",
                "Preset 4: Vacío Básico (Aire / Si)"
            ],
            state="readonly", width=42
        )
        self.cmb_preset.current(0)
        self.cmb_preset.pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Cargar Preset", command=self._apply_preset).pack(side=tk.LEFT, padx=3)

        table_container = ttk.Frame(self.tab_stack)
        table_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(table_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=canvas.yview)
        self.scrollable_layers_frame = ttk.Frame(canvas)

        self.scrollable_layers_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_layers_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.selected_layer_idx: Optional[int] = None
        self._refresh_layers_ui()

    def _refresh_layers_ui(self):
        for widget in self.scrollable_layers_frame.winfo_children():
            widget.destroy()

        headers = ["#", "Nombre", "Tipo de Medio", "Material Base", "Modelo Óptico", "Espesor d [nm]", "Límites n", "Parámetros", "Acciones"]
        header_frame = ttk.Frame(self.scrollable_layers_frame, padding=(5, 5))
        header_frame.pack(fill=tk.X)

        tooltips = {
            "Nombre": (
                "En esta columna debe ingresarse el nombre de la capa, evitar\n"
                "nombres repetidos."
            ),
            "Material Base": (
                "En esta columna se debe ingresar el nombre del material para cargar sus valores de n y k. "
                "La lista desplegable se compone de todos los materiales que esten cargados en la biblioteca de materiales.\n\n"
                "En caso de seleccionar algún modelo distinto de \"ninguno (tabulado)\" en la columna de modelo optico no se usará "
                "este material base y en cambio se ajustaran los parametros del modelo seleccionado."
            ),
            "Modelo Óptico": (
                "En esta columna se debe seleccionar el modelo optico a ajustar. El modelo optico unicamente se utilizará para "
                "la pestaña de ajuste elipsométrico, mientras que la pestaña de reflectancia y Jsc usará automaticamente el "
                "material base seleccionado para cada capa."
            ),
            "Espesor d [nm]": (
                "para determinar un rango de espesores a optimizar escribir: \"(valor_minimo,valor_maximo)\". "
                "Para elegír un valor fijo para el espesor escribir: \"(valor)\""
            ),
            "Límites n": (
                "Limita el valor máximo y minimo que puede tener n para cada capa, evitando que el modelo de valores ridiculos. "
                "Los mismos pueden ajustarse desde el boton de ajustar de cada copa. En la pestaña de ajustar también pueden "
                "ajustarse los valores maximos y minimos de los parámetros del modelo óptico."
            ),
        }

        widths = [4, 18, 16, 18, 18, 18, 14, 12, 10]
        for col_idx, (h_text, w) in enumerate(zip(headers, widths)):
            tip = tooltips.get(h_text)
            display_text = f"{h_text} (?)" if tip else h_text
            lbl = ttk.Label(
                header_frame,
                text=display_text,
                font=("Segoe UI", 9, "bold"),
                width=w,
                anchor=tk.W,
                cursor="hand2" if tip else ""
            )
            lbl.grid(row=0, column=col_idx, padx=4, sticky=tk.W)
            if tip:
                ToolTip(lbl, tip)

        separator = ttk.Separator(self.scrollable_layers_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=2)

        mat_keys = list(ftmm.materials.keys())

        for idx, layer in enumerate(self.layers):
            row_bg = "#e8f0fe" if idx == self.selected_layer_idx else ("#f8f9fa" if idx % 2 == 0 else "#ffffff")
            row_frame = tk.Frame(self.scrollable_layers_frame, bg=row_bg, padx=5, pady=4, relief=tk.SOLID, bd=1 if idx == self.selected_layer_idx else 0)
            row_frame.pack(fill=tk.X, pady=1)

            def make_select(i):
                return lambda e: self._select_layer(i)
            row_frame.bind("<Button-1>", make_select(idx))

            tk.Label(row_frame, text=f"#{idx}", width=4, anchor="w", bg=row_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=4)

            var_name = tk.StringVar(value=layer.name)
            def on_name_change(var, i):
                def callback(*args):
                    self.layers[i].name = var.get()
                return callback
            var_name.trace_add("write", on_name_change(var_name, idx))
            ent_name = ttk.Entry(row_frame, textvariable=var_name, width=18)
            ent_name.grid(row=0, column=1, padx=4)

            type_options = {"i": "Infinito (Incidente/Sustrato)", "c": "Capa Delgada (Finita)"}
            cmb_type = ttk.Combobox(row_frame, values=list(type_options.values()), state="readonly", width=16)
            cmb_type.set(type_options.get(layer.layer_type, "Capa Delgada (Finita)"))
            def on_type_change(i, cmb):
                val = cmb.get()
                self.layers[i].layer_type = "i" if "Infinito" in val else "c"
                if self.layers[i].layer_type == "i":
                    self.layers[i].d_mode = "inf"
                self._refresh_layers_ui()
                self._update_comp_stack_labels()
            cmb_type.bind("<<ComboboxSelected>>", lambda e, i=idx, c=cmb_type: on_type_change(i, c))
            cmb_type.grid(row=0, column=2, padx=4)

            cmb_mat = ttk.Combobox(row_frame, values=mat_keys, state="readonly", width=18)
            cmb_mat.set(layer.material if layer.material in mat_keys else (mat_keys[0] if mat_keys else "air"))
            def on_mat_change(i, cmb):
                self.layers[i].material = cmb.get()
                self._update_comp_stack_labels()
            cmb_mat.bind("<<ComboboxSelected>>", lambda e, i=idx, c=cmb_mat: on_mat_change(i, c))
            cmb_mat.grid(row=0, column=3, padx=4)

            model_options = {
                "none": "Ninguno (Tabulado)",
                "cauchy": "Cauchy (n)",
                "cauchy_absorbent": "Cauchy Absorbente (n, k)",
                "bruggeman": "Bruggeman EMA"
            }
            cmb_mod = ttk.Combobox(row_frame, values=list(model_options.values()), state="readonly", width=18)
            cmb_mod.set(model_options.get(layer.model_type, "Ninguno (Tabulado)"))
            def on_mod_change(i, cmb):
                sel = cmb.get()
                for k, v in model_options.items():
                    if v == sel:
                        self.layers[i].model_type = k
                        if k == "none":
                            self.layers[i].n_limits_enabled = False
                        break
                self._refresh_layers_ui()
            cmb_mod.bind("<<ComboboxSelected>>", lambda e, i=idx, c=cmb_mod: on_mod_change(i, c))
            cmb_mod.grid(row=0, column=4, padx=4)

            d_frame = tk.Frame(row_frame, bg=row_bg)
            d_frame.grid(row=0, column=5, padx=4, sticky="w")
            if layer.layer_type == "i":
                tk.Label(d_frame, text="Infinito (∞)", bg=row_bg, font=("Segoe UI", 9, "italic")).pack(side=tk.LEFT)
            else:
                ent_d = ttk.Entry(d_frame, width=15)
                if layer.d_mode == "range":
                    ent_d.insert(0, f"({layer.d_min:.1f}, {layer.d_max:.1f})")
                else:
                    ent_d.insert(0, f"{layer.d_val:.1f}")
                
                def on_d_change(e, i=idx, ent=ent_d):
                    val_str = ent.get().strip()
                    try:
                        if "(" in val_str or "," in val_str:
                            clean = val_str.replace("(", "").replace(")", "").split(",")
                            d1, d2 = float(clean[0]), float(clean[1])
                            self.layers[i].d_mode = "range"
                            self.layers[i].d_min, self.layers[i].d_max = min(d1, d2), max(d1, d2)
                        else:
                            self.layers[i].d_mode = "fixed"
                            self.layers[i].d_val = float(val_str)
                        self._update_comp_stack_labels()
                    except Exception:
                        pass
                ent_d.bind("<FocusOut>", on_d_change)
                ent_d.bind("<Return>", on_d_change)
                ent_d.pack(side=tk.LEFT)

            if layer.layer_type == "i":
                n_txt = "Infinito (∞)"
                is_adjustable = False
            elif layer.model_type == "none":
                n_txt = "No aplica"
                is_adjustable = False
            else:
                n_txt = f"({layer.n_min}, {layer.n_max})" if layer.n_limits_enabled else "Sin límite"
                is_adjustable = True

            tk.Label(row_frame, text=n_txt, width=14, anchor="w", bg=row_bg).grid(row=0, column=6, padx=4)

            state_p = "normal" if is_adjustable else "disabled"
            btn_params = ttk.Button(
                row_frame, text="⚙️ Ajustar", state=state_p, width=10,
                command=lambda l=layer: self._open_model_dialog(l)
            )
            btn_params.grid(row=0, column=7, padx=4)

            act_frame = tk.Frame(row_frame, bg=row_bg)
            act_frame.grid(row=0, column=8, padx=4)
            btn_up = ttk.Button(act_frame, text="▲", width=3, command=lambda i=idx: self._move_layer_direct(i, -1))
            btn_down = ttk.Button(act_frame, text="▼", width=3, command=lambda i=idx: self._move_layer_direct(i, 1))
            btn_del = ttk.Button(act_frame, text="✖", width=3, command=lambda i=idx: self._delete_layer_direct(i))
            btn_up.pack(side=tk.LEFT, padx=1)
            btn_down.pack(side=tk.LEFT, padx=1)
            btn_del.pack(side=tk.LEFT, padx=1)

        self._update_comp_stack_labels()

    def _select_layer(self, index: int):
        self.selected_layer_idx = index
        self._refresh_layers_ui()

    def _open_model_dialog(self, layer: LayerConfig):
        if layer.layer_type == "i" or layer.model_type == "none":
            return
        stack_layer_names = [l.name for l in self.layers if l != layer]
        dlg = ModelParamsDialog(self.root, layer, list(ftmm.materials.keys()), stack_layers=stack_layer_names)
        self.root.wait_window(dlg)
        if dlg.result:
            self._refresh_layers_ui()

    def _add_layer(self):
        new_layer = LayerConfig(
            name=f"Nueva_Capa_{len(self.layers)}",
            layer_type="c",
            material="SiO2",
            model_type="none",
            d_mode="range",
            d_min=10.0,
            d_max=100.0
        )
        if len(self.layers) >= 2:
            self.layers.insert(len(self.layers) - 1, new_layer)
        else:
            self.layers.append(new_layer)
        self.selected_layer_idx = len(self.layers) - 2 if len(self.layers) >= 2 else 0
        self._refresh_layers_ui()

    def _delete_selected_layer(self):
        if self.selected_layer_idx is not None and 0 <= self.selected_layer_idx < len(self.layers):
            self._delete_layer_direct(self.selected_layer_idx)

    def _delete_layer_direct(self, idx: int):
        if len(self.layers) <= 2:
            messagebox.showwarning("Atención", "El stack debe tener al menos medio incidente y sustrato.", parent=self.root)
            return
        del self.layers[idx]
        self.selected_layer_idx = min(idx, len(self.layers) - 1)
        self._refresh_layers_ui()

    def _move_layer(self, delta: int):
        if self.selected_layer_idx is not None:
            self._move_layer_direct(self.selected_layer_idx, delta)

    def _move_layer_direct(self, idx: int, delta: int):
        new_idx = idx + delta
        if 0 <= new_idx < len(self.layers):
            self.layers[idx], self.layers[new_idx] = self.layers[new_idx], self.layers[idx]
            self.selected_layer_idx = new_idx
            self._refresh_layers_ui()

    def _duplicate_layer(self):
        if self.selected_layer_idx is not None and 0 <= self.selected_layer_idx < len(self.layers):
            copied = copy.deepcopy(self.layers[self.selected_layer_idx])
            copied.name = f"{copied.name}_copia"
            self.layers.insert(self.selected_layer_idx + 1, copied)
            self.selected_layer_idx += 1
            self._refresh_layers_ui()

    def _apply_preset(self):
        preset_idx = self.cmb_preset.current()
        if preset_idx == 0:
            self._init_default_stack()
        elif preset_idx == 1:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", model_type="none", d_mode="inf"),
                LayerConfig(name="Al2O3_porosa", layer_type="c", material="Al2O3_porosa", model_type="none", d_mode="range", d_min=0.0, d_max=100.0),
                LayerConfig(name="TiO2_RTP_oxigeno", layer_type="c", material="TiO2_RTP_oxigeno", model_type="none", d_mode="range", d_min=0.0, d_max=100.0),
                LayerConfig(name="SiO2", layer_type="c", material="SiO2", model_type="none", d_mode="fixed", d_val=11.0),
                LayerConfig(name="Si", layer_type="i", material="Si", model_type="none", d_mode="inf")
            ]
        elif preset_idx == 2:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", model_type="none", d_mode="inf"),
                LayerConfig(name="MgF2", layer_type="c", material="MgF2", model_type="none", d_mode="range", d_min=0.0, d_max=150.0),
                LayerConfig(name="Si", layer_type="i", material="Si", model_type="none", d_mode="inf")
            ]
        elif preset_idx == 3:
            self.layers = [
                LayerConfig(name="air", layer_type="i", material="air", model_type="none", d_mode="inf"),
                LayerConfig(name="Si", layer_type="i", material="Si", model_type="none", d_mode="inf")
            ]
        self.selected_layer_idx = None
        self._refresh_layers_ui()
        self.lbl_status.config(text="Preset cargado exitosamente.")

    # ==========================================================================
    # PESTAÑA 2: AJUSTE ELIPSOMÉTRICO
    # ==========================================================================
    def _build_tab_elips(self):
        paned = ttk.PanedWindow(self.tab_elips, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=10)
        paned.add(left_frame, weight=1)

        file_frame = ttk.LabelFrame(left_frame, text="Datos Experimentales (.txt / .dat)", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.var_elips_file = tk.StringVar(value="")
        ent_file = ttk.Entry(file_frame, textvariable=self.var_elips_file, width=32)
        ent_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="Examinar...", command=self._browse_elips_file).pack(side=tk.RIGHT)

        default_data = os.path.join(SCRIPT_DIR, "../Files/8-10-26/Al2O3_nanotubos.N1.txt")
        if os.path.exists(default_data):
            self.var_elips_file.set(os.path.abspath(default_data))

        params_frame = ttk.LabelFrame(left_frame, text="Parámetros de Optimización Autorange", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        th0_def = str(getattr(ftmm, 'angulo_elipsometro', 69.5))
        grid_params = [
            ("Ángulo θ₀ [°]:", "ent_th0", th0_def),
            ("Num Starts (Semillas):", "ent_starts", "400"),
            ("Num Epochs (Épocas):", "ent_epochs", "150"),
            ("Learning Rate (lr):", "ent_lr", "1.0"),
            ("Loss_min:", "ent_chi2", "0.00065"),
            ("Intentos Autorange:", "ent_attempts", "10")
        ]

        self.elips_entries = {}
        for row_i, (label_txt, var_id, def_val) in enumerate(grid_params):
            ttk.Label(params_frame, text=label_txt).grid(row=row_i, column=0, sticky=tk.W, pady=3)
            ent = ttk.Entry(params_frame, width=14)
            ent.insert(0, def_val)
            ent.grid(row=row_i, column=1, sticky=tk.E, padx=5, pady=3)
            self.elips_entries[var_id] = ent

        self.var_use_cuda = tk.BooleanVar(value=torch.cuda.is_available())
        chk_cuda = ttk.Checkbutton(params_frame, text="Usar GPU CUDA", variable=self.var_use_cuda)
        if not torch.cuda.is_available():
            chk_cuda.config(state="disabled")
        chk_cuda.grid(row=len(grid_params), column=0, columnspan=2, sticky=tk.W, pady=5)

        out_frame = ttk.Frame(params_frame)
        out_frame.grid(row=len(grid_params)+1, column=0, columnspan=2, sticky=tk.EW, pady=5)
        ttk.Label(out_frame, text="Guardar salida en:").pack(side=tk.LEFT)
        self.var_elips_out = tk.StringVar(value="")
        ttk.Entry(out_frame, textvariable=self.var_elips_out, width=16).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(out_frame, text="...", width=3, command=self._browse_elips_out).pack(side=tk.RIGHT)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 5))

        self.btn_run_elips = ttk.Button(btn_frame, text="▶ Ejecutar Ajuste Elipsométrico", style="Accent.TButton", command=self._run_elips_fit)
        self.btn_run_elips.pack(fill=tk.X, pady=3)

        self.btn_cancel_elips = ttk.Button(btn_frame, text="⏹ Detener Ajuste", state="disabled", command=self._cancel_task)
        self.btn_cancel_elips.pack(fill=tk.X, pady=3)

        ttk.Separator(left_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        ttk.Button(left_frame, text="📂 Cargar Resultados Previos (.txt)", command=self._load_saved_elips_results).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="💾 Exportar Índice de Capa Seleccionada", command=self._export_fitted_index).pack(fill=tk.X, pady=3)

        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=3)

        self.plot_notebook = ttk.Notebook(right_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_plot_isic = ttk.Frame(self.plot_notebook)
        self.tab_plot_psid = ttk.Frame(self.plot_notebook)
        self.tab_plot_n = ttk.Frame(self.plot_notebook)

        self.plot_notebook.add(self.tab_plot_isic, text="Componentes Is / Ic")
        self.plot_notebook.add(self.tab_plot_psid, text="Ángulos Ψ / Δ")
        self.plot_notebook.add(self.tab_plot_n, text="Índice de Refracción n(λ)")

        self.fig_isic = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas_isic = FigureCanvasTkAgg(self.fig_isic, master=self.tab_plot_isic)
        self.canvas_isic.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_isic = NavigationToolbar2Tk(self.canvas_isic, self.tab_plot_isic)
        toolbar_isic.update()

        self.fig_psid = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas_psid = FigureCanvasTkAgg(self.fig_psid, master=self.tab_plot_psid)
        self.canvas_psid.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_psid = NavigationToolbar2Tk(self.canvas_psid, self.tab_plot_psid)
        toolbar_psid.update()

        self.fig_n = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas_n = FigureCanvasTkAgg(self.fig_n, master=self.tab_plot_n)
        self.canvas_n.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_n = NavigationToolbar2Tk(self.canvas_n, self.tab_plot_n)
        toolbar_n.update()

        summary_frame = ttk.LabelFrame(right_frame, text="Resultados del Ajuste", padding=8)
        summary_frame.pack(fill=tk.X, pady=(5, 0))

        self.lbl_elips_summary = ttk.Label(summary_frame, text="Aún no se ha ejecutado el ajuste.", font=("Segoe UI", 9))
        self.lbl_elips_summary.pack(anchor=tk.W)

    def _browse_elips_file(self):
        f = filedialog.askopenfilename(
            title="Seleccionar Datos Experimentales de Elipsometría",
            filetypes=[("Archivos de Texto (*.txt)", "*.txt"), ("Todos los archivos (*.*)", "*.*")],
            initialdir=os.path.join(SCRIPT_DIR, "../Files")
        )
        if f:
            self.var_elips_file.set(f)

    def _browse_elips_out(self):
        f = filedialog.asksaveasfilename(
            title="Guardar Resultados del Ajuste",
            defaultextension=".txt",
            filetypes=[("Archivos de Texto", "*.txt")],
            initialdir=os.path.join(SCRIPT_DIR, "../Files")
        )
        if f:
            self.var_elips_out.set(f)

    def _run_elips_fit(self):
        data_path = self.var_elips_file.get().strip()
        if not data_path or not os.path.exists(data_path):
            messagebox.showerror("Error", "Por favor selecciona un archivo experimental válido.", parent=self.root)
            return

        layer_names = [l.name for l in self.layers]
        layer_models = [l.get_layer_model() for l in self.layers]
        
        finite_layers = [l for l in self.layers if l.layer_type == "c"]
        if not finite_layers:
            messagebox.showerror("Error", "Debe existir al menos una capa delgada finita en el stack.", parent=self.root)
            return

        d_bounds = [l.get_d_spec() for l in finite_layers]
        n_max_limits = [l.get_n_limit_spec() for l in self.layers]

        try:
            th_0 = float(self.elips_entries["ent_th0"].get())
            num_starts = int(self.elips_entries["ent_starts"].get())
            num_epochs = int(self.elips_entries["ent_epochs"].get())
            lr = float(self.elips_entries["ent_lr"].get())
            chi2_min = float(self.elips_entries["ent_chi2"].get())
            max_attempts = int(self.elips_entries["ent_attempts"].get())
            use_cuda = self.var_use_cuda.get() and torch.cuda.is_available()
            salida_txt = self.var_elips_out.get().strip() or None
        except ValueError as e:
            messagebox.showerror("Error de Parámetro", f"Valor no numérico en los parámetros:\n{e}", parent=self.root)
            return

        self.btn_run_elips.config(state="disabled")
        self.btn_cancel_elips.config(state="normal")
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.start(10)
        self.lbl_status.config(text="Ejecutando ajuste elipsométrico con Autograd...")
        self.stop_requested = False

        old_stdout = sys.stdout
        sys.stdout = self.queue_writer

        def worker():
            try:
                self.log(f"\n=======================================================")
                self.log(f"Iniciando ajuste elipsométrico: {os.path.basename(data_path)}")
                self.log(f"Capas: {layer_names}")
                self.log(f"Espesores iniciales: {d_bounds}")
                self.log(f"Dispositivo: {'CUDA' if use_cuda else 'CPU'}")
                self.log(f"=======================================================\n")

                res = ftmm.ajuste_elipsometrico(
                    DATA_dir=data_path,
                    layer_names=layer_names,
                    layer_models=layer_models,
                    d_bounds=d_bounds,
                    n_max_limits=n_max_limits,
                    th_0=th_0,
                    salida_ajuste=salida_txt,
                    num_starts=num_starts,
                    num_epochs=num_epochs,
                    lr=lr,
                    chi2_min=chi2_min,
                    use_cuda=use_cuda,
                    max_attemps=max_attempts
                )

                wl_exp, Is_exp, Ic_exp, psi_exp, delta_exp, best_Ic, best_Is, best_thick, best_p, psi_fit, delta_fit = res

                self.root.after(0, lambda: self._on_elips_fit_finished(
                    wl_exp, Is_exp, Ic_exp, psi_exp, delta_exp, best_Ic, best_Is,
                    best_thick, best_p, psi_fit, delta_fit, layer_names, layer_models
                ))
            except Exception as e:
                err_msg = traceback.format_exc()
                self.log(f"\n[ERROR] Excepción durante el ajuste:\n{err_msg}")
                self.root.after(0, lambda: self._on_elips_fit_error(str(e)))
            finally:
                sys.stdout = old_stdout

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _cancel_task(self):
        self.stop_requested = True
        self.lbl_status.config(text="Cancelación solicitada...")
        self.log("\n[INFO] Solicitud de detención enviada.")

    def _on_elips_fit_error(self, err: str):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run_elips.config(state="normal")
        self.btn_cancel_elips.config(state="disabled")
        self.lbl_status.config(text="Error en la optimización.")
        messagebox.showerror("Error en el Ajuste", f"Ocurrió un error al ejecutar la optimización:\n{err}", parent=self.root)

    def _on_elips_fit_finished(self, wl_exp, Is_exp, Ic_exp, psi_exp, delta_exp, best_Ic, best_Is,
                               best_thick, best_p, psi_fit, delta_fit, layer_names, layer_models):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run_elips.config(state="normal")
        self.btn_cancel_elips.config(state="disabled")
        self.lbl_status.config(text="¡Ajuste elipsométrico completado con éxito!")

        self.last_elips_res = {
            'wl_exp': wl_exp, 'Is_exp': Is_exp, 'Ic_exp': Ic_exp,
            'psi_exp': psi_exp, 'delta_exp': delta_exp,
            'best_Ic': best_Ic, 'best_Is': best_Is,
            'best_thick': best_thick, 'best_params': best_p,
            'psi_fit': psi_fit, 'delta_fit': delta_fit,
            'layer_names': layer_names, 'layer_models': layer_models
        }

        self._plot_elips_results()

    def _plot_elips_results(self):
        if not self.last_elips_res:
            return

        res = self.last_elips_res
        wl = res['wl_exp']
        best_thick = res['best_thick']
        best_p = res['best_params']
        layer_names = res['layer_names']
        layer_models = res['layer_models']

        # 1. Gráfico Is / Ic
        self.fig_isic.clear()
        ax1 = self.fig_isic.add_subplot(2, 1, 1)
        ax2 = self.fig_isic.add_subplot(2, 1, 2)

        ax1.plot(wl, res['Is_exp'], 'b.', label='$I_s$ exp', markersize=3, alpha=0.5)
        ax1.plot(wl, res['best_Is'], 'b-', label='$I_s$ fit', linewidth=1.8)
        ax1.set_ylabel('$I_s$')
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        ax2.plot(wl, res['Ic_exp'], 'r.', label='$I_c$ exp', markersize=3, alpha=0.5)
        ax2.plot(wl, res['best_Ic'], 'r-', label='$I_c$ fit', linewidth=1.8)
        ax2.set_xlabel('Longitud de onda [nm]')
        ax2.set_ylabel('$I_c$')
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")
        self.fig_isic.tight_layout()
        self.canvas_isic.draw()

        # 2. Gráfico Psi / Delta
        self.fig_psid.clear()
        ax_psi = self.fig_psid.add_subplot(2, 1, 1)
        ax_del = self.fig_psid.add_subplot(2, 1, 2)

        ax_psi.plot(wl, res['psi_exp'], 'g.', label=r'$\Psi$ exp', markersize=3, alpha=0.5)
        ax_psi.plot(wl, res['psi_fit'], 'g-', label=r'$\Psi$ fit', linewidth=1.8)
        ax_psi.set_ylabel(r'$\Psi$ [°]')
        ax_psi.grid(True, linestyle="--", alpha=0.5)
        ax_psi.legend(loc="upper right")

        ax_del.plot(wl, res['delta_exp'], 'm.', label=r'$\Delta$ exp', markersize=3, alpha=0.5)
        ax_del.plot(wl, res['delta_fit'], 'm-', label=r'$\Delta$ fit', linewidth=1.8)
        ax_del.set_xlabel('Longitud de onda [nm]')
        ax_del.set_ylabel(r'$\Delta$ [°]')
        ax_del.grid(True, linestyle="--", alpha=0.5)
        ax_del.legend(loc="upper right")
        self.fig_psid.tight_layout()
        self.canvas_psid.draw()

        # 3. Gráfico n(λ)
        self.fig_n.clear()
        ax_n = self.fig_n.add_subplot(1, 1, 1)

        try:
            plotted_any = False
            try:
                n_list, d_list = ftmm.cargar_parametros_optimizados(best_p, best_thick, layer_names, layer_models)
                for i, name in enumerate(layer_names):
                    if i < len(layer_models) and layer_models[i] is not None:
                        d_val = d_list[i] if i < len(d_list) else 0.0
                        n_vals = n_list[i](wl)
                        ax_n.plot(wl, n_vals, linewidth=2, label=f"{name} (d = {d_val:.1f} nm)")
                        plotted_any = True
            except Exception:
                pass

            if not plotted_any and best_p:
                param_layers = [k for k, v in best_p.items() if isinstance(v, dict)]
                for i, name in enumerate(param_layers):
                    p_info = best_p[name]
                    m_type = str(p_info.get('model', '')).lower()
                    d_val = best_thick[i] if (best_thick is not None and i < len(best_thick)) else 0.0
                    if 'bruggeman' in m_type:
                        f_air = p_info.get('f_air', 0.5)
                        if isinstance(f_air, (list, tuple)):
                            f_air = f_air[0]
                        linked = p_info.get('linked_to', None)
                        if 'A' in p_info:
                            base_fn = cauchy_fn(float(p_info['A']), float(p_info.get('B', 0.0)), float(p_info.get('C', 0.0)))
                        elif linked and linked in best_p and 'A' in best_p[linked]:
                            base_fn = cauchy_fn(float(best_p[linked]['A']), float(best_p[linked].get('B', 0.0)), float(best_p[linked].get('C', 0.0)))
                        elif linked and linked in ftmm.materials:
                            base_fn = ftmm.materials[linked][0]
                        elif name in ftmm.materials:
                            base_fn = ftmm.materials[name][0]
                        else:
                            base_fn = constant_fn(1.5)
                        n_fn = brugg_fn(base_fn, constant_fn(1.0), float(f_air))
                        ax_n.plot(wl, n_fn(wl), linewidth=2, label=f"{name} (d = {d_val:.1f} nm, f_air = {float(f_air):.3f})")
                        if 'A' in p_info:
                            base_label = f"{linked or f'{name}_denso'} (base ajustada)"
                            ax_n.plot(wl, base_fn(wl), linewidth=1.5, linestyle="--", label=base_label)
                    elif 'cauchy' in m_type or 'A' in p_info:
                        n_fn = cauchy_fn(float(p_info.get('A', 1.5)), float(p_info.get('B', 0.0)), float(p_info.get('C', 0.0)))
                        ax_n.plot(wl, n_fn(wl), linewidth=2, label=f"{name} (d = {d_val:.1f} nm)")

            ax_n.set_xlabel('Longitud de onda [nm]')
            ax_n.set_ylabel('Índice de refracción n')
            ax_n.set_title('Índices de refracción optimizados')
            ax_n.grid(True, linestyle="--", alpha=0.5)
            handles, labels = ax_n.get_legend_handles_labels()
            if handles:
                ax_n.legend()
        except Exception as e:
            ax_n.text(0.5, 0.5, f"No se pudo graficar índice: {e}", ha='center')
        self.fig_n.tight_layout()
        self.canvas_n.draw()

        thick_str = ", ".join([f"{t:.2f} nm" for t in best_thick]) if best_thick is not None else "--"
        chi2_val = best_p.get('chi2_min', 'N/A')
        summary_txt = f"Espesores Óptimos: [{thick_str}] | Error Mínimo (MSE/Chi²): {chi2_val}\nParámetros: {best_p}"
        self.lbl_elips_summary.config(text=summary_txt)

    @staticmethod
    def _parse_fit_results_layers(file_path: str) -> Dict[str, Dict[str, Any]]:
        """Lee un archivo de resultados .txt y extrae con alta robustez las capas y sus parámetros."""
        layers_info = {}
        if not file_path or not os.path.exists(file_path):
            return layers_info
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_clean = line.strip()
                    if not line_clean.startswith('#'):
                        if not line_clean:
                            continue
                        break
                    if 'Capa' in line_clean and ':' in line_clean:
                        m = re.search(r"Capa\s+['\"]?([a-zA-Z0-9_\-\+\.\s]+?)['\"]?\s*:\s*(\{.*\})", line_clean)
                        if m:
                            name = m.group(1).strip()
                            dict_str = m.group(2).strip()
                            clean_dict_str = re.sub(r'(?:np\.)?float\d*\(|\btensor\(', '', dict_str).replace(')', '')
                            try:
                                p_dict = eval(dict_str, {
                                    "__builtins__": None, "True": True, "False": False,
                                    "None": None, "np": np, "float64": np.float64, "float": float, "int": int
                                })
                            except Exception:
                                try:
                                    p_dict = ast.literal_eval(clean_dict_str)
                                except Exception:
                                    p_dict = {}
                                    for kv in re.finditer(r"['\"]?(\w+)['\"]?\s*:\s*([^,\}]+)", dict_str):
                                        k_val, v_val = kv.group(1).strip(), kv.group(2).strip().strip("'\"")
                                        try:
                                            p_dict[k_val] = float(v_val)
                                        except ValueError:
                                            p_dict[k_val] = True if v_val.lower() == 'true' else (False if v_val.lower() == 'false' else v_val)
                            if p_dict:
                                layers_info[name] = p_dict
        except Exception:
            pass
        return layers_info

    def _load_saved_elips_results(self):
        f = filedialog.askopenfilename(
            title="Cargar Archivo de Resultados Elipsométricos",
            filetypes=[("Archivos de Texto (*.txt)", "*.txt"), ("Todos los archivos (*.*)", "*.*")],
            initialdir=os.path.join(SCRIPT_DIR, "../Files")
        )
        if not f:
            return

        try:
            res = leer_datos_guardados(f)
            data_file, wl_exp, Is_exp, best_Is, Ic_exp, best_Ic, psi_exp, psi_fit, delta_exp, delta_fit, best_thick, best_p = res

            # Parsear directamente las capas y modelos ópticos desde el archivo .txt
            file_layers_info = self._parse_fit_results_layers(f)
            if not file_layers_info and best_p:
                file_layers_info = {
                    k: v for k, v in best_p.items()
                    if isinstance(v, dict) and ('model' in v or 'A' in v or 'f_air' in v)
                }

            file_layer_names = list(file_layers_info.keys())
            file_layer_models = [{'model': file_layers_info[k].get('model', 'cauchy')} for k in file_layer_names]

            self.last_elips_res = {
                'source': 'file',
                'file_path': os.path.abspath(f),
                'data_file': data_file,
                'wl_exp': wl_exp, 'Is_exp': Is_exp, 'Ic_exp': Ic_exp,
                'psi_exp': psi_exp, 'delta_exp': delta_exp,
                'best_Ic': best_Ic, 'best_Is': best_Is,
                'best_thick': best_thick, 'best_params': best_p,
                'file_layers_info': file_layers_info,
                'psi_fit': psi_fit, 'delta_fit': delta_fit,
                'layer_names': file_layer_names,
                'layer_models': file_layer_models
            }

            self._plot_elips_results()
            layers_desc = f" ({', '.join(file_layer_names)})" if file_layer_names else ""
            self.lbl_status.config(text=f"Resultados cargados desde {os.path.basename(f)}{layers_desc}")
            messagebox.showinfo("Carga Exitosa", f"Se cargaron los resultados de ajuste desde:\n{f}\n\nCapas identificadas en el archivo: {file_layer_names}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo interpretar el archivo seleccionado:\n{e}", parent=self.root)

    def _export_fitted_index(self):
        if not self.last_elips_res:
            messagebox.showwarning("Atención", "Primero debes ejecutar o cargar un ajuste elipsométrico.", parent=self.root)
            return

        res = self.last_elips_res
        wl = res['wl_exp']
        best_p = res.get('best_params', {})

        # ======================================================================
        # CONDICIÓN REQUERIDA (IF):
        # En caso de que se hayan cargado los datos preajustados de algún stack,
        # identificar a partir del archivo los nombres de los materiales de las
        # capas, usar sus parámetros para calcular los índices de refracción
        # y exportar los mismos en archivo .txt.
        # ======================================================================
        if res.get('source') == 'file' or bool(res.get('file_path')):
            file_path = res.get('file_path', '')
            file_layers_info = res.get('file_layers_info')
            if not file_layers_info and file_path and os.path.exists(file_path):
                file_layers_info = self._parse_fit_results_layers(file_path)

            if not file_layers_info and best_p:
                file_layers_info = {
                    k: v for k, v in best_p.items()
                    if isinstance(v, dict) and ('model' in v or 'A' in v or 'f_air' in v)
                }

            if not file_layers_info:
                messagebox.showerror(
                    "Error",
                    f"No se pudieron identificar las capas en el archivo de datos cargado:\n{file_path}",
                    parent=self.root
                )
                return

            material_names = list(file_layers_info.keys())
            all_layer_params = file_layers_info
            dialog_title = f"Exportar Índice - Archivo: {os.path.basename(file_path)}"
            dialog_prompt = f"Capas identificadas del archivo '{os.path.basename(file_path)}':"

        else:
            # Caso en que se ejecutó el ajuste en vivo desde el stack óptico actual
            material_names = []
            for l in self.layers:
                if l.layer_type != 'i' and l.model_type != 'none':
                    material_names.append(l.name)
                    # Si la capa bruggeman ajustó un material base denso interno, permitir exportarlo también
                    p_info = best_p.get(l.name, {}) if best_p else {}
                    if 'A' in p_info:
                        base_mat_name = p_info.get('linked_to') or f"{l.name}_base"
                        if base_mat_name not in material_names:
                            material_names.append(base_mat_name)

            if not material_names:
                material_names = [k for k, v in best_p.items() if isinstance(v, dict) and ('model' in v or 'A' in v or 'f_air' in v)]

            if not material_names:
                messagebox.showwarning("Atención", "No hay capas con modelo óptico ajustado para exportar.", parent=self.root)
                return

            all_layer_params = best_p
            dialog_title = "Exportar Índice de Refracción"
            dialog_prompt = "Selecciona la capa del stack a exportar:"

        top = tk.Toplevel(self.root)
        top.title(dialog_title)
        top.geometry("450x230")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        ttk.Label(top, text=dialog_prompt, font=("Segoe UI", 10, "bold")).pack(pady=(12, 6))

        cmb = ttk.Combobox(top, values=material_names, state="readonly", width=32, font=("Segoe UI", 9))
        cmb.current(0)
        cmb.pack(pady=6)

        def calculate_n(mat_name):
            # Verificar si mat_name es la base ajustada de alguna capa Bruggeman
            for layer_k, p_k in all_layer_params.items():
                if isinstance(p_k, dict) and 'bruggeman' in str(p_k.get('model', '')).lower():
                    if p_k.get('linked_to') == mat_name and 'A' in p_k:
                        base_fn = cauchy_fn(float(p_k['A']), float(p_k.get('B', 0.0)), float(p_k.get('C', 0.0)))
                        return base_fn, base_fn(wl)

            p_info = all_layer_params.get(mat_name, {})
            m_type = str(p_info.get('model', '')).lower()

            if 'bruggeman' in m_type or 'f_air' in p_info:
                f_air = p_info.get('f_air', 0.5)
                if isinstance(f_air, (list, tuple)):
                    f_air = f_air[0]
                linked = p_info.get('linked_to')
                if 'A' in p_info:
                    base_fn = cauchy_fn(float(p_info['A']), float(p_info.get('B', 0.0)), float(p_info.get('C', 0.0)))
                elif linked and linked in all_layer_params and 'A' in all_layer_params[linked]:
                    lp = all_layer_params[linked]
                    base_fn = cauchy_fn(float(lp['A']), float(lp.get('B', 0.0)), float(lp.get('C', 0.0)))
                elif linked and linked in ftmm.materials:
                    base_fn = ftmm.materials[linked][0]
                elif mat_name in ftmm.materials:
                    base_fn = ftmm.materials[mat_name][0]
                else:
                    base_fn = constant_fn(1.5)
                n_fn = brugg_fn(base_fn, constant_fn(1.0), float(f_air))
            elif 'cauchy' in m_type or 'A' in p_info:
                A = float(p_info.get('A', 1.5))
                B = float(p_info.get('B', 0.0))
                C = float(p_info.get('C', 0.0))
                n_fn = cauchy_fn(A, B, C)
            else:
                sub_names = list(material_names)
                sub_models = [{'model': all_layer_params.get(k, {}).get('model', 'cauchy')} for k in sub_names]
                n_list, _ = ftmm.cargar_parametros_optimizados(all_layer_params, res.get('best_thick'), sub_names, sub_models)
                pos = sub_names.index(mat_name)
                n_fn = n_list[pos]

            n_calc = n_fn(wl)
            return n_fn, n_calc

        def do_export_single():
            selected_mat = cmb.get()
            out_path = filedialog.asksaveasfilename(
                title=f"Guardar Índice de Refracción - {selected_mat}",
                defaultextension=".txt",
                initialfile=f"{selected_mat}_indice.txt",
                filetypes=[("Archivos de Texto (*.txt)", "*.txt"), ("Todos los archivos (*.*)", "*.*")],
                parent=top
            )
            if not out_path:
                return

            try:
                n_fn, n_calc = calculate_n(selected_mat)
                guardar_indice_refraccion(wl, n_calc, selected_mat, salida_path=out_path)
                ftmm.materials[selected_mat] = (n_fn, constant_fn(0.0))
                self.custom_material_ranges[selected_mat] = (float(np.min(wl)), float(np.max(wl)))
                self.log(f"[INFO] Material '{selected_mat}' exportado a {out_path} y registrado en la biblioteca.")

                self._refresh_materials_listbox()
                self._refresh_layers_ui()
                top.destroy()
                messagebox.showinfo("Exportado", f"Índice de '{selected_mat}' guardado exitosamente en:\n{out_path}", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar el índice de '{selected_mat}':\n{e}", parent=top)

        def do_export_all():
            out_dir = filedialog.askdirectory(
                title="Seleccionar Carpeta para Exportar Todos los Índices",
                initialdir=os.path.join(SCRIPT_DIR, "../Files"),
                parent=top
            )
            if not out_dir:
                return

            saved_files = []
            try:
                for mat_name in material_names:
                    n_fn, n_calc = calculate_n(mat_name)
                    out_path = os.path.join(out_dir, f"{mat_name}_indice.txt")
                    guardar_indice_refraccion(wl, n_calc, mat_name, salida_path=out_path)
                    ftmm.materials[mat_name] = (n_fn, constant_fn(0.0))
                    self.custom_material_ranges[mat_name] = (float(np.min(wl)), float(np.max(wl)))
                    saved_files.append(out_path)
                    self.log(f"[INFO] Material '{mat_name}' exportado a {out_path} y registrado en la biblioteca.")

                self._refresh_materials_listbox()
                self._refresh_layers_ui()
                top.destroy()
                msg = f"Se exportaron {len(saved_files)} índices exitosamente en:\n{out_dir}\n\nArchivos:\n" + "\n".join([os.path.basename(p) for p in saved_files])
                messagebox.showinfo("Exportación Completa", msg, parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Ocurrió un error al exportar:\n{e}", parent=top)

        btn_box = ttk.Frame(top)
        btn_box.pack(pady=15)
        ttk.Button(btn_box, text="💾 Exportar Capa Seleccionada (.txt)", style="Accent.TButton", command=do_export_single).pack(side=tk.LEFT, padx=5)
        if len(material_names) > 1:
            ttk.Button(btn_box, text="📦 Exportar Todas las Capas", command=do_export_all).pack(side=tk.LEFT, padx=5)

    # ==========================================================================
    # PESTAÑA 3: OPTIMIZACIÓN DE ESPESORES Y REFLECTANCIA (CELDA SOLAR / Jsc)
    # ==========================================================================
    def _build_tab_refl(self):
        paned = ttk.PanedWindow(self.tab_refl, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=10)
        paned.add(left_frame, weight=1)

        cfg_frame = ttk.LabelFrame(left_frame, text="Condiciones de Iluminación y Espectro", padding=10)
        cfg_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(cfg_frame, text="Espectro Solar:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cmb_spectrum = ttk.Combobox(cfg_frame, values=["AM0 (Espacio)", "AM1.5G (Terrestre)", "Plano (Sin Ponderar)"], state="readonly", width=18)
        self.cmb_spectrum.current(0)
        self.cmb_spectrum.grid(row=0, column=1, sticky=tk.E, padx=5, pady=3)

        ttk.Label(cfg_frame, text="Polarización:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.cmb_pol = ttk.Combobox(cfg_frame, values=["s", "p"], state="readonly", width=18)
        self.cmb_pol.current(0)
        self.cmb_pol.grid(row=1, column=1, sticky=tk.E, padx=5, pady=3)

        ttk.Label(cfg_frame, text="Ángulo de Incidencia θ₀ [°]:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.ent_refl_th0 = ttk.Entry(cfg_frame, width=12)
        self.ent_refl_th0.insert(0, "0.0")
        self.ent_refl_th0.grid(row=2, column=1, sticky=tk.E, padx=5, pady=3)

        opt_box = ttk.LabelFrame(left_frame, text="Parámetros de Optimización", padding=10)
        opt_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(opt_box, text="Num Starts (Semillas):").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.ent_refl_starts = ttk.Entry(opt_box, width=12)
        self.ent_refl_starts.insert(0, "1000")
        self.ent_refl_starts.grid(row=0, column=1, sticky=tk.E, padx=5, pady=3)

        ttk.Label(opt_box, text="Num Epochs:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.ent_refl_epochs = ttk.Entry(opt_box, width=12)
        self.ent_refl_epochs.insert(0, "150")
        self.ent_refl_epochs.grid(row=1, column=1, sticky=tk.E, padx=5, pady=3)

        ttk.Label(opt_box, text="Learning Rate (lr):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.ent_refl_lr = ttk.Entry(opt_box, width=12)
        self.ent_refl_lr.insert(0, "1.0")
        self.ent_refl_lr.grid(row=2, column=1, sticky=tk.E, padx=5, pady=3)

        self.btn_run_refl = ttk.Button(left_frame, text="⚡ Calcular / Optimizar Reflectancia", style="Accent.TButton", command=self._run_refl_calc)
        self.btn_run_refl.pack(fill=tk.X, pady=4)

        ttk.Separator(left_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        curv_box = ttk.LabelFrame(left_frame, text="Curvas de Reflectancia Guardadas", padding=10)
        curv_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(curv_box, text="💾 Guardar Curva R(λ) Calculada", command=self._save_refl_curve).pack(fill=tk.X, pady=3)
        ttk.Button(curv_box, text="📂 Cargar y Superponer Curva R(λ)", command=self._load_compare_refl_curve).pack(fill=tk.X, pady=3)
        ttk.Button(curv_box, text="🧹 Limpiar Curvas Superpuestas", command=self._clear_comparison_curves).pack(fill=tk.X, pady=3)

        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=3)

        stats_frame = ttk.Frame(right_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.card_jsc = tk.Label(stats_frame, text="Jsc: -- mA/cm²", font=("Segoe UI", 12, "bold"), bg="#e8f5e9", fg="#2e7d32", padx=15, pady=8, relief=tk.GROOVE)
        self.card_jsc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.card_jmax = tk.Label(stats_frame, text="Jmax: -- mA/cm²", font=("Segoe UI", 11), bg="#f5f5f5", fg="#555", padx=15, pady=8, relief=tk.GROOVE)
        self.card_jmax.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.card_thick = tk.Label(stats_frame, text="Espesores: --", font=("Segoe UI", 11), bg="#f5f5f5", fg="#555", padx=15, pady=8, relief=tk.GROOVE)
        self.card_thick.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.fig_refl = Figure(figsize=(7, 5), dpi=100)
        self.canvas_refl = FigureCanvasTkAgg(self.fig_refl, master=right_frame)
        self.canvas_refl.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_refl = NavigationToolbar2Tk(self.canvas_refl, right_frame)
        toolbar_refl.update()

    def _run_refl_calc(self):
        stack = []
        for l in self.layers:
            spec = l.get_d_spec()
            stack.append([spec, l.material, l.layer_type])

        for l in self.layers:
            if l.material not in ftmm.materials:
                messagebox.showerror("Material Faltante", f"El material '{l.material}' no está definido en el diccionario.", parent=self.root)
                return

        spec_choice = self.cmb_spectrum.get()
        if "AM0" in spec_choice:
            weights = ftmm.weight_am0
        elif "AM1.5" in spec_choice:
            weights = ftmm.weight_am15
        else:
            weights = np.ones_like(ftmm.lams)

        pol = self.cmb_pol.get()
        try:
            th_0 = float(self.ent_refl_th0.get())
            num_starts = int(self.ent_refl_starts.get())
            num_epochs = int(self.ent_refl_epochs.get())
            lr = float(self.ent_refl_lr.get())
        except ValueError as e:
            messagebox.showerror("Error de Parámetro", f"Valor no numérico:\n{e}", parent=self.root)
            return

        self.btn_run_refl.config(state="disabled")
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.start(10)
        self.lbl_status.config(text="Calculando reflectancia / optimizando espesores...")

        old_stdout = sys.stdout
        sys.stdout = self.queue_writer

        def worker():
            try:
                self.log(f"\n[INFO] Calculando reflectancia para el stack: {[l.name for l in self.layers]}")
                use_cuda = torch.cuda.is_available()

                thick, R_curve = ftmm.optimizar_espesores(
                    layer_names=[l.name for l in self.layers],
                    stack=stack,
                    weights=weights,
                    th_0=th_0,
                    num_starts=num_starts,
                    num_epochs=num_epochs,
                    lr=lr,
                    use_cuda=use_cuda
                )

                jsc = ftmm.calcular_jsc(R_curve, weights=weights)
                jmax = np.trapezoid(weights, dx=ftmm.step)

                self.root.after(0, lambda: self._on_refl_finished(thick, R_curve, jsc, jmax))
            except Exception as e:
                err_msg = traceback.format_exc()
                self.log(f"\n[ERROR] Excepción en cálculo de reflectancia:\n{err_msg}")
                self.root.after(0, lambda: self._on_refl_error(str(e)))
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=worker, daemon=True).start()

    def _on_refl_error(self, err: str):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run_refl.config(state="normal")
        self.lbl_status.config(text="Error en el cálculo.")
        messagebox.showerror("Error", f"Ocurrió un error en el cálculo de reflectancia:\n{err}", parent=self.root)

    def _on_refl_finished(self, thick, R_curve, jsc, jmax):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run_refl.config(state="normal")
        self.lbl_status.config(text="¡Reflectancia calculada con éxito!")

        self.last_rt_res = {
            'thick': thick,
            'R_curve': R_curve,
            'jsc': jsc,
            'jmax': jmax,
            'lams': ftmm.lams
        }

        self.card_jsc.config(text=f"Jsc: {jsc:.2f} mA/cm²")
        self.card_jmax.config(text=f"Jmax: {jmax:.2f} mA/cm²")
        if isinstance(thick, (list, tuple, np.ndarray)):
            thick_str = ", ".join([f"{t:.1f} nm" for t in thick])
        else:
            thick_str = f"{thick} nm"
        self.card_thick.config(text=f"Espesores: [{thick_str}]")

        self._plot_reflectance()

    def _plot_reflectance(self):
        self.fig_refl.clear()
        ax = self.fig_refl.add_subplot(1, 1, 1)

        if self.last_rt_res:
            lams = self.last_rt_res['lams']
            R = self.last_rt_res['R_curve'] * 100.0
            jsc_val = self.last_rt_res['jsc']
            ax.plot(lams, R, 'b-', linewidth=2.2, label=f"Stack Actual (Jsc = {jsc_val:.2f} mA/cm²)")

        for label, c_lams, c_R in self.comparison_curves:
            c_R_perc = c_R * 100.0 if np.max(c_R) <= 1.0 else c_R
            ax.plot(c_lams, c_R_perc, linestyle="--", linewidth=1.6, label=label)

        ax.set_xlabel("Longitud de Onda λ [nm]", fontsize=10)
        ax.set_ylabel("Reflectancia R(λ) [%]", fontsize=10)
        ax.set_title("Curva de Reflectancia del Stack Óptico", fontsize=11, fontweight="bold")
        ax.set_ylim(-1, 101)
        ax.grid(True, linestyle="--", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", framealpha=0.9)
        self.fig_refl.tight_layout()
        self.canvas_refl.draw()

    def _save_refl_curve(self):
        if not self.last_rt_res:
            messagebox.showwarning("Atención", "No hay ninguna curva calculada para guardar.", parent=self.root)
            return

        out_path = filedialog.asksaveasfilename(
            title="Guardar Curva de Reflectancia",
            defaultextension=".txt",
            initialfile="R_curve_calculada.txt",
            filetypes=[("Archivos de Texto", "*.txt")],
            initialdir=os.path.join(SCRIPT_DIR, "../Files/reflectancias_optimas")
        )
        if not out_path:
            return

        try:
            ftmm.guardar_curva_R(
                lams=self.last_rt_res['lams'],
                R_curve=self.last_rt_res['R_curve'],
                path=out_path,
                thickness=self.last_rt_res['thick'],
                jsc=self.last_rt_res['jsc']
            )
            messagebox.showinfo("Guardado", f"Curva de reflectancia guardada en:\n{out_path}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la curva:\n{e}", parent=self.root)

    def _load_compare_refl_curve(self):
        f = filedialog.askopenfilename(
            title="Cargar Curva R(λ) para Comparar",
            filetypes=[("Archivos de Texto", "*.txt"), ("Todos los archivos", "*.*")],
            initialdir=os.path.join(SCRIPT_DIR, "../Files/reflectancias_optimas")
        )
        if not f:
            return

        try:
            c_lams, c_R = ftmm.leer_curva_R(f)
            label = os.path.basename(f).replace(".txt", "")
            self.comparison_curves.append((label, c_lams, c_R))
            self._plot_reflectance()
            self.lbl_status.config(text=f"Curva '{label}' agregada al gráfico.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo leer la curva de reflectancia:\n{e}", parent=self.root)

    def _clear_comparison_curves(self):
        self.comparison_curves.clear()
        self._plot_reflectance()
        self.lbl_status.config(text="Curvas superpuestas eliminadas.")

    # ==========================================================================
    # PESTAÑA 4: COMPARACIÓN DE STACKS Y MEDICIONES EXPERIMENTALES
    # ==========================================================================
    def _build_tab_comp(self):
        paned = ttk.PanedWindow(self.tab_comp, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=10)
        paned.add(left_frame, weight=1)

        # 1. Modo de Comparación
        mode_box = ttk.LabelFrame(left_frame, text="1. Modo de Comparación", padding=10)
        mode_box.pack(fill=tk.X, pady=(0, 8))

        self.var_comp_mode = tk.StringVar(value="exp")
        r_mode_exp = ttk.Radiobutton(mode_box, text="Stack 1 vs Mediciones Experimentales (DATA)", variable=self.var_comp_mode, value="exp", command=self._update_comp_ui_state)
        r_mode_exp.pack(anchor=tk.W, pady=2)
        r_mode_stk = ttk.Radiobutton(mode_box, text="Stack 1 vs Stack 2 (Simulación TMM)", variable=self.var_comp_mode, value="stacks", command=self._update_comp_ui_state)
        r_mode_stk.pack(anchor=tk.W, pady=2)

        # 2. Parámetro a Comparar
        param_box = ttk.LabelFrame(left_frame, text="2. Parámetro Óptico", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 8))

        self.var_comp_param = tk.StringVar(value="R")
        r_par_r = ttk.Radiobutton(param_box, text="Reflectancia R(λ) [Espectrofotómetro / TMM]", variable=self.var_comp_param, value="R", command=self._update_comp_ui_state)
        r_par_r.pack(anchor=tk.W, pady=2)
        r_par_is = ttk.Radiobutton(param_box, text="Elipsometría: Componentes Is / Ic", variable=self.var_comp_param, value="Is", command=self._update_comp_ui_state)
        r_par_is.pack(anchor=tk.W, pady=2)

        # 3. Stack 1 (Principal)
        st1_box = ttk.LabelFrame(left_frame, text="3. Stack 1 (Principal)", padding=10)
        st1_box.pack(fill=tk.X, pady=(0, 8))

        self.lbl_comp_st1 = ttk.Label(st1_box, text=format_stack_summary(self.layers), wraplength=320, font=("Segoe UI", 9, "bold"))
        self.lbl_comp_st1.pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(st1_box, text="🔄 Sincronizar con Pestaña 1", command=self._sync_comp_stack1).pack(fill=tk.X)

        # 4. Stack 2 (Secundario)
        self.st2_box = ttk.LabelFrame(left_frame, text="4. Stack 2 (Secundario)", padding=10)
        self.st2_box.pack(fill=tk.X, pady=(0, 8))

        self.lbl_comp_st2 = ttk.Label(self.st2_box, text=format_stack_summary(self.layers_stack2), wraplength=320, font=("Segoe UI", 9))
        self.lbl_comp_st2.pack(anchor=tk.W, pady=(0, 5))

        st2_btn_frame = ttk.Frame(self.st2_box)
        st2_btn_frame.pack(fill=tk.X)
        ttk.Button(st2_btn_frame, text="✏️ Editar Stack 2...", command=self._open_edit_stack2_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(st2_btn_frame, text="📋 Copiar Stack 1", command=self._copy_stack1_to_stack2).pack(side=tk.LEFT, padx=2)
        ttk.Button(st2_btn_frame, text="⇄ Invertir", command=self._swap_stacks).pack(side=tk.LEFT, padx=2)

        # 5. Archivo de Datos Experimentales (DATA)
        self.data_box = ttk.LabelFrame(left_frame, text="5. Datos Experimentales (DATA)", padding=10)
        self.data_box.pack(fill=tk.X, pady=(0, 8))

        self.var_comp_data_file = tk.StringVar(value="")
        # Archivo por defecto si existe
        def_r_path = os.path.join(SCRIPT_DIR, "../Files/8-10-26/reflectancias-8-10-26/3TIO2N1.TXT")
        if os.path.exists(def_r_path):
            self.var_comp_data_file.set(os.path.abspath(def_r_path))

        ent_dfile = ttk.Entry(self.data_box, textvariable=self.var_comp_data_file, width=28)
        ent_dfile.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(self.data_box, text="Examinar...", command=self._browse_comp_data).pack(side=tk.RIGHT)

        # 6. Ángulos de Incidencia
        ang_box = ttk.LabelFrame(left_frame, text="6. Ángulos de Incidencia [°]", padding=10)
        ang_box.pack(fill=tk.X, pady=(0, 8))

        th_el_def = str(getattr(ftmm, 'angulo_elipsometro', 69.5))
        th_cp_def = str(getattr(ftmm, 'angulo_comparacion', 0.0))

        row_a1 = ttk.Frame(ang_box)
        row_a1.pack(fill=tk.X, pady=2)
        ttk.Label(row_a1, text="Ángulo Elipsómetro θ₀:").pack(side=tk.LEFT)
        self.ent_comp_th_elips = ttk.Entry(row_a1, width=10)
        self.ent_comp_th_elips.insert(0, th_el_def)
        self.ent_comp_th_elips.pack(side=tk.RIGHT)

        row_a2 = ttk.Frame(ang_box)
        row_a2.pack(fill=tk.X, pady=2)
        ttk.Label(row_a2, text="Ángulo Comparación θ₀:").pack(side=tk.LEFT)
        self.ent_comp_th_comp = ttk.Entry(row_a2, width=10)
        self.ent_comp_th_comp.insert(0, th_cp_def)
        self.ent_comp_th_comp.pack(side=tk.RIGHT)

        # 7. Botones de Acción
        btn_action_frame = ttk.Frame(left_frame)
        btn_action_frame.pack(fill=tk.X, pady=(10, 0))

        self.btn_run_comp = ttk.Button(
            btn_action_frame, text="▶ Comparar en Gráfico Integrado", style="Accent.TButton",
            command=self._run_comparison_integrated
        )
        self.btn_run_comp.pack(fill=tk.X, pady=3)

        ttk.Button(
            btn_action_frame, text="🪟 Abrir en Ventana de Matplotlib (plt.show)",
            command=self._run_comparison_external
        ).pack(fill=tk.X, pady=3)

        # Panel Derecho: Gráfico Matplotlib Embebido
        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=3)

        self.fig_comp = Figure(figsize=(7, 5), dpi=100)
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=right_frame)
        self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_comp = NavigationToolbar2Tk(self.canvas_comp, right_frame)
        self.toolbar_comp.update()

        summary_comp = ttk.LabelFrame(right_frame, text="Detalles de Comparación", padding=8)
        summary_comp.pack(fill=tk.X, pady=(5, 0))

        self.lbl_comp_summary = ttk.Label(summary_comp, text="Selecciona el modo y ejecuta la comparación para visualizar los resultados.", font=("Segoe UI", 9))
        self.lbl_comp_summary.pack(anchor=tk.W)

        self._update_comp_ui_state()

    def _update_comp_ui_state(self):
        """Habilita o deshabilita paneles según el modo seleccionado."""
        mode = self.var_comp_mode.get()
        param = self.var_comp_param.get()

        if mode == "exp":
            self.data_box.pack(fill=tk.X, pady=(0, 8))
            self.st2_box.pack_forget()
            # Sugerir ruta según parámetro si está vacío o apuntando a otro
            cur = self.var_comp_data_file.get().strip()
            if param == "R":
                def_r = os.path.join(SCRIPT_DIR, "../Files/8-10-26/reflectancias-8-10-26/3TIO2N1.TXT")
                if (not cur or "Al2O3" in cur) and os.path.exists(def_r):
                    self.var_comp_data_file.set(os.path.abspath(def_r))
            else:
                def_is = os.path.join(SCRIPT_DIR, "../Files/8-10-26/Al2O3_nanotubos.N1.txt")
                if (not cur or "reflectancias" in cur) and os.path.exists(def_is):
                    self.var_comp_data_file.set(os.path.abspath(def_is))
        else:
            self.data_box.pack_forget()
            self.st2_box.pack(fill=tk.X, pady=(0, 8))

    def _update_comp_stack_labels(self):
        if hasattr(self, 'lbl_comp_st1') and self.lbl_comp_st1.winfo_exists():
            self.lbl_comp_st1.config(text=format_stack_summary(self.layers))
        if hasattr(self, 'lbl_comp_st2') and self.lbl_comp_st2.winfo_exists():
            self.lbl_comp_st2.config(text=format_stack_summary(self.layers_stack2))

    def _sync_comp_stack1(self):
        self._update_comp_stack_labels()
        self.lbl_status.config(text="Stack 1 sincronizado con la configuración de la Pestaña 1.")

    def _open_edit_stack2_dialog(self):
        dlg = EditStack2Dialog(self.root, self.layers_stack2, list(ftmm.materials.keys()))
        self.root.wait_window(dlg)
        if dlg.result:
            self.layers_stack2 = dlg.layers
            self._update_comp_stack_labels()
            self.lbl_status.config(text="Stack 2 actualizado.")

    def _copy_stack1_to_stack2(self):
        self.layers_stack2 = copy.deepcopy(self.layers)
        self._update_comp_stack_labels()
        self.lbl_status.config(text="Stack 1 copiado a Stack 2.")

    def _swap_stacks(self):
        self.layers, self.layers_stack2 = self.layers_stack2, self.layers
        self._refresh_layers_ui()
        self._update_comp_stack_labels()
        self.lbl_status.config(text="Stacks intercambiados (Stack 1 ⇄ Stack 2).")

    def _browse_comp_data(self):
        param = self.var_comp_param.get()
        title = "Seleccionar Archivo de Reflectancia" if param == "R" else "Seleccionar Archivo de Elipsometría"
        init_dir = os.path.join(SCRIPT_DIR, "../Files/8-10-26/reflectancias-8-10-26") if param == "R" else os.path.join(SCRIPT_DIR, "../Files")
        f = filedialog.askopenfilename(
            title=title,
            filetypes=[("Archivos de Texto", "*.txt *.dat *.TXT"), ("Todos los archivos", "*.*")],
            initialdir=init_dir
        )
        if f:
            self.var_comp_data_file.set(f)

    def _run_comparison_integrated(self):
        mode = self.var_comp_mode.get()
        param = self.var_comp_param.get()

        try:
            th_elips = float(self.ent_comp_th_elips.get())
            th_comp = float(self.ent_comp_th_comp.get())
            ftmm.angulo_elipsometro = th_elips
            ftmm.angulo_comparacion = th_comp
        except ValueError:
            messagebox.showerror("Error", "Los ángulos de incidencia deben ser valores numéricos.", parent=self.root)
            return

        st1 = layers_to_tmm_stack(self.layers)

        self.btn_run_comp.config(state="disabled")
        self.lbl_status.config(text="Ejecutando comparación...")

        try:
            if mode == "exp":
                data_file = self.var_comp_data_file.get().strip()
                if not data_file or not os.path.exists(data_file):
                    messagebox.showerror("Error", "Selecciona un archivo de datos experimentales válido.", parent=self.root)
                    self.btn_run_comp.config(state="normal")
                    return

                if param == "R":
                    R_data = np.loadtxt(data_file)
                    if R_data.ndim == 1:
                        R_exp = R_data
                        wl_exp = np.linspace(190, 900, len(R_exp))
                    else:
                        wl_exp = R_data[:, 0]
                        R_exp = R_data[:, 1]

                    RT = ftmm.RT_with_cache(None, st1, ftmm.materials, ftmm.lams, pol='s', th_0=0.0, thicks=None)
                    ref, trans = RT

                    self.fig_comp.clear()
                    ax = self.fig_comp.add_subplot(1, 1, 1)
                    ax.plot(ftmm.lams, ref, 'o-', ms=2, color="black", label="Simulación TMM (Stack 1)")
                    ax.plot(wl_exp, R_exp, 'o-', ms=2, color="darkcyan", label=f"R exp ({os.path.basename(data_file)})")
                    ax.set_xlabel(r"Longitud de onda ($\lambda$) [nm]", fontsize=10)
                    ax.set_ylabel("Reflectancia R", fontsize=10)
                    ax.set_title("Comparación de Reflectancia: Simulación vs Experimental", fontsize=11, fontweight="bold")
                    ax.set_xlim([300, 900])
                    ax.set_ylim([-0.02, 1.02])
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend(loc="upper right", framealpha=0.9)
                    self.fig_comp.tight_layout()
                    self.canvas_comp.draw()

                    self.lbl_comp_summary.config(
                        text=f"Comparación Reflectancia | Archivo: {os.path.basename(data_file)} | Puntos exp: {len(wl_exp)} | Rango: {wl_exp.min():.0f}-{wl_exp.max():.0f} nm"
                    )

                else:  # param in ('Is', 'Ic')
                    wl_exp, psi_exp, delta_exp, Ic_raw, Is_raw = np.loadtxt(data_file, skiprows=61, usecols=(0, 1, 2, 3, 4), max_rows=1334 - 61, unpack=True)
                    wl_exp, psi_exp, delta_exp, Ic_exp, Is_exp = interpolar_todo(wl_exp, psi_exp, delta_exp, Ic_raw, Is_raw)

                    th_0_rad = np.radians(th_elips)
                    n_list, d_list, c_list = stack2tmm(st1, ftmm.materials, ftmm.lams, add_inf=False)
                    res_s = coh_tmm('s', n_list, d_list, th_0_rad, ftmm.lams)
                    res_p = coh_tmm('p', n_list, d_list, th_0_rad, ftmm.lams)

                    r_s, r_p = res_s['r'], res_p['r']
                    rho = np.conj(r_p / r_s)
                    psi_rad = np.arctan(np.abs(rho))
                    delta_rad = np.angle(rho)
                    Is_sim = np.sin(2 * psi_rad) * np.sin(delta_rad)
                    Ic_sim = np.sin(2 * psi_rad) * np.cos(delta_rad)

                    self.fig_comp.clear()
                    ax1 = self.fig_comp.add_subplot(2, 1, 1)
                    ax2 = self.fig_comp.add_subplot(2, 1, 2)

                    ax1.plot(ftmm.lams, Is_sim, 'o-', ms=2, color="darkcyan", label="Is Simulación (Stack 1)")
                    ax1.plot(wl_exp, Is_exp, 'o', ms=2, color="black", alpha=0.7, label=f"Is exp ({os.path.basename(data_file)})")
                    ax1.set_ylabel("Is", fontsize=10)
                    ax1.set_xlim([np.min(wl_exp), np.max(wl_exp)])
                    ax1.grid(True, linestyle="--", alpha=0.5)
                    ax1.legend(loc="upper right", framealpha=0.9)

                    ax2.plot(ftmm.lams, Ic_sim, 'o-', ms=2, color="darkcyan", label="Ic Simulación (Stack 1)")
                    ax2.plot(wl_exp, Ic_exp, 'o', ms=2, color="black", alpha=0.7, label=f"Ic exp ({os.path.basename(data_file)})")
                    ax2.set_xlabel(r"Longitud de onda ($\lambda$) [nm]", fontsize=10)
                    ax2.set_ylabel("Ic", fontsize=10)
                    ax2.set_xlim([np.min(wl_exp), np.max(wl_exp)])
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    ax2.legend(loc="upper right", framealpha=0.9)

                    self.fig_comp.tight_layout()
                    self.canvas_comp.draw()

                    self.lbl_comp_summary.config(
                        text=f"Comparación Elipsométrica | θ₀={th_elips}° | Archivo: {os.path.basename(data_file)} | Puntos exp: {len(wl_exp)}"
                    )

            else:  # mode == "stacks"
                st2 = layers_to_tmm_stack(self.layers_stack2)

                if param == "R":
                    RT1 = ftmm.RT_with_cache(None, st1, ftmm.materials, ftmm.lams, pol='s', th_0=0.0, thicks=None)
                    RT2 = ftmm.RT_with_cache(None, st2, ftmm.materials, ftmm.lams, pol='s', th_0=0.0, thicks=None)
                    ref1, _ = RT1
                    ref2, _ = RT2

                    self.fig_comp.clear()
                    ax = self.fig_comp.add_subplot(1, 1, 1)
                    ax.plot(ftmm.lams, ref1, 'o-', ms=2, color="black", label="Simulación TMM (Stack 1)")
                    ax.plot(ftmm.lams, ref2, 'o-', ms=2, color="darkcyan", label="Simulación TMM (Stack 2)")
                    ax.set_xlabel(r"Longitud de onda ($\lambda$) [nm]", fontsize=10)
                    ax.set_ylabel("Reflectancia R", fontsize=10)
                    ax.set_title("Comparación de Reflectancia entre Stack 1 y Stack 2", fontsize=11, fontweight="bold")
                    ax.set_xlim([300, 900])
                    ax.set_ylim([-0.02, 1.02])
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend(loc="upper right", framealpha=0.9)
                    self.fig_comp.tight_layout()
                    self.canvas_comp.draw()

                    self.lbl_comp_summary.config(text="Comparación Reflectancia Stack 1 vs Stack 2 completada.")

                else:  # param in ('Is', 'Ic')
                    th_0_rad = np.radians(th_comp)
                    n1, d1, _ = stack2tmm(st1, ftmm.materials, ftmm.lams, add_inf=False)
                    n2, d2, _ = stack2tmm(st2, ftmm.materials, ftmm.lams, add_inf=False)

                    res_s1 = coh_tmm('s', n1, d1, th_0_rad, ftmm.lams)
                    res_p1 = coh_tmm('p', n1, d1, th_0_rad, ftmm.lams)
                    res_s2 = coh_tmm('s', n2, d2, th_0_rad, ftmm.lams)
                    res_p2 = coh_tmm('p', n2, d2, th_0_rad, ftmm.lams)

                    r_s1, r_p1 = res_s1['r'], res_p1['r']
                    r_s2, r_p2 = res_s2['r'], res_p2['r']

                    rho1 = np.conj(r_p1 / r_s1)
                    rho2 = np.conj(r_p2 / r_s2)

                    psi1 = np.arctan(np.abs(rho1))
                    psi2 = np.arctan(np.abs(rho2))
                    del1 = np.angle(rho1)
                    del2 = np.angle(rho2)

                    Is1 = np.sin(2 * psi1) * np.sin(del1)
                    Is2 = np.sin(2 * psi2) * np.sin(del2)
                    Ic1 = np.sin(2 * psi1) * np.cos(del1)
                    Ic2 = np.sin(2 * psi2) * np.cos(del2)

                    self.fig_comp.clear()
                    ax1 = self.fig_comp.add_subplot(2, 1, 1)
                    ax2 = self.fig_comp.add_subplot(2, 1, 2)

                    ax1.plot(ftmm.lams, Is1, 'o-', ms=2, color="darkcyan", label="Is Stack 1")
                    ax1.plot(ftmm.lams, Is2, 'o-', ms=2, color="darkorange", label="Is Stack 2")
                    ax1.set_ylabel("Is", fontsize=10)
                    ax1.set_xlim([np.min(ftmm.lams), np.max(ftmm.lams)])
                    ax1.grid(True, linestyle="--", alpha=0.5)
                    ax1.legend(loc="upper right", framealpha=0.9)

                    ax2.plot(ftmm.lams, Ic1, 'o-', ms=2, color="darkcyan", label="Ic Stack 1")
                    ax2.plot(ftmm.lams, Ic2, 'o-', ms=2, color="darkorange", label="Ic Stack 2")
                    ax2.set_xlabel(r"Longitud de onda ($\lambda$) [nm]", fontsize=10)
                    ax2.set_ylabel("Ic", fontsize=10)
                    ax2.set_xlim([np.min(ftmm.lams), np.max(ftmm.lams)])
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    ax2.legend(loc="upper right", framealpha=0.9)

                    self.fig_comp.tight_layout()
                    self.canvas_comp.draw()

                    self.lbl_comp_summary.config(text=f"Comparación Elipsométrica Stack 1 vs Stack 2 | θ₀={th_comp}° completada.")

            self.lbl_status.config(text="Comparación completada exitosamente.")
        except Exception as e:
            err_msg = traceback.format_exc()
            self.log(f"[ERROR] En comparación:\n{err_msg}")
            messagebox.showerror("Error en Comparación", f"Ocurrió un error al ejecutar la comparación:\n{e}", parent=self.root)
        finally:
            self.btn_run_comp.config(state="normal")

    def _run_comparison_external(self):
        st1 = layers_to_tmm_stack(self.layers)
        mode = self.var_comp_mode.get()
        param = self.var_comp_param.get()
        
        st2 = layers_to_tmm_stack(self.layers_stack2) if mode == "stacks" else None
        data_file = self.var_comp_data_file.get().strip() if mode == "exp" else None

        if mode == "exp" and (not data_file or not os.path.exists(data_file)):
            messagebox.showerror("Error", "Selecciona un archivo de datos experimentales válido.", parent=self.root)
            return

        try:
            th_elips = float(self.ent_comp_th_elips.get())
            th_comp = float(self.ent_comp_th_comp.get())
            ftmm.angulo_elipsometro = th_elips
            ftmm.angulo_comparacion = th_comp
        except ValueError:
            pass

        self.log(f"[INFO] Lanzando comparación externa en ventana de Matplotlib...")
        threading.Thread(
            target=lambda: ftmm.comparar_stacks(st1, stack2=st2, DATA=data_file, parametro=param),
            daemon=True
        ).start()

    # ==========================================================================
    # PESTAÑA 5: EXPLORADOR DE MATERIALES
    # ==========================================================================
    def _build_tab_mat(self):
        paned = ttk.PanedWindow(self.tab_mat, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=10)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Materiales Disponibles", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.lst_materials = tk.Listbox(left_frame, height=18, font=("Segoe UI", 9))
        self.lst_materials.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self._refresh_materials_listbox()
        self.lst_materials.bind("<<ListboxSelect>>", self._on_select_material)

        ttk.Button(left_frame, text="📈 Graficar Índice n y k", command=self._plot_selected_material).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="➕ Cargar Material desde Archivo (.nk / .txt)", command=self._load_custom_material).pack(fill=tk.X, pady=3)
        ttk.Button(left_frame, text="🗑️ Eliminar Material del Catálogo", command=self._delete_selected_material).pack(fill=tk.X, pady=3)
        self.lst_materials.bind("<Delete>", lambda e: self._delete_selected_material())

        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=2)

        self.fig_mat = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas_mat = FigureCanvasTkAgg(self.fig_mat, master=right_frame)
        self.canvas_mat.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_mat = NavigationToolbar2Tk(self.canvas_mat, right_frame)
        toolbar_mat.update()

    def _refresh_materials_listbox(self):
        self.lst_materials.delete(0, tk.END)
        for m in sorted(ftmm.materials.keys()):
            self.lst_materials.insert(tk.END, m)

    def _on_select_material(self, event):
        sel = self.lst_materials.curselection()
        if sel:
            mat_name = self.lst_materials.get(sel[0])
            self._plot_material(mat_name)

    def _plot_selected_material(self):
        sel = self.lst_materials.curselection()
        if sel:
            mat_name = self.lst_materials.get(sel[0])
            self._plot_material(mat_name)

    def _delete_selected_material(self):
        sel = self.lst_materials.curselection()
        if not sel:
            messagebox.showwarning("Atención", "Por favor, selecciona un material de la lista para eliminar.", parent=self.root)
            return

        mat_name = self.lst_materials.get(sel[0])

        if mat_name == "air":
            messagebox.showerror("Error", "No se puede eliminar el medio base 'air'.", parent=self.root)
            return

        if not messagebox.askyesno("Confirmar Eliminación", f"¿Estás seguro de que deseas eliminar el material '{mat_name}' del catálogo?", parent=self.root):
            return

        # Verificar si está en uso en alguna capa del stack
        used_layers = [l.name for l in self.layers if l.material == mat_name]
        used_layers2 = [l.name for l in self.layers_stack2 if l.material == mat_name]
        if used_layers or used_layers2:
            all_used = used_layers + [f"{l} (Stack 2)" for l in used_layers2]
            msg = f"El material '{mat_name}' está asignado en las siguientes capas:\n- " + "\n- ".join(all_used)
            msg += "\n\nSi continúas, estas capas cambiarán su material base a 'air'. ¿Deseas continuar?"
            if not messagebox.askyesno("Material en Uso", msg, parent=self.root):
                return
            for l in self.layers:
                if l.material == mat_name:
                    l.material = "air"
            for l in self.layers_stack2:
                if l.material == mat_name:
                    l.material = "air"

        if mat_name in ftmm.materials:
            del ftmm.materials[mat_name]

        if hasattr(self, 'custom_material_ranges') and mat_name in self.custom_material_ranges:
            del self.custom_material_ranges[mat_name]

        self._refresh_materials_listbox()
        self._refresh_layers_ui()
        self._update_comp_stack_labels()

        self.fig_mat.clear()
        self.canvas_mat.draw()

        self.log(f"[INFO] Material '{mat_name}' eliminado del catálogo.")
        messagebox.showinfo("Material Eliminado", f"El material '{mat_name}' fue eliminado del catálogo correctamente.", parent=self.root)

    def _get_material_range(self, mat_name: str) -> Tuple[float, float]:
        """Obtiene el rango (min, max) de longitud de onda [nm] específico del archivo o definición del material."""
        if hasattr(self, 'custom_material_ranges') and mat_name in self.custom_material_ranges:
            return self.custom_material_ranges[mat_name]

        if mat_name not in ftmm.materials:
            return float(np.min(ftmm.lams)), float(np.max(ftmm.lams))

        n_fn, k_fn = ftmm.materials[mat_name]

        def _extract_x(fn):
            if hasattr(fn, 'x') and fn.x is not None:
                return fn.x
            if hasattr(fn, '__closure__') and fn.__closure__:
                for cell in fn.__closure__:
                    try:
                        res = _extract_x(cell.cell_contents)
                        if res is not None:
                            return res
                    except Exception:
                        pass
            return None

        x_n = _extract_x(n_fn)
        x_k = _extract_x(k_fn)

        wl_min = None
        wl_max = None
        if x_n is not None and len(x_n) > 0:
            wl_min = float(np.min(x_n))
            wl_max = float(np.max(x_n))
        if x_k is not None and len(x_k) > 0:
            k_min = float(np.min(x_k))
            k_max = float(np.max(x_k))
            wl_min = k_min if wl_min is None else min(wl_min, k_min)
            wl_max = k_max if wl_max is None else max(wl_max, k_max)

        if wl_min is None or wl_max is None or wl_min >= wl_max:
            wl_min = float(np.min(ftmm.lams))
            wl_max = float(np.max(ftmm.lams))

        return wl_min, wl_max

    def _plot_material(self, mat_name: str):
        if mat_name not in ftmm.materials:
            return

        n_fn, k_fn = ftmm.materials[mat_name]
        wl_min, wl_max = self._get_material_range(mat_name)

        # Generar puntos de longitud de onda estrictamente dentro del rango medido/tabulado del archivo
        num_pts = int(np.clip(wl_max - wl_min + 1, 100, 2000))
        lams = np.linspace(wl_min, wl_max, num_pts)

        try:
            n_vals = n_fn(lams)
            k_vals = k_fn(lams)

            self.fig_mat.clear()
            ax1 = self.fig_mat.add_subplot(2, 1, 1)
            ax2 = self.fig_mat.add_subplot(2, 1, 2)

            ax1.plot(lams, n_vals, 'b-', linewidth=2, label=f"n({mat_name})")
            ax1.set_xlim(wl_min, wl_max)
            ax1.set_ylabel("Índice de Refracción n")
            ax1.set_title(f"Constantes Ópticas: {mat_name} (λ: {wl_min:.1f} - {wl_max:.1f} nm)", fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(lams, k_vals, 'r-', linewidth=2, label=f"k({mat_name})")
            ax2.set_xlim(wl_min, wl_max)
            ax2.set_xlabel("Longitud de Onda λ [nm]")
            ax2.set_ylabel("Coef. de Extinción k")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            self.fig_mat.tight_layout()
            self.canvas_mat.draw()
        except Exception as e:
            self.log(f"[ERROR] No se pudo graficar el material {mat_name}: {e}")

    def _load_custom_material(self):
        f = filedialog.askopenfilename(
            title="Cargar Archivo de Material",
            filetypes=[("Archivos de Material", "*.nk *.nkv *.in3 *.txt"), ("Todos los archivos", "*.*")],
            initialdir=ftmm.ruta_materiales
        )
        if not f:
            return

        base_name = os.path.splitext(os.path.basename(f))[0]
        top = tk.Toplevel(self.root)
        top.title("Nombre del Material")
        top.geometry("320x130")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        ttk.Label(top, text="Nombre para identificar el material:").pack(pady=8)
        ent = ttk.Entry(top, width=28)
        ent.insert(0, base_name)
        ent.pack(pady=5)

        def save_mat():
            mat_name = ent.get().strip()
            if not mat_name:
                return
            try:
                if f.endswith(".in3"):
                    ftmm.materials[mat_name] = load_interp_in3(f, unit='A')
                elif f.endswith(".txt"):
                    ftmm.materials[mat_name] = load_interp_only_n(f, skiprows=2)
                else:
                    ftmm.materials[mat_name] = load_interp(f, skiprows=1)

                r_min, r_max = self._get_material_range(mat_name)
                self.custom_material_ranges[mat_name] = (r_min, r_max)

                self._refresh_materials_listbox()
                self._refresh_layers_ui()
                top.destroy()
                self._plot_material(mat_name)
                messagebox.showinfo("Material Agregado", f"El material '{mat_name}' fue cargado exitosamente.\nRango λ: {r_min:.1f} - {r_max:.1f} nm", parent=self.root)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}", parent=top)

        ttk.Button(top, text="Guardar Material", command=save_mat).pack(pady=10)


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def main():
    root = tk.Tk()
    app = TMMInterfaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
