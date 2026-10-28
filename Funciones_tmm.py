import sys
import os
import re
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt

dirección_tmm = os.path.dirname(os.path.abspath(__file__))

if dirección_tmm not in sys.path:
    sys.path.insert(0, dirección_tmm)

from tmm_utils_Rodrigo import (
    load_interp, autorange_fit_ellipsometry_torch, load_fn, plot_js,
    cauchy_fn, constant_fn, brugg_fn, stack2tmm, calculate_RT_torch, transform_I_to_psi_delta,
    load_interp_in3, leer_datos_guardados, interpolar_todo, guardar_resultados_txt,
    guardar_indice_refraccion, load_interp_only_n, RT_with_cache
)

from tmm_core import coh_tmm

ruta_materiales = os.path.join(dirección_tmm, 'indices')

modelo_T1 = cauchy_fn(2.338, 1.906, 0.824)   # muestra T1 del paper (R-1)
f_T1 = 0.58
modelo_Al2O3_nanotubes = cauchy_fn(0.9378, 2.4589, -6.8318)

p_rtp_aire = os.path.join(dirección_tmm, '../Files/24-9-26/2026.09.24/indices/TiO2_RTP_aire_T1_indice.txt')
p_rtp_ox = os.path.join(dirección_tmm, '../Files/24-9-26/2026.09.24/indices/TiO2_RTP_oxigeno_T1_indice.txt')

materials = {
    'air': (constant_fn(1.0), constant_fn(0.0)),
    'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows=1)),
    'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows=1)),
    #'T1_densa': (modelo_T1, constant_fn(0.0)),
    #'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
    'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)),
    'MgF2': (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv', skiprows=1)),
    'vidrio': (load_interp(f'{ruta_materiales}/glass_thales.nk', skiprows=1)),
    'Al2O3': (load_interp(f'{ruta_materiales}/nkdata/optical/Al2O3.nkv', skiprows=1)),
    'SiO2': (load_interp(f'{ruta_materiales}/nkdata/optical/SiO2.nkv', skiprows=1)),
    'Si': (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nkv', skiprows=1)),
    'anatasa': (load_interp(f'{ruta_materiales}/nkdata/anatase.nkv', skiprows=1)),
    'aluminio': (load_interp_in3(f'{ruta_materiales}/nkdata/optical/aluminum.in3', unit='A')),
    #'Al2O3_nanotubes': (modelo_Al2O3_nanotubes, constant_fn(0.0)),
}

# if os.path.exists(p_rtp_aire):
#     materials['TiO2_RTP_aire'] = load_interp_only_n(p_rtp_aire, skiprows=2)
# if os.path.exists(p_rtp_ox):
#     materials['TiO2_RTP_oxigeno'] = load_interp_only_n(p_rtp_ox, skiprows=2)

#materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
#materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
# materials['mugre'] = (materials['Al2O3'][0], materials['aluminio'][1])
materials['SiO2_poroso'] = (brugg_fn(materials['SiO2'][0], constant_fn(1.0), 0.5265), materials['SiO2'][1])
materials['Al2O3_porosa'] = (brugg_fn(materials['Al2O3'][0], constant_fn(1.0), 0.5), materials.get('Al2O3_nanotubes', (None, constant_fn(0.0)))[1])
# materials['Al2O3_T1_densa'] = (brugg_fn(materials['Al2O3'][0], materials['T1_densa'][0], 0.5), materials['T1_densa'][1])

step = 1.0
lams = np.arange(300, 1201, step)
angulo_elipsometro=69.5
angulo_comparacion=0.0
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt',delimiter=",")(lams)/100

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams
weight_am0 = weight * am0
weight_am15 = weight * am15

def ajuste_elipsometrico(DATA_dir,layer_names,layer_models,d_bounds,n_max_limits,th_0=angulo_elipsometro,salida_ajuste=None,num_starts=400,
    num_epochs=150,lr=1,chi2_min=0.00065,use_cuda=True,max_attemps=10):

    wl_exp,psi_exp,delta_exp,Ic_exp_raw,Is_exp_raw = np.loadtxt(DATA_dir,skiprows=61,usecols=(0,1,2,3,4),max_rows=1334-61,unpack=True)
    wl_exp,psi_exp,delta_exp,Ic_exp,Is_exp = interpolar_todo(wl_exp,psi_exp,delta_exp,Ic_exp_raw,Is_exp_raw)
    n_list_np = []
    for i, name in enumerate(layer_names):
        if layer_models[i] is not None:
            n_list_np.append(np.ones_like(wl_exp, dtype=complex))
        else:
            n_fn, k_fn = materials[name]
            n_list_np.append(n_fn(wl_exp) + 1j * k_fn(wl_exp))

    n_list_torch = torch.tensor(np.array(n_list_np), dtype=torch.complex128)
    lams_torch = torch.tensor(wl_exp, dtype=torch.float64)
    Is_exp_torch = torch.tensor(Is_exp, dtype=torch.float64)
    Ic_exp_torch = torch.tensor(Ic_exp, dtype=torch.float64)
    th_0_rad = np.radians(th_0)

    best_thicknesses, best_Is_curve, best_Ic_curve, best_params = autorange_fit_ellipsometry_torch(
    n_list=n_list_torch,
    d_bounds=d_bounds,
    lams=lams_torch,
    Is_exp=Is_exp_torch,
    Ic_exp=Ic_exp_torch,
    th_0=th_0_rad,
    num_starts=num_starts,
    num_epochs=num_epochs,
    lr=lr,
    use_cuda=use_cuda,
    layer_models=layer_models,
    layer_names=layer_names,
    n_max_limits=n_max_limits,
    chi2_min=chi2_min,
    max_attempts=max_attemps)
    psi_fit, delta_fit = transform_I_to_psi_delta(best_Is_curve, best_Ic_curve)

    if salida_ajuste is not None:
        guardar_resultados_txt(
            salida_ajuste, DATA_dir, wl_exp, Is_exp, best_Is_curve, Ic_exp, best_Ic_curve,
            psi_exp, psi_fit, delta_exp, delta_fit, best_thicknesses, best_params
        )

    return wl_exp,Is_exp,Ic_exp,psi_exp,delta_exp,best_Ic_curve,best_Is_curve,best_thicknesses,best_params,psi_fit,delta_fit


def graficar_resultados(wl_exp,Is_exp,Ic_exp,psi_exp,delta_exp,best_Ic_curve,best_Is_curve,best_thicknesses,best_params,psi_fit,delta_fit,plot_IsIc=True,plot_psid=True):
    if plot_IsIc:
        plt.figure()
        plt.plot(wl_exp,Is_exp,label='Is_exp')
        plt.plot(wl_exp,best_Is_curve,label='Is_fit')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(wl_exp,Ic_exp,label='Ic_exp')
        plt.plot(wl_exp,best_Ic_curve,label='Ic_fit')
        plt.legend()
        plt.show()
    else :
        plt.figure()
        plt.plot(wl_exp,psi_exp,label='psi_exp')
        plt.plot(wl_exp,psi_fit,label='psi_fit')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(wl_exp,delta_exp,label='delta_exp')
        plt.plot(wl_exp,delta_fit,label='delta_fit')
        plt.legend()
        plt.show()


def cargar_parametros_optimizados (best_params,best_thicknesses,layer_names,layer_models):  
    n_list = []
    d_list = []
    finite_idx = 0
    for i, name in enumerate(layer_names):
        if layer_models is not None and i < len(layer_models) and layer_models[i] is not None:
            if best_thicknesses is not None and finite_idx < len(best_thicknesses):
                d_list.append(best_thicknesses[finite_idx])
            else:
                d_list.append(0.0)
            finite_idx += 1
            
            p_info = best_params.get(name, {}) if best_params else {}
            m_dict = layer_models[i] if isinstance(layer_models[i], dict) else {'model': str(layer_models[i])}
            m_type = m_dict.get('model', '')
            
            if 'bruggeman' in m_type:
                f_air = p_info.get('f_air', p_info.get('f_bounds', 0.5))
                if isinstance(f_air, (list, tuple)):
                    f_air = f_air[0]
                linked = p_info.get('linked_to', None)
                if not linked:
                    lm = m_dict.get('linked_material')
                    if isinstance(lm, str):
                        linked = lm
                    elif isinstance(lm, dict):
                        linked = lm.get('name')
                if not linked and (i + 1 < len(layer_names)):
                    linked = layer_names[i + 1]
                
                # Determinar función base
                if 'A' in p_info:
                    # El material base se ajustó internamente en la capa de Bruggeman
                    A, B, C = float(p_info['A']), float(p_info.get('B', 0.0)), float(p_info.get('C', 0.0))
                    if 'D' in p_info:
                        D, E, F = float(p_info['D']), float(p_info.get('E', 0.0)), float(p_info.get('F', 0.0))
                        base_fn = lambda w, a=A, b=B, c=C, d=D, e=E, f=F: cauchy_fn(a, b, c)(w) + 1j * cauchy_fn(d, e, f)(w)
                    else:
                        base_fn = cauchy_fn(A, B, C)
                elif linked and linked in best_params and 'A' in best_params[linked]:
                    lp = best_params[linked]
                    A, B, C = float(lp['A']), float(lp.get('B', 0.0)), float(lp.get('C', 0.0))
                    if 'D' in lp:
                        D, E, F = float(lp['D']), float(lp.get('E', 0.0)), float(lp.get('F', 0.0))
                        base_fn = lambda w, a=A, b=B, c=C, d=D, e=E, f=F: cauchy_fn(a, b, c)(w) + 1j * cauchy_fn(d, e, f)(w)
                    else:
                        base_fn = cauchy_fn(A, B, C)
                elif linked and linked in materials:
                    base_fn = materials[linked][0]
                elif name in materials:
                    base_fn = materials[name][0]
                else:
                    base_fn = constant_fn(1.5)
                materials[name] = (brugg_fn(base_fn, constant_fn(1.0), float(f_air)), constant_fn(0.0))
            elif 'cauchy' in m_type:
                A = float(p_info.get('A', 1.5))
                B = float(p_info.get('B', 0.0))
                C = float(p_info.get('C', 0.0))
                materials[name] = (cauchy_fn(A, B, C), constant_fn(0.0))
            
            if name in materials:
                n_list.append(materials[name][0])
        else:
            if name in materials:
                n_list.append(materials[name][0])
            d_list.append(np.inf)
    return n_list,d_list

def graficar_indice(material,best_params,best_thicknesses,layer_names,layer_models,wl_exp):
    n_list,d_list = cargar_parametros_optimizados(best_params,best_thicknesses,layer_names,layer_models)
    if material in layer_names:
        posicion_materials = layer_names.index(material)
        n, d = n_list[posicion_materials], d_list[posicion_materials]
    elif material in materials:
        n, d = materials[material][0], 'tabulado'
    else:
        raise ValueError(f"Material {material} no encontrado.")
        
    plt.figure(figsize=(8,6))
    label_txt = f'{material} ({d:.2f} nm)' if isinstance(d, (int, float)) and not np.isinf(d) else f'{material}'
    plt.plot(wl_exp,n(wl_exp),label=label_txt)
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Índice de refracción')
    plt.grid(True)
    plt.legend()
    plt.show()

def optimizar_espesores(layer_names,layer_models=None,d_bounds=None,n_max_limits=None,th_0=0.0,salida_ajuste=None,num_starts=2000,
    num_epochs=150,lr=1.0,chi2_min=0.00065,use_cuda=True,max_attemps=10,weights=None,stack=None):

    if stack is None:
        stack = []
        for i, name in enumerate(layer_names):
            if i == 0 or i == len(layer_names) - 1:
                stack.append([np.inf, name, 'i'])
            else:
                bound = d_bounds[i - 1] if d_bounds is not None and (i - 1) < len(d_bounds) else (0, 100)
                stack.append([bound, name, 'c'])

    w = weights if weights is not None else weight_am0
    th_rad = np.radians(th_0) if th_0 != 0.0 else 0.0

    thickness, R_curve = calculate_RT_torch(stack,
                                    materials,
                                    lams,
                                    weights=w, pol='s',
                                    th_0=th_rad,
                                    num_starts=num_starts, 
                                    num_epochs=num_epochs, 
                                    lr=lr, 
                                    use_cuda=use_cuda)

    return thickness,R_curve

def calcular_jsc (R_curve,weights=weight_am0):
    jmax = np.trapezoid(weights, dx=step)
    jsc = jmax - np.trapezoid(R_curve*weights, dx=step)
    return jsc

def guardar_curva_R(lams,R_curve,path,thickness,jsc):

    with open(path,'w') as f:
        f.write(f'grosores: {thickness}\n')
        f.write(f'corriente: {jsc} mA/cm2\n')
        f.write('wavelength\tReflectance\n')
        for i in range(len(lams)):
            f.write(f'{lams[i]}\t{R_curve[i]}\n')
    return None
def leer_curva_R(path):
    # skiprows=3 salta las 3 primeras líneas de metadatos/cabecera
    # unpack=True desempaqueta directamente las columnas en variables separadas
    lams, R_curve = np.loadtxt(path, skiprows=3, unpack=True)
    return lams, R_curve


def comparar_stacks(stack1,stack2=None,DATA=None,parametro='R'):
    #Esta función servira para comparar dos stacks o en caso del segundo ser None
    #comparar el primero con la información medida del elipsometro si el parametro
    #es 'Is' o 'Ic' o el espectrofotómetro si el parametro es 'R'.
    if stack2 is None:
        if parametro == 'R':
            R_exp = np.loadtxt(DATA)
            wl_exp = np.linspace(190, 900, len(R_exp))
            RT = RT_with_cache(None,stack1, materials, lams, pol='s', th_0=0.0, thicks=None)
            ref, trans = RT
            plt.figure()
            plt.plot(lams,ref,'o-',ms=2,color="black",label="simulación TMM")
            plt.plot(wl_exp,R_exp,'o-',ms=2,color="darkcyan",label="R_exp (muestra N5)")
            plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
            plt.ylabel("reflectancia R")
            plt.grid()
            plt.xlim([300, 900])
            plt.ylim([0, 1])
            plt.legend()
            plt.show()
        elif parametro=='Is' or parametro=='Ic':
            wl_exp,psi_exp,delta_exp,Ic_exp_raw,Is_exp_raw = np.loadtxt(DATA,skiprows=61,usecols=(0,1,2,3,4),max_rows=1334-61,unpack=True)
            wl_exp,psi_exp,delta_exp,Ic_exp,Is_exp = interpolar_todo(wl_exp,psi_exp,delta_exp,Ic_exp_raw,Is_exp_raw)
                    
            # 2. Convertir el stack al formato que requiere tmm
            n_list, d_list, c_list = stack2tmm(stack1, materials, lams, add_inf=False)
            th_0_rad = np.radians(angulo_elipsometro)
            # 3. Obtener los coeficientes de reflexión complejos r_s y r_p
            res_s = coh_tmm('s', n_list, d_list, th_0_rad, lams)
            res_p = coh_tmm('p', n_list, d_list, th_0_rad, lams)

            r_s = res_s['r']  # Amplitud compleja de reflexión s
            r_p = res_p['r']  # Amplitud compleja de reflexión p

            # 4. Calcular el ratio elipsométrico rho y los parámetros Psi y Delta
            rho = np.conj(r_p / r_s)
            psi_rad = np.arctan(np.abs(rho))
            delta_rad = np.angle(rho)  # o np.angle(-rho) según el convenio de signos


            # 5. Convertir a grados para graficar
            psi_deg = np.degrees(psi_rad)
            delta_deg = np.degrees(delta_rad)

            # 6. (Opcional) Si necesitas las componentes Is e Ic del elipsómetro:
            Is = (np.sin(2 * psi_rad) * np.sin(delta_rad))
            Ic = (np.sin(2 * psi_rad) * np.cos(delta_rad))

            # 7. Graficos

            fig,ax = plt.subplots(2,1,figsize=(12,10))
            ax[0].plot(lams, Is, 'o-', ms=2, color="darkcyan", label=r"Is")
            ax[0].plot(wl_exp,Is_exp,'o',ms=2,color='black',alpha=0.8, label='Is exp')
            ax[1].plot(lams, Ic, 'o-', ms=2, color="darkcyan", label=r"Ic")
            ax[1].plot(wl_exp,Ic_exp,'o',ms=2,color='black',alpha=0.8, label='Ic exp')
            ax[0].set_xlabel(r"Longitud de onda ($\lambda$) [nm]")
            ax[1].set_xlabel(r"Longitud de onda ($\lambda$) [nm]")
            ax[0].set_ylabel('Is')
            ax[1].set_ylabel('Ic')
            ax[0].set_xlim(np.min(wl_exp),np.max(wl_exp))
            ax[1].set_xlim(np.min(wl_exp),np.max(wl_exp))
            ax[0].grid()
            ax[1].grid()
            ax[0].legend()
            ax[1].legend()
            plt.show()
    else:
        if parametro == 'R':
            RT1 = RT_with_cache(None,stack1, materials, lams, pol='s', th_0=0.0, thicks=None)
            RT2 = RT_with_cache(None,stack2, materials, lams, pol='s', th_0=0.0, thicks=None)
            ref1, trans1 = RT1
            ref2, trans2 = RT2
            plt.figure()
            plt.plot(lams,ref1,'o-',ms=2,color="black",label="simulación TMM stack1")
            plt.plot(lams,ref2,'o-',ms=2,color="darkcyan",label="simulación TMM stack2")
            plt.xlabel(r"longitud de onda (\lambda) [nm]")
            plt.ylabel("reflectancia R")
            plt.grid()
            plt.xlim([300, 900])
            plt.ylim([0, 1])
            plt.legend()
            plt.show()

        elif parametro=='Is' or parametro=='Ic':
            n_list1, d_list1, c_list1 = stack2tmm(stack1, materials, lams, add_inf=False)
            n_list2, d_list2, c_list2 = stack2tmm(stack2, materials, lams, add_inf=False)
            th_0_rad = np.radians(angulo_comparacion)
            res_s1 = coh_tmm('s', n_list1, d_list1, th_0_rad, lams)
            res_p1 = coh_tmm('p', n_list1, d_list1, th_0_rad, lams)
            res_s2 = coh_tmm('s', n_list2, d_list2, th_0_rad, lams)
            res_p2 = coh_tmm('p', n_list2, d_list2, th_0_rad, lams)
            r_s1 = res_s1['r']
            r_p1 = res_p1['r']
            r_s2 = res_s2['r']
            r_p2 = res_p2['r']
            rho1 = np.conj(r_p1 / r_s1)
            rho2 = np.conj(r_p2 / r_s2)
            psi_rad1 = np.arctan(np.abs(rho1))
            psi_rad2 = np.arctan(np.abs(rho2))
            delta_rad1 = np.angle(rho1)
            delta_rad2 = np.angle(rho2)
            psi_deg1 = np.degrees(psi_rad1)
            psi_deg2 = np.degrees(psi_rad2)
            delta_deg1 = np.degrees(delta_rad1)
            delta_deg2 = np.degrees(delta_rad2)
            Is1 = (np.sin(2 * psi_rad1) * np.sin(delta_rad1))
            Is2 = (np.sin(2 * psi_rad2) * np.sin(delta_rad2))
            Ic1 = (np.sin(2 * psi_rad1) * np.cos(delta_rad1))
            Ic2 = (np.sin(2 * psi_rad2) * np.cos(delta_rad2))
            fig,ax = plt.subplots(2,1,figsize=(12,10))
            ax[0].plot(lams, Is1, 'o-', ms=2, color="darkcyan", label=r"Is stack1")
            ax[0].plot(lams, Is2, 'o-', ms=2, color="darkorange", label=r"Is stack2")
            ax[1].plot(lams, Ic1, 'o-', ms=2, color="darkcyan", label=r"Ic stack1")
            ax[1].plot(lams, Ic2, 'o-', ms=2, color="darkorange", label=r"Ic stack2")
            ax[0].set_xlabel(r"Longitud de onda (\lambda) [nm]")
            ax[1].set_xlabel(r"Longitud de onda (\lambda) [nm]")
            ax[0].set_ylabel('Is')
            ax[1].set_ylabel('Ic')
            ax[0].set_xlim(np.min(lams),np.max(lams))
            ax[1].set_xlim(np.min(lams),np.max(lams))
            ax[0].grid()
            ax[1].grid()
            ax[0].legend()
            ax[1].legend()
            plt.show()
            


        