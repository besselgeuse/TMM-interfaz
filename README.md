# ⚡ TMM Suite: Simulación Óptica, Ajuste Elipsométrico y Optimización de Capas Antirreflectantes (ARC)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-green.svg)]()

---

## 📖 Descripción General

**TMM Suite** es una plataforma de software con interfaz gráfica interactiva diseñada para el modelado, simulación y optimización de recubrimientos ópticos multicapa mediante el **Método de Matrices de Transferencia (Transfer Matrix Method - TMM)**. Desarrollada en el marco de investigación de tesis, su objetivo principal es el diseño y análisis de **películas antirreflectantes (ARC)** para celdas solares (de uso espacial y terrestre) y la caracterización óptica de películas delgadas mediante **ajuste de elipsometría espectroscópica**.

El motor de cálculo numérico está implementado sobre **PyTorch**, permitiendo aprovechar la diferenciación automática y la aceleración por hardware (**GPU vía CUDA** o **CPU con multithreading**).

---

## 🌟 Características Principales

### 1. 🧱 Gestión Dinámica del Stack Óptico (Capas)
* **Configuración visual de multicapas**: Medios semi-infinitos de entrada (e.g. aire, vacío) y salida (sustrato, e.g. GaAs, Si, InGaP, vidrio) junto a cualquier número de capas intermedias delgadas.
* **Control de espesores ($d$)**:
  * Espesor fijo (modo analítico directo).
  * Rango de espesores $[d_{min}, d_{max}]$ para búsqueda y optimización automática.
  * Medios semi-infinitos ($\infty$).
* **Presets de partida preconfigurados**:
  * *Preset 1*: Elipsometría $\text{Al}_2\text{O}_3$ (poroso / denso / $\text{SiO}_2$ / $\text{Si}$).
  * *Preset 2*: Celda solar con $\text{TiO}_2 + \text{Al}_2\text{O}_3$ (poroso / $\text{TiO}_2$ RTP / $\text{SiO}_2$ / $\text{Si}$).
  * *Preset 3*: Antirreflectante simple ($\text{MgF}_2$ / $\text{Si}$).
  * *Preset 4*: Sistema básico (Aire / $\text{Si}$).
* Herramientas de edición en vivo: agregar, eliminar, duplicar, subir y bajar capas en el stack.

### 2. 🔬 Modelos de Dispersión Óptica
* **Materiales tabulados**: Soporte para formatos estándar de constantes ópticas (`.nk`, `.nkv`, `.mat`, `.in3`).
* **Modelo Paramétrico de Cauchy**:
  $$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$
* **Modelo Cauchy Absorbente**:
  $$k(\lambda) = k_A + \frac{k_B}{\lambda^2} + \frac{k_C}{\lambda^4}$$
* **Aproximación de Medio Efectivo de Bruggeman (EMA)**:
  * Modelado de materiales porosos o mezclas compuestas.
  * Optimización de fracción de vacíos/volumétrica $f$ (modo rango o fijo).
  * Enlace con capas adyacentes o definición de materiales base personalizados.
* **Límites de índice de refracción**: Restricciones físicas configurables $[n_{min}, n_{max}]$ para evitar soluciones no físicas durante la optimización.

### 3. 🎯 Ajuste Elipsométrico Espectroscópico con PyTorch
* Carga de datos experimentales de elipsometría espectroscópica:
  * Componentes armónicos de intensidad: $I_s$ e $I_c$.
  * Ángulos elipsométricos derivados: $\Psi$ (Psi) y $\Delta$ (Delta).
* Optimización paramétrica basada en gradientes por tensores (PyTorch):
  * Ajuste simultáneo de espesores físicos y coeficientes ópticos de dispersión.
  * Algoritmo de rango dinámico (*Autorange*).
  * Visualización comparativa de curvas teóricas vs. experimentales y residuos del ajuste.

### 4. ☀️ Simulación de Reflectancia y Fotocorriente ($J_{sc}$)
* Cálculo de espectros de **Reflectancia ($R$)**, **Transmitancia ($T$)** y **Absorbancia ($A$)** para polarizaciones $s$, $p$ o luz no polarizada en función del ángulo de incidencia.
* Integración con espectros solares normalizados:
  * **AM0**: Espectro solar extraterrestre (aplicaciones espaciales / satelitales).
  * **AM1.5G**: Espectro solar terrestre estándar.
* Cálculo de la densidad de corriente de cortocircuito máxima fotogenerada ($J_{sc}$):
  $$J_{sc} = q \int \Phi_{\text{fotón}}(\lambda) \cdot [1 - R(\lambda)] \cdot \text{IQE}(\lambda) \, d\lambda$$
  donde se incorpora la Eficiencia Cuántica Interna ($\text{IQE}$) para sustratos como $\text{GaAs}$ y $\text{Si}$.

### 5. 📊 Comparación de Stacks y Datos Espectrofotométricos
* Comparación gráfica directa entre dos configuraciones de multicapas (**Stack 1** vs. **Stack 2**).
* Superposición de curvas experimentales de reflectancia espectrofotométrica.
* Cálculo de diferencias relativas y absolutas entre arquitecturas.

### 6. 📚 Biblioteca y Visualizador de Materiales
* Catálogo óptico con más de 100 materiales (incluyendo bases de datos SOPRA, Palik, etc.).
* Graficador interactivo de índice de refracción real $n(\lambda)$ y coeficiente de extinción $k(\lambda)$.
* Búsqueda y filtrado rápido por nombre de material.

---

## 📂 Estructura del Repositorio

```text
TMM-interfaz/
│
├── interfaz_tmm.py          # Interfaz gráfica de usuario principal (Tkinter / TTK)
├── tmm_core.py              # Núcleo de cálculo del Método de Matrices de Transferencia (TMM)
├── tmm_utils_Rodrigo.py     # Utilidades numéricas, modelos (Cauchy, Bruggeman) y optimizador
├── Funciones_tmm.py         # Funciones auxiliares y registro de catálogo de materiales
│
├── indices/                 # Base de datos de índices de refracción
│   ├── nkdata/              # Datos tabulados (.nk, .nkv, .mat, SOPRA, Palik)
│   ├── IQEGaAs2.txt         # Eficiencia cuántica interna de GaAs
│   ├── IQE_Si.txt           # Eficiencia cuántica interna de Silicio
│   └── ...
│
├── instalar_entorno.bat     # Instalador automático del entorno virtual en Windows
├── ejecutar_interfaz.bat    # Script de ejecución con 1 clic
├── requirements.txt         # Lista de librerías y dependencias necesarias
├── Guia.txt                 # Guía rápida inicial
└── README.md                # Documentación del proyecto
```

---

## 🛠️ Requisitos del Sistema

* **Sistema Operativo**: Windows 10 / 11 (recomendado para los scripts `.bat`), Linux o macOS.
* **Versión de Python**: **Python 3.12.x** (probado y validado en Python 3.12.2).
* **Hardware**:
  * CPU compatible con x86_64.
  * *(Opcional)* GPU NVIDIA con soporte CUDA para aceleración de ajustes numéricos complejos.

---

## 🚀 Instalación y Puesta en Marcha

### Opción A: Instalación Automática (Recomendada en Windows)

1. Clona o descarga el repositorio:
   ```cmd
   git clone https://github.com/besselgeuse/TMM-interfaz.git
   cd TMM-interfaz
   ```

2. Haz doble clic en el archivo **`instalar_entorno.bat`** (o ejecútalo desde la consola):
   * El script detectará tu instalación de Python 3.12.
   * Creará automáticamente un entorno virtual aislado en la carpeta `venv`.
   * Actualizará `pip` e instalará todas las dependencias listadas en `requirements.txt`.

3. Para abrir el programa, haz doble clic en **`ejecutar_interfaz.bat`**.

---

### Opción B: Instalación Manual

Si prefieres configurar el entorno manualmente desde una terminal:

1. **Crear el entorno virtual**:
   ```bash
   python -m venv venv
   ```

2. **Activar el entorno virtual**:
   * En Windows:
     ```cmd
     venv\Scripts\activate
     ```
   * En Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Lanzar la aplicación**:
   ```bash
   python interfaz_tmm.py
   ```

---

## 📦 Dependencias Principales

| Paquete | Versión Mínima | Uso en el Proyecto |
| :--- | :---: | :--- |
| **`numpy`** | `>= 1.24.0` | Operaciones matriciales, manejo de arreglos y cálculos vectoriales. |
| **`scipy`** | `>= 1.10.0` | Interpolación unidimensional de constantes ópticas (`interp1d`). |
| **`torch`** | `>= 2.0.0` | Tensores y optimización con descenso de gradiente (CPU y CUDA). |
| **`matplotlib`** | `>= 3.7.0` | Renderizado de gráficos interactivos embebidos en el canvas de Tkinter. |
| **`openpyxl`** | `>= 3.1.0` | Lectura y exportación de curvas y resultados a formato Excel (`.xlsx`). |

---

## 🧪 Flujo de Trabajo Típico

### 1. Simulación de Reflectancia de un Recubrimiento Antirreflectante (ARC)
1. Ve a la pestaña **🧱 Stack Óptico (Capas)**.
2. Selecciona un preset (por ejemplo, *Preset 3: Antirreflectante Simple*) o construye tu propia estructura capa por capa.
3. Dirígete a la pestaña **☀️ Reflectancia y Jsc**.
4. Define el rango de longitudes de onda de interés (e.g. 300 nm a 1200 nm) y el espectro solar deseado (**AM0** o **AM1.5G**).
5. Presiona **Calcular Reflectancia / Jsc** para obtener las curvas espectrales y la corriente fotogenerada.

### 2. Ajuste Elipsométrico de una Película Delgada
1. Configura el stack en la pestaña **🧱 Stack Óptico**:
   * Define el medio de entrada (e.g. `air`) y el sustrato (e.g. `Si`).
   * Configura la capa a caracterizar indicando su modelo (e.g. `cauchy` o `bruggeman`) y marcando los espesores con rangos $[d_{min}, d_{max}]$.
2. En la pestaña **🔬 Ajuste Elipsométrico**, carga el archivo experimental de mediciones.
3. Ajusta los hiperparámetros de optimización (épocas, learning rate, tolerancia).
4. Inicia el ajuste. La barra de progreso y la consola mostrarán la convergencia del error y los valores optimizados de los parámetros.

---

## 👥 Créditos y Contacto

* **Autor**: [besselgeuse](https://github.com/besselgeuse)
* Proyecto desarrollado para fines de investigación de tesis en películas delgadas ópticas y celdas solares.
