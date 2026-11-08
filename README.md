# HR Analytics: Employee Attrition Prediction

**Machine Learning Zoomcamp — Midterm Project**  
*Por: Romantic Maxwell*

---

## 1. Descripcion del Problema de Negocio

La rotacion o renuncia voluntaria de empleados (**Employee Attrition**) es uno de los desafios mas criticos y costosos para cualquier organizacion. Segun diversos estudios de Recursos Humanos (SHRM), reemplazar a un colaborador calificado cuesta entre el **50% y el 200% de su salario anual**, sumado a la perdida de conocimiento institucional, tiempo invertido en reclutamiento y disminucion de la moral en los equipos.

### Objetivo del Proyecto
Desarrollar una solucion integral de Machine Learning que prediga la probabilidad de que un empleado decida abandonar la empresa. Con estas predicciones, el departamento de Recursos Humanos (People Operations) puede:
- Identificar perfiles en riesgo de fuga de talento antes de que presenten su renuncia.
- Implementar planes de retencion personalizados (ajustes de carga laboral, revisiones salariales, promociones o capacitacion).
- Disminuir la tasa de rotacion voluntaria y optimizar el presupuesto operativo de la empresa.

### Dataset
Se utiliza el dataset de **IBM HR Analytics Employee Attrition** (disponible en `data/employee_attrition.csv`), el cual contiene **1,470 registros y 35 variables** que cubren aspectos demograficos, compensaciones, evaluaciones de desempeno y niveles de satisfaccion laboral.

---

## 2. Resumen del Analisis Exploratorio (EDA)

Durante el analisis en el notebook interactivo ([`notebook.ipynb`](notebook.ipynb)) se identificaron los siguientes patrones clave:
1. **Desbalance moderado de clases:** Aproximadamente el **16.1%** de los empleados renunciaron (`target = 1`), mientras que el **83.9%** permanecio en la compania (`target = 0`).
2. **Horas extras (`overtime`):** Es uno de los predictores mas fuertes; los empleados que realizan horas extras presentan una tasa de rotacion superior al **30%**, frente a menos del **10%** en quienes no las realizan.
3. **Ingreso mensual (`monthlyincome`):** Los colaboradores que abandonan la empresa tienen ingresos mensuales sensiblemente menores (mediana en ~$3,200 vs ~$5,200 en empleados retenidos).
4. **Distancia al hogar (`distancefromhome`):** Los trayectos diarios largos incrementan la probabilidad de renuncia.
5. **Satisfaccion laboral y del entorno:** Calificaciones bajas (1 o 2 sobre 4) se correlacionan directamente con mayor fuga de talento.

---

## 3. Comparativa y Afinamiento de Modelos

Se entrenaron y compararon 4 familias de algoritmos supervisados utilizando la metrica **ROC-AUC** debido al desbalance de clases:

| Modelo | Val ROC-AUC (Baseline) | 5-Fold CV ROC-AUC | Test ROC-AUC (Final) |
|---|:---:|:---:|:---:|
| **Logistic Regression** | 0.8308 | **0.8093 +/- 0.031** | **0.8053** |
| **Random Forest Classifier** | **0.8393** | 0.7992 +/- 0.022 | 0.7996 |
| **XGBoost Classifier** | 0.7808 | 0.7870 +/- 0.025 | 0.8033 |
| **Decision Tree Classifier** | 0.7584 | 0.7310 +/- 0.035 | 0.7240 |

### Seleccion del Modelo Final
Se selecciono **Logistic Regression** (con regularizacion `C=1.0`, solver `liblinear`) debido a:
- Mejor rendimiento promedio y consistencia en validacion cruzada y conjunto de prueba final (**AUC 0.8053**, **Accuracy 88.8%**).
- Alta interpretabilidad de los coeficientes para decisiones de negocio en Recursos Humanos.
- Inferencia ultra-rapida y bajo consumo de recursos en produccion.

---

## 4. Estructura del Repositorio

```text
mlzoomcamp-midterm-project/
├── data/
│   └── employee_attrition.csv   # Dataset original
├── notebook.ipynb                # Analisis exploratorio, modelado y tuning
├── train.py                     # Script para reentrenar y exportar model.bin
├── predict.py                   # Servicio web con FastAPI
├── test_predict.py              # Script cliente para probar inferencia
├── model.bin                    # Artefacto guardado (DictVectorizer + Modelo)
├── pyproject.toml               # Dependencias gestionadas con uv
├── uv.lock                      # Bloqueo determinista de dependencias
├── Dockerfile                   # Configuracion de contenedor
├── .gitignore                   # Exclusion de entornos y temporales
└── README.md                    # Documentacion del proyecto
```

---

## 5. Como Reproducir el Proyecto Localmente

### Requisitos previos
- Python 3.11 o 3.12 instalado.
- Gestor de paquetes `uv` instalado ([guia de instalacion](https://docs.astral.sh/uv/getting-started/installation/)):
  ```bash
  pip install uv
  ```

### 1. Clonar el repositorio y sincronizar el entorno
```bash
git clone <URL_DEL_REPOSITORIO>
cd mlzoomcamp-midterm-project
uv sync
```

### 2. Ejecutar el Notebook
Puedes iniciar Jupyter con `uv`:
```bash
uv run --with jupyter jupyter notebook
```
Y abrir `notebook.ipynb` para reproducir todos los graficos y entrenamientos.

### 3. Entrenar el modelo con el script
Para ejecutar el script de reentrenamiento que genera `model.bin`:
```bash
uv run python train.py
```

### 4. Iniciar el servicio web localmente
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

La API estara disponible en:
- Verificacion de salud: `http://localhost:9696/health`
- Documentacion interactiva Swagger: `http://localhost:9696/docs`

### 5. Probar la API local
En otra terminal ejecuta el script de prueba:
```bash
uv run python test_predict.py
```

O usando `curl`:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 29,
    "businesstravel": "Travel_Frequently",
    "dailyrate": 400,
    "department": "Sales",
    "distancefromhome": 25,
    "education": 2,
    "educationfield": "Marketing",
    "environmentsatisfaction": 1,
    "gender": "Female",
    "hourlyrate": 45,
    "jobinvolvement": 2,
    "joblevel": 1,
    "jobrole": "Sales Representative",
    "jobsatisfaction": 1,
    "maritalstatus": "Single",
    "monthlyincome": 2300,
    "monthlyrate": 12000,
    "numcompaniesworked": 5,
    "overtime": "Yes",
    "percentsalaryhike": 11,
    "performancerating": 3,
    "relationshipsatisfaction": 1,
    "stockoptionlevel": 0,
    "totalworkingyears": 4,
    "trainingtimeslastyear": 2,
    "worklifebalance": 1,
    "yearsatcompany": 2,
    "yearsincurrentrole": 1,
    "yearssincelastpromotion": 0,
    "yearswithcurrmanager": 1
  }'
```

Respuesta esperada:
```json
{
  "attrition_probability": 0.8142,
  "attrition": true,
  "risk_level": "Alto"
}
```

---

## 6. Despliegue con Docker

### 1. Construir la imagen Docker
```bash
docker build -t employee-attrition-app .
```

### 2. Ejecutar el contenedor
```bash
docker run --rm -p 9696:9696 employee-attrition-app
```

### 3. Probar el contenedor
En otra terminal ejecuta:
```bash
python test_predict.py
```
O realiza una peticion a `http://localhost:9696/health` para comprobar que el servicio esta activo.
