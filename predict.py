import pickle
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ruta del modelo entrenado
MODEL_PATH = Path("model.bin")

# Cargar vectorizador y modelo
with open(MODEL_PATH, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API para predecir el riesgo de renuncia o abandono de empleados (ML Zoomcamp Midterm Project)",
    version="1.0.0"
)

# Esquema de datos de entrada
class Employee(BaseModel):
    age: int = Field(default=35, description="Edad del empleado")
    businesstravel: str = Field(default="Travel_Rarely", description="Frecuencia de viajes laborales")
    dailyrate: int = Field(default=800, description="Tarifa diaria")
    department: str = Field(default="Research & Development", description="Departamento")
    distancefromhome: int = Field(default=5, description="Distancia al trabajo en km")
    education: int = Field(default=3, description="Nivel educativo (1-5)")
    educationfield: str = Field(default="Life Sciences", description="Campo de estudio")
    environmentsatisfaction: int = Field(default=3, description="Satisfaccion con el entorno (1-4)")
    gender: str = Field(default="Male", description="Genero")
    hourlyrate: int = Field(default=65, description="Tarifa por hora")
    jobinvolvement: int = Field(default=3, description="Nivel de involucramiento en el trabajo (1-4)")
    joblevel: int = Field(default=2, description="Nivel de puesto (1-5)")
    jobrole: str = Field(default="Research Scientist", description="Rol de trabajo")
    jobsatisfaction: int = Field(default=3, description="Satisfaccion laboral (1-4)")
    maritalstatus: str = Field(default="Single", description="Estado civil")
    monthlyincome: int = Field(default=5000, description="Ingreso mensual")
    monthlyrate: int = Field(default=15000, description="Tarifa mensual")
    numcompaniesworked: int = Field(default=2, description="Numero de empresas previas")
    overtime: str = Field(default="No", description="Horas extras (Yes/No)")
    percentsalaryhike: int = Field(default=14, description="Porcentaje de aumento salarial")
    performancerating: int = Field(default=3, description="Calificacion de desempeno (1-4)")
    relationshipsatisfaction: int = Field(default=3, description="Satisfaccion en relaciones (1-4)")
    stockoptionlevel: int = Field(default=1, description="Nivel de opciones sobre acciones (0-3)")
    totalworkingyears: int = Field(default=10, description="Anos totales trabajados")
    trainingtimeslastyear: int = Field(default=3, description="Veces capacitado el ultimo ano")
    worklifebalance: int = Field(default=3, description="Balance vida-trabajo (1-4)")
    yearsatcompany: int = Field(default=5, description="Anos en la empresa")
    yearsincurrentrole: int = Field(default=3, description="Anos en el rol actual")
    yearssincelastpromotion: int = Field(default=1, description="Anos desde la ultima promocion")
    yearswithcurrmanager: int = Field(default=3, description="Anos con el gerente actual")

@app.get("/health")
def health() -> Dict[str, str]:
    # Endpoint de verificacion de estado
    return {"status": "ok", "model": "LogisticRegression", "dataset": "IBM HR Employee Attrition"}

@app.post("/predict")
def predict(employee: Employee) -> Dict[str, Any]:
    # Normalizar registro a minusculas
    record = {k.lower(): v for k, v in employee.model_dump().items()}
    
    # Transformar caracteristicas y predecir
    X = dv.transform([record])
    probability = float(model.predict_proba(X)[0, 1])
    will_leave = bool(probability >= 0.5)
    
    return {
        "attrition_probability": round(probability, 4),
        "attrition": will_leave,
        "risk_level": "Alto" if probability >= 0.5 else ("Medio" if probability >= 0.25 else "Bajo")
    }
