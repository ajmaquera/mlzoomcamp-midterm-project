import requests

# URL del servicio web local
URL = "http://localhost:9696/predict"

# Ejemplo 1: Empleado con alto riesgo de abandono (horas extras, baja satisfaccion, soltero)
high_risk_employee = {
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
}

# Ejemplo 2: Empleado con bajo riesgo de abandono (estable, buena satisfaccion, sin horas extras)
low_risk_employee = {
    "age": 45,
    "businesstravel": "Non-Travel",
    "dailyrate": 1100,
    "department": "Research & Development",
    "distancefromhome": 3,
    "education": 4,
    "educationfield": "Life Sciences",
    "environmentsatisfaction": 4,
    "gender": "Male",
    "hourlyrate": 80,
    "jobinvolvement": 4,
    "joblevel": 3,
    "jobrole": "Healthcare Representative",
    "jobsatisfaction": 4,
    "maritalstatus": "Married",
    "monthlyincome": 9500,
    "monthlyrate": 18000,
    "numcompaniesworked": 1,
    "overtime": "No",
    "percentsalaryhike": 18,
    "performancerating": 3,
    "relationshipsatisfaction": 4,
    "stockoptionlevel": 2,
    "totalworkingyears": 18,
    "trainingtimeslastyear": 4,
    "worklifebalance": 3,
    "yearsatcompany": 12,
    "yearsincurrentrole": 8,
    "yearssincelastpromotion": 3,
    "yearswithcurrmanager": 8
}

def run_tests():
    print("Probando caso de alto riesgo de abandono:")
    try:
        res_high = requests.post(URL, json=high_risk_employee, timeout=5)
        print("  Respuesta:", res_high.json())
    except Exception as e:
        print(f"  Error de conexion (asegurate de que el servidor este corriendo): {e}")

    print("\nProbando caso de bajo riesgo de abandono:")
    try:
        res_low = requests.post(URL, json=low_risk_employee, timeout=5)
        print("  Respuesta:", res_low.json())
    except Exception as e:
        print(f"  Error de conexion: {e}")

if __name__ == "__main__":
    run_tests()
