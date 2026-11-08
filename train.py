import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Configuracion de rutas
DATA_PATH = Path("data/employee_attrition.csv")
MODEL_PATH = Path("model.bin")

def load_and_preprocess_data(path: Path):
    # Cargar dataset
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    
    # Crear variable objetivo binaria (Yes: 1, No: 0)
    df["target"] = (df["attrition"] == "Yes").astype(int)
    
    # Eliminar columnas irrelevantes o constantes
    drop_cols = ["attrition", "employeecount", "standardhours", "over18", "employeenumber"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df

def train():
    print("Iniciando entrenamiento del modelo de Employee Attrition...")
    df = load_and_preprocess_data(DATA_PATH)
    
    # Division en train y test (80% train, 20% test estratificado)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    
    y_train = df_train.pop("target").values
    y_test = df_test.pop("target").values
    
    train_dicts = df_train.to_dict(orient="records")
    test_dicts = df_test.to_dict(orient="records")
    
    # Vectorizacion de caracteristicas
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)
    
    # Modelo final seleccionado tras ajuste de hiperparametros
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluacion en conjunto de prueba
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Metricas en conjunto de test:")
    print(f"  - ROC-AUC:  {auc:.4f}")
    print(f"  - Accuracy: {acc:.4f}")
    
    # Guardar artefacto del modelo y vectorizador
    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")

if __name__ == "__main__":
    train()
