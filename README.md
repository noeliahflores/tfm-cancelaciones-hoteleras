# Demo Streamlit - TFM Cancelaciones Hoteleras

Esta carpeta contiene la demo interactiva del TFM sobre predicción de cancelaciones hoteleras.

## Estructura

```text
demo_streamlit/
├── app.py
├── requirements.txt
├── modelo/
│   ├── modelo_cancelaciones.pkl
│   ├── preprocesador_cancelaciones.pkl
│   ├── variables_modelo.pkl
│   ├── umbral_decision.pkl
│   └── metadata_modelo.json
└── datos/
    ├── hotel_bookings_dashboard.csv
    ├── resumen_economico.csv
    ├── impacto_por_hotel.csv
    ├── impacto_modelo.csv
    ├── overbooking_por_hotel.csv
    ├── escenarios_retorno.csv
    └── metricas_modelos.csv
```

## Cómo ejecutarla en local

Desde la carpeta `demo_streamlit`, ejecutar:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Qué incluye la demo

1. Dashboard de análisis hotelero.
2. Predicción de cancelación en tiempo real.
3. Estimación económica y simulador de overbooking controlado.

## Nota

Los archivos `.pkl` proceden del notebook final del TFM. Por eso se fijan versiones concretas en `requirements.txt`, especialmente `scikit-learn`, para evitar problemas al cargar el preprocesador.
