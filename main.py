from google.cloud import bigquery
from datetime import timedelta
import pandas as pd
from prophet import Prophet

def forecast_bigquery(request):
    client = bigquery.Client()

    # 1. Lê os dados do BigQuery
    QUERY = """
        SELECT DATE, VALUE
        FROM `seu_projeto.sua_dataset.sua_tabela_historico`
        WHERE VALUE IS NOT NULL
    """
    df = client.query(QUERY).to_dataframe()
    df.columns = ['ds', 'y']  # renomeia para Prophet

    # 2. Previsão com Prophet
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)  # 30 dias de forecast
    forecast = model.predict(future)

    # 3. Seleciona colunas importantes
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    result = result[result['ds'] > df['ds'].max()]  # somente dados futuros

    # 4. Escreve de volta no BigQuery
    result['ds'] = pd.to_datetime(result['ds'])
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("ds", "DATE"),
            bigquery.SchemaField("yhat", "FLOAT"),
            bigquery.SchemaField("yhat_lower", "FLOAT"),
            bigquery.SchemaField("yhat_upper", "FLOAT")
        ]
    )
    table_id = "seu_projeto.sua_dataset.forecast_result"
    job = client.load_table_from_dataframe(result, table_id, job_config=job_config)
    job.result()

    return 'Forecast atualizado com sucesso!'
