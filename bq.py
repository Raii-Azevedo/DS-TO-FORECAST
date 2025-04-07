import pandas as pd
from prophet import Prophet
from google.cloud import bigquery
from google.oauth2 import service_account

# Autenticação
credentials = service_account.Credentials.from_service_account_file(
    "caminho/para/credencial.json"
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Trazer histórico
query = """
SELECT 
  DATE as ds,
  VALUE as y
FROM `br-in-finance.dataset_original.tabela_original`
WHERE VALUE IS NOT NULL
"""
df = client.query(query).to_dataframe()
df['ds'] = pd.to_datetime(df['ds'])

# Treinamento e previsão
model = Prophet()
model.fit(df[["ds", "y"]])

future = model.make_future_dataframe(periods=12, freq='M')  # previsões para 12 meses
forecast = model.predict(future)

# Filtrar apenas os valores futuros
last_date = df["ds"].max()
forecast_filtered = forecast[forecast["ds"] > last_date]

# Selecionar apenas as colunas relevantes
final_df = forecast_filtered[["ds", "yhat", "yhat_upper", "yhat_lower"]]

# Enviar para o BigQuery
table_id = "br-in-finance.dataset_previsoes.previsao_demand_sensing"
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(final_df, table_id, job_config=job_config)
job.result()

print("Previsão futura enviada com sucesso para o BigQuery!")
