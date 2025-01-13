
# Aplicação de Previsão P&L usando Prophet

Este repositório contém um código para realizar análises preditivas de P&L (Profit and Loss) utilizando a biblioteca Prophet. O objetivo é gerar previsões a partir de dados históricos fornecidos pelo usuário e apresentar os resultados de forma visual e interativa com Streamlit.

## Objetivo
A aplicação permite:

1. Fazer upload de um arquivo contendo dados históricos (em formatos `.csv` ou `.xlsx`).
2. Gerar previsões de valores futuros com base nos dados carregados.
3. Visualizar as previsões em gráficos interativos e tabelas estilizadas.

---

## Fluxo do Código

### 1. **Upload do Arquivo**
O usuário pode fazer upload de arquivos `.csv` ou `.xlsx`. O código valida o formato e carrega os dados para análise.

- Exemplo de arquivo de entrada esperado:

```
| Data       | Valor |
|------------|-------|
| 01/01/2023 | 100   |
| 01/02/2023 | 150   |
| 01/03/2023 | 200   |
```

### 2. **Configuração do Forecast**
- O código permite ao usuário selecionar as colunas correspondentes à data e aos valores.
- Os dados históricos são preparados para o Prophet:
  - A coluna de data é renomeada para `ds`.
  - A coluna de valores é renomeada para `y`.

### 3. **Modelo Prophet**
- O Prophet é treinado com os dados históricos para gerar previsões.
- O modelo calcula valores de previsão (`yhat`) e intervalos de confiança (`yhat_upper` e `yhat_lower`).

### 4. **Preenchimento de Valores**
- Caso já existam valores históricos para certas datas, eles são mantidos.
- O forecast é realizado apenas para datas futuras ou lacunas nos dados.

### 5. **Gráficos e Tabelas**
- **Gráfico de Linhas:** Mostra os dados históricos, previsões e intervalos de confiança.
- **Tabela Estilizada:** Apresenta os valores previstos com destaque visual.

---

## Estrutura do Código

### Importação de Bibliotecas
O código utiliza:
- `streamlit` para criar a interface web.
- `pandas` para manipulação de dados.
- `prophet` para o modelo de previsão.
- `plotly` para visualização interativa.

### Fluxo Principal
1. **Carregamento de Dados**
   ```python
   if uploaded_file:
       if uploaded_file.name.endswith('.csv'):
           data = pd.read_csv(uploaded_file)
       elif uploaded_file.name.endswith(('.xlsx', '.xls')):
           data = pd.read_excel(uploaded_file)
   ```

2. **Configuração e Treinamento do Modelo**
   ```python
   forecast_data = data[[date_column, value_column]].rename(
       columns={date_column: "ds", value_column: "y"}
   )
   model = Prophet()
   model.fit(forecast_data)
   ```

3. **Preenchimento de Valores**
   ```python
   forecast['y_forecast'] = forecast.apply(
       lambda row: row['y'] if row['ds'] <= last_date else row['yhat'], axis=1
   )
   ```

4. **Gráficos**
   - Histórico: linha azul contínua.
   - Previsão: linha verde com intervalos de confiança.

5. **Tabela de Resultados**
   ```python
   st.dataframe(
       forecast_table.style.apply(color_forecast, axis=0).format("{:.2f}")
   )
   ```

---

## Como Usar

### Pré-requisitos
- Python 3.8+
- Dependências listadas no arquivo `requirements.txt`:

  ```
  streamlit
  pandas
  prophet
  plotly
  ```

### Passos
1. Instale as dependências com:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o aplicativo:
   ```bash
   streamlit run app.py
   ```
3. Faça o upload do arquivo de dados e visualize os resultados.

---

## Observações
- O código assume que as datas no arquivo de entrada estão no formato `DD/MM/AAAA`.
- Certifique-se de que o arquivo de entrada contém uma coluna de datas e uma de valores.

---

## Author
Raíssa Azevedo
