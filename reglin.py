import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Previsão de Custo Inicial", layout="wide")
st.title("Previsão Inicial de Custo para Franquia")
st.write("Informe o valor anual da franquia para calcular uma estimativa do custo inicial.")

# Carregar dados
dados = pd.read_csv("slr12.csv", sep=";")

# Treinar modelo de regressão
X = dados[['FrqAnual']]
y = dados['CusInic']
modelo = LinearRegression().fit(X, y)

# Adicionando um botão para exibir a explicação sobre o relatório
with st.expander("ℹ️ Como funciona este relatório?", expanded=False):
    st.markdown(
        """
        Este relatório foi desenvolvido para fornecer uma estimativa do custo inicial para abrir uma franquia 
        com base no valor anual informado pelo usuário. O modelo de regressão linear utilizado é treinado com 
        dados históricos e permite prever o custo inicial necessário.

        **Funcionamento:**
        1. O usuário insere o **valor anual** da franquia.
        2. O modelo de regressão faz uma **previsão** do custo inicial com base nos dados fornecidos.
        3. Um **gráfico de dispersão** exibe a relação entre o valor anual e o custo inicial para ilustrar os dados.

        """
    )

# Criando duas colunas para a Tabela de Dados e o Gráfico
st.write("---")  # Separador para organização visual
col1, col2 = st.columns(2, gap="large")  # Aumentando o espaçamento entre colunas

# --- Bloco: Tabela de Dados ---
with col1:
    st.markdown(
        """
        <div style='border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9;'>
            <h3 style='text-align: center; margin-bottom: 10px; color: black;'>Tabela de Dados</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(dados, height=400, use_container_width=True)  # Exibindo a tabela com rolagem e altura fixa

# --- Bloco: Gráfico de Dispersão ---
with col2:
    st.markdown(
        """
        <div style='border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9;'>
            <h3 style='text-align: center; margin-bottom: 10px; color: black;'>Gráfico de Dispersão</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(5, 4))  # Definindo o tamanho do gráfico para coincidir com a tabela
    ax.scatter(X, y, color='blue', label='Dados Originais')
    ax.plot(X, modelo.predict(X), color='red', label='Linha de Regressão')
    ax.set_xlabel('Valor Anual (R$)')
    ax.set_ylabel('Custo Inicial (R$)')
    ax.legend()
    st.pyplot(fig)

# --- Bloco: Entrada do Valor Anual e Previsão ---
st.write("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.header("Valor Anual da Franquia:")
    novo_valor = st.number_input("Insira Novo Valor", min_value=1.0, max_value=999999.0, value=1500.0, step=0.01)

    if st.button("Calcular Previsão", key="btn_previsao"):
        dados_novo_valor = pd.DataFrame([[novo_valor]], columns=['FrqAnual'])
        prev = modelo.predict(dados_novo_valor)
        st.success(f"A previsão do custo inicial é de **R$ {prev[0]:,.2f}**")

# --- Rodapé ---
st.write("---")
st.markdown(
    """
    <div style='text-align: left; margin-top: 10px; line-height: 1.0;'>
        <p style='font-size: 16px; font-weight: bold; margin: 0 0 8px 0;'>Projeto: Prevendo Custos para Abrir Franquia (Regressão)</p>
        <p style='font-size: 14px; margin: 0 0 5px 0;'>Desenvolvido por:</p>
        <p style='font-size: 20px; color: #4CAF50; font-weight: bold; margin: 0;'>Cláudio Ferreira Neves</p>
        <p style='font-size: 16px; color: #555; margin: 0;'>Especialista em Análise de Dados, RPA e AI</p>
        <p style='font-size: 14px; margin: 10px 0 5px 0;'>Ferramentas utilizadas: Python, Streamlit, Pandas, Scikit-learn, Matplotlib</p>
        <p style='font-size: 12px; color: #777; margin: 0;'>© 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
