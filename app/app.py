import re
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Análise de Sentimentos - Mídias Sociais", layout="wide")

def carregar_css(arquivo):
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

carregar_css("app/style/style.css")

# Paleta Pastel Editorial (contraste escuro para fundo claro)
COR_TEXTO = "#2b1d2e"
COR_GRADE = "rgba(43, 29, 46, 0.08)"
PALETA_SENTIMENTOS = {
    "Positivo": "#e58ab0",
    "Neutro": "#b39ddb",
    "Negativo": "#7c588c",
}
PALETA_GRAFICO = ["#e58ab0", "#d4739d", "#b39ddb", "#9575cd", "#f2b5d0", "#7c588c"]

# 1. Pipeline de Pré-processamento PLN
STOPWORDS = set(ENGLISH_STOP_WORDS)

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+|#\w+", "", texto)
    texto = re.sub(r"[^a-z\s]", "", texto)
    tokens = texto.split()
    tokens_limpos = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens_limpos)

# 2. Carregamento do Corpus
@st.cache_data
def carregar_dados():
    df = pd.read_csv("data/Tweets.csv")
    mapa_sentimento = {
        "positive": "Positivo",
        "neutral": "Neutro",
        "negative": "Negativo",
    }
    df["sentimento"] = df["airline_sentiment"].map(mapa_sentimento)
    df["texto_limpo"] = df["text"].apply(limpar_texto)
    return df.dropna(subset=["sentimento", "texto_limpo"])

df = carregar_dados()

# 3. Treinamento Supervisionado (TF-IDF + Naive Bayes)
@st.cache_resource
def treinar_modelo(dados):
    X = dados["texto_limpo"]
    y = dados["sentimento"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.5)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acuracia = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=["Positivo", "Neutro", "Negativo"])
    return pipeline, acuracia, cm

modelo, acuracia, matriz_confusao = treinar_modelo(df)

# 4. Barra Lateral
st.sidebar.header("Social Listening")
marcas = ["Todas as Marcas"] + sorted(df["airline"].dropna().unique().tolist())
marca_selecionada = st.sidebar.selectbox("Filtrar Marca Monitorada", marcas)

df_filtrado = df if marca_selecionada == "Todas as Marcas" else df[df["airline"] == marca_selecionada]

st.sidebar.markdown("---")
st.sidebar.metric("Total de Menções (Corpus)", f"{len(df_filtrado):,}")
st.sidebar.caption(f"Acurácia do Modelo Naive Bayes: **{acuracia:.1%}**")

# 5. Cabeçalho Principal
st.title("Análise Preditiva de Sentimentos em Mídias Sociais")
st.markdown("Mineração de texto aplicada ao monitoramento de marca, classificação de polaridade e alerta de crises.")

aba1, aba2, aba3 = st.tabs([
    "Simulador em Tempo Real (PLN)",
    "Social Listening & Crises",
    "Matriz TF-IDF & Métricas",
])

# ==========================================
# ABA 1: SIMULADOR DE PREDIÇÃO EM TEMPO REAL
# ==========================================
with aba1:
    st.subheader("Classificar Novo Comentário ou Publicação")
    texto_usuario = st.text_area(
        "Digite ou cole uma publicação em inglês para testar o modelo:",
        value="@UnitedAirline My flight was delayed for 4 hours and nobody helped me at the gate. Terrible service!",
        height=100,
    )

    if st.button("Analisar Sentimento", type="primary") or texto_usuario:
        texto_processado = limpar_texto(texto_usuario)
        tokens_exibicao = texto_processado.split()

        predicao = modelo.predict([texto_processado])[0]
        probs = modelo.predict_proba([texto_processado])[0]
        classes = list(modelo.classes_)

        col_res1, col_res2 = st.columns([1, 1.4])

        with col_res1:
            confianca = max(probs) * 100
            st.metric("Polaridade Prevista", predicao, delta=f"{confianca:.1f}% de confiança")
            st.markdown("**Etapas de Pré-processamento (PLN):**")
            st.caption(f"Texto normalizado e sem stopwords: `{texto_processado}`")
            st.caption(f"Tokens extraídos ({len(tokens_exibicao)}): `{tokens_exibicao}`")

        with col_res2:
            df_prob = pd.DataFrame({
                "Sentimento": classes,
                "Probabilidade (%)": [p * 100 for p in probs],
            }).sort_values("Probabilidade (%)", ascending=True)

            fig_prob = px.bar(
                df_prob,
                x="Probabilidade (%)",
                y="Sentimento",
                orientation="h",
                color="Sentimento",
                color_discrete_map=PALETA_SENTIMENTOS,
                title="Distribuição de Probabilidade do Classificador",
            )
            fig_prob.update_layout(
                showlegend=False,
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COR_TEXTO),
                xaxis=dict(range=[0, 100], gridcolor=COR_GRADE),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

# ==========================================
# ABA 2: SOCIAL LISTENING & GESTÃO DE CRISES
# ==========================================
with aba2:
    c1, c2, c3 = st.columns(3)
    pct_neg = (df_filtrado["sentimento"] == "Negativo").mean() * 100
    pct_pos = (df_filtrado["sentimento"] == "Positivo").mean() * 100
    pct_neu = (df_filtrado["sentimento"] == "Neutro").mean() * 100

    c1.metric("Menções Positivas", f"{pct_pos:.1f}%")
    c2.metric("Menções Neutras", f"{pct_neu:.1f}%")
    c3.metric("Alerta de Crise (Negativas)", f"{pct_neg:.1f}%", delta="Monitoramento ativo", delta_color="inverse")

    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Polaridade por Marca")
        df_marca = df.groupby(["airline", "sentimento"], as_index=False).size()
        fig_marca = px.bar(
            df_marca,
            x="airline",
            y="size",
            color="sentimento",
            barmode="group",
            color_discrete_map=PALETA_SENTIMENTOS,
            labels={"airline": "Marca", "size": "Qtd. Publicações", "sentimento": "Sentimento"},
        )
        fig_marca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COR_TEXTO),
            xaxis=dict(gridcolor=COR_GRADE),
            yaxis=dict(gridcolor=COR_GRADE),
        )
        st.plotly_chart(fig_marca, use_container_width=True)

    with g2:
        st.subheader("Principais Motivos de Reclamação")
        df_motivos = (
            df_filtrado["negativereason"]
            .dropna()
            .value_counts()
            .head(6)
            .reset_index()
        )
        df_motivos.columns = ["Motivo", "Ocorrências"]
        fig_motivos = px.bar(
            df_motivos.sort_values("Ocorrências"),
            x="Ocorrências",
            y="Motivo",
            orientation="h",
            color="Motivo",
            color_discrete_sequence=PALETA_GRAFICO,
        )
        fig_motivos.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COR_TEXTO),
            xaxis=dict(gridcolor=COR_GRADE),
        )
        st.plotly_chart(fig_motivos, use_container_width=True)

# ==========================================
# ABA 3: BASTIDORES DO MODELO (TF-IDF)
# ==========================================
with aba3:
    m1, m2 = st.columns(2)

    with m1:
        st.subheader("Matriz de Confusão (Dados de Teste)")
        fig_cm = px.imshow(
            matriz_confusao,
            text_auto=True,
            x=["Positivo", "Neutro", "Negativo"],
            y=["Positivo", "Neutro", "Negativo"],
            labels=dict(x="Previsto pelo Modelo", y="Real"),
            color_continuous_scale=["#f9eff4", "#b39ddb", "#c85284"],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COR_TEXTO),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with m2:
        st.subheader("Termos de Maior Peso (TF-IDF) por Classe")
        classe_escolha = st.selectbox("Selecione a polaridade:", ["Positivo", "Neutro", "Negativo"])
        vectorizer = modelo.named_steps["tfidf"]
        clf = modelo.named_steps["clf"]
        idx_classe = list(clf.classes_).index(classe_escolha)
        nomes_termos = vectorizer.get_feature_names_out()
        pesos = clf.feature_log_prob_[idx_classe]

        top_indices = pesos.argsort()[-10:]
        df_termos = pd.DataFrame({
            "Termo": [nomes_termos[i] for i in top_indices],
            "Peso Log-Prob": [pesos[i] for i in top_indices],
        })
        fig_termos = px.bar(
            df_termos,
            x="Peso Log-Prob",
            y="Termo",
            orientation="h",
            color_discrete_sequence=["#d4739d"],
        )
        fig_termos.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COR_TEXTO),
            xaxis=dict(gridcolor=COR_GRADE),
        )
        st.plotly_chart(fig_termos, use_container_width=True)
