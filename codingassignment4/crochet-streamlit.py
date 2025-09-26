def main():
    import streamlit as st
    import pandas as pd
    from dotenv import load_dotenv
    from supabase import create_client
    import os
    import matplotlib.pyplot as plt
    import nltk
    from nltk.corpus import stopwords
    import re
    from sklearn.cluster import KMeans  
    from sklearn.decomposition import PCA
    import seaborn as sns
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import TfidfVectorizer


    st.title("Crochet Pattern Webscraping")
    st.write("This dashboard displays crochet patterns scraped from Lovecrafts.")
    st.write("Author: Hannah Wilson")

    # displaying the data table
    st.header("Crochet Patterns Data")
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    supabase = create_client(url, key)

    response = supabase.table('crochetpattern').select("*").execute()
    patterns = pd.DataFrame(response.data)
    st.dataframe(patterns)

    # displaying distribution of languages

    patterns["languages"] = patterns["subtitle"].str.replace("Downloadable PDF", "", regex=False).str.strip()
    patterns_split = patterns["languages"].str.split(", ").explode()
    patterns_split = patterns_split[patterns_split != ""]
    language_counts = patterns_split.value_counts()

    st.subheader("Number of patterns by language")
    fig, ax = plt.subplots()
    ax.bar(language_counts.index, language_counts.values)
    ax.set_ylabel("Number of Patterns")
    ax.set_xlabel("Languages")
    ax.set_title("Distribution of Languages in Crochet Patterns")
    ax.set_xticks(range(len(language_counts)), language_counts.index, rotation=45, ha="right")

    st.pyplot(fig)

    # clustering pattern titles
    # preprocess data

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    vectorizer = TfidfVectorizer()
    patterns['cleaned_pattern_name'] = patterns['pattern_name'].apply(preprocess)
    

    k = 4
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    X = vectorizer.fit_transform(patterns['cleaned_pattern_name'])
    patterns['kmeans_cluster'] = kmeans_model.fit_predict(X)

    def get_top_words(model, vectorizer, n_terms=5):
        terms = vectorizer.get_feature_names_out()
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        for i in range(model.n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :n_terms]]
            print(f"Cluster {i}: {' '.join(top_terms)}")
            for idx in order_centroids[i, :n_terms]:
                print(f"{terms[idx]}")

    get_top_words(kmeans_model, vectorizer)
    
    pcs = PCA(n_components=2)
    x_pca = pcs.fit_transform(X.toarray())
    st.subheader("KMeans Clustering of Pattern Names")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=patterns["kmeans_cluster"], palette="Set2", ax=ax)
    ax.set_title("KMeans Clustering of Crochet Pattern Names")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

    def generate_word_cloud(cluster_data, cluster_num):
        cluster_docs = patterns[patterns["kmeans_cluster"] == cluster_num]["cleaned_pattern_name"]
        text = ' '.join(cluster_docs)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Word Cloud for Cluster {cluster_num}")
        st.pyplot(fig)

    st.subheader("Word Clouds for Each Cluster")
    for i in range(k):
        generate_word_cloud(patterns, i)
    
    # creating chart of most common words in pattern titles
    clean_name = patterns['cleaned_pattern_name']
    patterns_split = clean_name.str.split().explode()
    word_counts = patterns_split.value_counts().head(20)
    st.subheader("Most Common Words in Pattern Names")
    fig, ax = plt.subplots()
    ax.bar(word_counts.index, word_counts.values)
    ax.set_ylabel("Count")
    ax.set_xlabel("Words")
    ax.set_title("Top 20 Most Common Words in Crochet Pattern Names")
    ax.set_xticks(range(len(word_counts)), word_counts.index, rotation=45, ha="right")
    st.pyplot(fig)

if __name__ == "__main__":
    main()