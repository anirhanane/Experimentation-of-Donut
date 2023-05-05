import streamlit as st
import pandas as pd
import gdown
import os

@st.cache(allow_output_mutation=True)
def load_document():
        embeddings = ""
        return embeddings

@st.cache(allow_output_mutation=True)
def load_dataset():
        df = ""
        return df

def create_html(result):
    output = f""
    spans = []
    for token, score in result["tokens"]:
        color = None
        if score >= 0.1:
            color = "#fdd835"
        elif score >= 0.075:
            color = "#ffeb3b"
        elif score >= 0.05:
            color = "#ffee58"
        elif score >= 0.02:
            color = "#fff59d"

        spans.append((token, score, color))

    if result["score"] >= 0.05 and not [color for _, _, color in spans if color]:
        mscore = max([score for _, score, _ in spans])
        spans = [(token, score, "#fff59d" if score == mscore else color) for token, score, color in spans]

    for token, _, color in spans:
        if color:
            output += f"<span style='background-color: {color}'>{token}</span> "
        else:
            output += f"{token} "
    return output



def main():
    print("hello")
    st.set_page_config(layout="wide", page_title="Document Understading")
    with open('css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.title("Document Understading")
    st.image("images/01_Logo_datavalue_small.png")
    st.sidebar.image("images/01_Logo_datavalue_small.png")
    st.sidebar.markdown("Developed by Chenaf Anir Hanane using [Streamlit](https://www.streamlit.io)", unsafe_allow_html=True)
    st.sidebar.markdown("Current Version: 0.0.1")
    query = st.sidebar.text_input("Query")
    #num_results = st.sidebar.number_input("Number of Results", 1, 2000, 20)
    #ignore_search_words = st.sidebar.checkbox("Ignore Search Words")
    # provide options to either do extraction of information  or VQA
    parsing_tab, vqa_tab = st.tabs(["Parsing", "VQA"])
    embeddings = load_document()
    df = load_dataset()
    if st.sidebar.button("Apply"):
        st.markdown("Done", unsafe_allow_html=True)
        #res = embeddings.explain(query, limit = num_results)
        """html_txt = [create_html(r) for r in res]
        indices = [int(index['id']) for index in res]
        scores = [index['score'] for index in res]
        texts = [index['text'] for index in res]
        y = df.iloc[indices]
        y['PlayerLine'] = html_txt
        # y = df[df.index.isin(indices)].drop(["Dataline", "PlayerLinenumber"], axis=1)

        y["similarity"] = scores
        y = y.drop(["Dataline"], axis=1)
        if ignore_search_words == True:
            words = query.split()
            for word in words:
                y = y[~y["PlayerLine"].str.contains(word.lower(), case=False)]
        st.markdown(y.to_markdown(), unsafe_allow_html=True) """


if __name__=="__main__":
    main()
