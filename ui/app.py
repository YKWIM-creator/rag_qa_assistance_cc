import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CUNY Assistant", page_icon="🎓", layout="centered")
st.title("🎓 CUNY Student Assistant")
st.caption("Ask questions about CUNY programs, admissions, financial aid, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"- [{s.get('title', s['url'])}]({s['url']}) — *{s.get('school', '')}*")

if prompt := st.chat_input("Ask a question about CUNY..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching CUNY documents..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": prompt}, timeout=30)
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.exceptions.ConnectionError:
                answer = "Could not connect to the CUNY assistant API. Make sure the backend is running."
                sources = []
            except Exception as e:
                answer = f"An error occurred: {str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s.get('title', s['url'])}]({s['url']}) — *{s.get('school', '')}*")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
