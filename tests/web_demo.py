import json
import streamlit as st
import requests

url = 'http://192.168.101.30:8081/chat'

histroy = []

data = {
    "history": [],
    "query": "",
    "model": "Baichuan-13B-Chat",
    "params": {"max_new_tokens": 1024,"do_sample": False,"temperature": 0.65,"top_p": 0.8,"repetition_penalty": 1.01}
}

def gen_chat(data):
    
    try:
        r = requests.post(url,json=data).json()
        
        return r["runtime"], r["result"], r["msg"]
    except Exception as e:
        print(e)
        
        return "","",""


st.set_page_config(page_title="Accelerated infer" + data["model"])
st.title(data["model"])

def clear_chat_history():
    data["history"] = []
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是" + data["model"] + "，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        #print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            
            q,a = "",""
            for his_ in messages:
                if his_["role"] == "user":
                    q = his_["content"]
                else:
                    a = his_["content"]
                    data["history"].append(dict( (("q",q),("a",a)) ))
                            
            data["query"] = prompt
            print(data)
            runtime_,result_,msg_ = gen_chat(data)
            placeholder.markdown(f"time:{runtime_}\r\n{result_}")
        messages.append({"role": "assistant", "content": result_})
        #print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
