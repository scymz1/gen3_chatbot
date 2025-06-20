from fastapi import FastAPI, Body
from pydantic import BaseModel
from llama_cpp import Llama
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import json
import os
from datetime import datetime
from uuid import uuid4
import requests
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableMap
from langchain_core.output_parsers import StrOutputParser


# app = FastAPI()
app = FastAPI(root_path="/chatbot")


hostname = os.environ.get("HOSTNAME", "localhost")
frontend_origin = f"https://{hostname}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin,
        "http://localhost:3000",  # 本地开发环境 React/Vite 常用端口
        "http://127.0.0.1:3000",
        "http://localhost",       # 通配情况
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# llm = Llama.from_pretrained(
# 	repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
# 	filename="Meta-Llama-3-8B-Instruct.Q2_K.gguf",
# )
# Init model
# llm = Llama(
#     model_path="./model/Meta-Llama-3-8B-Instruct-Q2_K.gguf",
#     n_ctx=2048,
#     chat_format="llama-3"
# )


# Init DB
engine = create_engine("sqlite:///chat_history.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_histories"
    conversation_id = Column(String, primary_key=True)
    user_id = Column(String, primary_key=True)
    messages_json = Column(Text)

class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id = Column(String, primary_key=True)
    user_id = Column(String)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Input format
class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    question: str

def truncate_history(messages, max_turns=5):
    """
    从对话记录中截取最近 max_turns 个 user + assistant 对
    系统消息保留
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    dialogue_msgs = [m for m in messages if m["role"] != "system"]
    truncated = dialogue_msgs[-max_turns * 2:]  # user + assistant
    return system_msgs + truncated

@app.post("/chat")
def chat(req: ChatRequest):
    db = SessionLocal()

    # 读取历史消息
    history = db.query(ChatHistory).filter_by(
        user_id=req.user_id,
        conversation_id=req.conversation_id
    ).first()

    if history:
        messages = json.loads(history.messages_json)
    else:
        messages = [{"role": "system", "content": "You are a helpful assistant for biomedical researchers."}]

    messages.append({"role": "user", "content": req.question})
    messages = truncate_history(messages, max_turns=5)

    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])

    # ✅ 加载 Guppy 数据
    try:
        case_df = pd.read_json("case_data.json")
    except Exception as e:
        print("Warning: failed to load case_data.json:", e)
        case_df = pd.DataFrame()

    try:
        follow_df = pd.read_json("follow_up_data.json")
    except Exception as e:
        print("Warning: failed to load follow_up_data.json:", e)
        follow_df = pd.DataFrame()

    llm = ChatOllama(model="llama3", temperature=0)

    # Prompt 用于分类问题类型
    router_prompt = ChatPromptTemplate.from_template(
        """You are a biomedical assistant. 
    Classify the user question into one of the following categories:
    - structured_query: if it's about patient data, case status, follow-up info, statistics, or anything derived from a dataset
    - llm_chat: if it's a general biomedical or health knowledge question

    Question: {input}
    Respond only with "structured_query" or "llm_chat".
    """
    )

    # LLM 做分类
    router_chain = router_prompt | llm | StrOutputParser()

    # 两个子链
    structured_chain = (
        PromptTemplate.from_template(
            "You are a biomedical data analyst. Use the following structured data to answer the user's question.\n\n"
            "Case data sample:\n{case_preview}\n\n"
            "Follow-up data sample:\n{follow_preview}\n\n"
            "Conversation history:\n{history}\n\n"
            "Question: {input}"
        )
        .partial(
            case_preview=case_df.head(5).to_string(),
            follow_preview=follow_df.head(5).to_string(),
            history=history_str
        )
        | llm
        | StrOutputParser()
    )


    # 用户当前提问
    current_question = messages[-1]["content"]

    chat_prompt = PromptTemplate.from_template(
        "You are a helpful biomedical assistant.\n"
        "Conversation history:\n{history}\n\n"
        "Now continue the conversation. User just asked:\n{input}"
    )

    llm_chat_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt.partial(history=history_str)
    )

    # 分类后执行对应子链
    full_chain = RunnableMap({
        "input": lambda x: x["input"],
        "category": lambda x: router_chain.invoke({"input": x["input"]}),
    }) | RunnableBranch(
        (lambda x: x["category"] == "structured_query", structured_chain),
        llm_chat_chain
    )

    # 执行 chain（替代原来的 router_chain.run(...)）
    answer = full_chain.invoke({"input": req.question})
    messages.append({"role": "assistant", "content": answer})

    # ✅ 写入历史
    if history:
        history.messages_json = json.dumps(messages)
    else:
        db.add(ChatHistory(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            messages_json=json.dumps(messages)
        ))

    db.commit()
    db.close()

    return {"response": answer}

@app.post("/new_conversation")
def new_conversation(data: dict):
    db = SessionLocal()
    conversation_id = str(uuid4())
    title = f"Chat on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    user_id = data["user_id"]
    conv = Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        title=title
    )
    db.add(conv)
    db.commit()
    db.close()
    return {
        "conversation_id": conversation_id,
        "title": title,
        "time": datetime.now().strftime('%Y-%m-%d %H:%M')
    }

@app.get("/conversations")
def get_conversations(user_id: str):
    db = SessionLocal()
    convs = db.query(Conversation).filter_by(user_id=user_id).order_by(Conversation.created_at.desc()).all()
    db.close()
    return [
        {
            "conversation_id": c.conversation_id,
            "title": c.title,
            "time": c.created_at.strftime('%Y-%m-%d %H:%M')
        }
        for c in convs
    ]

@app.get("/history")
def get_history(user_id: str, conversation_id: str):
    db = SessionLocal()
    history = db.query(ChatHistory).filter_by(
        user_id=user_id,
        conversation_id=conversation_id
    ).first()
    db.close()
    if history:
        return {"messages": json.loads(history.messages_json)}
    else:
        return {"messages": [
            {"role": "system", "content": "You are a helpful assistant for biomedical researchers."}
        ]}

@app.post("/rename_conversation")
def rename_conversation(data: dict = Body(...)):
    db = SessionLocal()
    conv = db.query(Conversation).filter_by(
        user_id=data["user_id"],
        conversation_id=data["conversation_id"]
    ).first()
    if conv:
        conv.title = data["title"]
        db.commit()
    db.close()
    return {"success": True}

@app.middleware("http")
async def log_path(request, call_next):
    print(">>> Request path:", request.url.path)
    return await call_next(request)
