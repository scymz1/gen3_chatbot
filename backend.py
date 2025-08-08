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
from fastapi.responses import StreamingResponse
from time import sleep
import re
from typing import List, Optional
import io
import contextlib
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # 或 logging.DEBUG 以查看更多日志
    stream=sys.stdout,   # 输出到 stdout，这样 kubectl logs 才能抓到
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    guppy_access: Optional[List[str]] = []  # ✅ 新增字段

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

    # 加载数据
    try:
        case_df = pd.read_json("case_data.json")
    except:
        case_df = pd.DataFrame()

    try:
        follow_df = pd.read_json("follow_up_data.json")
    except:
        follow_df = pd.DataFrame()

    if req.guppy_access:
        # 提取 "Project_demo" 格式的 project_id
        allowed_project_ids = []
        for path in req.guppy_access:
            match = re.match(r"^/programs/([^/]+)/projects/([^/]+)$", path)
            if match:
                program, project = match.groups()
                allowed_project_ids.append(f"{program}-{project}")
        print('allowed_project_ids', allowed_project_ids)
        if not case_df.empty and 'project_id' in case_df.columns:
            case_df = case_df[case_df['project_id'].isin(allowed_project_ids)]
        else:
            print("Warning: 'project_id' column missing in case_df")
            case_df = pd.DataFrame()

        if not follow_df.empty and 'project_id' in follow_df.columns:
            follow_df = follow_df[follow_df['project_id'].isin(allowed_project_ids)]
        else:
            print("Warning: 'project_id' column missing in follow_df")
            follow_df = pd.DataFrame()

    llm = ChatOllama(model="llama3", temperature=0)

    # 分类
    router_prompt = ChatPromptTemplate.from_template(
        """You are a biomedical assistant working inside a research data portal called 'UFCDC Portal'. 
        This portal is built on top of the Gen3 platform, which is commonly used for managing and exploring biomedical research data.
        The UFCDC Portal allows users to explore and query biomedical datasets via chat and visual interfaces.

        Classify the user question into one of the following categories:
        - structured_query: if it's about patient data, case status, follow-up info, statistics, or anything derived from a dataset
        - llm_chat: if it's a general biomedical or health knowledge question, OR about this portal (e.g. its purpose, usage, features, or Gen3-based design)

        Question: {input}
        Respond only with "structured_query" or "llm_chat".
        """
    )

    router_chain = router_prompt | llm | StrOutputParser()

    category = router_chain.invoke({"input": req.question}).strip()
    logger.info('category: %s', category)
    print('category', category)

    def stream_generator():
        final_response = ""
        logger.info('which category: %s', category)
        if category == "structured_query":
            code_prompt = PromptTemplate.from_template(
                """You are a Python data analyst. 
                You are provided two pandas DataFrames: `case_df` and `follow_df`.

                Here are the first few rows of `case_df`:
                {case_head}

                And the first few rows of `follow_df`:
                {follow_head}

                Write Python code to answer the following question from the user:
                {question}

                Wrap the result in `print(...)` so the answer is displayed.

                Only return valid Python code. Do not explain. Do not add ```python. Just return code only.
                """
            ).partial(
                case_head=case_df.head(5).to_string(index=False),
                follow_head=follow_df.head(5).to_string(index=False),
            )
            generated_code = llm.invoke(code_prompt.format(question=req.question)).content
            logger.info("generated_code: %s", generated_code)
            print('generated_code', generated_code)
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                local_vars = {"case_df": case_df, "follow_df": follow_df}
                try:
                    exec(generated_code, {}, local_vars)
                except Exception as e:
                    yield f"[Error executing code: {e}]"
                    return
                output = buf.getvalue()
                print(">> Execution output:", output)
                logger.info(">> Execution output: %s", output)

            explain_prompt = PromptTemplate.from_template(
                """The result of executing the following Python code is:

                {output}

                Now explain the answer to the user based on the original question:
                {question}

                Do not repeat the code. Just give a clear answer.
                """
            )
            explain_response = llm.stream(explain_prompt.format(output=output, question=req.question))
            for chunk in explain_response:
                token = chunk.content
                final_response += token
                yield token

        else:
            prompt = (
                "You are a helpful biomedical assistant.\n"
                f"Conversation history:\n{history_str}\n\n"
                f"Now continue the conversation. User just asked:\n{req.question}"
            )
            for chunk in llm.stream(prompt):
                token = chunk.content
                final_response += token
                yield token

        messages.append({"role": "assistant", "content": final_response})
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

    return StreamingResponse(stream_generator(), media_type="text/plain")


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
