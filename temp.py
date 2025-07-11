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

case_df = pd.read_json("case_data.json")
follow_df = pd.read_json("follow_up_data.json")
history_str = ""
structured_prompt = PromptTemplate.from_template(
    "You are a biomedical data analyst. Use the following structured data to answer the user's question.\n\n"
    "Case data:\n{case_data}\n\n"
    "Follow-up data:\n{follow_data}\n\n"
    "Conversation history:\n{history}\n\n"
    "Question: {input}"
).partial(
    case_data=case_df.to_string(index=False),
    follow_data=follow_df.to_string(index=False),
    history=history_str
)

prompt = structured_prompt.format(input="how many patient whose primary site is Breast?")
print("Prompt length (chars):", len(prompt))
print("Prompt snippet:\n", prompt[:1000])  # 看看前 1000 个字符