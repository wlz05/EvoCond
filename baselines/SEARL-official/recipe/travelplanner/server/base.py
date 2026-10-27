# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from abc import ABC, abstractmethod

class BaseServer(ABC):
    def __init__(self, name: str, port: int):
        self.app = FastAPI(title=f"{name} query service")
        self.port = port
        self.data = None

    @abstractmethod
    def load_data(self):
        pass

    def run(self):
        print(f'Starting {self.app.title}... listening at 127.0.0.1:{self.port}')
        uvicorn.run(self.app, host="127.0.0.1", port=self.port)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
