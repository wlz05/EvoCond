# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from base import BaseServer
import re
import pandas as pd

class DistanceRequest(BaseModel):
    origin: str = Field(..., description='Origin')
    destination: str = Field(..., description='Destination')
    mode: str = Field(default="driving", description='Transport mode')

class DistanceResponse(BaseModel):
    origin: str
    destination: str
    duration: Optional[str]
    distance: Optional[str]
    cost: Optional[float]

class DistanceServer(BaseServer):
    def __init__(self):
        super().__init__('distance', 8006)
        self.load_data()
        self.setup_routes()

    def load_data(self):
        self.data = pd.read_csv("envs/reward_score/database/googleDistanceMatrix/distance.csv")

    def extract_before_parenthesis(self, s):
        match = re.search(r'^(.*?)\([^)]*\)', s)
        return match.group(1) if match else s

    def setup_routes(self):
        @self.app.post("/distance", response_model=DistanceResponse)
        async def query_distance(request: DistanceRequest):
            try:
                origin = self.extract_before_parenthesis(request.origin)
                destination = self.extract_before_parenthesis(request.destination)

                result = self.data[
                    (self.data['origin'] == origin) &
                    (self.data['destination'] == destination)
                ]

                if len(result) == 0:
                    return DistanceResponse(
                        origin=origin,
                        destination=destination,
                        duration=None,
                        distance=None,
                        cost=None
                    )

                row = result.iloc[0]
                duration = row['duration']
                distance = row['distance']

                cost = None
                if 'day' not in duration:
                    distance_value = float(distance.replace("km","").replace(",",""))
                    if request.mode == 'driving':
                        cost = distance_value * 0.05
                    elif request.mode == 'taxi':
                        cost = distance_value

                return DistanceResponse(
                    origin=origin,
                    destination=destination,
                    duration=duration,
                    distance=distance,
                    cost=cost
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server = DistanceServer()
    server.run()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
