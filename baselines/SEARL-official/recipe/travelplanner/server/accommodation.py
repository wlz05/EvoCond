# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from base import BaseServer
import pandas as pd

class AccommodationRequest(BaseModel):
    city: str = Field(..., description='City name')

class AccommodationResponse(BaseModel):
    name: str
    price: float
    room_type: str
    house_rules: Optional[str] = None
    minimum_nights: int
    maximum_occupancy: int
    review_rate: float
    city: str

class AccommodationServer(BaseServer):
    def __init__(self):
        super().__init__('accommodation', 8009)
        self.load_data()
        self.setup_routes()

    def load_data(self):
        self.data = pd.read_csv("envs/reward_score/database/accommodations/clean_accommodations_2022.csv")

    def setup_routes(self):
        @self.app.post("/accommodations", response_model=List[AccommodationResponse])
        async def query_accommodations(request: AccommodationRequest):
            try:
                results = self.data[self.data["city"] == request.city]

                if len(results) == 0:
                    return []

                return [
                    AccommodationResponse(
                        name=row["NAME"],
                        price=float(row["price"]),
                        room_type=row["room type"],
                        house_rules=str(row["house_rules"]) if pd.notna(row["house_rules"]) else None,
                        minimum_nights=int(row["minimum nights"]),
                        maximum_occupancy=int(row["maximum occupancy"]),
                        review_rate=float(row["review rate number"]),
                        city=row["city"]
                    ) for _, row in results.iterrows()
                ]

            except Exception as e:
                print(f"Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server = AccommodationServer()
    server.run()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
