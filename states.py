# To amke a graph first we will create a state

import os

#1) Typed Dict (Most common way to create a state)

from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int

#2) Pydantic Model
# it is good for data validation and settings management using python type annotations

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic: str
    summary: str = ""
    score: int

    @field_validator
    def score_positive(cls, value):
        if value < 0:
            raise ValueError("Score must be a positive integer")
        return value
    
#3) Python Dataclass
# it is good for creating classes that are primarily used to store data with less boilerplate code
#but it is used rarely

from dataclasses import dataclass,field

@dataclass
class State:
    topic: str
    summary: str = ""
    score: int = 0

    def __post_init__(self):
        if self.score < 0:
            raise ValueError("Score must be a positive integer")
        