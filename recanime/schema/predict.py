from typing import List, Text

from pydantic import BaseModel


class AnimeAttributes(BaseModel):
    Comedy: float = 0.0
    Action: float = 0.0
    Fantasy: float = 0.0
    Drama: float = 0.0
    Romance: float = 0.0
    Sci_Fi: float = 0.0
    Shounen: float = 0.0
    Adventure: float = 0.0
    School: float = 0.0
    Supernatural: float = 0.0
    Slice_of_Life: float = 0.0
    Ecchi: float = 0.0
    Magic: float = 0.0
    Seinen: float = 0.0
    Mystery: float = 0.0
    Mecha: float = 0.0
    Super_Power: float = 0.0
    Music: float = 0.0
    Historical: float = 0.0
    Harem: float = 0.0
    Military: float = 0.0
    Shoujo: float = 0.0
    Psychological: float = 0.0
    Sports: float = 0.0


class AnimeInfo(BaseModel):
    anime_id: Text
    anime_name: Text
    genres: List[Text]
    synopsis: Text


class PredictResults(BaseModel):
    predict_score: float
    anime_info: AnimeInfo
