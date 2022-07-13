from typing import List, Text
from pathlib import Path

import numpy as np
import pandas as pd

from recanime.document_store.base import AnimeStoreBase
from recanime.schema.predict import AnimeAttributes, AnimeInfo


class ExcelAnimeStore(AnimeStoreBase):
    def __init__(self, excel_path: Path) -> None:
        self.anime_df = pd.read_csv(
            excel_path,
            low_memory=False,
            usecols=["MAL_ID", "Name", "Genres", "sypnopsis"]
        )
        self.anime_df.rename({"MAL_ID": "anime_id"}, inplace=True, axis=1)
        self.anime_df['Genres'] = self.anime_df['Genres'].str.split(pat=",", expand=False)
        self.anime_df['Genres'] = self.anime_df['Genres'].apply(
            lambda genres: [genre.strip() for genre in genres]
        )

    async def filter_anime_ids_by_attributes(
        self,
        attributes: AnimeAttributes,
    ) -> List[Text]:
        """Implement the base class method."""

        attributes_dict = attributes.dict()
        for key, value in attributes_dict.items():
            if value <= 0:
                attributes_dict.pop(key)

        selected_attributes = set(attributes_dict.keys())
        selected_animes_df = self.anime_df[['anime_id', 'Genres']].copy(deep=True)
        selected_animes_df['Genres'] = selected_animes_df['Genres'].apply(
            lambda x: np.nan if len(selected_attributes & set(x)) == 0 else x
        )
        selected_animes_df.dropna(inplace=True)
        selected_animes_dict = selected_animes_df.to_dict(orient='list')
        selected_anime_ids = selected_animes_dict['anime_id']

        return selected_anime_ids

    async def get_anime_informations_by_anime_id(
        self, anime_ids: List[Text]
    ) -> List[AnimeInfo]:
        """Implement the base class method."""

        selected_animes_df = self.anime_df[
            self.anime_df["anime_id"].isin(anime_ids)
        ].copy(deep=True)
        selected_animes_df.rename({"Name": "anime_name"}, inplace=True, axis=1)
        selected_animes_df.rename({"Genres": "genres"}, inplace=True, axis=1)
        selected_animes = selected_animes_df.to_dict(orient='records')

        return [AnimeInfo(**selected_anime) for selected_anime in selected_animes]
