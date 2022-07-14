from abc import ABC, abstractmethod
from ctypes import Union
from typing import Dict, List, Text

import numpy as np

from recanime.anime_store.base import AnimeStoreBase
from recanime.schema.predict import AnimeAttributes, PredictResults


class AnimeRecBase(ABC):
    def __init__(
        self,
        genre2genre_idx: Dict[Text, int],
        genre_idx2genre: Dict[int, Text],
        user_with_vector_dict: Dict[Text, Union[int, np.ndarray]],
    ) -> None:
        self.genre2genre_idx = genre2genre_idx
        self.genre_idx2genre = genre_idx2genre
        self.genre_num = len(self.genre2genre_idx)

        self.user_with_vector_dict = user_with_vector_dict
        self.user_id2user_id_idx = {x: i for i, x in enumerate(user_with_vector_dict['user_id'])}
        self.user_id_idx2user_id = {i: x for i, x in enumerate(user_with_vector_dict['user_id'])}
        self.user_feature_matrix = np.vstack(user_with_vector_dict['genre_vector'])

    async def map_to_existed_user(self, attributes: AnimeAttributes) -> int:
        """Get the user's latent factor by mapping the attributes from existed user."""
        input_latent_factor: np.ndarray = np.zeros((self.genre_num), dtype=float)
        for genre, score in attributes.dict().items():
            input_latent_factor[self.genre2genre_idx[genre]] = score

        most_similar_user_id = self.user_id_idx2user_id[
            np.argmax(np.inner(self.user_feature_matrix, input_latent_factor))
        ]

        return most_similar_user_id

    @abstractmethod
    async def predict(
        self,
        anime_store: AnimeStoreBase,
        attributes: AnimeAttributes,
        top_k: int
    ) -> List[PredictResults]:
        """Retrieve the recommendation by the given attributes."""

        raise NotImplementedError()
