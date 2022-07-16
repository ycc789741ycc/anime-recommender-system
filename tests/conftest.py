import json
import torch
import pickle
import pytest

from typing import Dict

from recanime.anime_store.store import AnimeStore
from recanime.recommender.ranking_base_filter.model import FactorizationMachineModel
from recanime.schema.predict import AnimeInfo
from recanime.schema.user import ExistedUserAttributesVector


@pytest.fixture(scope="session")
def factorization_machine_encode_config():
    INPUT_DIR = './model/ranking_base'

    with open(INPUT_DIR + "/encode_config.json") as f:
        encoder_config = json.load(f)

    return encoder_config


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("factorization_machine_encode_config")
def factorization_machine_model(factorization_machine_encode_config: Dict):
    INPUT_DIR = './model/ranking_base'

    user2user_encoded = factorization_machine_encode_config['user2user_encoded']
    anime2anime_encoded = factorization_machine_encode_config['anime2anime_encoded']

    n_users = len(user2user_encoded)
    n_animes = len(anime2anime_encoded)
    field_dims = [n_users, n_animes]

    model = FactorizationMachineModel(field_dims=field_dims, embed_dim=32)
    model.load_state_dict(torch.load(INPUT_DIR + "/model_state_dict.pt"))

    return model


@pytest.fixture(scope="session")
def existed_user_attributes_vector():
    INPUT_DIR = './data'
    with open(INPUT_DIR + "/user_with_vector_dict.pickle", 'rb') as f:
        user_with_vector_dict = pickle.load(f)

    with open(INPUT_DIR + "/genre_index_mapping.pickle", 'rb') as f:
        genre_index_mapping = pickle.load(f)

    return ExistedUserAttributesVector(
        user_ids=user_with_vector_dict['user_id'],
        user_attributes_vector=user_with_vector_dict['genre_vector'],
        attribute_map_to_vector_position=genre_index_mapping["genre2genre_idx"],
        vector_position_map_to_attribute=genre_index_mapping["genre_idx2genre"]
    )


@pytest.fixture(scope="session")
def anime_store():
    return AnimeStore(anime_infos=ANIME_INFOS)


ANIME_INFOS = [
    AnimeInfo(
        anime_id="1",
        anime_name="Cowboy Bebop",
        genres=["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Space"],
        synopsis="In the year 2071, humanity..."
    ),
    AnimeInfo(
        anime_id="6",
        anime_name="Trigun",
        genres=["Action", "Sci-Fi", "Adventure", "Comedy", "Drama", "Shounen"],
        synopsis="Vash the Stampede is the man..."
    ),
    AnimeInfo(
        anime_id="8",
        anime_name="Bouken Ou Beet",
        genres=["Adventure", "Fantasy", "Shounen", "Supernatural"],
        synopsis="It is the dark century and the people..."
    )
]
