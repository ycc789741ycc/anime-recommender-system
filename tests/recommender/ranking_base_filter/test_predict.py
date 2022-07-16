import logging
from typing import Dict

import pytest

from recanime.recommender.ranking_base_filter.model import FactorizationMachineModel
from recanime.recommender.ranking_base_filter.predict import RankingBaseAnimeRec
from recanime.schema.predict import AnimeInfo, AnimeAttributes
from recanime.schema.user import ExistedUserAttributesVector


logger = logging.getLogger("pytest")


@pytest.fixture(scope="session")
@pytest.mark.usefixtures(
    "existed_user_attributes_vector",
    "factorization_machine_model",
    "factorization_machine_encode_config"
)
def ranking_base_anime_rec(
    existed_user_attributes_vector: ExistedUserAttributesVector,
    factorization_machine_model: FactorizationMachineModel,
    factorization_machine_encode_config: Dict
):
    return RankingBaseAnimeRec(
        existed_user_attributes_vector=existed_user_attributes_vector,
        model=factorization_machine_model,
        model_encode_config=factorization_machine_encode_config
    )


@pytest.mark.usefixtures("ranking_base_anime_rec", "anime_store")
async def test_predict(
    ranking_base_anime_rec: RankingBaseAnimeRec,
    anime_store: AnimeInfo
):
    result = await ranking_base_anime_rec.predict(
        anime_store=anime_store,
        attributes=AnimeAttributes(
            Action=10
        ),
        top_k=1
    )

    logger.info(result)
    assert "Action" in result[0].anime_info.genres
