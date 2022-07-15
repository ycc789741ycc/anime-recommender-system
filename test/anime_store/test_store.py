import logging

from recanime.anime_store.store import AnimeStore
from recanime.schema.predict import AnimeInfo, AnimeAttributes
from test.conftest import ANIME_INFOS


logger = logging.getLogger("pytest")


anime_store = AnimeStore(anime_infos=ANIME_INFOS)


async def test_filter_anime_ids_by_attributes():
    assert_target = ["1", "6"]

    action_result = await anime_store.filter_anime_ids_by_attributes(
        AnimeAttributes(
            Action=5,
            Sci_Fi=3
        )
    )
    action_result.sort(key=lambda x: int(x))

    assert action_result == assert_target, "Wrong result!"


async def test_get_anime_informations_by_anime_id():
    assert_target = [
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

    action_result = await anime_store.get_anime_informations_by_anime_id(
        ["6", "8"]
    )

    assert action_result == assert_target, "Wrong result!"
