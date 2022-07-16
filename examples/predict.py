import asyncio
import pprint


from recanime.anime_store.excel_store import ExcelAnimeStore
from recanime.recommender.ranking_base_filter.model import FactorizationMachineModel
from recanime.recommender.ranking_base_filter.predict import RankingBaseAnimeRec
from recanime.schema.user import ExistedUserAttributesVector
from recanime.schema.predict import AnimeAttributes
from recanime.utils.model_utils import get_fm_model, get_fm_encoder_config
from recanime.utils.rec_utils import get_existed_user_attributes_vector


MODEL_INPUT_DIR = './model/ranking_base'
DATA_INPUT_DIR = './data'


excel_anime_store = ExcelAnimeStore(
    excel_path=DATA_INPUT_DIR + "/anime_with_synopsis.csv"
)


model: FactorizationMachineModel = get_fm_model()
model.eval()
model_encoder_config = get_fm_encoder_config()
existed_user_attributes_vector: ExistedUserAttributesVector = get_existed_user_attributes_vector()
ranking_base_anime_rec = RankingBaseAnimeRec(
    model=model,
    model_encode_config=model_encoder_config,
    existed_user_attributes_vector=existed_user_attributes_vector
)


async def main():
    input_feature = {"Romance": 4, "School": 4, "Super Power": 2}
    results = await ranking_base_anime_rec.predict(
        anime_store=excel_anime_store,
        attributes=AnimeAttributes(**input_feature),
        top_k=5
    )
    pprint.pprint([result.dict() for result in results])


if __name__ == "__main__":
    asyncio.run(main())
