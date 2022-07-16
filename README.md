# Anime recommender system build by PyTorch
## Description  
This package shows how to implement anime recommender system by Pytorch.  
## Dataset  
* [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)  
## Installation  
```bash
pip install recanime
```
This github repo has contains the pretrained model and animes until 2020, is about to total 10MB.  
Download `model` and `data` or just clone down this whole github repo, and put `model` and `data` 
in the first level of your work directory.  
## Quick Start  
Prepare: The input format is define in `class AnimeAttributes`, which is inherit from [pydantic](https://pydantic-docs.helpmanual.io/) BaseModel.  
```python
from recanime.anime_store.excel_store import ExcelAnimeStore
from recanime.recommender.ranking_base_filter.model import FactorizationMachineModel
from recanime.recommender.ranking_base_filter.predict import RankingBaseAnimeRec
from recanime.schema.user import ExistedUserAttributesVector
from recanime.schema.predict import AnimeAttributes
from recanime.utils.model_utils import get_fm_model, get_fm_encoder_config
from recanime.utils.rec_utils import get_existed_user_attributes_vector

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
```
Predict  
```python
input_feature = {"Romance": 4, "School": 4, "Super Power": 2}
results = await ranking_base_anime_rec.predict(
    anime_store=excel_anime_store,
    attributes=AnimeAttributes(**input_feature),
    top_k=5
)
```
Result
```python
import pprint

pprint.pprint([result.dict() for result in results])
```
```
[{'anime_info': {'anime_id': '9790',
                 'anime_name': 'Sora no Otoshimono: Tokeijikake no Angeloid',
                 'genres': ['Comedy',
                            'Drama',
                            'Ecchi',
                            'Harem',
                            'Romance',
                            'Sci-Fi',
                            'Shounen',
                            'Supernatural'],
                 'synopsis': 'ovie adaptation of the Sora no Otoshimono manga...'
                },
  'predict_score': 1.0},
 {'anime_info': {'anime_id': '10110',
                 'anime_name': 'Mayo Chiki!',
                 'genres': ['Harem', 'Comedy', 'Romance', 'Ecchi', 'School'],
                 'synopsis': 'Due to his mother and sister, who both love...'
                },
  'predict_score': 1.0},
 {'anime_info': {'anime_id': '35480',
                 'anime_name': 'Karadasagashi',
                 'genres': ['Horror', 'School', 'Shounen'],
                 'synopsis': '"Hey, Asuka... search for my body." At school in...'
                },
  'predict_score': 1.0},
 {'anime_info': {'anime_id': '10080',
                 'anime_name': 'Kami nomi zo Shiru Sekai II',
                 'genres': ['Comedy',
                            'Harem',
                            'Romance',
                            'Shounen',
                            'Supernatural'],
                 'synopsis': 'Keima Katsuragi, the "God of Conquest," returns...'
                },
  'predict_score': 1.0},
 {'anime_info': {'anime_id': '10045',
                 'anime_name': 'Seishun Fuufu Monogatari: Koiko no Mainichi',
                 'genres': ['Comedy', 'Drama', 'Ecchi', 'Romance', 'Seinen'],
                 'synopsis': 'Based on the romance manga "Koiko no Mainichi ...'
                },
  'predict_score': 1.0}]
```