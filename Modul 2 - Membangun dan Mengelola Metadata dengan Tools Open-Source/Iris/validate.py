import mlflow
from mlflow.models import validate_serving_input  # validasi input cocok atau tidak
from mlflow.models import convert_input_example_to_serving_input # convert input sesuai format yang dibutuhkan
import pandas as pd

INPUT_EXAMPLE = [
  [
    6.6,
    3,
    4.4,
    1.4
  ],
  [
    6.9,
    3.1,
    4.9,
    1.5
  ],
  [
    6.2,
    3.4,
    5.4,
    2.3
  ],
  [
    5.7,
    2.8,
    4.5,
    1.3
  ],
  [
    5.5,
    2.6,
    4.4,
    1.2
  ]
]

model_uri = "runs:/c790c495d7f142b2b737e5538ca7a45a/model"  # Ganti dengan RUN_ID yang sesuai
# serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# print(validate_serving_input(model_uri=model_uri, serving_input=serving_payload))


# testing predict
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
predict = loaded_model.predict(pd.DataFrame([[6.7, 3.1, 5.6, 2.4]]))
print(predict)