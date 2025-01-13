id='gpt_$(date +"%M%S")'
pytest test/code -m "not training" >> test_code_$id.logs
python GPT/tune.py --experiment_name $id
pytest tests/model -m "not training" >> test_model_$id.logs
python GPT/batch_predict.py --experiment_name $id --input_batch ["My name is", "your name is", "her name is"]
python GPT/serve.py --experiment_name $id
