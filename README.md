# text-mining-laporgub

# Install library
pip install -r requirements.txt

# ACTIVATE
.\.venv\Scripts\Activate

# COMPILE
 python src/preprocess.py
 python src/train.py
 python src/predict.py

# RUN
streamlit run app.py