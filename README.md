
Getting started:

ssh claas@vid-gpu1.inf.cs.cmu.edu
source ~/thesis/bin/activate

python3 src/data/_crawlers/website_crawler.py
slurm -i eth0
top
tmux a
psql -U postgres -d gdelt_social_video

pg_dump -U postgres gdelt_social_video > ~/data/other/database_backups/dump_20181006
scp claas@vid-gpu1.inf.cs.cmu.edu:~/dump_20180919 data/
scp data/other/database_backups/dump_20180916 claas@vid-gpu1.inf.cs.cmu.edu:~/dump_20180916
psql -U postgres gdelt_social_video < data/dump_20180919

Stopping, restarting postgres:
pg_ctl -D /usr/local/var/postgres stop
pg_ctl -D /usr/local/var/postgres start

./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
export DYLD_LIBRARY_PATH="/usr/local/cuda/lib"

Make sure the Project is in your PYTHONPATH, otherwise the src wont be importable

video_news_classification
==============================

Linking video footage of events to news articles reporting on them

install_name_tool -change @rpath/libcusolver.8.0.dylib /usr/local/cuda/lib/libcusolver.8.0.dylib -change @rpath/libcudart.8.0.dylib /usr/local/cuda/lib/libcudart.8.0.dylib -change @rpath/libcublas.8.0.dylib /usr/local/cuda/lib/libcublas.8.0.dylib /Users/claasmeiners/.virtualenvs/video_article_retrieval/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
install_name_tool -change @rpath/libcudart.8.0.dylib /usr/local/cuda/lib/libcudart.8.0.dylib -change @rpath/libcublas.8.0.dylib /usr/local/cuda/lib/libcublas.8.0.dylib -change @rpath/libcudnn.5.dylib /usr/local/cuda/lib/libcudnn.5.dylib -change @rpath/libcufft.8.0.dylib /usr/local/cuda/lib/libcufft.8.0.dylib -change @rpath/libcurand.8.0.dylib /usr/local/cuda/lib/libcurand.8.0.dylib -change @rpath/libcudart.8.0.dylib /usr/local/cuda/lib/libcudart.8.0.dylib /Users/claasmeiners/.virtualenvs/video_article_retrieval/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
