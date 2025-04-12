import os

###################### DATA INGESTION ############################

# Raw data paths
RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"

# Processed data paths
PROCESSED_DIR = "artifacts/processed"
ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv")
ANIMESYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv")

# Processed DataFrames
X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR, "rating_df.csv")
DF = os.path.join(PROCESSED_DIR, "anime_df.csv")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "synopsis_df.csv")

# Encoded/Decoded mappings for users and animes
USER2USER_ENCODED = os.path.join(PROCESSED_DIR, "user2user_encoded.pkl")
USER2USER_DECODED = os.path.join(PROCESSED_DIR, "user2user_decoded.pkl")

ANIME2ANIME_ENCODED = os.path.join(PROCESSED_DIR, "anim2anime_encoded.pkl")
ANIME2ANIME_DECODED = os.path.join(PROCESSED_DIR, "anim2anime_decoded.pkl")

###################### MODEL TRAINING ###########################

# Model paths
MODEL_DIR = "artifacts/model"
WEIGHTS_DIR = "artifacts/weights"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "anime_weights.pkl")
USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "user_weights.pkl")
CHECKPOINT_FILE_PATH = "artifacts/model_checkpoint/weights.weights.h5"

###################### CONFIGURE THE PATHS #######################

# Make it easier to import the paths
path_anime_df = DF
path_rating_df = RATING_DF
path_synopsis_df = SYNOPSIS_DF
