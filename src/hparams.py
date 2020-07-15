from effortless_config import Config, setting


class config(Config):
    groups = ["vanilla", "melgan"]

    TYPE = setting(default="vanilla", vanilla="vanilla", melgan="melgan")

    # MELGAN PARAMETERS
    INPUT_SIZE = 128
    NGF = 32
    N_RES_G = 3

    HOP_LENGTH = 256

    RATIOS = setting(default=[1, 1, 1, 2, 1, 1, 1],
                     vanilla=[1, 1, 1, 2, 1, 1, 1],
                     melgan=[8, 8, 2, 2])

    NUM_D = 3
    NDF = 16
    N_LAYER_D = 4
    DOWNSAMP_D = 4

    # AUTOENCODER
    CHANNELS = [128, 256, 256, 512, 512, 512, 128, 32]
    KERNEL = 5
    EXTRACT_LOUDNESS = True
    AUGMENT = setting(default=5, vanilla=5, melgan=1)

    # CLASSIFIER
    CLASSIFIER_CHANNELS = [16, 64, 256]
    CLASSIFIER_LIN_SIZE = [256, 64, 2]

    # TRAIN PARAMETERS
    PATH_PREPEND = "./runs/"
    SAMPRATE = 16000
    N_SIGNAL = setting(default=2**15, vanilla=2**15, melgan=2**14)
    EPOCH = 1000
    BATCH = 1
    LR = 1e-4
    NAME = "untitled"
    CKPT = None

    WAV_LOC = None
    FILE_LIST = None
    LMDB_LOC = "./preprocessed"

    BACKUP = 10000
    EVAL = 1000

    # INCREMENTAL GENERATION
    USE_CACHED_PADDING = False
    BUFFER_SIZE = 1024


if __name__ == "__main__":
    print(config)
