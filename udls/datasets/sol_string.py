from .. import DomainAdaptationDataset, SimpleDataset

SolV4folders = [
    "/fast-2/datasets/Solv4_strings_wav/audio/Cello",
    "/fast-2/datasets/Solv4_strings_wav/audio/Contrabass",
    "/fast-2/datasets/Solv4_strings_wav/audio/Violin",
    "/fast-2/datasets/Solv4_strings_wav/audio/Viola"
]


def Solv4Strings_DomainAdaptation(out_database_location, preprocess_function):
    return DomainAdaptationDataset(out_database_location, SolV4folders,
                                   preprocess_function, "*.wav", 1e11)


def Solv4Strings_Simple(out_database_location, preprocess_function):
    return SimpleDataset(out_database_location, SolV4folders,
                         preprocess_function, "*.wav", 1e11)
