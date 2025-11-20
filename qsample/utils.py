import dill as pickle

def save(data, path):
    """Save data to path

    **Attributes:**

    path : str
        File path to save to
    data : *
        Data to write to path
    """
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def load(path):
    """Load data from path

    **Attributes:**

    path : str
        File path to load from
        
    **Returns:**
    *
        Loaded data
    """
    with open(path, 'wb') as fp:
        data = pickle.load(fp)
    return data
