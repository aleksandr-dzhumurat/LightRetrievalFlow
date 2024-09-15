import re
from typing import List

import pandas as pd

def clean_text(txt):
    cleaned_text = re.sub(r'\n+', ' ', txt)
    cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
    return cleaned_text


def read_csv_as_dicts(csv_filepath) -> List:
    """ input_filename = os.path.join(root_dir, 'data', 'pipelines-data', config['content_file_name'])
    """
    df = pd.read_csv(csv_filepath)
    csv_entries = df.to_dict(orient='records')
    print('Num entries: %d' % len(csv_entries))
    return csv_entries