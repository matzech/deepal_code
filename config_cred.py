import json

try:
    with open('config_cred.json', 'r') as file:
        paths = json.load(file)
        img_folder_positives = paths["img_folder_positives"]
        img_folder_all = paths["img_folder_all"]
        res_folder = paths["res_folder"]
except ImportError:
    img_folder_positives = "data"
    img_folder_all = "data_w_negatives/"
    res_folder = "results_paper_pub"
