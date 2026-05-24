import os
import requests
import json
import time

# vai criar um arquivo onde será guardado as imagens baixadas
# diretórios
DIR_AMOSTRAS = "./amostras"
DIR_OBJETOS = "./alvo"

DIR_DATASETS = "./datasets/custom"

DIR_TRAIN_IMAGES = f"{DIR_DATASETS}/train"
DIR_VAL_IMAGES = f"{DIR_DATASETS}/val"

DIR_ANNOTATIONS = f"{DIR_DATASETS}/annotations"

os.makedirs(DIR_DATASETS, exist_ok=True)

os.makedirs(DIR_AMOSTRAS, exist_ok=True)
os.makedirs(DIR_OBJETOS, exist_ok=True)

os.makedirs(DIR_TRAIN_IMAGES, exist_ok=True)
os.makedirs(DIR_VAL_IMAGES, exist_ok=True)

os.makedirs(DIR_ANNOTATIONS, exist_ok=True)

# nova requisição usando HTTP request (testado no postman)
# https://unsplash.com/napi/search?query=montanhas&per_page=20&plus=none
def getURL():
    array = []
    palavras_chave = ['pessoas','montanhas','natureza','paisagens','urbanismo','rodovias'] #palavras chave para buscar imagens
    for palavra in palavras_chave:
        for x in range(30):
            URL = "https://unsplash.com/napi/search/photos?query={0}&per_page=20&page={1}&plus=none".format(palavra,x)
            response = requests.get(URL, timeout=15)
            if response.status_code != 200:
                continue
            data = response.json()
            for x in data['results']:
                array.append(x['urls']['regular'])
    array = list(set(array))
    return array

def getImages(array):
    vals = len(os.listdir(DIR_AMOSTRAS))
    for x in range(len(array)):
        try:
            url = array[x]
            response = requests.get(
                url,
                stream=True,
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
                timeout=15
            )
            if not response.ok:
                print("Erro:", response.status_code)
                continue
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                print("URL não é imagem:", url)
                continue
            nome_arquivo = './amostras/img{0}.jpg'.format(x + vals)
            with open(nome_arquivo, 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
            print("Imagem salva:", nome_arquivo)
        except Exception as e:
            print("Erro ao baixar:", e)
        time.sleep(0.2)

array = getURL()
getImages(array)
print(array)