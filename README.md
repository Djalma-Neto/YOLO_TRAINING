# Treinamento de detector com YOLOX

Este repositório inclui uma integração com o YOLOX oficial para gerar um dataset sintético a partir de `amostras/` (fundos) e `alvo/` (objetos PNG), converter as anotações para COCO e treinar um detector. O código da integração está em `yolox_migration/`.

## Primeiro treinamento

Execute os comandos no PowerShell, a partir da raiz deste repositório, com o ambiente virtual já ativado. É necessário ter PyTorch com CUDA funcional para treinar na GPU. A instalação do PyTorch depende da versão CUDA compatível com a máquina; use o seletor oficial em [pytorch.org](https://pytorch.org/get-started/locally/).

Instale as dependências do pipeline e o runtime YOLOX já incluído neste repositório:

```powershell
pip install -r yolox_migration\requirements.txt
pip install -e third_party\YOLOX
```

Gere, valide e converta o dataset:

```powershell
python -m yolox_migration.cli generate
python -m yolox_migration.cli validate
python -m yolox_migration.cli coco
```

Então inicie o primeiro treino com YOLOX-S:

```powershell
python -m yolox_migration.cli train --model yolox-s --epochs 100 --batch-size 8 --size 640 --device cuda
```

Por padrão, a configuração gera 1.000 imagens, com cerca de 15% negativas, e divide os fundos entre treino e validação para manter imagens derivadas do mesmo fundo na mesma partição. Para gerar imagens apenas positivas, altere `negative_fraction` para `0.0` em `yolox_migration/config.yaml`. PNGs planos em `alvo/` são tratados como uma classe; para várias classes, organize-os em subpastas com o nome de cada classe e ajuste `classes` na configuração.

O treino acima começa sem pesos pré-treinados. Para usar transferência de aprendizado, acrescente `--pretrained C:\caminho\para\checkpoint.pth` ao comando, após confirmar que o checkpoint é compatível e que sua licença permite esse uso.

## Testar o modelo pela webcam

Exporte o checkpoint treinado para ONNX e inicie a detecção em vídeo ao vivo. O retângulo verde contorna a caixa de detecção do objeto; YOLOX é um detector e não produz a silhueta pixel a pixel.

```powershell
python -m yolox_migration.cli export C:\caminho\para\best_ckpt.pth --model yolox-s --size 640
python -m yolox_migration.webcam --model C:\caminho\para\best_ckpt.onnx
```

O preview espelhado usa a câmera de índice `0` por padrão; pressione `Q` ou `Esc` para sair. Para outra câmera, use `--camera 1`. Para diminuir ou aumentar a confiança mínima, use `--threshold 0.4`. A inferência usa CUDA pelo ONNX Runtime quando o provider CUDA está instalado; caso contrário, usa CPU. A velocidade depende do hardware, da resolução e do tamanho do modelo. O utilitário precisa de OpenCV com suporte a janela de vídeo (`opencv-python`), além de ONNX Runtime.

## Usar em outra máquina sem internet

Antes de desconectar a máquina que prepara o Git, faça commit e push das alterações deste projeto. Na nova máquina, clone o seu repositório e copie também as pastas `amostras/` e `alvo/`: elas são ignoradas pelo Git e não vêm no clone. O ambiente virtual da nova máquina precisa ter, previamente, as dependências Python do pipeline e do YOLOX, incluindo PyTorch com CUDA funcional. Se alguma biblioteca estiver faltando, será necessário instalá-la a partir de um wheelhouse local; sem isso, o `pip` precisará de acesso a um índice de pacotes.

Com as dependências já instaladas, o ambiente virtual ativado, as imagens copiadas e estando na raiz do projeto, instale o runtime vendorizado sem consultar a internet:

```powershell
pip install --no-deps --no-build-isolation -e third_party\YOLOX
```

Gere as imagens sintéticas, valide os rótulos, converta para COCO e treine sem pesos pré-treinados:

```powershell
python -m yolox_migration.cli generate
python -m yolox_migration.cli validate
python -m yolox_migration.cli coco
python -m yolox_migration.cli train --model yolox-s --epochs 100 --batch-size 8 --size 640 --device cuda
```

Não execute `download.py` nesse fluxo: ele baixa fundos da web. Sem `--pretrained`, o treinamento começa com pesos aleatórios e não precisa baixar checkpoints. O código do YOLOX usado pelo treino está em `third_party/YOLOX/`, portanto não é necessário clonar o repositório upstream. A geração, validação e conversão são locais; o treino requer uma GPU CUDA disponível.

## Arquivos gerados e licença

O dataset novo é criado em `datasets/yolox_custom/`. A pasta `datasets/` já está no `.gitignore`, assim como `alvo/`, `amostras/` e `venv/`; imagens, rótulos e demais estruturas de dados geradas não entram no repositório. O código em `yolox_migration/` e o runtime mínimo licenciado do YOLOX em `third_party/YOLOX/` ficam no repositório para uso futuro. Checkpoints e logs de treino em `third_party/YOLOX/YOLOX_outputs/` são ignorados.

O código mantido neste estado do repositório é o YOLOX oficial vendorizado em `third_party/YOLOX/` e a integração/pipeline em `yolox_migration/`, ambos sob Apache-2.0; a licença completa está em `LICENSE` e a licença/proveniência upstream em `third_party/YOLOX/LICENSE` e `third_party/YOLOX/UPSTREAM.md`. Arquivos MMYOLO antigos podem continuar presentes em clones locais, mas são ignorados e não fazem parte do conteúdo atual versionado.

A licença do código não concede direitos sobre imagens, datasets ou checkpoints usados no treinamento. Verifique as licenças desses materiais separadamente antes de distribuir ou comercializar modelos. O histórico Git anterior ainda contém commits MMYOLO/GPL; esta alteração limpa o estado atual do repositório, mas não apaga esses commits históricos.
