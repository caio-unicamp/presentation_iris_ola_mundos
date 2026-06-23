# 🤖 MNIST Interativo: IA vs. Alunos
Este projeto é uma demonstração interativa de Inteligência Artificial e Visão Computacional, desenvolvida especialmente para apresentações educacionais voltadas a alunos do Ensino Fundamental II e Ensino Médio. Uma iniciativa do grupo de estudos Iris em parceria com a Olá Mundos!

A aplicação utiliza a webcam para capturar números desenhados à mão em tempo real, processa a imagem replicando as condições exatas do clássico dataset MNIST, e utiliza um modelo de Rede Neural (via ONNX Runtime) para adivinhar o dígito. A interface foi pensada para ser gamificada, desafiando os alunos a tentarem "enganar" a máquina.

## ✨ Funcionalidades
* **Inferência em Tempo Real:** Captura e processamento instantâneo via OpenCV.
* **Processamento Visual Educativo:** Exibe na tela a "Visão da IA" (a matriz 28x28 pixels processada), ajudando a explicar conceitos de *Feature Extraction* e binarização.
* **Download Automático do Modelo:** O repositório é mantido leve. O modelo `mnist-8.onnx` é baixado automaticamente do repositório oficial da Microsoft/ONNX apenas na primeira execução.
* **Leveza:** Não exige a instalação de frameworks pesados de Deep Learning (como PyTorch ou TensorFlow) na máquina de apresentação.

## 🚀 Como replicar este projeto

### Pré-requisitos
* Python 3.8 ou superior.
* Uma webcam funcional conectada ao computador.
* [uv](https://github.com/astral-sh/uv) instalado no sistema.
   * **Mac/Linux:**
   ```bash
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
   ```
   * **Windows (PowerShell):** 
   ```bash
   `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   ```

### Instalação

1. Clone este repositório:
   ```bash
   git clone git@github.com:[SEU_USUARIO]/presentation_iris_ola_mundos.git
   cd presentation_iris_ola_mundos
   ```

2. Ative o ambiente virtual
    ```bash
    ## No Mac/Linux
    source .venv/bin/activate
    ## No Windows
    .venv\Scripts\activate
    ```

3. Instale as dependências
    ```bash
    uv pip install -r requirements.txt
    ```
## 🎮 Como Usar

Com o ambiente virtual ativado, basta executar o script principal:
```bash
python pipeline.py
```

Na primeira execução, o script baixará o arquivo mnist-8.onnx automaticamente (cerca de 25MB). A câmera será iniciada e uma janela chamada "IA vs Alunos" se abrirá. Peça para desenharem um número de 0 a 9 em um pedaço de papel branco usando uma caneta escura (preta ou azul de preferência). Posicione o papel na **ZONA DE ESCANEAMENTO** (quadrado verde no centro da tela). Acompanhe a predição e o nível de certeza da IA na tela!

Para encerrar a aplicação, pressione a tecla q com a janela do vídeo selecionada.


## ⚖️ Créditos e Licença

O modelo de Inteligência Artificial utilizado neste projeto (`mnist-8.onnx`) não é distribuído diretamente neste repositório. Ele é baixado dinamicamente a partir do ONNX Model Zoo, que disponibiliza seus modelos sob a licença **Apache 2.0**.