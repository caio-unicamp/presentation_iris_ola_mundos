import cv2
import numpy as np
import onnxruntime as ort
import os
import urllib.request


def processar_imagem_para_mnist(roi):
    """
    Isola o desenho do papel, centraliza pelo bounding box, adiciona padding e redimensiona para 28x28 pixels, formatando o tensor para consumo do modelo ONNX.

    @param roi: Região de interesse da imagem
    """
    # Converter para tons de cinza e aplicar desfoque
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Binarização de Otsu e Inversão
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, thresh
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Ignora ruídos muito pequenos (ajustado para a ROI)
    if w < 15 or h < 15:
        return None, thresh

    # Recortar e redimensionar mantendo a proporção (20x20)
    digit = thresh[y:y+h, x:x+w]
    
    if w > h:
        new_w = 20
        new_h = int(20 * (h / w))
    else:
        new_h = 20
        new_w = int(20 * (w / h))
        
    # Evita erro de dimensão 0
    if new_w == 0 or new_h == 0:
        return None, thresh
        
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 5. Padding para 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    
    digit_padded = cv2.copyMakeBorder(digit_resized, pad_top, pad_bottom, pad_left, pad_right, 
                                      cv2.BORDER_CONSTANT, value=0)
    
    # Normalizar para o modelo ONNX
    digit_normalized = digit_padded.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(digit_normalized, axis=0), axis=0)
    
    return input_tensor, digit_padded

def baixar_modelo_se_necessario(caminho_modelo="mnist-8.onnx"):
    """
    Verifica a existência do arquivo .onnx localmente e o baixa caso necessário.

    @param caminho_modelo: path do modelo baixado ao rodar a pipeline
    """
    # URL oficial do ONNX Model Zoo migrada para o huggingface
    url = "https://huggingface.co/onnxmodelzoo/mnist-8/resolve/main/mnist-8.onnx"
    
    if not os.path.exists(caminho_modelo):
        print("Modelo ONNX não encontrado localmente.")
        print(f"Baixando de {url} (isso vai acontecer apenas na primeira vez)...")
        try:
            urllib.request.urlretrieve(url, caminho_modelo)
            print("Download concluído com sucesso!")
        except Exception as e:
            print(f"Erro ao tentar baixar o modelo: {e}")
            return False
            
    return True

def main():
    """
    Gerencia o loop de captura de vídeo, a interface gráfica do OpenCV e a inferência.
    """
    # Garante que o modelo está baixado
    if not baixar_modelo_se_necessario("mnist-8.onnx"):
        print("Não foi possível obter o modelo. Encerrando...")
        return

    # Carrega o modelo pré-treinado
    try:
        session = ort.InferenceSession("mnist-8.onnx")
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Erro ao carregar o modelo ONNX: {e}")
        return
    # Inicia a webcam
    indice_camera = 0 
    cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"ERRO: Não foi possível acessar a câmera no índice {indice_camera}.")
        print("Verifique se ela está conectada, se não está sendo usada por outro app, ou tente outro índice.")
        return

    # Configuração inicial da janela
    cv2.namedWindow("IA vs Alunos", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("IA vs Alunos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Variável para controlar o estado da tela cheia
    estado_tela_cheia = [True]
    # Função que capta os cliques do mouse
    def evento_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Verifica se o clique foi dentro da área do botão (x entre 10 e 180, y entre 10 e 50)
            if 10 <= x <= 180 and 10 <= y <= 50:
                estado_tela_cheia[0] = not estado_tela_cheia[0] # Inverte o estado
                
                if estado_tela_cheia[0]:
                    cv2.setWindowProperty("IA vs Alunos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("IA vs Alunos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Conecta o mouse à janela
    cv2.setMouseCallback("IA vs Alunos", evento_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelhar a imagem para agir como um espelho natural
        height, width, _ = frame.shape

        # Definir a zona de escaneamento (ROI) no centro da tela
        box_size = 250
        x_start = width // 2 - box_size // 2
        y_start = height // 2 - box_size // 2
        x_end = x_start + box_size
        y_end = y_start + box_size

        # Recortar a ROI da imagem original
        roi = frame[y_start:y_end, x_start:x_end]

        # Desenhar a interface na tela principal
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(frame, "ZONA DE ESCANEAMENTO", (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Botão de tela cheia
        cv2.rectangle(frame, (10, 10), (180, 50), (220, 220, 220), -1) # Fundo do botão
        cv2.rectangle(frame, (10, 10), (180, 50), (50, 50, 50), 2)     # Borda do botão
        
        texto_botao = "MODO JANELA" if estado_tela_cheia[0] else "TELA CHEIA"
        cv2.putText(frame, texto_botao, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Processar a imagem recortada
        input_tensor, imagem_debug = processar_imagem_para_mnist(roi)

        if input_tensor is not None:
            # Fazer a predição
            results = session.run(None, {input_name: input_tensor})[0][0]
            
            # Calcular probabilidades usando Softmax para mostrar a "certeza"
            exp_scores = np.exp(results - np.max(results))
            probabilities = exp_scores / np.sum(exp_scores)
            
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities) * 100

            # Exibir o palpite e a certeza da IA apenas se tiver confiança razoável
            if confidence > 30:
                texto_resultado = f"Acho que é o numero: {prediction}"
                texto_confianca = f"Certeza: {confidence:.1f}%"
                
                # Cor dinâmica: Verde se tem certeza, Laranja se está em dúvida
                cor = (0, 255, 0) if confidence > 80 else (0, 165, 255)
                
                cv2.putText(frame, texto_resultado, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 3)
                cv2.putText(frame, texto_confianca, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

            # Exibir a "Visão da IA" no canto da tela
            # Redimensiona o 28x28 para 150x150 sem borrar (INTER_NEAREST) para dar efeito de pixel art
            debug_resized = cv2.resize(imagem_debug, (150, 150), interpolation=cv2.INTER_NEAREST)
            debug_colored = cv2.cvtColor(debug_resized, cv2.COLOR_GRAY2BGR)
            
            # Colar no canto superior direito
            frame[20:170, width-170:width-20] = debug_colored
            cv2.putText(frame, "VISAO DA IA", (width-170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar o resultado final
        cv2.imshow("IA vs Alunos", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
