import os
import cv2
import numpy as np

IMAGE_DIRS_TO_CHECK = ['/media/digas/Win_Dados/Faculdades/Fatec/TCC/codigos/GestureRecon/Gestures/Images/Train']
BLUR_THRESHOLD = 100.0  # Ajuste este valor. Menor que 100 é geralmente borrado.
DARK_THRESHOLD = 60    # Brilho médio abaixo disso é escuro.
BRIGHT_THRESHOLD = 200 # Brilho médio acima disso é claro demais.

def check_blurriness(image_path, threshold=100.0):
    """
    Verifica se uma imagem está potencialmente borrada.
    Retorna True se for borrada, False caso contrário.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, 0 # Não conseguiu ler a imagem
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var < threshold, laplacian_var
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return False, 0


def check_brightness(image_path, dark_threshold=70, bright_threshold=180):
    """
    Verifica o brilho médio da imagem.
    Retorna 'dark', 'bright' ou 'ok'.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 'error', 0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < dark_threshold:
            return 'dark', mean_brightness
        elif mean_brightness > bright_threshold:
            return 'bright', mean_brightness
        else:
            return 'ok', mean_brightness
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return 'error', 0

if __name__ == '__main__':
    print("Iniciando análise de qualidade das imagens...")
    
    bad_images = []

    for img_dir in IMAGE_DIRS_TO_CHECK:
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(img_dir, filename)
                
                # Checa se está borrada
                is_blurry, blur_value = check_blurriness(image_path, BLUR_THRESHOLD)
                if is_blurry:
                    report = f"[BORRADA] {image_path} (Valor: {blur_value:.2f})"
                    print(report)
                    bad_images.append(report)
                    continue # Já marcou como ruim, vai para a próxima
                
                # Checa brilho
                brightness_status, bright_value = check_brightness(image_path, DARK_THRESHOLD, BRIGHT_THRESHOLD)
                if brightness_status == 'dark':
                    report = f"[ESCURA] {image_path} (Valor: {bright_value:.2f})"
                    print(report)
                    bad_images.append(report)
                elif brightness_status == 'bright':
                    report = f"[CLARA DEMAIS] {image_path} (Valor: {bright_value:.2f})"
                    print(report)
                    bad_images.append(report)

    print("\n--- Relatório Final de Imagens com Potenciais Problemas ---")
    if not bad_images:
        print("Nenhuma imagem com problemas óbvios foi encontrada!")
    else:
        for report in bad_images:
            print(report)
    
    print("\nAnálise concluída. Revise as imagens listadas e apague-as se necessário.")
    print("Lembre-se de apagar o arquivo .txt correspondente na pasta 'Labels'!")