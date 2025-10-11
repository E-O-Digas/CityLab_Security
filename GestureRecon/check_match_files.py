import os

# --- Configure suas pastas aqui ---
base_path = 'Gestures'
image_folders = [os.path.join(base_path, 'Images', 'Train'), os.path.join(base_path, 'Images', 'Val')]
label_folders = [os.path.join(base_path, 'Labels', 'Train'), os.path.join(base_path, 'Labels', 'Val')]
# ------------------------------------

def find_unmatched_files():
    for img_folder, lbl_folder in zip(image_folders, label_folders):
        print(f"--- Verificando a pasta: {img_folder} ---")
        
        image_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_folder) if f.lower().endswith('.txt')}

        # Imagens que não têm anotação
        images_without_labels = image_files - label_files
        if images_without_labels:
            print(f"\n[AVISO] As seguintes IMAGENS não têm um arquivo de anotação (.txt):")
            for img_name in sorted(list(images_without_labels)):
                print(f"  - {os.path.join(img_folder, img_name)}.*")

        # Anotações que não têm imagem
        labels_without_images = label_files - image_files
        if labels_without_images:
            print(f"\n[AVISO] Os seguintes ARQUIVOS DE ANOTAÇÃO não têm uma imagem correspondente:")
            for lbl_name in sorted(list(labels_without_images)):
                print(f"  - {os.path.join(lbl_folder, lbl_name)}.txt")
        
        print("\nVerificação concluída para esta pasta.\n")

if __name__ == '__main__':
    find_unmatched_files()