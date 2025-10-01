import os
import torch
import cv2

class Preprocessor:
    def rename_file(folder: str) -> None:
        filenames = sorted(os.listdir(folder))
        for i, file in enumerate(filenames):
            old_path = os.path.join(folder, file)
            ext = os.path.splitext(file)[-1]
            new_name = f'image_{i}{ext}'
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)

    def remove_class(folder: str) -> None:
        for file in os.listdir(folder):
            path_file = os.path.join(folder, file)
            content = open(path_file, 'r').read()
            if content:
                open(path_file, 'w').writelines(content[2:])

    def get_data_image(list_file_path: list) -> torch.Tensor:
        images = []
        for file_path in list_file_path:
            img_numpy = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB) #H * W * C
            images.append(torch.from_numpy(img_numpy).permute(2, 0, 1).float() / 255.0) # C * H * W
        images = torch.stack(images, dim=0)
        return images

    def get_data_label(list_file_path: list) -> torch.Tensor:
        labels = []
        for file in list_file_path:
            text = open(file, 'r').read().split()
            content = [float(number) for number in text] if text else [0.0] * 4
            labels.append(content)
        labels = torch.Tensor(labels)
        return labels
