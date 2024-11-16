import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): train, test veya valid klasörlerinden biri.
            transform (callable, optional): Uygulanacak veri dönüşümleri (augmentation).
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.image_files = sorted(os.listdir(self.images_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        
        # Dosya sayılarının eşleştiğinden emin olun
        assert len(self.image_files) == len(self.label_files), "Görüntü ve etiket sayıları eşleşmiyor!"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Görüntü dosyasını yükle
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Etiket dosyasını yükle
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        boxes, fire_count = self.parse_label_file(label_path)
        
        # Veri dönüşümleri (augmentation) uygulanacaksa uygula
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, fire_count
    
    def parse_label_file(self, label_path):
        """
        Etiket dosyasını oku ve veriyi çıkar.
        Her satır: yangın_id x_center y_center width height
        """
        boxes = []
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                fire_id = int(parts[0])  # Yangın sayısı gibi kullanacağız
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Veriyi listeye ekle
                boxes.append([fire_id, x_center, y_center, width, height])
        
        # Yangın sayısı, etiket dosyasındaki satırların toplamıdır
        fire_count = len(boxes)
        
        # Tensor'a çevir
        return torch.tensor(boxes, dtype=torch.float32), fire_count


# Örnek kullanım
if __name__ == "__main__":
    # Dataset oluştur
    train_dataset = FireDataset(root_dir='train')
    
    # DataLoader ile batch halinde verileri çekelim
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Batch içerisindeki verileri kontrol edelim
    for images, boxes, fire_counts in train_loader:
        print("Görüntü boyutu:", images.shape)
        print("Yangın Alanları (Bounding Boxes):", boxes)
        print("Görüntüdeki Yangın Sayısı:", fire_counts)
        break
