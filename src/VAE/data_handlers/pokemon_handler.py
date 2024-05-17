from src.VAE.utils.imports import *

POKEMON_IMAGE_PATH = "/Users/jordan/Data/pokemon_dataset/images/"
POKEMON_DATA_PATH = "/Users/jordan/Data/pokemon_dataset/pokemon.csv"

class PokemonDataset(Dataset):
    def __init__(self):
        self.image_dir = POKEMON_IMAGE_PATH
        self.image_paths = sorted(self._find_files_(self.image_dir))
        self.pokemon_df = pd.read_csv(POKEMON_DATA_PATH)
        self.pokemon_df.set_index("Name", inplace=True)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pokemon_name = image_path.split("/")[-1].replace('.png', '')
        pokemon_type = self.pokemon_df.loc[pokemon_name]["Type1"]

        x = io.imread(image_path)
        x = torch.tensor(x).type(torch.FloatTensor)
        x = x[:, :, :3]
        xmin, xmax = torch.min(x), torch.max(x)
        x_norm = (x - xmin) / (xmax - xmin)

        x = torch.reshape(x_norm, (3, 120, 120))

        return x, pokemon_type
    
    def _find_files_(self, image_dir, pattern="*.png"):
        img_path_list = []
        for root, dirnames, filenames in os.walk(image_dir):
            for filename in fnmatch.filter(filenames, pattern):
                img_path_list.append(os.path.join(root, filename))
        
        return img_path_list


if __name__ == "__main__":
    dataset = PokemonDataset()
    print(len(dataset))

    print(dataset.pokemon_df.head())


    x, name = dataset[random.randrange(0, len(dataset))]
    print(x.shape)
    print(name)

    plt.figure(figsize=(5, 5))
    plt.imshow(torch.reshape(x, (120, 120, 3)))
    plt.title(name)
    plt.show()